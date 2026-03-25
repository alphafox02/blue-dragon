// Copyright 2026 CEMAXECUTER LLC
//
// bladeRF BLE FPGA backend: hybrid FPGA+host BLE decoder.
//
// The FPGA handles wideband DSP: PFB channelizer (80 Msps, 64 bins),
// FM discriminator (CORDIC), and burst detection.  During active bursts,
// raw FM demod samples are streamed to the host over USB as 128-bit words.
//
// The host performs protocol-level decoding: interpolation resampling
// (SPS 1.25 -> 1.0), bit slicing, access address search, data whitening
// removal, and CRC-24 verification.  This avoids the SPS=1.25 limitation
// of the FPGA bit slicer (no timing recovery in fixed-point VHDL).
//
// Packet format (128-bit words, header first):
//   Word 0: Header
//     [127:120] magic    = 0xBE
//     [119:112] channel  (0-63, = PFB bin index directly; 64-point FFT)
//     [111:104] phy      (0=1M, 1=2M, 2=Coded)
//     [103:96]  flags    (reserved)
//     [95:80]   length   (PDU byte count)
//     [79:72]   rssi     (signed dBm)
//     [71]      crc_ok
//     [70:64]   reserved
//     [63:56]   aa_errors
//     [55:0]    reserved
//   Word 1..N: PDU bytes (16 bytes per word, last word zero-padded)

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bd_protocol::ble::{BlePacket, BlePhy};
use bd_protocol::Timespec;

type BladerfDevice = c_void;

const BLADERF_MODULE_RX: c_int = 0;
const BLADERF_CHANNEL_RX_0: c_int = 0;
const BLADERF_FORMAT_SC16_Q11: c_int = 0;
const BLADERF_FORMAT_SC8_Q7: c_int = 4;

const FPGA_MAGIC: u8 = 0xBE;

// 128-bit word = 16 bytes
const WORD_SIZE: usize = 16;

// Max burst FM samples: ~600 samples for a max-length BLE packet at SPS=1.25
// ceil(600*2 / 16) = 75 payload words + 1 header
const MAX_PKT_WORDS: usize = 76;

// Max BLE channel: 0-39 (FPGA maps PFB bins to BLE channel numbers)
const MAX_BLE_CHAN: u8 = 39;

// BLE advertising access address
const BLE_ADV_AA: u32 = 0x8E89BED6;

// ---------------------------------------------------------------------------
// Software BLE demodulator for FPGA burst FM samples
// ---------------------------------------------------------------------------

/// Demodulate a burst of FM samples into BLE packets.
/// Returns decoded packets (may be 0 or 1 per burst).
fn demod_burst(fm_samples: &[i16], ble_channel: u8) -> Vec<DemodPacket> {
    if fm_samples.len() < 60 {
        return Vec::new(); // Too short for preamble + AA + min PDU
    }

    // Step 1: CFO correction via arithmetic mean.
    // The DC offset of the FM output equals the carrier frequency offset.
    // Subtracting the mean centers the GFSK swing around zero.
    let mean: f64 = fm_samples.iter().map(|&s| s as f64).sum::<f64>() / fm_samples.len() as f64;
    let mut corrected = Vec::with_capacity(fm_samples.len());
    for &s in fm_samples {
        corrected.push(s as f64 - mean);
    }

    // Step 2+3: Try all 5 possible starting phases for the resampler.
    // The burst detector fires at an unknown phase relative to symbol timing.
    // Exhaustive search over 5 phase offsets finds the correct alignment.
    let sps_numer = 5u32;
    let sps_denom = 4u32;
    let target = bit_reverse_32(BLE_ADV_AA);
    let mut results = Vec::new();
    let mut global_best_errors = 32u32;

    for phase_offset in 0..sps_numer {
        // Resample with this starting phase
        let mut phase = phase_offset;
        let mut bits = Vec::with_capacity(fm_samples.len());
        let mut prev = corrected[0];

        for &cur in &corrected[1..] {
            phase += sps_denom;
            if phase >= sps_numer {
                let frac = (sps_numer as f64 - (phase - sps_denom) as f64) / sps_denom as f64;
                let interp = prev * (1.0 - frac) + cur * frac;
                bits.push(if interp > 0.0 { 1u8 } else { 0u8 });
                phase -= sps_numer;
            }
            prev = cur;
        }

        if bits.len() < 40 { continue; } // Need preamble + AA minimum

        // AA search
        let mut sr: u32 = 0;
        for i in 0..bits.len() {
            sr = (sr << 1) | (bits[i] as u32);
            if i < 31 { continue; }

            let xor = sr ^ target;
            let errors = xor.count_ones();
            if errors < global_best_errors {
                global_best_errors = errors;
            }
            if errors <= 2 {
                let pdu_start = i + 1;
                if let Some(pkt) = decode_pdu(&bits[pdu_start..], ble_channel, errors as u8) {
                    results.push(pkt);
                    // Found a valid AA match at this phase -- don't search further phases
                    // to avoid duplicate detections
                    break;
                }
            }
        }
        if !results.is_empty() { break; }
    }

    // Debug: report best AA match quality per burst
    static DEBUG_CNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let cnt = DEBUG_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if cnt < 200 && (ble_channel == 37 || global_best_errors <= 4) {
        eprintln!("    [demod ch{}] best_aa_err={} found={} samples={}",
            ble_channel, global_best_errors, results.len(), fm_samples.len());
    }

    results
}

/// Decoded packet from software demod (before building BlePacket)
struct DemodPacket {
    pdu_bytes: Vec<u8>,
    crc_ok: bool,
    crc_computed: u32,
    crc_received: u32,
    aa_errors: u8,
}

/// Decode PDU + CRC from bit stream after AA match.
fn decode_pdu(bits: &[u8], ble_channel: u8, aa_errors: u8) -> Option<DemodPacket> {
    if bits.len() < 16 {
        return None; // Need at least 2 header bytes
    }

    // Initialize whitening LFSR (BLE spec: d0=1, d1..d6=channel)
    let mut lfsr: u8 = ((ble_channel & 0x3F) << 1) | 1;

    // Dewhiten and assemble bytes, feeding dewhitened bits to CRC
    let mut crc: u32 = 0xAAAAAA; // Reflected init
    let mut pdu_bytes = Vec::new();
    let mut byte_val: u8 = 0;
    let mut bit_idx = 0;

    // We don't know the length yet -- read header first (2 bytes = 16 bits)
    let max_bits = bits.len().min(258 * 8 + 24); // max PDU + CRC

    for i in 0..max_bits {
        if i >= bits.len() { break; }

        // Dewhiten
        let whiten_bit = lfsr & 1;
        let dewhitened = bits[i] ^ whiten_bit;

        // Step LFSR (Fibonacci: feedback = d0 XOR d4)
        let feedback = (lfsr & 1) ^ ((lfsr >> 4) & 1);
        lfsr = (feedback << 6) | (lfsr >> 1);

        // Check if we're in PDU or CRC region
        let pdu_len = if pdu_bytes.len() >= 2 {
            pdu_bytes[1] as usize + 2 // length field + 2 header bytes
        } else {
            258 // unknown, assume max
        };

        let pdu_bits = pdu_len * 8;
        let crc_end = pdu_bits + 24;

        if i < pdu_bits {
            // PDU region: feed to CRC and assemble byte
            // CRC: reflected LFSR, poly 0xDA6000
            let fb = ((crc & 1) ^ (dewhitened as u32)) as u32;
            crc >>= 1;
            if fb != 0 {
                crc ^= 0xDA6000;
            }

            // Assemble byte LSB-first
            byte_val |= dewhitened << (bit_idx & 7);
            bit_idx += 1;
            if bit_idx & 7 == 0 {
                pdu_bytes.push(byte_val);
                byte_val = 0;

                // Check length after 2 header bytes
                if pdu_bytes.len() == 2 {
                    let length = pdu_bytes[1] as usize;
                    if length > 255 {
                        return None;
                    }
                }
            }
        } else if i < crc_end {
            // CRC region: collect received CRC bits
            // (CRC engine continues to process -- if all correct, result = 0)
            let fb = ((crc & 1) ^ (dewhitened as u32)) as u32;
            crc >>= 1;
            if fb != 0 {
                crc ^= 0xDA6000;
            }
        } else {
            break;
        }
    }

    if pdu_bytes.len() < 2 {
        return None;
    }

    // CRC check: if all bits are correct, the CRC register should be 0
    let crc_ok = crc == 0;

    Some(DemodPacket {
        pdu_bytes,
        crc_ok,
        crc_computed: crc,
        crc_received: 0, // We use self-checking property
        aa_errors,
    })
}

fn bit_reverse_32(mut v: u32) -> u32 {
    let mut r: u32 = 0;
    for _ in 0..32 {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

fn ble_chan_to_freq_mhz(ch: u8) -> u32 {
    match ch {
        37 => 2402,
        38 => 2426,
        39 => 2480,
        c if c <= 10 => 2404 + (c as u32) * 2,
        c if c <= 36 => 2406 + (c as u32) * 2,
        _ => 2402,
    }
}


extern "C" {
    fn bladerf_open(device: *mut *mut BladerfDevice, identifier: *const c_char) -> c_int;
    fn bladerf_close(device: *mut BladerfDevice);
    fn bladerf_set_frequency(dev: *mut BladerfDevice, ch: c_int, frequency: u64) -> c_int;
    fn bladerf_set_bandwidth(
        dev: *mut BladerfDevice,
        ch: c_int,
        bandwidth: c_uint,
        actual: *mut c_uint,
    ) -> c_int;
    fn bladerf_set_gain(dev: *mut BladerfDevice, ch: c_int, gain: c_int) -> c_int;
    fn bladerf_set_gain_mode(dev: *mut BladerfDevice, ch: c_int, mode: c_int) -> c_int;
    fn bladerf_set_sample_rate(
        dev: *mut BladerfDevice,
        ch: c_int,
        rate: c_uint,
        actual: *mut c_uint,
    ) -> c_int;
    fn bladerf_sync_config(
        dev: *mut BladerfDevice,
        layout: c_int,
        format: c_int,
        num_buffers: c_uint,
        buffer_size: c_uint,
        num_transfers: c_uint,
        stream_timeout: c_uint,
    ) -> c_int;
    fn bladerf_sync_rx(
        dev: *mut BladerfDevice,
        samples: *mut c_void,
        num_samples: c_uint,
        metadata: *mut c_void,
        timeout_ms: c_uint,
    ) -> c_int;
    fn bladerf_enable_module(dev: *mut BladerfDevice, ch: c_int, enable: bool) -> c_int;
    fn bladerf_enable_feature(dev: *mut BladerfDevice, feature: c_uint, enable: bool) -> c_int;
}

const BLADERF_GAIN_MGC: c_int = 1;
const BLADERF_FEATURE_OVERSAMPLE: c_uint = 1;

/// Convert BLE channel number (0-39) to center frequency in MHz.
/// Handle for reading burst FM samples from a bladeRF with BLE FPGA image.
pub struct BladerfBleHandle {
    dev: *mut BladerfDevice,
    pub running: Arc<AtomicBool>,
    // Receive buffer: we receive SC16 "samples" from libbladeRF but they're
    // actually 128-bit packet words packed as 4 x i16 pairs (8 x i16 = 16 bytes).
    rx_buf: Vec<i16>,
    buf_size: usize,
    // Residual bytes from previous recv that didn't complete a packet
    residual: Vec<u8>,
    pkt_count: u64,
    overflow_count: u64,
}

unsafe impl Send for BladerfBleHandle {}

impl BladerfBleHandle {
    /// Open bladeRF for BLE FPGA packet reception.
    ///
    /// The FPGA image must already be loaded (the BLE channelizer image).
    /// `sample_rate` and `center_freq` configure the AD9361 for the BLE band.
    /// The FPGA processes the IQ internally; we only read decoded packets.
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: i32,
    ) -> Result<Self, String> {
        // The BLE FPGA image requires 80 Msps oversample mode (8-bit SC8_Q7).
        // The 64-point PFB gives 1.25 MHz bins covering all 40 BLE channels.
        // The FPGA's ADC input path applies shift_right(8) to undo the AD9361's
        // left-justification of 8-bit samples, which is only correct in oversample
        // mode. Running at other sample rates would produce garbage.
        if sample_rate != 80_000_000 {
            return Err(format!(
                "bladerf-ble requires 80 Msps (-C 80), got {} Msps. \
                 The FPGA BLE channelizer needs oversample mode for full 40-channel coverage.",
                sample_rate / 1_000_000
            ));
        }

        let instance = parse_instance(iface)
            .ok_or_else(|| format!("invalid bladerf-ble interface: '{}'", iface))?;

        let identifier = CString::new(format!("*:instance={}", instance))
            .map_err(|e| format!("CString error: {}", e))?;

        unsafe {
            let mut dev: *mut BladerfDevice = ptr::null_mut();
            let r = bladerf_open(&mut dev, identifier.as_ptr());
            if r != 0 {
                return Err(format!("bladerf_open failed: {}", r));
            }

            // Enable oversample mode for 80 Msps (8-bit SC8 on AD9361)
            let use_sc8 = if sample_rate > 61_440_000 {
                let r = bladerf_enable_feature(dev, BLADERF_FEATURE_OVERSAMPLE, true);
                if r != 0 {
                    eprintln!("WARNING: bladerf_enable_feature(OVERSAMPLE) failed: {}", r);
                }
                r == 0
            } else {
                false
            };
            let format = if use_sc8 {
                eprintln!("  oversample mode: SC8_Q7");
                BLADERF_FORMAT_SC8_Q7
            } else {
                eprintln!("  normal mode: SC16_Q11");
                let bw = sample_rate.min(56_000_000);
                bladerf_set_bandwidth(dev, BLADERF_CHANNEL_RX_0, bw, ptr::null_mut());
                BLADERF_FORMAT_SC16_Q11
            };

            bladerf_set_frequency(dev, BLADERF_CHANNEL_RX_0, center_freq);
            let mut actual_rate: c_uint = 0;
            bladerf_set_sample_rate(
                dev,
                BLADERF_CHANNEL_RX_0,
                sample_rate,
                &mut actual_rate,
            );
            eprintln!("  actual sample rate: {} Hz", actual_rate);
            bladerf_set_gain_mode(dev, BLADERF_CHANNEL_RX_0, BLADERF_GAIN_MGC);
            bladerf_set_gain(dev, BLADERF_CHANNEL_RX_0, gain);

            // Small buffers for sparse packet data. The FPGA only writes to
            // the FIFO when a BLE packet is decoded, so large buffers never
            // fill and USB transfers time out.
            // Short timeout (500ms) so the recv loop checks Ctrl-C promptly.
            let buf_size: c_uint = 1024;
            let r = bladerf_sync_config(
                dev,
                BLADERF_MODULE_RX,
                format,
                2,
                buf_size,
                1,
                1000,
            );
            if r != 0 {
                bladerf_close(dev);
                return Err(format!("bladerf_sync_config failed: {}", r));
            }

            let r = bladerf_enable_module(dev, BLADERF_MODULE_RX, true);
            if r != 0 {
                bladerf_close(dev);
                return Err(format!("bladerf_enable_module failed: {}", r));
            }

            log::info!(
                "bladeRF-BLE open (instance={}, {} MHz, {} MS/s, gain={} dB)",
                instance,
                center_freq / 1_000_000,
                sample_rate / 1_000_000,
                gain,
            );

            Ok(Self {
                dev,
                running: Arc::new(AtomicBool::new(true)),
                rx_buf: vec![0i16; buf_size as usize * 2],
                buf_size: buf_size as usize,
                residual: Vec::with_capacity(MAX_PKT_WORDS * WORD_SIZE),
                pkt_count: 0,
                overflow_count: 0,
            })
        }
    }

    /// Receive decoded BLE packets from the FPGA.
    ///
    /// Calls bladerf_sync_rx to get a buffer of 128-bit words, then scans
    /// for packet headers (magic 0xBE) and assembles complete packets.
    /// Returns a Vec of decoded BlePacket structs.
    pub fn recv_packets(&mut self) -> Vec<BlePacket> {
        let n = unsafe {
            bladerf_sync_rx(
                self.dev,
                self.rx_buf.as_mut_ptr() as *mut c_void,
                self.buf_size as c_uint,
                ptr::null_mut(),
                500,
            )
        };
        if n != 0 {
            return Vec::new();
        }

        // Convert i16 buffer to raw bytes (the FPGA packs data as raw bits,
        // not actual SC16 samples)
        let raw_bytes: Vec<u8> = self.rx_buf[..self.buf_size * 2]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();

        // Prepend any residual bytes from previous call
        let mut data = Vec::with_capacity(self.residual.len() + raw_bytes.len());
        data.extend_from_slice(&self.residual);
        data.extend_from_slice(&raw_bytes);
        self.residual.clear();

        let mut packets = Vec::new();
        let mut pos = 0;

        while pos + WORD_SIZE <= data.len() {
            // Scan for magic byte at word boundary
            if data[pos + 15] != FPGA_MAGIC {
                // Not a header word -- skip to next word
                // (128-bit words are 16 bytes; header magic is at byte offset 15
                // because VHDL packs MSB-first: bits[127:120] = byte[15])
                pos += WORD_SIZE;
                continue;
            }

            // Parse header word (big-endian bit layout from FPGA)
            // Byte layout in little-endian USB transfer:
            //   byte[0]     = bin_idx(7:2) | pad(1:0)
            //   byte[1..3]  = crc_received (24 bits, LE)
            //   byte[4..6]  = crc_computed (24 bits, LE)
            //   byte[7]     = aa_errors
            //   byte[8]     = crc_ok(bit7) | reserved(bits 6:0)
            //   byte[9]     = rssi (signed)
            //   byte[10..11] = length (16-bit LE)
            //   byte[12]    = flags
            //   byte[13]    = phy
            //   byte[14]    = channel
            //   byte[15]    = magic (0xBE)
            let channel = data[pos + 14];
            let phy_byte = data[pos + 13];
            let length = u16::from_le_bytes([data[pos + 10], data[pos + 11]]) as usize;
            let rssi = data[pos + 9] as i8;
            let crc_ok = (data[pos + 8] & 0x80) != 0;

            // CRC diagnostic fields
            let bin_idx = (data[pos] >> 2) & 0x3F;
            let crc_received = u32::from_le_bytes([data[pos + 1], data[pos + 2], data[pos + 3], 0]) & 0xFFFFFF;
            let crc_computed = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], 0]) & 0xFFFFFF;

            // Heartbeat diagnostic: phy=0xFF is a heartbeat marker from FPGA
            if phy_byte == 0xFF {
                // Heartbeat: header + 1 payload word with debug counters
                let total = 2 * WORD_SIZE;
                if pos + total <= data.len() {
                    let pay = pos + WORD_SIZE;
                    // Counters are little-endian 32-bit (Altera DCFIFO 128->32 reads LSBs first)
                    let pfb_cnt = u32::from_le_bytes([data[pay+12], data[pay+13], data[pay+14], data[pay+15]]);
                    let burst_cnt = u32::from_le_bytes([data[pay+8], data[pay+9], data[pay+10], data[pay+11]]);
                    let fpga_pkt_cnt = u32::from_le_bytes([data[pay+4], data[pay+5], data[pay+6], data[pay+7]]);
                    // Diagnostics: ADC non-zero count and AA match count
                    let adc_nz = u16::from_le_bytes([data[pay+2], data[pay+3]]);
                    let aa_cnt = u16::from_le_bytes([data[pay+0], data[pay+1]]);
                    let seq = data[pos + 12]; // flags field = seq#
                    eprintln!("  [FPGA heartbeat #{}] pfb_valid={}, burst_start={}, pkt_eop={}, adc_nz={}, aa_found={}",
                        seq, pfb_cnt, burst_cnt, fpga_pkt_cnt, adc_nz, aa_cnt);
                    pos += total;
                } else {
                    self.residual.extend_from_slice(&data[pos..]);
                    break;
                }
                continue;
            }

            // Validate header: length is in bytes (FM samples * 2)
            // Max burst: ~600 samples * 2 = 1200 bytes
            if channel > MAX_BLE_CHAN || length == 0 || length > 1200 {
                pos += WORD_SIZE;
                continue;
            }

            // Calculate number of payload words: ceil(length / 16)
            let payload_words = (length + WORD_SIZE - 1) / WORD_SIZE;
            let total_bytes = (1 + payload_words) * WORD_SIZE;

            if pos + total_bytes > data.len() {
                // Incomplete burst -- save residual for next call
                self.residual.extend_from_slice(&data[pos..]);
                break;
            }

            // Extract FM samples from payload (16-bit LE signed, serialized as byte pairs)
            let payload_start = pos + WORD_SIZE;
            let num_samples = length / 2;
            let mut fm_samples = Vec::with_capacity(num_samples);
            for i in 0..num_samples {
                let lo = data[payload_start + i * 2] as u16;
                let hi = data[payload_start + i * 2 + 1] as u16;
                fm_samples.push((lo | (hi << 8)) as i16);
            }

            // Software demodulation: CFO correction, interpolation, AA search, CRC
            // Debug: show burst stats
            if self.pkt_count < 200 && num_samples > 20 {
                // Histogram: count samples in ranges
                let mut neg_big = 0i32; // < -8000
                let mut neg_sm = 0i32;  // -8000..0
                let mut pos_sm = 0i32;  // 0..8000
                let mut pos_big = 0i32; // > 8000
                let mut sum: f64 = 0.0;
                for &s in &fm_samples {
                    sum += s as f64;
                    if s < -8000 { neg_big += 1; }
                    else if s < 0 { neg_sm += 1; }
                    else if s < 8000 { pos_sm += 1; }
                    else { pos_big += 1; }
                }
                let mean = sum / num_samples as f64;
                eprintln!("  [burst] ch={} bin={} n={} mean={:.0} hist: <-8k:{} -8k..0:{} 0..8k:{} >8k:{}",
                    channel, bin_idx, num_samples, mean, neg_big, neg_sm, pos_sm, pos_big);
            }

            let demod_results = demod_burst(&fm_samples, channel);

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();

            for dpkt in demod_results {
                let freq = ble_chan_to_freq_mhz(channel);
                let pdu_len = dpkt.pdu_bytes.len();

                // Build raw packet: AA + PDU + CRC placeholder
                let mut raw = Vec::with_capacity(4 + pdu_len + 3);
                raw.extend_from_slice(&BLE_ADV_AA.to_le_bytes());
                raw.extend_from_slice(&dpkt.pdu_bytes);
                raw.extend_from_slice(&[0x00, 0x00, 0x00]);

                let pkt = BlePacket {
                    aa: BLE_ADV_AA,
                    rssi_db: rssi as i32,
                    noise_db: -100,
                    freq,
                    len: raw.len(),
                    timestamp: Timespec {
                        tv_sec: now.as_secs() as u64,
                        tv_nsec: now.subsec_nanos() as u64,
                    },
                    crc_checked: true,
                    crc_valid: dpkt.crc_ok,
                    is_data: channel <= 36,
                    conn_valid: false,
                    phy: BlePhy::Phy1M,
                    ext_header: None,
                    data: raw,
                };

                if self.pkt_count < 200 {
                    eprintln!("  [demod] ch={} len={} crc_ok={} freq={} MHz bin={} aa_err={} fm_samples={}",
                        channel, pdu_len, dpkt.crc_ok, freq, bin_idx, dpkt.aa_errors, num_samples);
                }
                packets.push(pkt);
                self.pkt_count += 1;
            }
            pos += total_bytes;
        }

        // Save any remaining partial data as residual
        if pos < data.len() && self.residual.is_empty() {
            self.residual.extend_from_slice(&data[pos..]);
        }

        packets
    }

    pub fn pkt_count(&self) -> u64 {
        self.pkt_count
    }

    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    pub fn set_gain(&self, gain: f64) {
        unsafe {
            bladerf_set_gain(self.dev, BLADERF_CHANNEL_RX_0, gain as c_int);
        }
    }
}

impl Drop for BladerfBleHandle {
    fn drop(&mut self) {
        unsafe {
            // Signal stop first, then disable module to unblock any pending sync_rx
            self.running.store(false, std::sync::atomic::Ordering::Relaxed);
            bladerf_enable_module(self.dev, BLADERF_MODULE_RX, false);
            bladerf_close(self.dev);
        }
    }
}

fn parse_instance(iface: &str) -> Option<u32> {
    // "bladerf-ble" -> 0 (default)
    // "bladerf-ble0" -> 0
    // "bladerf-ble1" -> 1
    let suffix = iface.strip_prefix("bladerf-ble")?;
    if suffix.is_empty() {
        Some(0)
    } else {
        suffix.parse().ok()
    }
}
