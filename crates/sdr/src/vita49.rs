// Copyright 2026 CEMAXECUTER LLC
//
// VITA 49 (VRT) UDP input - receives IQ samples via VRT signal/context packets.
//
// Auto-detects sample rate, center frequency, and IQ format from VRT IF context
// packets (type 0x4). Falls back to user-supplied values if no context arrives.
//
// Pull-based: recv_into_i16() blocks until samples are available, matching the
// pattern used by USRP/HackRF/bladeRF backends.
//
// Usage: -i vita49 or -i vita49:192.168.1.100:5000
// Default bind: 0.0.0.0:4991

use std::net::UdpSocket;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const DEFAULT_PORT: u16 = 4991;
const VRT_MAX_PKT: usize = 65536;

// VRT packet types
const VRT_TYPE_SIGNAL_DATA_NO_SID: u8 = 0x0;
const VRT_TYPE_SIGNAL_DATA: u8 = 0x1;
const VRT_TYPE_IF_CONTEXT: u8 = 0x4;
const VRT_TYPE_EXT_CONTEXT: u8 = 0x5;

// CIF0 field bit positions (VITA 49.0 Section 7.1.5)
const CIF_RF_REF_FREQ: u32 = 1 << 27;
const CIF_SAMPLE_RATE: u32 = 1 << 21;
const CIF_DATA_FORMAT: u32 = 1 << 15;

/// CIF0 field table: (bit flag, size in 32-bit words), MSB-first order.
const CIF_FIELDS: &[(u32, u32)] = &[
    (1 << 31, 1), // change indicator
    (1 << 30, 1), // reference point ID
    (1 << 29, 2), // bandwidth
    (1 << 28, 2), // IF reference frequency
    (CIF_RF_REF_FREQ, 2), // RF reference frequency
    (1 << 26, 2), // RF/IF frequency offset
    (1 << 25, 2), // IF band offset
    (1 << 24, 1), // reference level
    (1 << 23, 1), // gain
    (1 << 22, 1), // over-range count
    (CIF_SAMPLE_RATE, 2), // sample rate
    (1 << 20, 2), // timestamp adjustment
    (1 << 19, 1), // timestamp calibration time
    (1 << 18, 1), // temperature
    (1 << 17, 2), // device identifier
    (1 << 16, 1), // state/event indicators
    (CIF_DATA_FORMAT, 2), // data packet payload format
];

use crate::IqFormat;

pub struct Vita49Handle {
    sock: UdpSocket,
    pkt_buf: Vec<u8>,
    /// Residual i16 samples from previous packet not yet consumed
    residual: Vec<i16>,
    residual_offset: usize,
    /// Reusable buffer for format conversion when payload exceeds remaining buf space
    convert_buf: Vec<i16>,
    sample_rate: u32,
    center_freq: u64,
    iq_format: IqFormat,
    gap_count: AtomicU64,
    last_seq: u8,
    have_seq: bool,
}

impl Vita49Handle {
    /// Open a VITA 49 UDP receiver.
    ///
    /// `iface` format: "vita49", "vita49:PORT", or "vita49:IP:PORT"
    /// `sample_rate` and `center_freq_hz` are defaults; overridden by context packets.
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq_hz: u64,
        _gain: f64,
    ) -> Result<Self, String> {
        let (bind_addr, port) = parse_endpoint(iface)?;
        let bind = format!("{}:{}", bind_addr, port);

        let sock = UdpSocket::bind(&bind)
            .map_err(|e| format!("vita49: bind {}: {}", bind, e))?;

        // 500ms timeout so recv_into_i16 returns periodically for stop checks
        sock.set_read_timeout(Some(Duration::from_millis(500)))
            .map_err(|e| format!("vita49: set_read_timeout: {}", e))?;

        // 4 MB receive buffer for bursty traffic
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = sock.as_raw_fd();
            let requested: libc::c_int = 4 * 1024 * 1024;
            unsafe {
                libc::setsockopt(
                    fd,
                    libc::SOL_SOCKET,
                    libc::SO_RCVBUF,
                    &requested as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                );
                let mut actual: libc::c_int = 0;
                let mut len: libc::socklen_t =
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t;
                libc::getsockopt(
                    fd,
                    libc::SOL_SOCKET,
                    libc::SO_RCVBUF,
                    &mut actual as *mut _ as *mut libc::c_void,
                    &mut len,
                );
                if actual < requested {
                    eprintln!(
                        "vita49: WARNING: SO_RCVBUF {} bytes (requested {}). \
                         Increase with: sudo sysctl net.core.rmem_max={}",
                        actual, requested, requested
                    );
                }
            }
        }

        eprintln!("vita49: listening on {}", bind);

        Ok(Self {
            sock,
            pkt_buf: vec![0u8; VRT_MAX_PKT],
            residual: Vec::with_capacity(8192),
            residual_offset: 0,
            convert_buf: vec![0i16; VRT_MAX_PKT / 2],
            sample_rate,
            center_freq: center_freq_hz,
            iq_format: IqFormat::Ci16,
            gap_count: AtomicU64::new(0),
            last_seq: 0,
            have_seq: false,
        })
    }

    /// Receive IQ samples into caller's i16 buffer. Returns number of complex
    /// samples written (buf filled with interleaved I,Q,I,Q,...).
    /// Blocks until at least one packet arrives or timeout.
    pub fn recv_into_i16(&mut self, buf: &mut [i16]) -> usize {
        let mut written = 0usize;

        // Drain residual first (use offset to avoid memmove)
        let residual_remaining = self.residual.len() - self.residual_offset;
        if residual_remaining > 0 {
            let n = residual_remaining.min(buf.len());
            buf[..n].copy_from_slice(
                &self.residual[self.residual_offset..self.residual_offset + n],
            );
            self.residual_offset += n;
            if self.residual_offset == self.residual.len() {
                self.residual.clear();
                self.residual_offset = 0;
            }
            written = n;
            if written >= buf.len() {
                return written / 2;
            }
        }

        // Keep receiving until buffer is full or timeout
        loop {
            let len = match self.sock.recv(&mut self.pkt_buf) {
                Ok(n) => n,
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    break;
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {
                    continue;
                }
                Err(_) => break,
            };

            let (vrt_start, vrt_len) = vrl_strip(&self.pkt_buf[..len]);
            if vrt_len < 4 {
                continue;
            }
            let vrt_end = vrt_start + vrt_len;

            let w0 = u32::from_be_bytes([
                self.pkt_buf[vrt_start],
                self.pkt_buf[vrt_start + 1],
                self.pkt_buf[vrt_start + 2],
                self.pkt_buf[vrt_start + 3],
            ]);
            let pkt_type = ((w0 >> 28) & 0xF) as u8;

            // Handle context packets (parse without copying pkt_buf)
            if pkt_type == VRT_TYPE_IF_CONTEXT || pkt_type == VRT_TYPE_EXT_CONTEXT {
                if let Some(ctx) = parse_context(&self.pkt_buf[vrt_start..vrt_end]) {
                    self.apply_context(ctx);
                }
                continue;
            }

            // Parse signal data
            let vrt_slice = &self.pkt_buf[vrt_start..vrt_end];
            let (payload_off, payload_bytes) = match parse_signal_header(vrt_slice) {
                Some(v) => v,
                None => continue,
            };

            // Sequence gap detection
            let count = ((w0 >> 16) & 0xF) as u8;
            if self.have_seq {
                let expected = (self.last_seq + 1) & 0xF;
                if count != expected {
                    self.gap_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.last_seq = count;
            self.have_seq = true;

            let payload_start = vrt_start + payload_off;
            let payload_end = payload_start + payload_bytes;
            let payload = &self.pkt_buf[payload_start..payload_end];

            // Calculate how many i16 values this payload produces
            let total_values = match self.iq_format {
                IqFormat::Ci8 => payload.len(),
                IqFormat::Ci16 => payload.len() / 2,
                IqFormat::Cf32 => payload.len() / 4,
            };
            if total_values == 0 {
                continue;
            }

            let space = buf.len() - written;

            if total_values <= space {
                // Fast path: convert directly into caller's buf (no intermediate alloc)
                let dst = &mut buf[written..written + total_values];
                match self.iq_format {
                    IqFormat::Ci8 => { convert_ci8_into(payload, dst); }
                    IqFormat::Ci16 => { convert_ci16_be_into(payload, dst); }
                    IqFormat::Cf32 => { convert_cf32_be_into(payload, dst); }
                }
                written += total_values;
            } else {
                // Overflow: fill remaining buf, spill rest into residual
                let dst = &mut buf[written..];
                let direct = match self.iq_format {
                    IqFormat::Ci8 => convert_ci8_into(payload, dst),
                    IqFormat::Ci16 => convert_ci16_be_into(payload, dst),
                    IqFormat::Cf32 => convert_cf32_be_into(payload, dst),
                };
                written += direct;

                // Convert the overflow portion into convert_buf, then save as residual
                let overflow_values = total_values - direct;
                if overflow_values > 0 {
                    let overflow_payload = match self.iq_format {
                        IqFormat::Ci8 => &payload[direct..],
                        IqFormat::Ci16 => &payload[direct * 2..],
                        IqFormat::Cf32 => &payload[direct * 4..],
                    };
                    if self.convert_buf.len() < overflow_values {
                        self.convert_buf.resize(overflow_values, 0);
                    }
                    let n = match self.iq_format {
                        IqFormat::Ci8 => convert_ci8_into(overflow_payload, &mut self.convert_buf),
                        IqFormat::Ci16 => convert_ci16_be_into(overflow_payload, &mut self.convert_buf),
                        IqFormat::Cf32 => convert_cf32_be_into(overflow_payload, &mut self.convert_buf),
                    };
                    self.residual.clear();
                    self.residual_offset = 0;
                    self.residual.extend_from_slice(&self.convert_buf[..n]);
                }
            }

            if written >= buf.len() {
                break;
            }
        }

        written / 2 // return complex sample count
    }

    pub fn max_samps(&self) -> usize {
        // VRT packets are typically 1-8K samples; 16K is a safe max
        16384
    }

    pub fn overflow_count(&self) -> u64 {
        self.gap_count.load(Ordering::Relaxed)
    }

    pub fn set_gain(&self, _gain: f64) {
        // no-op: VITA 49 source controls gain externally
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn center_frequency(&self) -> u64 {
        self.center_freq
    }


    /// Apply auto-detected metadata from a VRT context packet.
    /// Logs on first detection and on any subsequent change.
    fn apply_context(&mut self, ctx: ContextInfo) {
        if let Some(sr) = ctx.sample_rate {
            let new_rate = sr as u32;
            if new_rate != self.sample_rate {
                self.sample_rate = new_rate;
                eprintln!("vita49: auto-detected sample_rate={} Hz ({} Msps)",
                    sr as u64, sr as u64 / 1_000_000);
            }
        }
        if let Some(freq) = ctx.center_freq {
            let new_freq = freq as u64;
            if new_freq != self.center_freq {
                self.center_freq = new_freq;
                eprintln!("vita49: auto-detected rf_freq={} Hz ({} MHz)",
                    new_freq, new_freq / 1_000_000);
            }
        }
        if let Some(fmt) = ctx.format {
            if fmt != self.iq_format {
                self.iq_format = fmt;
                eprintln!("vita49: auto-detected format={:?}", fmt);
            }
        }
    }
}

/// Metadata auto-detected from VRT context packets
struct ContextInfo {
    sample_rate: Option<f64>,
    center_freq: Option<f64>,
    format: Option<IqFormat>,
}

/// Strip VRL wrapper ("VRLP" magic). Returns (vrt_offset, vrt_len).
/// VRL frame: 8-byte header ("VRLP" + frame size) + VRT packet + 4-byte trailer.
fn vrl_strip(pkt: &[u8]) -> (usize, usize) {
    if pkt.len() >= 12 {
        let magic = u32::from_be_bytes([pkt[0], pkt[1], pkt[2], pkt[3]]);
        if magic == 0x56524C50 {
            let vrt_start = 8;
            let vrt_len = pkt.len() - 8 - 4; // exclude header and trailer
            return (vrt_start, vrt_len);
        }
    }
    (0, pkt.len())
}

/// Parse VRT signal data header. Returns (payload_offset, payload_bytes).
fn parse_signal_header(vrt: &[u8]) -> Option<(usize, usize)> {
    if vrt.len() < 4 {
        return None;
    }

    let w0 = u32::from_be_bytes([vrt[0], vrt[1], vrt[2], vrt[3]]);
    let pkt_type = ((w0 >> 28) & 0xF) as u8;
    let class_id = (w0 >> 27) & 1;
    let trailer = (w0 >> 26) & 1;
    let tsi = (w0 >> 22) & 0x3;
    let tsf = (w0 >> 20) & 0x3;
    let pkt_size_w = (w0 & 0xFFFF) as usize;

    if pkt_type != VRT_TYPE_SIGNAL_DATA_NO_SID && pkt_type != VRT_TYPE_SIGNAL_DATA {
        return None;
    }

    let pkt_size_bytes = pkt_size_w * 4;
    if pkt_size_bytes > vrt.len() || pkt_size_bytes < 4 {
        return None;
    }

    let mut hdr_words: usize = 1;
    if pkt_type == VRT_TYPE_SIGNAL_DATA {
        hdr_words += 1;
    }
    if class_id != 0 {
        hdr_words += 2;
    }
    if tsi != 0 {
        hdr_words += 1;
    }
    if tsf != 0 {
        hdr_words += 2;
    }

    let trailer_words = if trailer != 0 { 1 } else { 0 };
    if hdr_words + trailer_words >= pkt_size_w {
        return None;
    }

    let payload_words = pkt_size_w - hdr_words - trailer_words;
    Some((hdr_words * 4, payload_words * 4))
}

/// Parse VRT IF context packet for sample rate, RF frequency, and data format.
fn parse_context(vrt: &[u8]) -> Option<ContextInfo> {
    if vrt.len() < 8 {
        return None;
    }

    let w0 = u32::from_be_bytes([vrt[0], vrt[1], vrt[2], vrt[3]]);
    let class_id = (w0 >> 27) & 1;
    let tsi = (w0 >> 22) & 0x3;
    let tsf = (w0 >> 20) & 0x3;
    let pkt_size_w = (w0 & 0xFFFF) as usize;

    if pkt_size_w * 4 > vrt.len() || pkt_size_w < 2 {
        return None;
    }

    let mut off_w: usize = 1;
    off_w += 1; // stream ID (always present in context)
    if class_id != 0 {
        off_w += 2;
    }
    if tsi != 0 {
        off_w += 1;
    }
    if tsf != 0 {
        off_w += 2;
    }

    if off_w >= pkt_size_w {
        return None;
    }

    let cif0 = read_be32(vrt, off_w);
    off_w += 1;

    let mut info = ContextInfo {
        sample_rate: None,
        center_freq: None,
        format: None,
    };

    for &(bit, words) in CIF_FIELDS {
        if cif0 & bit == 0 {
            continue;
        }
        if off_w + words as usize > pkt_size_w {
            return None;
        }

        if bit == CIF_RF_REF_FREQ {
            // 64-bit signed fixed-point Hz, 20-bit fraction (VITA 49.0 section 7.1.5.6)
            let val = read_be64_signed(vrt, off_w);
            info.center_freq = Some(val as f64 / (1i64 << 20) as f64);
        } else if bit == CIF_SAMPLE_RATE {
            // 64-bit signed fixed-point Hz, 20-bit fraction (VITA 49.0 section 7.1.5.12)
            let val = read_be64_signed(vrt, off_w);
            info.sample_rate = Some(val as f64 / (1i64 << 20) as f64);
        } else if bit == CIF_DATA_FORMAT {
            let fmt_w1 = read_be32(vrt, off_w);
            info.format = parse_data_format(fmt_w1);
        }

        off_w += words as usize;
    }

    if info.sample_rate.is_some() || info.center_freq.is_some() || info.format.is_some() {
        Some(info)
    } else {
        None
    }
}

fn parse_data_format(fmt_word1: u32) -> Option<IqFormat> {
    let real_cpx = (fmt_word1 >> 29) & 0x3;
    let dif = (fmt_word1 >> 24) & 0x1F;
    let data_bits = (fmt_word1 & 0x1F) + 1;

    if real_cpx != 1 {
        return None;
    }

    match dif {
        0 => match data_bits {
            8 => Some(IqFormat::Ci8),
            16 => Some(IqFormat::Ci16),
            _ => None,
        },
        14 => Some(IqFormat::Cf32),
        _ => None,
    }
}

// IQ conversion: delegate to shared crate::convert functions
use crate::convert::{ci8_to_i16 as convert_ci8_into, ci16_be_to_i16 as convert_ci16_be_into, cf32_be_to_i16 as convert_cf32_be_into};

fn read_be32(data: &[u8], word_offset: usize) -> u32 {
    let off = word_offset * 4;
    u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Read signed 64-bit big-endian value from two consecutive 32-bit words.
/// VITA 49 frequency and sample rate fields are signed fixed-point.
fn read_be64_signed(data: &[u8], word_offset: usize) -> i64 {
    let hi = read_be32(data, word_offset) as u64;
    let lo = read_be32(data, word_offset + 1) as u64;
    ((hi << 32) | lo) as i64
}

fn parse_endpoint(iface: &str) -> Result<(String, u16), String> {
    let rest = iface.strip_prefix("vita49").unwrap_or(iface);

    if rest.is_empty() {
        return Ok(("0.0.0.0".to_string(), DEFAULT_PORT));
    }

    let rest = rest.strip_prefix(':').ok_or_else(|| {
        format!("invalid vita49 endpoint: '{}' (use vita49, vita49:PORT, or vita49:IP:PORT)", iface)
    })?;

    if let Ok(port) = rest.parse::<u16>() {
        return Ok(("0.0.0.0".to_string(), port));
    }

    if let Some((ip, port_str)) = rest.rsplit_once(':') {
        let port = port_str
            .parse::<u16>()
            .map_err(|_| format!("invalid port in vita49 endpoint: '{}'", iface))?;
        return Ok((ip.to_string(), port));
    }

    Err(format!(
        "invalid vita49 endpoint: '{}' (use vita49, vita49:PORT, or vita49:IP:PORT)",
        iface
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_endpoint_default() {
        let (addr, port) = parse_endpoint("vita49").unwrap();
        assert_eq!(addr, "0.0.0.0");
        assert_eq!(port, DEFAULT_PORT);
    }

    #[test]
    fn test_parse_endpoint_port_only() {
        let (addr, port) = parse_endpoint("vita49:5000").unwrap();
        assert_eq!(addr, "0.0.0.0");
        assert_eq!(port, 5000);
    }

    #[test]
    fn test_parse_endpoint_ip_port() {
        let (addr, port) = parse_endpoint("vita49:192.168.1.100:4991").unwrap();
        assert_eq!(addr, "192.168.1.100");
        assert_eq!(port, 4991);
    }

    #[test]
    fn test_parse_endpoint_invalid() {
        assert!(parse_endpoint("vita49xyz").is_err());
    }

    #[test]
    fn test_vrl_strip_no_wrapper() {
        let pkt = [0x10, 0x00, 0x00, 0x04]; // VRT signal data, 4 words
        let (off, len) = vrl_strip(&pkt);
        assert_eq!(off, 0);
        assert_eq!(len, 4);
    }

    #[test]
    fn test_vrl_strip_with_wrapper() {
        // VRLP magic + 4 bytes frame size + VRT data + 4 byte trailer
        let mut pkt = vec![0x56, 0x52, 0x4C, 0x50]; // "VRLP"
        pkt.extend_from_slice(&[0x00, 0x00, 0x00, 0x10]); // frame size
        pkt.extend_from_slice(&[0x10, 0x00, 0x00, 0x04]); // VRT header
        pkt.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // trailer
        let (off, len) = vrl_strip(&pkt);
        assert_eq!(off, 8);
        assert_eq!(len, 4); // 16 total - 8 header - 4 trailer = 4
    }

    #[test]
    fn test_parse_signal_header_type0() {
        // Type 0 (no stream ID), no class, no timestamp, 4 words total
        let mut pkt = vec![0u8; 16]; // 4 words
        let w0: u32 = (0x0 << 28) | 4; // type=0, size=4
        pkt[0..4].copy_from_slice(&w0.to_be_bytes());
        let (off, payload) = parse_signal_header(&pkt).unwrap();
        assert_eq!(off, 4);   // 1 header word
        assert_eq!(payload, 12); // 3 payload words
    }

    #[test]
    fn test_parse_signal_header_type1_with_ts() {
        // Type 1 (stream ID), TSI=1, TSF=1, 10 words total
        let mut pkt = vec![0u8; 40]; // 10 words
        let w0: u32 = (0x1 << 28) | (1 << 22) | (1 << 20) | 10; // type=1, tsi=1, tsf=1, size=10
        pkt[0..4].copy_from_slice(&w0.to_be_bytes());
        let (off, payload) = parse_signal_header(&pkt).unwrap();
        // header: 1 (base) + 1 (stream_id) + 1 (tsi) + 2 (tsf) = 5 words = 20 bytes
        assert_eq!(off, 20);
        assert_eq!(payload, 20); // 5 payload words
    }

    #[test]
    fn test_parse_signal_header_rejects_context() {
        let mut pkt = vec![0u8; 16];
        let w0: u32 = (0x4 << 28) | 4; // type=4 (IF context)
        pkt[0..4].copy_from_slice(&w0.to_be_bytes());
        assert!(parse_signal_header(&pkt).is_none());
    }

    #[test]
    fn test_parse_context_sample_rate() {
        // Construct a minimal IF context packet with sample rate = 40 MHz
        // Type 4, stream_id present, CIF0 with only sample_rate bit set
        let pkt_size_w: u32 = 5; // header(1) + stream_id(1) + cif0(1) + sample_rate(2)
        let w0: u32 = (0x4 << 28) | pkt_size_w;
        let stream_id: u32 = 0x12345678;
        let cif0: u32 = CIF_SAMPLE_RATE;
        // 40 MHz = 40_000_000 Hz, fixed-point with 20-bit fraction: 40_000_000 << 20
        let rate_fixed: i64 = 40_000_000i64 << 20;

        let mut pkt = Vec::new();
        pkt.extend_from_slice(&w0.to_be_bytes());
        pkt.extend_from_slice(&stream_id.to_be_bytes());
        pkt.extend_from_slice(&cif0.to_be_bytes());
        pkt.extend_from_slice(&(rate_fixed as u64).to_be_bytes());

        let ctx = parse_context(&pkt).unwrap();
        let sr = ctx.sample_rate.unwrap();
        assert!((sr - 40_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_context_rf_freq() {
        // RF frequency = 2441 MHz
        let pkt_size_w: u32 = 5;
        let w0: u32 = (0x4 << 28) | pkt_size_w;
        let stream_id: u32 = 0;
        let cif0: u32 = CIF_RF_REF_FREQ;
        let freq_fixed: i64 = 2_441_000_000i64 << 20;

        let mut pkt = Vec::new();
        pkt.extend_from_slice(&w0.to_be_bytes());
        pkt.extend_from_slice(&stream_id.to_be_bytes());
        pkt.extend_from_slice(&cif0.to_be_bytes());
        pkt.extend_from_slice(&(freq_fixed as u64).to_be_bytes());

        let ctx = parse_context(&pkt).unwrap();
        let freq = ctx.center_freq.unwrap();
        assert!((freq - 2_441_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_data_format_ci16() {
        // real_cpx=1 (complex), dif=0 (signed int), data_bits=16 (item_size=15)
        let word: u32 = (1 << 29) | (0 << 24) | 15;
        assert_eq!(parse_data_format(word), Some(IqFormat::Ci16));
    }

    #[test]
    fn test_parse_data_format_cf32() {
        // real_cpx=1 (complex), dif=14 (float32), data_bits=32 (item_size=31)
        let word: u32 = (1 << 29) | (14 << 24) | 31;
        assert_eq!(parse_data_format(word), Some(IqFormat::Cf32));
    }

    #[test]
    fn test_parse_data_format_ci8() {
        let word: u32 = (1 << 29) | (0 << 24) | 7;
        assert_eq!(parse_data_format(word), Some(IqFormat::Ci8));
    }

    #[test]
    fn test_parse_data_format_real_rejected() {
        // real_cpx=0 (real, not complex) should be rejected
        let word: u32 = (0 << 29) | (0 << 24) | 15;
        assert_eq!(parse_data_format(word), None);
    }

    #[test]
    fn test_convert_ci8_into() {
        let payload = [0x7Fu8, 0x80, 0x00, 0xFF]; // 127, -128, 0, -1
        let mut dst = [0i16; 4];
        let n = convert_ci8_into(&payload, &mut dst);
        assert_eq!(n, 4);
        assert_eq!(dst[0], 127 << 8);   // 32512
        assert_eq!(dst[1], -128 << 8);  // -32768
        assert_eq!(dst[2], 0);
        assert_eq!(dst[3], -1 << 8);    // -256
    }

    #[test]
    fn test_convert_ci16_be_into() {
        // Two i16 values: 0x1234 and 0xFEDC (big-endian)
        let payload = [0x12, 0x34, 0xFE, 0xDC];
        let mut dst = [0i16; 2];
        let n = convert_ci16_be_into(&payload, &mut dst);
        assert_eq!(n, 2);
        assert_eq!(dst[0], 0x1234);
        assert_eq!(dst[1], -292i16); // 0xFEDC as i16
    }

    #[test]
    fn test_convert_cf32_be_into() {
        // float32 1.0 in big-endian: 0x3F800000
        let payload = [0x3F, 0x80, 0x00, 0x00];
        let mut dst = [0i16; 1];
        let n = convert_cf32_be_into(&payload, &mut dst);
        assert_eq!(n, 1);
        assert_eq!(dst[0], 32767); // 1.0 * 32767
    }

    #[test]
    fn test_read_be64_signed_positive() {
        // 40 MHz in fixed-point: 40_000_000 << 20
        let val: i64 = 40_000_000i64 << 20;
        let bytes = (val as u64).to_be_bytes();
        let mut data = vec![0u8; 8];
        data.copy_from_slice(&bytes);
        let result = read_be64_signed(&data, 0);
        assert_eq!(result, val);
    }

    #[test]
    fn test_parse_context_with_preceding_fields() {
        // Context with bandwidth (2 words) + IF ref freq (2 words) + RF ref freq (2 words)
        // CIF0 bits: bandwidth(29) + IF_ref(28) + RF_ref(27)
        let cif0: u32 = (1 << 29) | (1 << 28) | CIF_RF_REF_FREQ;
        let pkt_size_w: u32 = 3 + 2 + 2 + 2; // hdr+sid+cif0 + bw + if_freq + rf_freq
        let w0: u32 = (0x4 << 28) | pkt_size_w;

        let mut pkt = Vec::new();
        pkt.extend_from_slice(&w0.to_be_bytes());         // header
        pkt.extend_from_slice(&0u32.to_be_bytes());        // stream_id
        pkt.extend_from_slice(&cif0.to_be_bytes());        // CIF0
        pkt.extend_from_slice(&0u64.to_be_bytes());        // bandwidth (skip)
        pkt.extend_from_slice(&0u64.to_be_bytes());        // IF ref freq (skip)
        let freq_fixed: i64 = 2_402_000_000i64 << 20;
        pkt.extend_from_slice(&(freq_fixed as u64).to_be_bytes()); // RF ref freq

        let ctx = parse_context(&pkt).unwrap();
        let freq = ctx.center_freq.unwrap();
        assert!((freq - 2_402_000_000.0).abs() < 1.0);
        assert!(ctx.sample_rate.is_none()); // not in CIF0
    }
}
