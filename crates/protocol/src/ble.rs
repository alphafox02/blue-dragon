use crate::{freq_to_channel, Timespec};

pub const BLE_ADV_AA: u32 = 0x8E89BED6;
pub const BLE_SPS: usize = 2; // samples per symbol
pub const BLE_AA_BITS: usize = 32;
pub const BLE_AA_TLEN: usize = BLE_AA_BITS * BLE_SPS; // 64 samples
pub const BLE_CORR_THRESH: f32 = 0.6;
pub const BLE_MAX_HD: u32 = 4; // max hamming distance for AA match

// Pre-computed 127-bit whitening sequence (7-bit maximal-length LFSR, period 127)
// All 40 BLE channels use the same sequence at different offsets
static WHITENING: [u8; 127] = [
    1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1,
    0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
    0, 1,
];

// Per-channel starting offset into the whitening sequence
static WHITENING_INDEX: [u8; 40] = [
    70, 62, 120, 111, 77, 46, 15, 101, 66, 39, 31, 26, 80, 83, 125, 89, 10, 35,
    8, 54, 122, 17, 33, 0, 58, 115, 6, 94, 86, 49, 52, 20, 40, 27, 84, 90, 63,
    112, 47, 102,
];

/// Whitening bit lookup: index into pre-computed sequence at channel-specific offset
#[inline]
pub fn whitening_bit(channel: u32, bit_position: u32) -> u8 {
    WHITENING[((WHITENING_INDEX[channel as usize] as u32 + bit_position) % 127) as usize]
}

/// Reflect (bit-reverse) a 24-bit value
fn reflect24(mut v: u32) -> u32 {
    let mut result: u32 = 0;
    for _ in 0..24 {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// BLE CRC-24 implementation
/// Polynomial: x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1
/// Reflected polynomial for right-shifting CRC: 0xDA6000
/// Init: 0x555555 for advertising channels, CRC init from CONNECT_IND for data
/// Init value must be bit-reversed for the reflected CRC algorithm
pub fn crc24(data: &[u8], init: u32) -> u32 {
    let mut crc = reflect24(init & 0xFFFFFF);

    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xDA6000;
            } else {
                crc >>= 1;
            }
        }
    }

    crc & 0xFFFFFF
}

/// AA correlator state: pre-computed template for BLE advertising access address
pub struct AaCorrelator {
    template: [f32; BLE_AA_TLEN],
    template_norm: f32,
}

impl AaCorrelator {
    pub fn new() -> Self {
        let mut template = [0.0f32; BLE_AA_TLEN];
        let mut sum_sq = 0.0f32;
        let aa = BLE_ADV_AA;

        for i in 0..BLE_AA_BITS {
            let val: f32 = if (aa >> i) & 1 == 1 { 1.0 } else { -1.0 };
            for j in 0..BLE_SPS {
                template[i * BLE_SPS + j] = val;
                sum_sq += 1.0; // val^2 is always 1
            }
        }

        Self {
            template,
            template_norm: sum_sq.sqrt(),
        }
    }

    /// AA correlator: find BLE advertising packets by correlating the analog demod
    /// signal against the known access address pattern.
    pub fn correlate(
        &self,
        demod: &[f32],
        freq: u32,
        timestamp: Timespec,
        check_crc: bool,
    ) -> Option<BlePacket> {
        if demod.len() < BLE_AA_TLEN + 80 {
            return None;
        }

        // Slide template across demod, find best normalized correlation
        // Start from sample 1 to skip potentially wild first sample
        let search_end = demod.len() - BLE_AA_TLEN;
        let mut best_score: f32 = 0.0;
        let mut best_idx: usize = 0;

        for i in 1..=search_end {
            let mut dot: f32 = 0.0;
            let mut win_sq: f32 = 0.0;
            for j in 0..BLE_AA_TLEN {
                let s = demod[i + j];
                dot += s * self.template[j];
                win_sq += s * s;
            }
            let win_norm = win_sq.sqrt();
            if win_norm > 0.001 {
                let score = dot / (win_norm * self.template_norm);
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }
        }

        if best_score < BLE_CORR_THRESH {
            return None;
        }

        // Try both sample phases for bit extraction, pick lowest hamming distance
        let mut best_phase: usize = 0;
        let mut best_hd: u32 = 33;

        for phase in 0..BLE_SPS {
            let mut aa: u32 = 0;
            for i in 0..32 {
                let idx = best_idx + phase + i * BLE_SPS;
                if idx < demod.len() && demod[idx] > 0.0 {
                    aa |= 1u32 << i;
                }
            }
            let hd = (aa ^ BLE_ADV_AA).count_ones();
            if hd < best_hd {
                best_hd = hd;
                best_phase = phase;
            }
        }

        if best_hd > BLE_MAX_HD {
            return None;
        }

        // Extract bits starting at AA position with best phase
        let bit_start = best_idx + best_phase;
        let max_bits = (demod.len() - bit_start) / BLE_SPS;

        if max_bits < 32 + 16 {
            return None;
        }

        // Slice bits from analog signal
        let mut bits = vec![0u8; max_bits];
        for i in 0..max_bits {
            let idx = bit_start + i * BLE_SPS;
            bits[i] = if demod[idx] > 0.0 { 1 } else { 0 };
        }

        let channel = freq_to_channel(freq);

        // Extract length byte (PDU byte 1) with dewhitening
        let mut header_len: u8 = 0;
        for j in 0..8u32 {
            let whiten_bit = whitening_bit(channel, 8 + j);
            if (32 + 8 + j as usize) < max_bits {
                header_len |= (bits[32 + 8 + j as usize] ^ whiten_bit) << j;
            }
        }

        // Validate: BLE advertising PDU payload max 37 bytes
        let needed_bits = 32 + 16 + (header_len as usize) * 8 + 24;
        if header_len > 37 || needed_bits > max_bits {
            return None;
        }

        // Build packet data: AA(4) + header(2) + payload(header_len) + CRC(3)
        let pkt_len = 4 + 2 + header_len as usize + 3;
        let mut data = vec![0u8; pkt_len.max(64)];

        // AA bytes (LE)
        data[0] = (BLE_ADV_AA >> 0) as u8;
        data[1] = (BLE_ADV_AA >> 8) as u8;
        data[2] = (BLE_ADV_AA >> 16) as u8;
        data[3] = (BLE_ADV_AA >> 24) as u8;

        // Dewhiten and pack remaining bytes
        for i in 0..(pkt_len - 4) {
            let mut byte: u8 = 0;
            for j in 0..8u32 {
                let wb = whitening_bit(channel, i as u32 * 8 + j);
                let bit_idx = 32 + i * 8 + j as usize;
                if bit_idx < max_bits {
                    byte |= (bits[bit_idx] ^ wb) << j;
                }
            }
            data[i + 4] = byte;
        }

        // CRC validation
        let mut crc_checked = false;
        let mut crc_valid = false;
        if check_crc {
            let crc_init = 0x555555u32; // correlator only finds advertising AA
            let crc_len = pkt_len - 4 - 3;
            let computed_crc = crc24(&data[4..4 + crc_len], crc_init);
            let received_crc = data[pkt_len - 3] as u32
                | ((data[pkt_len - 2] as u32) << 8)
                | ((data[pkt_len - 1] as u32) << 16);
            crc_checked = true;
            crc_valid = computed_crc == received_crc;
        }

        Some(BlePacket {
            aa: BLE_ADV_AA,
            freq,
            len: pkt_len,
            data: data[..pkt_len].to_vec(),
            timestamp,
            rssi_db: 0,
            noise_db: 0,
            crc_checked,
            crc_valid,
            is_data: false,
            conn_valid: false,
        })
    }
}

impl Default for AaCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

/// BLE packet structure
#[derive(Debug, Clone)]
pub struct BlePacket {
    pub aa: u32,
    pub rssi_db: i32,
    pub noise_db: i32,
    pub freq: u32,
    pub len: usize,
    pub timestamp: Timespec,
    pub crc_checked: bool,
    pub crc_valid: bool,
    pub is_data: bool,
    pub conn_valid: bool,
    pub data: Vec<u8>,
}

/// Preamble-first BLE detection: checks preamble pattern, extracts AA candidates,
/// dewhitens, and validates length against burst size.
pub fn ble_burst(
    bits: &[u8],
    freq: u32,
    timestamp: Timespec,
    check_crc: bool,
    mut crc_init_fn: impl FnMut(u32) -> Option<(u32, bool)>,
) -> Option<BlePacket> {
    let bits_len = bits.len();

    // Check preamble pattern: alternating bits
    if bits_len < 48 {
        return None;
    }
    if !(bits[0] == bits[2] && bits[2] == bits[4]
        && bits[1] == bits[3] && bits[3] == bits[5])
    {
        return None;
    }

    let channel = freq_to_channel(freq);

    let mut smallest_delta: u32 = 0xffffffff;
    let mut smallest_offset: usize = 0;
    let mut smallest_aa: u32 = 0;
    let mut smallest_header_len: u8 = 0;

    // Try three candidates for AA start position
    for i in 6..9 {
        if i + 32 + 16 >= bits_len {
            continue;
        }
        let mut aa: u32 = 0;
        for j in 0..32 {
            aa |= (bits[i + j] as u32) << j;
        }
        let mut header_len: u8 = 0;
        for j in 0..8u32 {
            let whiten_bit = whitening_bit(channel, 8 + j);
            header_len |= (bits[i + 32 + 8 + j as usize] ^ whiten_bit) << j;
        }
        let bit_len = 8 + 32 + 16 + (header_len as u32) * 8 + 24;
        let delta = (bits_len as i32) - (bit_len as i32);
        if delta > 0 && (delta as u32) < smallest_delta {
            smallest_delta = delta as u32;
            smallest_offset = i;
            smallest_aa = aa;
            smallest_header_len = header_len;
        }
    }

    if smallest_delta >= 20 {
        return None;
    }

    // Build packet
    let pkt_len = 4 + 2 + smallest_header_len as usize + 3;
    let mut data = vec![0u8; pkt_len.max(64)];

    data[0] = (smallest_aa >> 0) as u8;
    data[1] = (smallest_aa >> 8) as u8;
    data[2] = (smallest_aa >> 16) as u8;
    data[3] = (smallest_aa >> 24) as u8;

    for i in 0..(pkt_len - 4) {
        let mut byte: u8 = 0;
        for j in 0..8u32 {
            let wb = whitening_bit(channel, i as u32 * 8 + j);
            let bit_idx = smallest_offset + 32 + i * 8 + j as usize;
            if bit_idx < bits_len {
                byte |= (bits[bit_idx] ^ wb) << j;
            }
        }
        data[i + 4] = byte;
    }

    let is_data = smallest_aa != BLE_ADV_AA;

    // CRC validation
    let mut crc_checked = false;
    let mut crc_valid = false;
    let mut conn_valid = false;

    if check_crc {
        if smallest_aa == BLE_ADV_AA {
            // Advertising AA: always use 0x555555
            let crc_len = pkt_len - 4 - 3;
            let computed = crc24(&data[4..4 + crc_len], 0x555555);
            let received = data[pkt_len - 3] as u32
                | ((data[pkt_len - 2] as u32) << 8)
                | ((data[pkt_len - 1] as u32) << 16);
            crc_checked = true;
            crc_valid = computed == received;
        } else {
            // Data channel: look up connection for CRC init
            if let Some((init, valid)) = crc_init_fn(smallest_aa) {
                let crc_len = pkt_len - 4 - 3;
                let computed = crc24(&data[4..4 + crc_len], init);
                let received = data[pkt_len - 3] as u32
                    | ((data[pkt_len - 2] as u32) << 8)
                    | ((data[pkt_len - 1] as u32) << 16);
                crc_checked = true;
                crc_valid = computed == received;
                conn_valid = valid;
            }
            // else: unknown connection, skip CRC check
        }
    }

    data.truncate(pkt_len);

    Some(BlePacket {
        aa: smallest_aa,
        freq,
        len: pkt_len,
        data,
        timestamp,
        rssi_db: 0,
        noise_db: 0,
        crc_checked,
        crc_valid,
        is_data,
        conn_valid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc24_known_value() {
        // CRC-24/BLE check value: 0xC25A56
        // Input: "123456789" (ASCII bytes)
        let data = b"123456789";
        let result = crc24(data, 0x555555);
        assert_eq!(result, 0xC25A56, "CRC-24/BLE check value mismatch: got 0x{:06X}", result);
    }

    #[test]
    fn test_reflect24() {
        assert_eq!(reflect24(0x555555), 0xAAAAAA);
        assert_eq!(reflect24(0xAAAAAA), 0x555555);
        assert_eq!(reflect24(0x000001), 0x800000);
    }

    #[test]
    fn test_whitening_bit_range() {
        // All values should be 0 or 1
        for ch in 0..40 {
            for bit in 0..256 {
                let wb = whitening_bit(ch, bit);
                assert!(wb <= 1, "whitening_bit({}, {}) = {}", ch, bit, wb);
            }
        }
    }

    #[test]
    fn test_aa_correlator_construction() {
        let corr = AaCorrelator::new();
        // Template norm should be sqrt(64) = 8.0
        assert!((corr.template_norm - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_freq_to_channel() {
        use crate::freq_to_channel;
        assert_eq!(freq_to_channel(2402), 37);
        assert_eq!(freq_to_channel(2426), 38);
        assert_eq!(freq_to_channel(2480), 39);
        assert_eq!(freq_to_channel(2404), 0);
        assert_eq!(freq_to_channel(2406), 1);
        assert_eq!(freq_to_channel(2424), 10);
        assert_eq!(freq_to_channel(2428), 11);
        assert_eq!(freq_to_channel(2478), 36);
    }

    /// End-to-end test: construct a known BLE advertising packet,
    /// whiten it, convert to bit array, and verify ble_burst decodes it
    /// with correct CRC.
    #[test]
    fn test_ble_burst_known_packet() {
        use crate::freq_to_channel;

        let freq = 2402u32; // advertising channel 37
        let channel = freq_to_channel(freq);
        assert_eq!(channel, 37);

        // Build a simple ADV_NONCONN_IND: type=0x02, length=12
        // AdvA = 11:22:33:44:55:66, AdvData = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]
        let pdu_header: [u8; 2] = [0x02, 0x0C]; // type=2, len=12
        let adv_a: [u8; 6] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let adv_data: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        // PDU = header + AdvA + AdvData
        let mut pdu = Vec::new();
        pdu.extend_from_slice(&pdu_header);
        pdu.extend_from_slice(&adv_a);
        pdu.extend_from_slice(&adv_data);
        assert_eq!(pdu.len(), 14);

        // Compute CRC-24 over PDU
        let crc = crc24(&pdu, 0x555555);
        let crc_bytes = [
            (crc & 0xFF) as u8,
            ((crc >> 8) & 0xFF) as u8,
            ((crc >> 16) & 0xFF) as u8,
        ];

        // Full packet bytes: AA(4) + PDU(14) + CRC(3) = 21 bytes
        let aa_bytes = [0xD6u8, 0xBE, 0x89, 0x8E]; // LE encoding of 0x8E89BED6

        // Convert to bit array (LSB first within each byte)
        let mut all_bytes = Vec::new();
        all_bytes.extend_from_slice(&aa_bytes);
        all_bytes.extend_from_slice(&pdu);
        all_bytes.extend_from_slice(&crc_bytes);

        // Convert bytes to bits (LSB first)
        let mut bits_raw = Vec::new();
        for &byte in &all_bytes {
            for j in 0..8 {
                bits_raw.push((byte >> j) & 1);
            }
        }

        // Apply whitening to PDU+CRC bits (not AA)
        let mut bits_whitened = Vec::new();
        // Preamble: 8 alternating bits (for AA starting with 0: 01010101)
        bits_whitened.extend_from_slice(&[0, 1, 0, 1, 0, 1, 0, 1]);
        // AA bits (not whitened)
        bits_whitened.extend_from_slice(&bits_raw[..32]);
        // PDU+CRC bits (whitened)
        for i in 32..bits_raw.len() {
            let wb = whitening_bit(channel, (i - 32) as u32);
            bits_whitened.push(bits_raw[i] ^ wb);
        }

        // Add a few extra trailing bits (burst catcher always produces overshoot)
        for _ in 0..16 {
            bits_whitened.push(0);
        }

        // ble_burst should find this packet
        let ts = crate::Timespec { tv_sec: 0, tv_nsec: 0 };
        let pkt = ble_burst(&bits_whitened, freq, ts, true, |_aa| None);
        assert!(pkt.is_some(), "ble_burst failed to find known packet");
        let pkt = pkt.unwrap();
        assert_eq!(pkt.aa, BLE_ADV_AA);
        assert!(pkt.crc_checked, "CRC was not checked");
        assert!(pkt.crc_valid, "CRC INVALID: packet data={:02X?}", &pkt.data);

        // Verify dewhitened data matches original
        assert_eq!(pkt.data[4], 0x02, "PDU type mismatch");
        assert_eq!(pkt.data[5], 0x0C, "PDU length mismatch");
        assert_eq!(&pkt.data[6..12], &adv_a, "AdvA mismatch");
        assert_eq!(&pkt.data[12..18], &adv_data, "AdvData mismatch");
    }

    /// Test the AA correlator with a synthetic BLE signal
    #[test]
    fn test_correlator_known_packet() {
        use crate::freq_to_channel;

        let freq = 2402u32; // channel 37
        let channel = freq_to_channel(freq);

        // Build same packet as above
        let pdu = [0x02u8, 0x0C, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
                   0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let crc = crc24(&pdu, 0x555555);
        let crc_bytes = [
            (crc & 0xFF) as u8,
            ((crc >> 8) & 0xFF) as u8,
            ((crc >> 16) & 0xFF) as u8,
        ];

        // Build full packet bytes
        let aa_bytes = [0xD6u8, 0xBE, 0x89, 0x8E];
        let mut all_bytes = Vec::new();
        all_bytes.extend_from_slice(&aa_bytes);
        all_bytes.extend_from_slice(&pdu);
        all_bytes.extend_from_slice(&crc_bytes);

        // Convert to bits (LSB first)
        let mut bits_raw = Vec::new();
        for &byte in &all_bytes {
            for j in 0..8 {
                bits_raw.push((byte >> j) & 1);
            }
        }

        // Apply whitening to PDU+CRC
        let mut bits_pdu_whitened = Vec::new();
        for i in 0..bits_raw.len() {
            if i < 32 {
                bits_pdu_whitened.push(bits_raw[i]); // AA not whitened
            } else {
                let wb = whitening_bit(channel, (i - 32) as u32);
                bits_pdu_whitened.push(bits_raw[i] ^ wb);
            }
        }

        // Build a synthetic demod signal: +1.0 for bit 1, -1.0 for bit 0
        // Each bit repeated BLE_SPS=2 times
        // Prepend some preamble + silence
        let mut demod = Vec::new();
        // 20 samples of silence
        for _ in 0..20 {
            demod.push(0.0f32);
        }
        // Preamble (8 bits * 2 sps = 16 samples)
        let preamble = [0u8, 1, 0, 1, 0, 1, 0, 1];
        for &b in &preamble {
            let val = if b == 1 { 1.0 } else { -1.0 };
            for _ in 0..BLE_SPS {
                demod.push(val);
            }
        }
        // Whitened packet bits
        for &b in &bits_pdu_whitened {
            let val = if b == 1 { 1.0 } else { -1.0 };
            for _ in 0..BLE_SPS {
                demod.push(val);
            }
        }
        // Trail with some zeros
        for _ in 0..40 {
            demod.push(0.0f32);
        }

        let corr = AaCorrelator::new();
        let ts = crate::Timespec { tv_sec: 0, tv_nsec: 0 };
        let pkt = corr.correlate(&demod, freq, ts, true);
        assert!(pkt.is_some(), "correlator failed to find known packet");
        let pkt = pkt.unwrap();
        assert_eq!(pkt.aa, BLE_ADV_AA);
        assert!(pkt.crc_checked, "CRC was not checked");
        assert!(pkt.crc_valid, "CRC INVALID on correlator: data={:02X?}", &pkt.data);
    }
}
