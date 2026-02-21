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
/// AA correlator state: pre-computed template for BLE advertising access address.
/// Parameterized by samples-per-symbol to support LE 1M (SPS=2) and LE 2M (SPS=1).
pub struct AaCorrelator {
    template: Vec<f32>,
    template_norm: f32,
    sps: usize,
}

impl AaCorrelator {
    /// Create a correlator for the given samples-per-symbol rate.
    /// SPS=2 for LE 1M (default), SPS=1 for LE 2M.
    pub fn with_sps(sps: usize) -> Self {
        let tlen = BLE_AA_BITS * sps;
        let mut template = vec![0.0f32; tlen];
        let mut sum_sq = 0.0f32;
        let aa = BLE_ADV_AA;

        for i in 0..BLE_AA_BITS {
            let val: f32 = if (aa >> i) & 1 == 1 { 1.0 } else { -1.0 };
            for j in 0..sps {
                template[i * sps + j] = val;
                sum_sq += 1.0;
            }
        }

        Self {
            template,
            template_norm: sum_sq.sqrt(),
            sps,
        }
    }

    pub fn new() -> Self {
        Self::with_sps(BLE_SPS)
    }

    /// AA correlator: find BLE advertising packets by correlating the analog demod
    /// signal against the known access address pattern.
    /// `max_pdu_len`: 37 for legacy advertising, 251 for extended (LE 2M).
    /// `phy`: PHY type to set on the resulting packet.
    pub fn correlate(
        &self,
        demod: &[f32],
        freq: u32,
        timestamp: Timespec,
        check_crc: bool,
    ) -> Option<BlePacket> {
        self.correlate_with_params(demod, freq, timestamp, check_crc, 37, BlePhy::Phy1M)
    }

    /// AA correlator for LE 2M: allows extended PDU length.
    pub fn correlate_2m(
        &self,
        demod: &[f32],
        freq: u32,
        timestamp: Timespec,
        check_crc: bool,
    ) -> Option<BlePacket> {
        self.correlate_with_params(demod, freq, timestamp, check_crc, 251, BlePhy::Phy2M)
    }

    fn correlate_with_params(
        &self,
        demod: &[f32],
        freq: u32,
        timestamp: Timespec,
        check_crc: bool,
        max_pdu_len: u8,
        phy: BlePhy,
    ) -> Option<BlePacket> {
        let tlen = self.template.len();
        if demod.len() < tlen + 80 {
            return None;
        }

        // Slide template across demod, find best normalized correlation
        let search_end = demod.len() - tlen;
        let mut best_score: f32 = 0.0;
        let mut best_idx: usize = 0;

        for i in 1..=search_end {
            let mut dot: f32 = 0.0;
            let mut win_sq: f32 = 0.0;
            for j in 0..tlen {
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

        // Try sample phases for bit extraction, pick lowest hamming distance
        let mut best_phase: usize = 0;
        let mut best_hd: u32 = 33;

        for phase in 0..self.sps {
            let mut aa: u32 = 0;
            for i in 0..32 {
                let idx = best_idx + phase + i * self.sps;
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
        let max_bits = (demod.len() - bit_start) / self.sps;

        if max_bits < 32 + 16 {
            return None;
        }

        let mut bits = vec![0u8; max_bits];
        for i in 0..max_bits {
            let idx = bit_start + i * self.sps;
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

        let needed_bits = 32 + 16 + (header_len as usize) * 8 + 24;
        if header_len > max_pdu_len || needed_bits > max_bits {
            return None;
        }

        // Build packet data: AA(4) + header(2) + payload(header_len) + CRC(3)
        let pkt_len = 4 + 2 + header_len as usize + 3;
        let mut data = vec![0u8; pkt_len.max(64)];

        data[0] = (BLE_ADV_AA >> 0) as u8;
        data[1] = (BLE_ADV_AA >> 8) as u8;
        data[2] = (BLE_ADV_AA >> 16) as u8;
        data[3] = (BLE_ADV_AA >> 24) as u8;

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
            let crc_init = 0x555555u32;
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
            phy,
            ext_header: None,
        })
    }
}

impl Default for AaCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

/// BLE PHY type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlePhy {
    Phy1M,
    Phy2M,
    PhyCoded,
}

impl Default for BlePhy {
    fn default() -> Self {
        BlePhy::Phy1M
    }
}

/// AuxPtr: pointer to secondary advertising channel
#[derive(Debug, Clone, Copy)]
pub struct AuxPtr {
    pub channel: u8,      // 0-39
    pub offset_usec: u32, // offset to next packet in microseconds
    pub phy: BlePhy,      // PHY of secondary channel
}

/// Extended Advertising header (Common Extended Header Format)
/// Parsed from ADV_EXT_IND (PDU type 7) and AUX_ADV_IND payloads
#[derive(Debug, Clone)]
pub struct ExtAdvHeader {
    pub adv_mode: u8,                // 0=non-conn/non-scan, 1=connectable, 2=scannable
    pub adv_a: Option<[u8; 6]>,      // advertiser address
    pub target_a: Option<[u8; 6]>,   // target address
    pub aux_ptr: Option<AuxPtr>,     // pointer to secondary channel
    pub adi: Option<u16>,            // advertising data info (DID + SID)
    pub tx_power: Option<i8>,        // TX power in dBm
}

/// Parse the Common Extended Header from an ADV_EXT_IND PDU.
/// `pdu` starts at the first byte after AA (i.e., PDU header byte 0).
/// Returns None if the PDU type is not 7 or if parsing fails.
pub fn parse_ext_adv(pdu: &[u8]) -> Option<ExtAdvHeader> {
    // PDU header: byte 0 = type(4 bits) | ChSel | TxAdd | RxAdd | rfu
    //             byte 1 = length
    if pdu.len() < 3 {
        return None;
    }

    let pdu_type = pdu[0] & 0x0F;
    if pdu_type != 7 {
        return None; // not ADV_EXT_IND
    }

    let pdu_len = pdu[1] as usize;
    if pdu_len < 1 || pdu.len() < 2 + pdu_len {
        return None;
    }

    // Extended header starts at pdu[2]
    let ext_hdr = &pdu[2..2 + pdu_len];

    // First byte: extended header length (6 bits) + AdvMode (2 bits)
    let ext_hdr_len = (ext_hdr[0] & 0x3F) as usize;
    let adv_mode = (ext_hdr[0] >> 6) & 0x03;

    if ext_hdr_len < 1 || ext_hdr.len() < 1 + ext_hdr_len {
        return None;
    }

    // Flags byte determines which optional fields are present
    let flags = ext_hdr[1];
    let mut pos: usize = 2; // skip ext_hdr_len byte and flags byte

    // Fields appear in fixed order when their flag bit is set:
    // bit 0: AdvA (6 bytes)
    // bit 1: TargetA (6 bytes)
    // bit 2: CTEInfo (1 byte) -- skip
    // bit 3: ADI (2 bytes)
    // bit 4: AuxPtr (3 bytes)
    // bit 5: SyncInfo (18 bytes) -- skip
    // bit 6: TxPower (1 byte)

    let end = 1 + ext_hdr_len; // bounds within ext_hdr

    let adv_a = if flags & 0x01 != 0 {
        if pos + 6 > end { return None; }
        let mut addr = [0u8; 6];
        addr.copy_from_slice(&ext_hdr[pos..pos + 6]);
        pos += 6;
        Some(addr)
    } else {
        None
    };

    let target_a = if flags & 0x02 != 0 {
        if pos + 6 > end { return None; }
        let mut addr = [0u8; 6];
        addr.copy_from_slice(&ext_hdr[pos..pos + 6]);
        pos += 6;
        Some(addr)
    } else {
        None
    };

    // CTEInfo (bit 2)
    if flags & 0x04 != 0 {
        if pos + 1 > end { return None; }
        pos += 1;
    }

    let adi = if flags & 0x08 != 0 {
        if pos + 2 > end { return None; }
        let val = ext_hdr[pos] as u16 | ((ext_hdr[pos + 1] as u16) << 8);
        pos += 2;
        Some(val)
    } else {
        None
    };

    let aux_ptr = if flags & 0x10 != 0 {
        if pos + 3 > end { return None; }
        let b0 = ext_hdr[pos] as u32;
        let b1 = ext_hdr[pos + 1] as u32;
        let b2 = ext_hdr[pos + 2] as u32;
        let raw = b0 | (b1 << 8) | (b2 << 16);
        pos += 3;

        let channel = (raw & 0x3F) as u8;
        let ca = ((raw >> 6) & 0x01) as u8;
        let _ = ca; // clock accuracy, not used
        let offset_units = (raw >> 7) & 0x01; // 0=30us, 1=300us
        let aux_offset = (raw >> 8) & 0x1FFF;
        let aux_phy = (raw >> 21) & 0x07;

        let offset_usec = if offset_units == 0 {
            aux_offset * 30
        } else {
            aux_offset * 300
        };

        let phy = match aux_phy {
            0 => BlePhy::Phy1M,
            1 => BlePhy::Phy2M,
            2 => BlePhy::PhyCoded,
            _ => BlePhy::Phy1M, // reserved, default to 1M
        };

        Some(AuxPtr { channel, offset_usec, phy })
    } else {
        None
    };

    // SyncInfo (bit 5)
    if flags & 0x20 != 0 {
        if pos + 18 > end { return None; }
        pos += 18;
    }

    let tx_power = if flags & 0x40 != 0 {
        if pos + 1 > end { return None; }
        let val = ext_hdr[pos] as i8;
        pos += 1;
        let _ = pos;
        Some(val)
    } else {
        None
    };

    Some(ExtAdvHeader {
        adv_mode,
        adv_a,
        target_a,
        aux_ptr,
        adi,
        tx_power,
    })
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
    pub phy: BlePhy,
    pub ext_header: Option<ExtAdvHeader>,
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

    // Parse extended advertising header for ADV_EXT_IND (PDU type 7)
    let ext_header = if crc_valid && data.len() > 4 {
        parse_ext_adv(&data[4..])
    } else {
        None
    };

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
        phy: BlePhy::Phy1M,
        ext_header,
    })
}

/// LE 2M preamble-first BLE detection.
/// Same logic as ble_burst() but adapted for LE 2M PHY:
/// - 16-bit preamble (alternating, vs 8-bit for LE 1M)
/// - Bits are at SPS=1 (re-sliced from the 2 Msps demod)
/// - Max PDU length 251 bytes (extended advertising)
pub fn ble_burst_2m(
    bits: &[u8],
    freq: u32,
    timestamp: Timespec,
    check_crc: bool,
    mut crc_init_fn: impl FnMut(u32) -> Option<(u32, bool)>,
) -> Option<BlePacket> {
    let bits_len = bits.len();

    // Check 16-bit preamble: alternating bits (need at least 12 matching)
    if bits_len < 64 {
        return None;
    }
    if !(bits[0] == bits[2] && bits[2] == bits[4] && bits[4] == bits[6]
        && bits[1] == bits[3] && bits[3] == bits[5] && bits[5] == bits[7]
        && bits[8] == bits[10] && bits[9] == bits[11]
        && bits[0] != bits[1])
    {
        return None;
    }

    let channel = freq_to_channel(freq);

    let mut smallest_delta: u32 = 0xffffffff;
    let mut smallest_offset: usize = 0;
    let mut smallest_aa: u32 = 0;
    let mut smallest_header_len: u8 = 0;

    // Try candidates for AA start position (after 16-bit preamble)
    for i in 14..18 {
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
        // LE 2M: preamble is 16 bits
        let bit_len = 16 + 32 + 16 + (header_len as u32) * 8 + 24;
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

    // Max PDU length 251 for extended advertising
    if smallest_header_len > 251 {
        return None;
    }

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

    let mut crc_checked = false;
    let mut crc_valid = false;
    let mut conn_valid = false;

    if check_crc {
        if smallest_aa == BLE_ADV_AA {
            let crc_len = pkt_len - 4 - 3;
            let computed = crc24(&data[4..4 + crc_len], 0x555555);
            let received = data[pkt_len - 3] as u32
                | ((data[pkt_len - 2] as u32) << 8)
                | ((data[pkt_len - 1] as u32) << 16);
            crc_checked = true;
            crc_valid = computed == received;
        } else {
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
        }
    }

    data.truncate(pkt_len);

    let ext_header = if crc_valid && data.len() > 4 {
        parse_ext_adv(&data[4..])
    } else {
        None
    };

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
        phy: BlePhy::Phy2M,
        ext_header,
    })
}

/// LE Coded PHY preamble: 80 symbols of repeating `00111100` (10 repetitions).
/// At SPS=2 the preamble is 160 samples.
const CODED_PREAMBLE_PATTERN: [u8; 8] = [0, 0, 1, 1, 1, 1, 0, 0];
const CODED_PREAMBLE_REPS: usize = 10;
const CODED_PREAMBLE_SYMBOLS: usize = 8 * CODED_PREAMBLE_REPS; // 80

/// Find LE Coded preamble in demod signal.
/// Returns the sample index where the preamble starts, or None.
pub fn find_coded_preamble(demod: &[f32], sps: usize) -> Option<usize> {
    let preamble_samples = CODED_PREAMBLE_SYMBOLS * sps;
    if demod.len() < preamble_samples + 100 {
        return None;
    }

    // Build template
    let mut template = Vec::with_capacity(preamble_samples);
    for _ in 0..CODED_PREAMBLE_REPS {
        for &bit in &CODED_PREAMBLE_PATTERN {
            let val = if bit == 1 { 1.0f32 } else { -1.0 };
            for _ in 0..sps {
                template.push(val);
            }
        }
    }

    let template_norm: f32 = (template.len() as f32).sqrt();
    let search_end = (demod.len() - preamble_samples).min(preamble_samples * 2);

    let mut best_score: f32 = 0.0;
    let mut best_idx: usize = 0;

    for i in 0..search_end {
        let mut dot: f32 = 0.0;
        let mut win_sq: f32 = 0.0;
        for j in 0..preamble_samples {
            let s = demod[i + j];
            dot += s * template[j];
            win_sq += s * s;
        }
        let win_norm = win_sq.sqrt();
        if win_norm > 0.001 {
            let score = dot / (win_norm * template_norm);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
    }

    if best_score > 0.5 {
        Some(best_idx)
    } else {
        None
    }
}

/// LE Coded PHY burst decoder.
///
/// Structure:
/// - Preamble: 80 symbols (`00111100` x10)
/// - FEC Block 1 (S=8): AA(32 bits) + CI(2 bits) + TERM1(3 bits) = 37 data bits
///   -> conv encode -> 74 coded bits -> pattern map S=8 -> 592 symbols
/// - FEC Block 2 (S=CI): PDU + CRC(24 bits) + TERM2(3 bits)
///   -> conv encode -> pattern map at S=2 or S=8
///
/// PCAP format for LE Coded: AA(4) + CI(1) + PDU + CRC(3)
pub fn ble_coded_burst(
    demod: &[f32],
    freq: u32,
    timestamp: Timespec,
    sps: usize,
    check_crc: bool,
    mut crc_init_fn: impl FnMut(u32) -> Option<(u32, bool)>,
) -> Option<BlePacket> {
    use crate::fec;

    // Find preamble
    let preamble_start = find_coded_preamble(demod, sps)?;
    let preamble_samples = CODED_PREAMBLE_SYMBOLS * sps;

    // FEC Block 1 starts after preamble
    let block1_start = preamble_start + preamble_samples;

    // FEC Block 1: 37 data bits -> 74 coded bits -> 592 symbols at S=8
    let block1_symbols = 74 * 8; // 592
    let block1_samples = block1_symbols * sps;

    if block1_start + block1_samples > demod.len() {
        return None;
    }

    // Extract soft symbols for Block 1
    let block1_soft: Vec<f32> = (0..block1_symbols)
        .map(|i| {
            let idx = block1_start + i * sps;
            if sps == 2 {
                (demod[idx] + demod[idx + 1]) / 2.0
            } else {
                demod[idx]
            }
        })
        .collect();

    // Soft demap at S=8, then Viterbi decode
    let soft_coded_block1 = fec::pattern_demap_s8(&block1_soft);
    let decoded_block1 = fec::viterbi_decode(&soft_coded_block1);

    // decoded_block1 should be 37 bits: AA(32) + CI(2) + TERM1(3)
    if decoded_block1.len() < 34 {
        return None;
    }

    // Extract AA (32 bits, LSB first within each byte)
    let mut aa: u32 = 0;
    for i in 0..32 {
        aa |= (decoded_block1[i] as u32) << i;
    }

    // Verify AA: must be advertising AA or a known connection AA
    // For coded PHY, only advertising AA is expected on primary channels
    let aa_hd = (aa ^ BLE_ADV_AA).count_ones();

    if aa_hd > BLE_MAX_HD {
        return None;
    }
    let aa = BLE_ADV_AA; // snap to advertising AA if close enough

    // Extract CI (2 bits): coding indicator
    let ci_val = decoded_block1[32] | (decoded_block1[33] << 1);
    let coding_s = match ci_val {
        0 => 8usize, // S=8 (125 kbps)
        1 => 2usize, // S=2 (500 kbps)
        _ => return None, // CI=2,3 are reserved
    };

    // FEC Block 2 starts after Block 1
    let block2_start = block1_start + block1_samples;

    // We don't know the PDU length yet. Extract enough symbols for max PDU (251) + CRC(3) + TERM2(3).
    // At S=2: (257 + 3) * 8 * 2 / 2 = up to ~2080 symbols
    // At S=8: much more. Limit to available samples.
    let remaining_samples = if block2_start < demod.len() {
        demod.len() - block2_start
    } else {
        return None;
    };
    let remaining_symbols = remaining_samples / sps;

    if remaining_symbols < coding_s * 20 {
        return None; // too short for even a minimal PDU
    }

    // Extract soft symbols for Block 2
    let block2_soft: Vec<f32> = (0..remaining_symbols)
        .map(|i| {
            let idx = block2_start + i * sps;
            if sps == 2 && idx + 1 < demod.len() {
                (demod[idx] + demod[idx + 1]) / 2.0
            } else if idx < demod.len() {
                demod[idx]
            } else {
                0.0
            }
        })
        .collect();

    // Soft demap at S=coding_s
    let soft_coded_block2 = if coding_s == 8 {
        fec::pattern_demap_s8(&block2_soft)
    } else {
        fec::pattern_demap_s2(&block2_soft)
    };

    // Viterbi decode Block 2
    let decoded_block2 = fec::viterbi_decode(&soft_coded_block2);

    // decoded_block2 = PDU header(2 bytes) + PDU payload(len bytes) + CRC(3 bytes) + TERM2(3 bits)
    // First, extract PDU header to get length
    if decoded_block2.len() < 16 {
        return None;
    }

    let channel = freq_to_channel(freq);

    // Dewhiten the decoded bits to get PDU header
    // Note: whitening is applied after FEC decoding in LE Coded
    let mut header_bytes = [0u8; 2];
    for byte_idx in 0..2 {
        let mut byte_val: u8 = 0;
        for bit_idx in 0..8u32 {
            let data_bit = decoded_block2[byte_idx * 8 + bit_idx as usize];
            let wb = whitening_bit(channel, byte_idx as u32 * 8 + bit_idx);
            byte_val |= (data_bit ^ wb) << bit_idx;
        }
        header_bytes[byte_idx] = byte_val;
    }

    let pdu_length = header_bytes[1] as usize;
    if pdu_length > 251 {
        return None;
    }

    // Total decoded bits needed: header(16) + payload(pdu_length*8) + CRC(24)
    let total_pdu_crc_bits = 16 + pdu_length * 8 + 24;
    if decoded_block2.len() < total_pdu_crc_bits {
        return None;
    }

    // Dewhiten all PDU + CRC bits
    let mut pdu_crc_bytes = vec![0u8; 2 + pdu_length + 3];
    for byte_idx in 0..(2 + pdu_length + 3) {
        let mut byte_val: u8 = 0;
        for bit_idx in 0..8u32 {
            let data_bit = decoded_block2[byte_idx * 8 + bit_idx as usize];
            let wb = whitening_bit(channel, byte_idx as u32 * 8 + bit_idx);
            byte_val |= (data_bit ^ wb) << bit_idx;
        }
        pdu_crc_bytes[byte_idx] = byte_val;
    }

    // Build packet data: AA(4) + CI(1) + PDU(header + payload) + CRC(3)
    let pkt_len = 4 + 1 + 2 + pdu_length + 3;
    let mut data = vec![0u8; pkt_len];

    // AA bytes (little-endian)
    data[0] = (aa >> 0) as u8;
    data[1] = (aa >> 8) as u8;
    data[2] = (aa >> 16) as u8;
    data[3] = (aa >> 24) as u8;

    // CI byte
    data[4] = ci_val;

    // PDU + CRC (already dewhitened)
    data[5..].copy_from_slice(&pdu_crc_bytes);

    let is_data = aa != BLE_ADV_AA;

    // CRC validation
    let mut crc_checked = false;
    let mut crc_valid = false;
    let mut conn_valid = false;

    if check_crc {
        // For LE Coded, CRC is computed over PDU (after dewhitening, before CRC bytes)
        // CRC init depends on AA
        let crc_data = &pdu_crc_bytes[..2 + pdu_length]; // PDU header + payload
        let received_crc = pdu_crc_bytes[2 + pdu_length] as u32
            | ((pdu_crc_bytes[2 + pdu_length + 1] as u32) << 8)
            | ((pdu_crc_bytes[2 + pdu_length + 2] as u32) << 16);

        if aa == BLE_ADV_AA {
            let computed = crc24(crc_data, 0x555555);
            crc_checked = true;
            crc_valid = computed == received_crc;
        } else {
            if let Some((init, valid)) = crc_init_fn(aa) {
                let computed = crc24(crc_data, init);
                crc_checked = true;
                crc_valid = computed == received_crc;
                conn_valid = valid;
            }
        }
    }

    let ext_header = if crc_valid && pdu_crc_bytes.len() > 0 {
        parse_ext_adv(&pdu_crc_bytes[..2 + pdu_length])
    } else {
        None
    };

    Some(BlePacket {
        aa,
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
        phy: BlePhy::PhyCoded,
        ext_header,
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

    /// End-to-end test: construct a known BLE LE 2M advertising packet,
    /// whiten it, convert to bit array, and verify ble_burst_2m decodes it
    /// with correct CRC.
    #[test]
    fn test_ble_burst_2m_known_packet() {
        use crate::freq_to_channel;

        let freq = 2402u32;
        let channel = freq_to_channel(freq);
        assert_eq!(channel, 37);

        // Build ADV_NONCONN_IND: type=0x02, length=12
        let pdu_header: [u8; 2] = [0x02, 0x0C];
        let adv_a: [u8; 6] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let adv_data: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        let mut pdu = Vec::new();
        pdu.extend_from_slice(&pdu_header);
        pdu.extend_from_slice(&adv_a);
        pdu.extend_from_slice(&adv_data);

        let crc = crc24(&pdu, 0x555555);
        let crc_bytes = [
            (crc & 0xFF) as u8,
            ((crc >> 8) & 0xFF) as u8,
            ((crc >> 16) & 0xFF) as u8,
        ];

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

        // Build LE 2M bit stream: 16-bit preamble + AA(not whitened) + PDU+CRC(whitened)
        let mut bits_2m = Vec::new();
        // 16-bit alternating preamble
        for i in 0..16 {
            bits_2m.push((i & 1) as u8);
        }
        // AA bits (not whitened)
        bits_2m.extend_from_slice(&bits_raw[..32]);
        // PDU+CRC bits (whitened)
        for i in 32..bits_raw.len() {
            let wb = whitening_bit(channel, (i - 32) as u32);
            bits_2m.push(bits_raw[i] ^ wb);
        }
        // Trailing bits
        for _ in 0..16 {
            bits_2m.push(0);
        }

        let ts = crate::Timespec { tv_sec: 0, tv_nsec: 0 };
        let pkt = ble_burst_2m(&bits_2m, freq, ts, true, |_aa| None);
        assert!(pkt.is_some(), "ble_burst_2m failed to find known packet");
        let pkt = pkt.unwrap();
        assert_eq!(pkt.aa, BLE_ADV_AA);
        assert_eq!(pkt.phy, BlePhy::Phy2M);
        assert!(pkt.crc_checked, "CRC not checked");
        assert!(pkt.crc_valid, "CRC INVALID: data={:02X?}", &pkt.data);
        assert_eq!(pkt.data[4], 0x02, "PDU type mismatch");
        assert_eq!(pkt.data[5], 0x0C, "PDU length mismatch");
    }

    /// Test LE 2M AA correlator with synthetic signal
    #[test]
    fn test_correlator_2m_known_packet() {
        use crate::freq_to_channel;

        let freq = 2402u32;
        let channel = freq_to_channel(freq);

        let pdu = [0x02u8, 0x0C, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
                   0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let crc = crc24(&pdu, 0x555555);
        let crc_bytes = [
            (crc & 0xFF) as u8,
            ((crc >> 8) & 0xFF) as u8,
            ((crc >> 16) & 0xFF) as u8,
        ];

        let aa_bytes = [0xD6u8, 0xBE, 0x89, 0x8E];
        let mut all_bytes = Vec::new();
        all_bytes.extend_from_slice(&aa_bytes);
        all_bytes.extend_from_slice(&pdu);
        all_bytes.extend_from_slice(&crc_bytes);

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
                bits_pdu_whitened.push(bits_raw[i]);
            } else {
                let wb = whitening_bit(channel, (i - 32) as u32);
                bits_pdu_whitened.push(bits_raw[i] ^ wb);
            }
        }

        // Build synthetic demod at SPS=1 (LE 2M)
        let mut demod = Vec::new();
        for _ in 0..10 { demod.push(0.0f32); } // silence
        // Preamble (16 bits * SPS=1)
        let preamble_2m = [0u8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        for &b in &preamble_2m {
            demod.push(if b == 1 { 1.0 } else { -1.0 });
        }
        // Whitened packet bits at SPS=1
        for &b in &bits_pdu_whitened {
            demod.push(if b == 1 { 1.0 } else { -1.0 });
        }
        for _ in 0..20 { demod.push(0.0f32); }

        let corr = AaCorrelator::with_sps(1);
        let ts = crate::Timespec { tv_sec: 0, tv_nsec: 0 };
        let pkt = corr.correlate_2m(&demod, freq, ts, true);
        assert!(pkt.is_some(), "2M correlator failed");
        let pkt = pkt.unwrap();
        assert_eq!(pkt.aa, BLE_ADV_AA);
        assert_eq!(pkt.phy, BlePhy::Phy2M);
        assert!(pkt.crc_valid, "CRC INVALID on 2M correlator");
    }

    /// Test Extended Advertising header parsing
    #[test]
    fn test_parse_ext_adv() {
        // ADV_EXT_IND: PDU type 7, with AdvA + AuxPtr
        // ext_hdr_len = 1(flags) + 6(AdvA) + 3(AuxPtr) = 10
        // pdu_len = 1(ext_hdr_len byte) + ext_hdr_len = 11

        let mut pdu = Vec::new();
        pdu.push(0x07); // PDU type 7 (ADV_EXT_IND)
        pdu.push(11);   // pdu_len

        // Extended header
        pdu.push(10 | (0 << 6)); // ext_hdr_len=10, adv_mode=0
        pdu.push(0x11);          // flags: AdvA(bit0) + AuxPtr(bit4)

        // AdvA (6 bytes)
        pdu.extend_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66]);

        // AuxPtr (3 bytes): channel=5, ca=0, offset_units=0, aux_offset=100, phy=0 (1M)
        // bits: [5:0]=channel, [6]=ca, [7]=offset_units, [20:8]=aux_offset, [23:21]=phy
        let aux_raw: u32 = 5 | (0 << 6) | (0 << 7) | (100 << 8) | (0 << 21);
        pdu.push((aux_raw & 0xFF) as u8);
        pdu.push(((aux_raw >> 8) & 0xFF) as u8);
        pdu.push(((aux_raw >> 16) & 0xFF) as u8);

        let result = parse_ext_adv(&pdu);
        assert!(result.is_some(), "parse_ext_adv failed");
        let hdr = result.unwrap();
        assert_eq!(hdr.adv_mode, 0);
        assert_eq!(hdr.adv_a, Some([0x11, 0x22, 0x33, 0x44, 0x55, 0x66]));
        assert!(hdr.aux_ptr.is_some());
        let aux = hdr.aux_ptr.unwrap();
        assert_eq!(aux.channel, 5);
        assert_eq!(aux.offset_usec, 100 * 30); // offset_units=0 -> 30us
        assert_eq!(aux.phy, BlePhy::Phy1M);
    }

    /// Test parse_ext_adv returns None for non-type-7 PDU
    #[test]
    fn test_parse_ext_adv_wrong_type() {
        let pdu = [0x00, 0x06, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06]; // type 0, not 7
        assert!(parse_ext_adv(&pdu).is_none());
    }

    /// Test LE Coded PHY end-to-end with synthetic signal
    #[test]
    fn test_ble_coded_burst_synthetic() {
        use crate::fec;
        use crate::freq_to_channel;

        let freq = 2402u32;
        let channel = freq_to_channel(freq);
        let sps = 2usize;

        // Build a minimal ADV packet
        let pdu_header: [u8; 2] = [0x02, 0x06]; // type=2, len=6
        let adv_a: [u8; 6] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];

        let mut pdu = Vec::new();
        pdu.extend_from_slice(&pdu_header);
        pdu.extend_from_slice(&adv_a);

        let crc = crc24(&pdu, 0x555555);
        let crc_bytes = [
            (crc & 0xFF) as u8,
            ((crc >> 8) & 0xFF) as u8,
            ((crc >> 16) & 0xFF) as u8,
        ];

        // AA bits (LSB first)
        let aa_bytes = [0xD6u8, 0xBE, 0x89, 0x8E];
        let mut aa_bits = Vec::new();
        for &byte in &aa_bytes {
            for j in 0..8 {
                aa_bits.push((byte >> j) & 1);
            }
        }

        // CI = 0 (S=8 for PDU)
        let ci_bits = vec![0u8, 0]; // CI=0 means S=8

        // FEC Block 1: AA(32) + CI(2), conv_encode adds 3 flush bits (= TERM1)
        let mut block1_data = Vec::new();
        block1_data.extend_from_slice(&aa_bits);
        block1_data.extend_from_slice(&ci_bits);
        let block1_coded = fec::conv_encode(&block1_data);
        let block1_symbols = fec::pattern_map_s8(&block1_coded);

        // FEC Block 2: PDU + CRC (whitened), then FEC + pattern map at S=8
        let mut pdu_crc = Vec::new();
        pdu_crc.extend_from_slice(&pdu);
        pdu_crc.extend_from_slice(&crc_bytes);

        // Whiten PDU+CRC, conv_encode adds 3 flush bits (= TERM2)
        let mut pdu_crc_bits = Vec::new();
        for (byte_idx, &byte) in pdu_crc.iter().enumerate() {
            for bit_idx in 0..8u32 {
                let raw_bit = (byte >> bit_idx) & 1;
                let wb = whitening_bit(channel, byte_idx as u32 * 8 + bit_idx);
                pdu_crc_bits.push(raw_bit ^ wb);
            }
        }

        let block2_coded = fec::conv_encode(&pdu_crc_bits);
        let block2_symbols = fec::pattern_map_s8(&block2_coded);

        // Build full demod signal at SPS=2
        let mut demod = Vec::new();

        // Preamble: 80 symbols of 00111100, each repeated SPS=2 times
        for _ in 0..CODED_PREAMBLE_REPS {
            for &bit in &CODED_PREAMBLE_PATTERN {
                let val = if bit == 1 { 1.0f32 } else { -1.0 };
                for _ in 0..sps { demod.push(val); }
            }
        }

        // FEC Block 1 symbols at SPS=2
        for &sym in &block1_symbols {
            let val = if sym == 1 { 1.0f32 } else { -1.0 };
            for _ in 0..sps { demod.push(val); }
        }

        // FEC Block 2 symbols at SPS=2
        for &sym in &block2_symbols {
            let val = if sym == 1 { 1.0f32 } else { -1.0 };
            for _ in 0..sps { demod.push(val); }
        }

        // Trail
        for _ in 0..40 { demod.push(0.0f32); }

        let ts = crate::Timespec { tv_sec: 0, tv_nsec: 0 };
        let pkt = ble_coded_burst(&demod, freq, ts, sps, true, |_| None);
        assert!(pkt.is_some(), "ble_coded_burst failed to decode known packet");
        let pkt = pkt.unwrap();
        assert_eq!(pkt.aa, BLE_ADV_AA);
        assert_eq!(pkt.phy, BlePhy::PhyCoded);
        assert!(pkt.crc_checked, "CRC not checked");
        assert!(pkt.crc_valid, "CRC INVALID on coded burst: data={:02X?}", &pkt.data);
        // Verify CI byte is present in output
        assert_eq!(pkt.data[4], 0, "CI should be 0 (S=8)");
        // Verify dewhitened PDU
        assert_eq!(pkt.data[5], 0x02, "PDU type mismatch");
        assert_eq!(pkt.data[6], 0x06, "PDU length mismatch");
    }
}
