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

