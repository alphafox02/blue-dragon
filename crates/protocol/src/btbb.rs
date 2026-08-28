// Copyright 2025-2026 CEMAXECUTER LLC

use std::collections::{HashMap, HashSet};

const MAX_BARKER_ERRORS: u8 = 1;
const DEFAULT_AC: u64 = 0xcc7b7268ff614e1b;
const PN: u64 = 0x83848D96BBCC54FC;

static BARKER_DISTANCE: [u8; 128] = [
    3, 3, 3, 2, 3, 2, 2, 1, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 3,
    3, 2, 2, 1, 2, 1, 1, 0, 3, 3, 3, 2, 3, 2, 2, 1, 3, 3, 3, 2, 3, 2, 2, 1, 2, 3, 3, 3, 3, 3, 3, 2,
    2, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 3, 1, 2, 2, 3, 2, 3, 3, 3, 0, 1, 1, 2, 1, 2, 2, 3,
    3, 3, 3, 2, 3, 2, 2, 1, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 3,
];

static BARKER_CORRECT: [u64; 128] = [
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0x4e00000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0xb000000000000000,
    0x4e00000000000000,
];

// Syndrome check tables (pre-computed for BCH(64,30) code)
include!("sw_check_tables.rs");

fn air_to_host8(air_order: &[u8], bits: usize) -> u8 {
    let mut host_order: u8 = 0;
    for i in 0..bits {
        host_order |= (air_order[i] & 1) << i;
    }
    host_order
}

fn air_to_host64(air_order: &[u8], bits: usize) -> u64 {
    let mut host_order: u64 = 0;
    for i in 0..bits {
        host_order |= (air_order[i] as u64 & 1) << i;
    }
    host_order
}

fn gen_syndrome(codeword: u64) -> u64 {
    let mut syndrome = codeword & 0xffffffff;
    let mut cw = codeword >> 32;
    syndrome ^= SW_CHECK_TABLE4[(cw & 0xff) as usize];
    cw >>= 8;
    syndrome ^= SW_CHECK_TABLE5[(cw & 0xff) as usize];
    cw >>= 8;
    syndrome ^= SW_CHECK_TABLE6[(cw & 0xff) as usize];
    cw >>= 8;
    syndrome ^= SW_CHECK_TABLE7[(cw & 0xff) as usize];
    syndrome
}

/// Syndrome map for Classic BT access code error correction
pub struct SyndromeMap {
    map: HashMap<u64, u64>,
}

impl SyndromeMap {
    pub fn new(max_bit_errors: u32) -> Self {
        let mut sm = SyndromeMap {
            map: HashMap::new(),
        };
        for depth in 1..=max_bit_errors {
            sm.cycle(0, 0, depth, DEFAULT_AC);
        }
        sm
    }

    fn cycle(&mut self, error: u64, start: usize, depth: u32, codeword: u64) {
        let depth = depth - 1;
        for i in start..58 {
            let new_error = (1u64 << i) | error;
            if depth > 0 {
                self.cycle(new_error, i + 1, depth, codeword);
            } else {
                let syndrome = gen_syndrome(codeword ^ new_error);
                self.map.insert(syndrome, new_error);
            }
        }
    }

    fn find(&self, syndrome: u64) -> Option<u64> {
        self.map.get(&syndrome).copied()
    }
}

/// Classic BT packet detection result
#[derive(Debug, Clone)]
pub struct ClassicBtPacket {
    pub lap: u32,
    pub ac_errors: u8,
    /// Bit offset of the 64-bit sync word within the detector input.
    pub sync_offset: usize,
    pub rssi_db: i32,
    pub noise_db: i32,
    pub freq: u32,
    pub timestamp: crate::Timespec,
    pub raw_header: [u8; 7],
    pub has_header: bool,
    /// Raw (whitened, coded) payload bits after the 54-bit header, up to the end
    /// of the burst. Used for UAP disambiguation and FHS decode.
    pub payload: Vec<u8>,
    /// Recovered upper address part, once the tracker has converged.
    pub uap: Option<u8>,
    /// True only when the UAP was confirmed by a payload CRC or an FHS packet.
    pub uap_verified: bool,
    /// Non-significant address part, only from a decoded FHS packet.
    pub nap: Option<u16>,
    /// Decoded 18-bit header, once the UAP is known.
    pub header: Option<BtHeader>,
    /// De-whitened, de-FEC'd payload bytes (header + body) once decoded. Empty
    /// until the UAP is known and the packet type carries a payload.
    pub decoded_payload: Vec<u8>,
    /// Whether the decoded payload's CRC checked out under the recovered UAP.
    pub crc_ok: bool,
    /// Full 28-bit master clock (CLK27-0) for this packet, once an FHS from the
    /// device has anchored it. Low 2 bits are approximate (FHS carries CLK27-2).
    pub clkn: Option<u32>,
}

/// Find a Classic BT access code in a bit stream.
/// Returns (lap, ac_offset, ac_errors) if found, None otherwise.
pub fn find_ac(
    stream: &[u8],
    max_ac_errors: u8,
    syndrome_map: &SyndromeMap,
) -> Option<(u32, usize, u8)> {
    let search_length = stream.len();
    if search_length < 64 {
        return None;
    }

    // Barker code sliding window
    let mut barker = air_to_host8(&stream[57..], 6);
    barker <<= 1;

    for count in 0..(search_length - 64) {
        let symbols = &stream[count..];
        barker >>= 1;
        if count + 63 < search_length {
            barker |= (symbols[63] & 1) << 6;
        } else {
            break;
        }

        if BARKER_DISTANCE[barker as usize] <= MAX_BARKER_ERRORS {
            let mut syncword = air_to_host64(symbols, 64);

            // Correct the barker code
            let corrected_barker = BARKER_CORRECT[(syncword >> 57) as usize & 0x7F];
            syncword = (syncword & 0x01ffffffffffffff) | corrected_barker;

            let codeword = syncword ^ PN;
            let syndrome = gen_syndrome(codeword);
            let mut ac_errors: u8 = 0;

            if syndrome != 0 {
                if let Some(error) = syndrome_map.find(syndrome) {
                    syncword ^= error;
                    ac_errors = error.count_ones() as u8;
                } else {
                    continue; // unfixable
                }
            }

            if ac_errors <= max_ac_errors {
                let lap = ((syncword >> 34) & 0xffffff) as u32;
                return Some((lap, count, ac_errors));
            }
        }
    }

    None
}

/// Detect Classic BT and build a packet structure
pub fn detect(
    bits: &[u8],
    freq: u32,
    rssi: i32,
    noise: i32,
    timestamp: crate::Timespec,
    syndrome_map: &SyndromeMap,
) -> Option<ClassicBtPacket> {
    let (lap, ac_offset, ac_errors) = find_ac(bits, 1, syndrome_map)?;

    let mut pkt = ClassicBtPacket {
        lap,
        ac_errors,
        sync_offset: ac_offset,
        rssi_db: rssi,
        noise_db: noise,
        freq,
        timestamp,
        raw_header: [0; 7],
        has_header: false,
        payload: Vec::new(),
        uap: None,
        uap_verified: false,
        nap: None,
        header: None,
        decoded_payload: Vec::new(),
        crc_ok: false,
        clkn: None,
    };

    // A full access code has a 64-bit sync word followed by a 4-bit trailer.
    // `find_ac` returns the sync-word offset, not the preamble/access-code start.
    let header_start = ac_offset + 68;
    if header_start + 54 <= bits.len()
        && header_fec_disagreements(&bits[header_start..header_start + 54]) <= 6
    {
        pkt.has_header = true;
        for i in 0..54 {
            pkt.raw_header[i / 8] |= (bits[header_start + i] & 1) << (i % 8);
        }
        // Everything after the header is the (whitened, coded) payload. Cap it
        // at a DH5's worth of bits so a runaway burst can't balloon the packet.
        let payload_start = header_start + 54;
        if payload_start < bits.len() {
            let end = bits.len().min(payload_start + 2790);
            pkt.payload = bits[payload_start..end].iter().map(|&b| b & 1).collect();
        }
    }

    Some(pkt)
}

/// Count header symbols whose three 1/3-FEC copies are not unanimous. This is
/// a cheap way to distinguish a real header from noise following an ID packet;
/// majority decoding still corrects the accepted disagreements.
fn header_fec_disagreements(bits54: &[u8]) -> usize {
    bits54
        .chunks_exact(3)
        .filter(|triplet| !(triplet[0] == triplet[1] && triplet[1] == triplet[2]))
        .count()
}

/// Decoded Classic BT packet header (18 bits after 1/3 FEC), valid only once a
/// UAP is known (the UAP seeds the HEC that gates a correct decode).
#[derive(Debug, Clone, Copy)]
pub struct BtHeader {
    pub lt_addr: u8,  // 3 bits: active-member (logical transport) address
    pub pkt_type: u8, // 4 bits: packet type (POLL/NULL/FHS/DM1/DH1/DH3/DH5/...)
    pub flow: u8,     // 1 bit
    pub arqn: u8,     // 1 bit
    pub seqn: u8,     // 1 bit
    pub hec: u8,      // 8-bit header error check (as received)
    /// CLK[6:1] whitening index that produced a UAP-consistent decode (a coarse
    /// clock estimate, 0..63).
    pub clk6: u8,
}

/// Human-readable BR/EDR packet-type name for a decoded 4-bit type field.
/// (SCO/eSCO type codes overlap ACL codes; this table names the common ACL set.)
pub fn pkt_type_name(t: u8) -> &'static str {
    match t & 0x0f {
        0x0 => "NULL",
        0x1 => "POLL",
        0x2 => "FHS",
        0x3 => "DM1",
        0x4 => "DH1",
        0x5 => "HV1/DH1e",
        0x6 => "HV2/2-DH1",
        0x7 => "HV3/3-DH1",
        0x8 => "DV/2-DH3",
        0x9 => "AUX1/3-DH3",
        0xa => "DM3",
        0xb => "DH3",
        0xc => "2-DH5/EV4",
        0xd => "3-DH5/EV5",
        0xe => "DM5",
        0xf => "DH5",
        _ => "?",
    }
}

/// Enhanced Data Rate bits per symbol for ACL packet type codes. Because these
/// codes overlap SCO/eSCO meanings, callers must also validate the DPSK sync.
pub fn edr_bits_per_symbol(pkt_type: u8) -> Option<usize> {
    match pkt_type & 0x0f {
        0x6 | 0x8 | 0xc => Some(2),
        0x7 | 0x9 | 0xd => Some(3),
        _ => None,
    }
}

/// EDR-capable header interpretations for all 64 possible whitening clocks.
pub fn edr_header_candidates(raw: &[u8; 7]) -> Vec<(u8, BtHeader)> {
    let bits54 = unpack_54(raw);
    let mut candidates = Vec::new();
    for clk6 in 0u8..64 {
        let (h10, hec) = dewhiten_fec(&bits54, clk6);
        let pkt_type = ((h10 >> 3) & 0x0f) as u8;
        if edr_bits_per_symbol(pkt_type).is_none() {
            continue;
        }
        let uap = uap_from_hec(h10, hec);
        if let Some(header) = decode_header_at(raw, uap, clk6) {
            candidates.push((uap, header));
        }
    }
    candidates
}

/// Header candidates with one bounded correction for a non-unanimous 1/3-FEC
/// triplet. This is only used after the fixed EDR sync has validated the burst;
/// a payload CRC must still prove the selected UAP and clock.
pub fn edr_header_candidates_corrected(raw: &[u8; 7]) -> Vec<(u8, BtHeader)> {
    let bits54 = unpack_54(raw);
    let mut majority = [0u8; 18];
    for (index, bit) in majority.iter_mut().enumerate() {
        let offset = index * 3;
        *bit = u8::from(bits54[offset] + bits54[offset + 1] + bits54[offset + 2] >= 2);
    }

    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut corrections: Vec<usize> = (0..18)
        .filter(|&index| {
            let offset = index * 3;
            !(bits54[offset] == bits54[offset + 1] && bits54[offset + 1] == bits54[offset + 2])
        })
        .collect();
    corrections.push(18); // unmodified majority result
    for clk6 in 0u8..64 {
        let whitening = whitening_bits(clk6, 18);
        let mut decoded = majority;
        for (bit, mask) in decoded.iter_mut().zip(whitening) {
            *bit ^= mask;
        }
        for &correction in &corrections {
            if correction < 18 {
                decoded[correction] ^= 1;
            }
            let mut h10 = 0u16;
            let mut hec = 0u8;
            for bit in 0..10 {
                h10 |= (decoded[bit] as u16) << bit;
            }
            for bit in 0..8 {
                hec |= decoded[10 + bit] << bit;
            }
            let pkt_type = ((h10 >> 3) & 0x0f) as u8;
            if edr_bits_per_symbol(pkt_type).is_some() {
                let uap = uap_from_hec(h10, hec);
                if seen.insert((uap, clk6, h10)) {
                    candidates.push((
                        uap,
                        BtHeader {
                            lt_addr: (h10 & 0x7) as u8,
                            pkt_type,
                            flow: ((h10 >> 7) & 1) as u8,
                            arqn: ((h10 >> 8) & 1) as u8,
                            seqn: ((h10 >> 9) & 1) as u8,
                            hec,
                            clk6,
                        },
                    ));
                }
            }
            if correction < 18 {
                decoded[correction] ^= 1;
            }
        }
    }
    candidates
}

// Bluetooth's 127-bit whitening sequence and CLK[6:1] start offsets. These
// vectors are shared with libbtbb's bluetooth_packet.c.
const WHITENING_INDICES: [usize; 64] = [
    99, 85, 17, 50, 102, 58, 108, 45, 92, 62, 32, 118, 88, 11, 80, 2, 37, 69, 55, 8, 20, 40, 74,
    114, 15, 106, 30, 78, 53, 72, 28, 26, 68, 7, 39, 113, 105, 77, 71, 25, 84, 49, 57, 44, 61, 117,
    10, 1, 123, 124, 22, 125, 111, 23, 42, 126, 6, 112, 76, 24, 48, 43, 116, 0,
];
const WHITENING_DATA: [u8; 127] = [
    1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
];

/// Generate `n` bits of BR data whitening for a CLK[6:1] value.
fn whitening_bits(clk1_6: u8, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut index = WHITENING_INDICES[(clk1_6 & 0x3f) as usize];
    for _ in 0..n {
        out.push(WHITENING_DATA[index]);
        index = (index + 1) % WHITENING_DATA.len();
    }
    out
}

/// Whitening bits `start..start+n` of the continuous per-packet LFSR (header +
/// payload share one sequence seeded at packet start).
fn whitening_slice(clk1_6: u8, start: usize, n: usize) -> Vec<u8> {
    whitening_bits(clk1_6, start + n).split_off(start)
}

fn reverse8(mut b: u8) -> u8 {
    b = ((b & 0xF0) >> 4) | ((b & 0x0F) << 4);
    b = ((b & 0xCC) >> 2) | ((b & 0x33) << 2);
    b = ((b & 0xAA) >> 1) | ((b & 0x55) << 1);
    b
}

/// Reverse the HEC LFSR to recover the UAP that would produce `hec` over the
/// 10-bit header payload (the BT spec seeds the HEC LFSR with reverse(UAP)).
/// Same algorithm as libbtbb `btbb_uap_from_header`.
fn uap_from_hec(header_10: u16, hec: u8) -> u8 {
    let mut reg = hec;
    for i in (0..10).rev() {
        if reg & 0x80 != 0 {
            reg ^= 0x65;
        }
        let data_bit = ((header_10 >> i) & 1) as u8;
        reg = (reg << 1) | (((reg >> 7) ^ data_bit) & 1);
    }
    reverse8(reg)
}

/// Unpack the 54 FEC-encoded header bits (7 bytes, LSB-first) into a bit vector.
fn unpack_54(raw: &[u8; 7]) -> [u8; 54] {
    let mut bits = [0u8; 54];
    for i in 0..54 {
        bits[i] = (raw[i / 8] >> (i % 8)) & 1;
    }
    bits
}

/// Undo the 1/3-rate header FEC, dewhiten with CLK[6:1] = `clk`, and split the
/// result into the 10-bit header payload and 8-bit HEC.
fn dewhiten_fec(bits54: &[u8; 54], clk: u8) -> (u16, u8) {
    let mut bits18 = [0u8; 18];
    for i in 0..18 {
        let b0 = bits54[i * 3];
        let b1 = bits54[i * 3 + 1];
        let b2 = bits54[i * 3 + 2];
        bits18[i] = if (b0 + b1 + b2) >= 2 { 1 } else { 0 };
    }
    let wh = whitening_bits(clk, 18);
    for i in 0..18 {
        bits18[i] ^= wh[i];
    }
    let mut header_10: u16 = 0;
    for j in 0..10 {
        header_10 |= (bits18[j] as u16) << j;
    }
    let mut hec: u8 = 0;
    for j in 0..8 {
        hec |= bits18[10 + j] << j;
    }
    (header_10, hec)
}

/// Decode a header once the UAP is known: find the CLK[6:1] whose de-whitened,
/// FEC-decoded header yields a HEC consistent with `uap`, and return the parsed
/// fields. Returns None if no clock candidate validates (corrupt header).
pub fn decode_header(raw: &[u8; 7], uap: u8) -> Option<BtHeader> {
    for clk in 0u8..64 {
        if let Some(header) = decode_header_at(raw, uap, clk) {
            return Some(header);
        }
    }
    None
}

fn decode_header_at(raw: &[u8; 7], uap: u8, clk6: u8) -> Option<BtHeader> {
    let bits54 = unpack_54(raw);
    let (header_10, hec) = dewhiten_fec(&bits54, clk6);
    if uap_from_hec(header_10, hec) != uap {
        return None;
    }
    Some(BtHeader {
        lt_addr: (header_10 & 0x7) as u8,
        pkt_type: ((header_10 >> 3) & 0xf) as u8,
        flow: ((header_10 >> 7) & 1) as u8,
        arqn: ((header_10 >> 8) & 1) as u8,
        seqn: ((header_10 >> 9) & 1) as u8,
        hec,
        clk6,
    })
}

/// Duration of one Bluetooth slot (CLK1 tick period) in nanoseconds.
const SLOT_NS: u64 = 625_000;

/// For a single header, return the UAP implied at each of the 64 CLK[6:1]
/// values: `out[clk6] = uap`. This is the per-packet constraint the
/// clock-consistency tracker intersects across time.
fn header_candidates(raw: &[u8; 7]) -> [u8; 64] {
    let bits54 = unpack_54(raw);
    let mut out = [0u8; 64];
    for clk in 0u8..64 {
        let (h10, hec) = dewhiten_fec(&bits54, clk);
        out[clk as usize] = uap_from_hec(h10, hec);
    }
    out
}

fn ts_ns(a: &crate::Timespec, anchor: &crate::Timespec) -> i64 {
    (a.tv_sec as i64 - anchor.tv_sec as i64) * 1_000_000_000
        + (a.tv_nsec as i64 - anchor.tv_nsec as i64)
}

fn clk6_from_delta(phase: u8, delta_ns: i64) -> u8 {
    let half_slot = SLOT_NS as i64 / 2;
    let slots = if delta_ns >= 0 {
        (delta_ns + half_slot) / SLOT_NS as i64
    } else {
        (delta_ns - half_slot) / SLOT_NS as i64
    };
    (phase as i64 + slots).rem_euclid(64) as u8
}

/// Per-LAP clock-consistency state. Surviving (phase, uap) pairs, where `phase`
/// is CLK[6:1] at the anchor timestamp. A packet at time t predicts
/// clk6 = (phase + slots_since_anchor) & 0x3f, and only pairs whose predicted
/// clk6 yields the packet's implied UAP survive. Voting alone can't break the
/// ~6-way UAP aliasing; the timing constraint is what does it.
#[derive(Clone)]
struct Piconet {
    anchor: crate::Timespec,
    cands: Vec<(u8, u8)>, // (phase, uap)
    packets: u32,
    uap: Option<u8>,
    uap_verified: bool,
    crc_confirmations: u8,
    reanchors: u32,
    inconsistent_headers: u8,
    announced: bool,
    /// Full clock anchor from an FHS: (CLK27-0 at anchor, anchor timestamp).
    clk_anchor: Option<(u32, crate::Timespec)>,
}

impl Piconet {
    fn new(ts: &crate::Timespec, cand0: &[u8; 64]) -> Self {
        let cands = (0u8..64).map(|c| (c, cand0[c as usize])).collect();
        Piconet {
            anchor: ts.clone(),
            cands,
            packets: 1,
            uap: None,
            uap_verified: false,
            crc_confirmations: 0,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: false,
            clk_anchor: None,
        }
    }

    fn observe(&mut self, ts: &crate::Timespec, cand: &[u8; 64]) {
        if self.packets == 0 {
            self.anchor = ts.clone();
            self.cands = (0u8..64)
                .filter_map(|phase| {
                    let candidate = cand[phase as usize];
                    if !self.uap_verified || self.uap == Some(candidate) {
                        Some((phase, candidate))
                    } else {
                        None
                    }
                })
                .collect();
            self.packets = u32::from(!self.cands.is_empty());
            self.inconsistent_headers = 0;
            return;
        }
        self.packets += 1;
        let dns = ts_ns(ts, &self.anchor);
        let previous = self.cands.clone();
        self.cands.retain(|&(phase, uap)| {
            let clk6 = clk6_from_delta(phase, dns);
            cand[clk6 as usize] == uap
        });
        if self.cands.is_empty() {
            self.cands = previous;
            self.inconsistent_headers = self.inconsistent_headers.saturating_add(1);
            // Preserve strong clock evidence through a run of corrupt headers.
            // Ambiguous tracks still reanchor quickly so a bad first packet
            // cannot hold the tracker indefinitely.
            if self.uap_verified {
                return;
            }
            let reanchor_after = if self.cands.len() == 1 { 64 } else { 8 };
            if self.inconsistent_headers < reanchor_after {
                return;
            }
            self.anchor = ts.clone();
            self.cands = (0u8..64).map(|c| (c, cand[c as usize])).collect();
            self.reanchors += 1;
            self.inconsistent_headers = 0;
            self.uap = None;
            self.uap_verified = false;
            self.crc_confirmations = 0;
            return;
        }
        self.inconsistent_headers = 0;
        let first = self.cands[0].1;
        if self.cands.iter().all(|&(_, u)| u == first) {
            self.uap = Some(first);
        } else {
            self.uap = None;
        }
    }

    fn leader_uaps(&self) -> Vec<u8> {
        let mut u: Vec<u8> = self.cands.iter().map(|&(_, x)| x).collect();
        u.sort_unstable();
        u.dedup();
        u
    }

    fn clk6_at(&self, ts: &crate::Timespec) -> Option<u8> {
        let &(phase, _) = self.cands.first()?;
        if self.cands.len() != 1 || self.packets == 0 {
            return None;
        }
        let dns = ts_ns(ts, &self.anchor);
        Some(clk6_from_delta(phase, dns))
    }
}

/// Tracks UAP recovery per LAP across all channels using inter-packet clock
/// consistency. Feed every detected Classic BT header with its timestamp.
/// Result of trying to decode a candidate's payload.
enum PayloadCheck {
    /// Packet type carries no CRC (or is unknown): can't judge this candidate.
    Unsupported,
    /// Not even a full payload header present: can't judge.
    Incomplete,
    /// Header parsed but the length field wants more bits than we captured.
    /// A correct candidate never does this, so it's a wrong one.
    Overrun,
    /// Fully decoded: (header+body data bits, received CRC).
    Decoded(Vec<u8>, u16),
}

/// Undo 2/3 FEC when present, dewhiten the result starting after the 18 logical
/// header bits, then split it into (header+body) data bits and the received CRC.
fn decode_payload(raw_coded: &[u8], clk6: u8, pkt_type: u8) -> PayloadCheck {
    let (multi, fec23, has_crc) = match acl_type_info(pkt_type) {
        Some(v) => v,
        None => return PayloadCheck::Unsupported,
    };
    if !has_crc {
        return PayloadCheck::Unsupported;
    }
    let mut data: Vec<u8> = if fec23 {
        fec23_decode_bits(raw_coded, raw_coded.len() / 15 * 10)
    } else {
        raw_coded.to_vec()
    };
    let wh = whitening_slice(clk6, 18, data.len());
    for (bit, mask) in data.iter_mut().zip(wh) {
        *bit ^= mask;
    }

    let hdr_bits = if multi { 16 } else { 8 };
    let ph = match parse_payload_header(&data, multi) {
        Some(h) => h,
        None => return PayloadCheck::Incomplete,
    };
    let body_bits = ph.length as usize * 8;
    let need = hdr_bits + body_bits + 16;
    if data.len() < need {
        return PayloadCheck::Overrun;
    }
    let hdr_body = data[..hdr_bits + body_bits].to_vec();
    let mut rx_crc: u16 = 0;
    for j in 0..16 {
        rx_crc |= (data[hdr_bits + body_bits + j] as u16 & 1) << j;
    }
    PayloadCheck::Decoded(hdr_body, rx_crc)
}

impl Piconet {
    // Drop any (phase, uap) candidate whose payload CRC doesn't check out for
    // this packet. This is what finishes the job the header alone can't: the
    // last two-or-so UAP candidates collapse to one. The packet type comes from
    // each candidate's own header decode, so we don't need the UAP up front.
    fn disambiguate(&mut self, ts: &crate::Timespec, raw_header: &[u8; 7], raw_coded: &[u8]) {
        let dns = ts_ns(ts, &self.anchor);
        if dns < 0 {
            return;
        }
        let header_cands = self.cands.clone();
        let header_uap = self.uap;
        let header_verified = self.uap_verified;
        let header_crc_confirmations = self.crc_confirmations;
        let bits54 = unpack_54(raw_header);
        let slots = ((dns as u64 + SLOT_NS / 2) / SLOT_NS) as u8;
        let mut crc_matched = false;
        self.cands.retain(|&(phase, uap)| {
            let clk6 = phase.wrapping_add(slots) & 0x3f;
            let (h10, _) = dewhiten_fec(&bits54, clk6);
            let pkt_type = ((h10 >> 3) & 0xf) as u8;
            match decode_payload(raw_coded, clk6, pkt_type) {
                PayloadCheck::Decoded(hdr_body, rx_crc) => {
                    let valid = btcrc(&hdr_body, uap) == rx_crc;
                    crc_matched |= valid;
                    valid
                }
                PayloadCheck::Overrun => false,
                _ => true,
            }
        });
        if self.cands.is_empty() {
            // A corrupt or truncated payload must not erase timing evidence
            // already accepted from this packet's FEC-protected header.
            self.cands = header_cands;
            self.uap = header_uap;
            self.uap_verified = header_verified;
            self.crc_confirmations = header_crc_confirmations;
            return;
        }
        let first = self.cands[0].1;
        if self.cands.iter().all(|&(_, u)| u == first) {
            self.uap = Some(first);
            if crc_matched {
                self.crc_confirmations = self.crc_confirmations.saturating_add(1);
                self.uap_verified |= self.crc_confirmations >= 2;
            }
        }
    }
}

#[derive(Default)]
pub struct PiconetTracker {
    map: HashMap<u32, Piconet>,
}

impl PiconetTracker {
    pub fn new() -> Self {
        PiconetTracker {
            map: HashMap::new(),
        }
    }

    /// Feed one header for a LAP. Returns the UAP once only one value is left
    /// standing.
    pub fn observe(&mut self, lap: u32, raw: &[u8; 7], ts: &crate::Timespec) -> Option<u8> {
        let cand = header_candidates(raw);
        match self.map.get_mut(&lap) {
            Some(pn) => pn.observe(ts, &cand),
            None => {
                self.map.insert(lap, Piconet::new(ts, &cand));
            }
        }
        self.map.get(&lap).and_then(|p| p.uap)
    }

    /// Feed a full detection: the 54-bit header plus the packet's raw (whitened,
    /// coded) payload bits. Runs the clock constraint, then the CRC pass when
    /// payload bits are supplied. This is the entry point the pipeline uses.
    pub fn feed(
        &mut self,
        lap: u32,
        raw_header: &[u8; 7],
        payload: Option<&[u8]>,
        ts: &crate::Timespec,
    ) -> Option<u8> {
        self.observe(lap, raw_header, ts);
        if let (Some(pn), Some(pl)) = (self.map.get_mut(&lap), payload) {
            if !pl.is_empty() {
                pn.disambiguate(ts, raw_header, pl);
            }
        }
        self.map.get(&lap).and_then(|p| p.uap)
    }

    pub fn uap(&self, lap: u32) -> Option<u8> {
        self.map.get(&lap).and_then(|p| p.uap)
    }

    /// Anchor a LAP's full clock from an FHS (CLK27-2 field) at time `ts`.
    pub fn set_clock_anchor(&mut self, lap: u32, clk27_2: u32, ts: &crate::Timespec) {
        if let Some(pn) = self.map.get_mut(&lap) {
            // FHS gives CLK27-2; low 2 bits unknown, treated as 0.
            pn.clk_anchor = Some(((clk27_2 << 2) & 0x0fff_ffff, ts.clone()));
        }
    }

    /// Full 28-bit clock for a LAP at time `ts`, advanced from its FHS anchor.
    /// CLK ticks every 312.5 us.
    pub fn clkn_at(&self, lap: u32, ts: &crate::Timespec) -> Option<u32> {
        let pn = self.map.get(&lap)?;
        let (base, anchor) = pn.clk_anchor.as_ref()?;
        let dns = ts_ns(ts, anchor);
        if dns < 0 {
            return Some(*base);
        }
        let ticks = (dns as u64 / 312_500) as u32; // 312.5 us per CLK tick
        Some(base.wrapping_add(ticks) & 0x0fff_ffff)
    }

    /// Record a known-good UAP for a LAP (e.g. straight from an FHS packet), so
    /// later non-FHS packets are enriched without waiting for convergence.
    pub fn set_uap(&mut self, lap: u32, uap: u8) {
        let pn = self.map.entry(lap).or_insert_with(|| Piconet {
            anchor: crate::Timespec::default(),
            cands: (0u8..64).map(|phase| (phase, uap)).collect(),
            packets: 0,
            uap: Some(uap),
            uap_verified: true,
            crc_confirmations: 2,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: true,
            clk_anchor: None,
        });
        pn.uap = Some(uap);
        pn.uap_verified = true;
        pn.crc_confirmations = 2;
        pn.announced = true;
        pn.cands.retain(|&(_, candidate)| candidate == uap);
        if pn.cands.is_empty() {
            pn.cands = (0u8..64).map(|phase| (phase, uap)).collect();
            pn.packets = 0;
        }
    }

    fn set_uap_at(&mut self, lap: u32, uap: u8, clk6: u8, ts: &crate::Timespec) {
        let pn = self.map.entry(lap).or_insert_with(|| Piconet {
            anchor: ts.clone(),
            cands: Vec::new(),
            packets: 1,
            uap: Some(uap),
            uap_verified: true,
            crc_confirmations: 2,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: false,
            clk_anchor: None,
        });
        pn.anchor = ts.clone();
        pn.cands = vec![(clk6, uap)];
        pn.packets = pn.packets.max(1);
        pn.uap = Some(uap);
        pn.uap_verified = true;
        pn.crc_confirmations = 2;
    }

    /// Record CRC evidence for a UAP and whitening clock. EDR tries several
    /// bounded demodulation variants, so a single 16-bit CRC is not enough to
    /// publish an address; a later, clock-consistent packet must confirm it.
    pub fn confirm_uap_at(&mut self, lap: u32, uap: u8, clk6: u8, ts: &crate::Timespec) -> bool {
        if let Some(pn) = self.map.get_mut(&lap) {
            if pn.uap_verified {
                return pn.uap == Some(uap);
            }

            let dns = ts_ns(ts, &pn.anchor);
            let clock_matches = if dns != 0 && pn.cands.len() == 1 && pn.uap == Some(uap) {
                clk6_from_delta(pn.cands[0].0, dns) == clk6
            } else {
                false
            };
            if clock_matches {
                pn.crc_confirmations = pn.crc_confirmations.saturating_add(1).max(1);
                pn.uap_verified = pn.crc_confirmations >= 2;
                return pn.uap_verified;
            }

            pn.anchor = ts.clone();
            pn.cands = vec![(clk6, uap)];
            pn.packets = pn.packets.max(1);
            pn.uap = Some(uap);
            pn.uap_verified = false;
            pn.crc_confirmations = 1;
            return false;
        }

        self.map.insert(
            lap,
            Piconet {
                anchor: ts.clone(),
                cands: vec![(clk6, uap)],
                packets: 1,
                uap: Some(uap),
                uap_verified: false,
                crc_confirmations: 1,
                reanchors: 0,
                inconsistent_headers: 0,
                announced: false,
                clk_anchor: None,
            },
        );
        false
    }

    fn decode_tracked_header(
        &self,
        lap: u32,
        raw: &[u8; 7],
        ts: &crate::Timespec,
    ) -> Option<BtHeader> {
        let pn = self.map.get(&lap)?;
        decode_header_at(raw, pn.uap?, pn.clk6_at(ts)?)
    }

    pub fn uap_verified(&self, lap: u32) -> bool {
        self.map.get(&lap).is_some_and(|pn| pn.uap_verified)
    }

    /// Returns true the first time a LAP's UAP is reported, so callers can log
    /// it once instead of on every packet.
    pub fn mark_announced(&mut self, lap: u32) -> bool {
        match self.map.get_mut(&lap) {
            Some(pn) if pn.uap.is_some() && pn.uap_verified && !pn.announced => {
                pn.announced = true;
                true
            }
            _ => false,
        }
    }

    /// Surviving UAP candidates for a LAP (deduped, sorted).
    pub fn candidate_uap_set(&self, lap: u32) -> Vec<u8> {
        self.map
            .get(&lap)
            .map(|p| {
                let mut u = p.leader_uaps();
                u.sort_unstable();
                u.dedup();
                u
            })
            .unwrap_or_default()
    }

    /// Feed a packet's raw (whitened, coded) payload bits for CRC-based UAP
    /// disambiguation. Call after `observe`; packet type is taken from the
    /// header per candidate.
    pub fn disambiguate(
        &mut self,
        lap: u32,
        raw_header: &[u8; 7],
        raw_coded: &[u8],
        ts: &crate::Timespec,
    ) -> Option<u8> {
        if let Some(pn) = self.map.get_mut(&lap) {
            pn.disambiguate(ts, raw_header, raw_coded);
            return pn.uap;
        }
        None
    }

    /// Number of surviving UAP candidates for a LAP (1 == fully converged).
    pub fn candidate_uaps(&self, lap: u32) -> usize {
        self.map
            .get(&lap)
            .map(|p| {
                let mut u = p.leader_uaps();
                u.sort_unstable();
                u.dedup();
                u.len()
            })
            .unwrap_or(0)
    }
}

// Payload side: CRC-16, 2/3 FEC, payload header, FHS. The BR ordering is
// checked against independent libbtbb vectors; callers accept decoded payloads
// only after their over-the-air CRC validates.

/// Baseband CRC-16 (CRC-CCITT g(D)=D^16+D^12+D^5+1), processed LSB-first over
/// `bits`, register seeded with the UAP (spec 7.1.2). Returns the 16-bit CRC.
fn btcrc(bits: &[u8], uap: u8) -> u16 {
    let mut reg: u16 = (reverse8(uap) as u16) << 8;
    for &b in bits {
        let inbit = (b & 1) as u16;
        let lsb = reg & 1;
        reg >>= 1;
        if (lsb ^ inbit) != 0 {
            reg ^= 0x8408; // bit-reversed 0x1021
        }
    }
    reg
}

// 2/3 rate FEC: (15,10) shortened Hamming. Codeword layout is data in bits
// 0..9, parity in bits 10..14 (the on-air order). The per-data-bit parity
// patterns below are the generator matrix rows, matching libbtbb's unfec23.
const FEC23_PARITY: [u8; 10] = [11, 22, 7, 14, 28, 19, 13, 26, 31, 21];

/// Parity (5 bits) for a 10-bit data word.
fn fec23_parity(data10: u16) -> u8 {
    let mut p = 0u8;
    for i in 0..10 {
        if (data10 >> i) & 1 != 0 {
            p ^= FEC23_PARITY[i];
        }
    }
    p
}

/// Encode 10 data bits into a 15-bit codeword (data in bits 0..9, parity 10..14).
#[cfg(test)]
fn fec23_encode10(data10: u16) -> u32 {
    (data10 as u32 & 0x3ff) | ((fec23_parity(data10) as u32) << 10)
}

/// Decode a 15-bit codeword, correcting a single-bit error, returning 10 data bits.
fn fec23_decode15(cw15: u32) -> u16 {
    let mut data = (cw15 & 0x3ff) as u16;
    let parity_rx = ((cw15 >> 10) & 0x1f) as u8;
    let syn = parity_rx ^ fec23_parity(data);
    if syn != 0 {
        // Error in a data bit shows up as that bit's parity pattern; error in a
        // parity bit shows up as a single set syndrome bit (data unaffected).
        if let Some(i) = FEC23_PARITY.iter().position(|&p| p == syn) {
            data ^= 1 << i;
        }
    }
    data & 0x3ff
}

/// Decode a 2/3-FEC bit stream (LSB-first, 15 coded bits -> 10 data bits per
/// block) into `n_data` data bits.
fn fec23_decode_bits(coded: &[u8], n_data: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_data);
    let mut i = 0;
    while out.len() < n_data && i + 15 <= coded.len() {
        let mut cw: u32 = 0;
        for j in 0..15 {
            cw |= ((coded[i + j] & 1) as u32) << j;
        }
        let data = fec23_decode15(cw);
        for j in 0..10 {
            if out.len() == n_data {
                break;
            }
            out.push(((data >> j) & 1) as u8);
        }
        i += 15;
    }
    out
}

/// Encode `data` bits with 2/3 FEC (10 -> 15 per block, LSB-first). Zero-pads
/// the final partial block. Test/round-trip helper mirror of the decoder.
#[cfg(test)]
fn fec23_encode_bits(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let mut d: u16 = 0;
        for j in 0..10 {
            if i + j < data.len() {
                d |= (data[i + j] as u16 & 1) << j;
            }
        }
        let cw = fec23_encode10(d);
        for j in 0..15 {
            out.push(((cw >> j) & 1) as u8);
        }
        i += 10;
    }
    out
}

/// Decoded ACL payload header.
#[derive(Debug, Clone, Copy)]
pub struct PayloadHeader {
    pub llid: u8,
    pub flow: u8,
    pub length: u16,
    pub multi_slot: bool,
}

/// (multi_slot header, uses 2/3 FEC, carries a CRC) for an ACL packet type.
fn acl_type_info(pkt_type: u8) -> Option<(bool, bool, bool)> {
    match pkt_type & 0x0f {
        0x2 => Some((false, true, true)), // FHS: 2/3 FEC, CRC (handled separately)
        0x3 => Some((false, true, true)), // DM1
        0x4 => Some((false, false, true)), // DH1
        0x9 => Some((true, false, false)), // AUX1: no CRC
        0xa => Some((true, true, true)),  // DM3
        0xb => Some((true, false, true)), // DH3
        0xe => Some((true, true, true)),  // DM5
        0xf => Some((true, false, true)), // DH5
        _ => None,
    }
}

/// Parse the payload header from de-whitened, FEC-decoded payload data bits.
pub fn parse_payload_header(data: &[u8], multi_slot: bool) -> Option<PayloadHeader> {
    let need = if multi_slot { 16 } else { 8 };
    if data.len() < need {
        return None;
    }
    let llid = (data[0] & 1) | ((data[1] & 1) << 1);
    let flow = data[2] & 1;
    let (length, lbits) = if multi_slot {
        let mut l: u16 = 0;
        for j in 0..10 {
            l |= (data[3 + j] as u16 & 1) << j;
        }
        (l, 10)
    } else {
        let mut l: u16 = 0;
        for j in 0..5 {
            l |= (data[3 + j] as u16 & 1) << j;
        }
        (l, 5)
    };
    let _ = lbits;
    Some(PayloadHeader {
        llid,
        flow,
        length,
        multi_slot,
    })
}

/// Full device address recovered from an FHS packet.
#[derive(Debug, Clone, Copy)]
pub struct FhsInfo {
    pub lap: u32,
    pub uap: u8,
    pub nap: u16,
    pub class_of_device: u32,
    pub lt_addr: u8,
    pub clk27_2: u32,
}

impl FhsInfo {
    /// Full 48-bit BD_ADDR (NAP:UAP:LAP).
    pub fn bd_addr(&self) -> u64 {
        ((self.nap as u64) << 32) | ((self.uap as u64) << 24) | (self.lap as u64)
    }
}

/// Parse a 144-bit FHS payload body (LSB-first per field, spec Table 6.6).
pub fn parse_fhs_body(body: &[u8]) -> Option<FhsInfo> {
    if body.len() < 144 {
        return None;
    }
    let field = |start: usize, len: usize| -> u32 {
        let mut v: u32 = 0;
        for j in 0..len {
            v |= (body[start + j] as u32 & 1) << j;
        }
        v
    };
    Some(FhsInfo {
        lap: field(34, 24),
        uap: field(64, 8) as u8,
        nap: field(72, 16) as u16,
        class_of_device: field(88, 24),
        lt_addr: field(112, 3) as u8,
        clk27_2: field(115, 26),
    })
}

/// Decode an FHS packet's payload (2/3 FEC over a 144-bit body + 16-bit CRC),
/// de-whitened at `clk6`. Returns the address only if the CRC checks out under
/// the UAP the FHS itself carries.
pub fn decode_fhs_payload(raw_coded: &[u8], clk6: u8) -> Option<FhsInfo> {
    // 144 body + 16 CRC = 160 info bits -> 240 coded bits under 2/3 FEC.
    if raw_coded.len() < 240 {
        return None;
    }
    let mut data = fec23_decode_bits(&raw_coded[..240], 160);
    if data.len() < 160 {
        return None;
    }
    let wh = whitening_slice(clk6, 18, data.len());
    for (bit, mask) in data.iter_mut().zip(wh) {
        *bit ^= mask;
    }
    let info = parse_fhs_body(&data[..144])?;
    let mut rx_crc: u16 = 0;
    for j in 0..16 {
        rx_crc |= (data[144 + j] as u16 & 1) << j;
    }
    if btcrc(&data[..144], info.uap) != rx_crc {
        return None;
    }
    Some(info)
}

/// Run a detection through the tracker and fill in uap / header / nap when they
/// become known. An FHS packet resolves the full address immediately; otherwise
/// the clock-consistency + CRC tracker fills in the UAP over several packets.
/// Try to read a self-verifying FHS packet: for each clock the header could
/// have used, if it decodes as an FHS whose payload CRC (seeded by the UAP the
/// FHS itself carries) checks out and whose LAP/UAP agree with the header, we
/// have the whole address from this one packet. Returns (uap, fhs, clk6).
fn try_fhs(raw_header: &[u8; 7], payload: &[u8], lap: u32) -> Option<(u8, FhsInfo, u8)> {
    if payload.len() < 240 {
        return None;
    }
    let bits54 = unpack_54(raw_header);
    for clk6 in 0u8..64 {
        let (h10, hec) = dewhiten_fec(&bits54, clk6);
        if ((h10 >> 3) & 0xf) as u8 != 0x2 {
            continue; // not an FHS at this clock
        }
        let uap = uap_from_hec(h10, hec);
        if let Some(fhs) = decode_fhs_payload(payload, clk6) {
            if fhs.uap == uap && fhs.lap == lap {
                return Some((uap, fhs, clk6));
            }
        }
    }
    None
}

pub fn enrich(pkt: &mut ClassicBtPacket, tracker: &mut PiconetTracker) -> bool {
    if !pkt.has_header {
        return false;
    }
    // An FHS packet resolves the full address on its own - no need to wait for
    // the clock-consistency tracker to converge.
    if let Some((uap, fhs, clk6)) = try_fhs(&pkt.raw_header, &pkt.payload, pkt.lap) {
        pkt.uap = Some(uap);
        pkt.uap_verified = true;
        pkt.nap = Some(fhs.nap);
        pkt.header = decode_header_at(&pkt.raw_header, uap, clk6);
        tracker.set_uap_at(pkt.lap, uap, clk6, &pkt.timestamp);
        tracker.set_clock_anchor(pkt.lap, fhs.clk27_2, &pkt.timestamp);
        pkt.clkn = tracker.clkn_at(pkt.lap, &pkt.timestamp);
        return tracker.mark_announced(pkt.lap);
    }

    // A CRC-bearing BR payload is stronger evidence than the timing-only
    // header track. Search all whitening clocks independently so earlier ID
    // packets or corrupt headers cannot prevent a real ACL packet from
    // recovering the UAP. Two clock-consistent CRC matches are still required
    // before the address or payload is exposed.
    if let Some((uap, header, decoded_payload)) = unique_br_crc_candidate(pkt) {
        if tracker.confirm_uap_at(pkt.lap, uap, header.clk6, &pkt.timestamp) {
            pkt.uap = Some(uap);
            pkt.uap_verified = true;
            pkt.header = Some(header);
            pkt.decoded_payload = decoded_payload;
            pkt.crc_ok = true;
            pkt.clkn = tracker.clkn_at(pkt.lap, &pkt.timestamp);
            return tracker.mark_announced(pkt.lap);
        }
        return false;
    }

    // A header-only UAP estimate can still have several surviving clock
    // phases. Keep feeding those tentative estimates until CRC/FHS verifies
    // them; otherwise decode_tracked_header() can never select a clock.
    let known_uap = tracker.uap(pkt.lap);
    let uap = if tracker.uap_verified(pkt.lap) {
        tracker.observe(pkt.lap, &pkt.raw_header, &pkt.timestamp);
        known_uap
    } else {
        let payload = if pkt.payload.is_empty() {
            None
        } else {
            Some(pkt.payload.as_slice())
        };
        tracker.feed(pkt.lap, &pkt.raw_header, payload, &pkt.timestamp)
    };
    if let Some(u) = uap {
        pkt.uap = Some(u);
        pkt.uap_verified = tracker.uap_verified(pkt.lap);
        pkt.header = tracker.decode_tracked_header(pkt.lap, &pkt.raw_header, &pkt.timestamp);
        // The tracker already counted this packet's CRC. Do not expose its
        // candidate payload until another packet independently confirms UAP.
        if pkt.uap_verified {
            fill_decoded_payload(pkt, u);
        }
        pkt.clkn = tracker.clkn_at(pkt.lap, &pkt.timestamp);
    }
    tracker.mark_announced(pkt.lap)
}

fn unique_br_crc_candidate(pkt: &ClassicBtPacket) -> Option<(u8, BtHeader, Vec<u8>)> {
    let bits54 = unpack_54(&pkt.raw_header);
    let mut match_found = None;
    for clk6 in 0u8..64 {
        let (h10, hec) = dewhiten_fec(&bits54, clk6);
        let pkt_type = ((h10 >> 3) & 0xf) as u8;
        if pkt_type == 0x2 {
            continue;
        }
        let uap = uap_from_hec(h10, hec);
        let PayloadCheck::Decoded(bits, rx_crc) = decode_payload(&pkt.payload, clk6, pkt_type)
        else {
            continue;
        };
        if btcrc(&bits, uap) != rx_crc {
            continue;
        }
        if match_found.is_some() {
            return None;
        }
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (index, &bit) in bits.iter().enumerate() {
            bytes[index / 8] |= (bit & 1) << (index % 8);
        }
        match_found = Some((
            uap,
            BtHeader {
                lt_addr: (h10 & 0x7) as u8,
                pkt_type,
                flow: ((h10 >> 7) & 1) as u8,
                arqn: ((h10 >> 8) & 1) as u8,
                seqn: ((h10 >> 9) & 1) as u8,
                hec,
                clk6,
            },
            bytes,
        ));
    }
    match_found
}

/// Once the UAP is known, de-whiten/de-FEC the payload of a CRC-bearing ACL
/// packet and stash the bytes. The clock is chosen by CRC: we try each clock
/// whose header agrees with the UAP and keep the one whose payload CRC checks
/// out, which sidesteps the single-packet clock ambiguity.
fn fill_decoded_payload(pkt: &mut ClassicBtPacket, uap: u8) {
    let bits54 = unpack_54(&pkt.raw_header);
    let tracked_clk = pkt.header.map(|h| h.clk6);
    let mut clocks: Vec<u8> = tracked_clk.into_iter().collect();
    clocks.extend((0u8..64).filter(|clk6| Some(*clk6) != tracked_clk));
    for clk6 in clocks {
        let (h10, hec) = dewhiten_fec(&bits54, clk6);
        if uap_from_hec(h10, hec) != uap {
            continue;
        }
        let pkt_type = ((h10 >> 3) & 0xf) as u8;
        if pkt_type == 0x2 {
            continue; // FHS is surfaced structurally (uap/nap)
        }
        if let PayloadCheck::Decoded(bits, rx_crc) = decode_payload(&pkt.payload, clk6, pkt_type) {
            if btcrc(&bits, uap) == rx_crc {
                let mut bytes = vec![0u8; bits.len().div_ceil(8)];
                for (i, &b) in bits.iter().enumerate() {
                    bytes[i / 8] |= (b & 1) << (i % 8);
                }
                pkt.crc_ok = true;
                pkt.decoded_payload = bytes;
                pkt.header = decode_header_at(&pkt.raw_header, uap, clk6);
                return;
            }
        }
    }
}

fn decode_edr_payload_bits(raw: &[u8], uap: u8, clk6: u8) -> Option<Vec<u8>> {
    let whitening = whitening_slice(clk6, 18, raw.len());
    let data: Vec<u8> = raw
        .iter()
        .zip(whitening)
        .map(|(&bit, mask)| bit ^ mask)
        .collect();
    let payload_header = parse_payload_header(&data, true)?;
    let body_end = 16 + payload_header.length as usize * 8;
    for mic_bits in [0usize, 32] {
        let crc_start = body_end + mic_bits;
        if data.len() < crc_start + 16 {
            continue;
        }
        let mut received_crc = 0u16;
        for bit in 0..16 {
            received_crc |= (data[crc_start + bit] as u16 & 1) << bit;
        }
        let calculated_crc = btcrc(&data[..crc_start], uap);
        if calculated_crc == received_crc {
            return Some(data[..crc_start].to_vec());
        }
    }
    None
}

/// Decode an EDR payload by trying every whitening start phase (all 127) instead
/// of the clock-fixed offset. A clipped header can leave the payload-start symbol
/// and thus the whitening phase ambiguous; trying each and letting the CRC decide
/// recovers packets the clock path misses. Returns the dewhitened payload bits.
fn decode_edr_payload_bits_any_phase(raw: &[u8], uap: u8) -> Option<Vec<u8>> {
    for phase in 0..WHITENING_DATA.len() {
        let data: Vec<u8> = raw
            .iter()
            .enumerate()
            .map(|(i, &bit)| bit ^ WHITENING_DATA[(phase + i) % WHITENING_DATA.len()])
            .collect();
        let Some(payload_header) = parse_payload_header(&data, true) else {
            continue;
        };
        let body_end = 16 + payload_header.length as usize * 8;
        for mic_bits in [0usize, 32] {
            let crc_start = body_end + mic_bits;
            if data.len() < crc_start + 16 {
                continue;
            }
            let mut received_crc = 0u16;
            for bit in 0..16 {
                received_crc |= (data[crc_start + bit] as u16 & 1) << bit;
            }
            if btcrc(&data[..crc_start], uap) == received_crc {
                return Some(data[..crc_start].to_vec());
            }
        }
    }
    None
}

/// Attach an EDR payload located by whitening-phase search. For the wideband path
/// when a clipped header yields no clock-consistent candidate. The payload CRC is
/// the sole authority.
pub fn enrich_edr_candidate_any_phase(
    pkt: &mut ClassicBtPacket,
    raw_bits: &[u8],
    uap: u8,
    header: BtHeader,
) -> bool {
    if edr_bits_per_symbol(header.pkt_type).is_none() {
        return false;
    }
    let Some(decoded) = decode_edr_payload_bits_any_phase(raw_bits, uap) else {
        return false;
    };
    let mut bytes = vec![0u8; decoded.len().div_ceil(8)];
    for (index, &bit) in decoded.iter().enumerate() {
        bytes[index / 8] |= (bit & 1) << (index % 8);
    }
    pkt.payload = raw_bits.to_vec();
    pkt.uap = Some(uap);
    pkt.uap_verified = true;
    pkt.header = Some(header);
    pkt.decoded_payload = bytes;
    pkt.crc_ok = true;
    true
}

/// Decode only the two-byte EDR payload header for diagnostics. No identity or
/// payload is accepted from this result; the full EDR CRC remains authoritative.
pub fn edr_payload_header(raw: &[u8], clk6: u8) -> Option<PayloadHeader> {
    let whitening = whitening_slice(clk6, 18, raw.len());
    let data: Vec<u8> = raw
        .iter()
        .zip(whitening)
        .map(|(&bit, mask)| bit ^ mask)
        .collect();
    parse_payload_header(&data, true)
}

/// Attach a synchronization-validated EDR bitstream to a packet. The bits must
/// begin at the two-byte EDR payload header and remain whitened as transmitted.
pub fn enrich_edr_payload(pkt: &mut ClassicBtPacket, raw_bits: &[u8]) -> bool {
    let (uap, header) = match (pkt.uap, pkt.header) {
        (Some(uap), Some(header)) if edr_bits_per_symbol(header.pkt_type).is_some() => {
            (uap, header)
        }
        _ => return false,
    };
    enrich_edr_candidate(pkt, raw_bits, uap, header)
}

/// Attach an EDR payload using an explicit header interpretation. The caller
/// may obtain candidates from `edr_header_candidates`; CRC validation here is
/// the authority that selects the correct clock and UAP.
pub fn enrich_edr_candidate(
    pkt: &mut ClassicBtPacket,
    raw_bits: &[u8],
    uap: u8,
    header: BtHeader,
) -> bool {
    if edr_bits_per_symbol(header.pkt_type).is_none() {
        return false;
    }
    let decoded = match decode_edr_payload_bits(raw_bits, uap, header.clk6) {
        Some(decoded) => decoded,
        None => return false,
    };

    let mut bytes = vec![0u8; decoded.len().div_ceil(8)];
    for (index, &bit) in decoded.iter().enumerate() {
        bytes[index / 8] |= (bit & 1) << (index % 8);
    }
    pkt.payload = raw_bits.to_vec();
    pkt.uap = Some(uap);
    pkt.uap_verified = true;
    pkt.header = Some(header);
    pkt.decoded_payload = bytes;
    pkt.crc_ok = true;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syndrome_map_creation() {
        let sm = SyndromeMap::new(1);
        assert!(!sm.map.is_empty());
    }

    #[test]
    fn test_barker_distance_table() {
        // Index 39 (binary 0100111) should have distance 0 (perfect barker match)
        assert_eq!(BARKER_DISTANCE[39], 0);
        // Index 88 (binary 1011000) should have distance 0 (inverted barker)
        assert_eq!(BARKER_DISTANCE[88], 0);
        // Sanity: all distances should be 0-3
        assert!(BARKER_DISTANCE.iter().all(|&d| d <= 3));
    }

    // --- UAP recovery / header decode round-trip -------------------------

    /// Test-side encoder: build a 7-byte FEC-encoded, whitened header for a
    /// known (uap, clk, fields). HEC is found by brute force so it is exactly
    /// the value uap_from_hec() inverts, keeping the test independent of a
    /// hand-derived forward HEC.
    fn make_raw_header(
        uap: u8,
        clk: u8,
        lt_addr: u8,
        pkt_type: u8,
        flow: u8,
        arqn: u8,
        seqn: u8,
    ) -> [u8; 7] {
        let header_10: u16 = (lt_addr as u16 & 0x7)
            | ((pkt_type as u16 & 0xf) << 3)
            | ((flow as u16 & 1) << 7)
            | ((arqn as u16 & 1) << 8)
            | ((seqn as u16 & 1) << 9);
        let hec = (0u16..256)
            .map(|h| h as u8)
            .find(|&h| uap_from_hec(header_10, h) == uap)
            .expect("some HEC maps to this uap");

        let mut bits18 = [0u8; 18];
        for (j, bit) in bits18.iter_mut().enumerate().take(10) {
            *bit = ((header_10 >> j) & 1) as u8;
        }
        for j in 0..8 {
            bits18[10 + j] = (hec >> j) & 1;
        }
        let wh = whitening_bits(clk, 18);
        for i in 0..18 {
            bits18[i] ^= wh[i];
        }
        // 1/3 FEC: repeat each whitened bit 3x
        let mut bits54 = [0u8; 54];
        for i in 0..18 {
            bits54[i * 3] = bits18[i];
            bits54[i * 3 + 1] = bits18[i];
            bits54[i * 3 + 2] = bits18[i];
        }
        // Pack LSB-first into 7 bytes
        let mut raw = [0u8; 7];
        for i in 0..54 {
            raw[i / 8] |= (bits54[i] & 1) << (i % 8);
        }
        raw
    }

    #[test]
    fn test_hec_uap_roundtrip() {
        // uap_from_hec must invert the HEC for every uap (bijection per header).
        for uap in [0u8, 1, 0x3c, 0xa5, 0xff] {
            let h10: u16 = 0x2ab;
            let hec = (0u16..256)
                .map(|h| h as u8)
                .find(|&h| uap_from_hec(h10, h) == uap)
                .expect("bijection");
            assert_eq!(uap_from_hec(h10, hec), uap);
        }
    }

    #[test]
    fn test_dewhiten_fec_at_true_clock() {
        // At the true CLK[6:1], dewhiten+FEC must reproduce the exact fields.
        let (uap, clk) = (0xa5u8, 17u8);
        let raw = make_raw_header(uap, clk, 0b101, 0x4 /*DH1*/, 1, 0, 1);
        let bits54 = unpack_54(&raw);
        let (h10, hec) = dewhiten_fec(&bits54, clk);
        assert_eq!(uap_from_hec(h10, hec), uap);
        assert_eq!((h10 & 0x7) as u8, 0b101); // lt_addr
        assert_eq!(((h10 >> 3) & 0xf) as u8, 0x4); // type = DH1
        assert_eq!(((h10 >> 7) & 1) as u8, 1); // flow
                                               // decode_header finds *a* consistent clock (may alias on one packet),
                                               // but must still return a structurally valid header.

        assert!(decode_header(&raw, uap).is_some());
    }

    #[test]
    fn test_header_fec_quality_distinguishes_repetition_from_noise() {
        let raw = make_raw_header(0x67, 23, 1, 0x04, 1, 0, 1);
        let bits = unpack_54(&raw);
        assert_eq!(header_fec_disagreements(&bits), 0);

        let alternating: Vec<u8> = (0..54).map(|index| (index & 1) as u8).collect();
        assert_eq!(header_fec_disagreements(&alternating), 18);
    }

    #[test]
    fn test_edr_candidates_recover_one_bad_fec_triplet() {
        let (uap, clk) = (0x67u8, 23u8);
        let mut raw = make_raw_header(uap, clk, 1, 0x0d, 1, 0, 1);
        // Two physical errors defeat majority voting for logical header bit 5.
        for physical_bit in [15usize, 16] {
            raw[physical_bit / 8] ^= 1 << (physical_bit % 8);
        }

        let candidates = edr_header_candidates_corrected(&raw);
        assert!(candidates.iter().any(|&(candidate_uap, header)| {
            candidate_uap == uap && header.clk6 == clk && header.pkt_type == 0x0d
        }));
    }

    fn mk_ts(ns: u64) -> crate::Timespec {
        crate::Timespec {
            tv_sec: ns / 1_000_000_000,
            tv_nsec: ns % 1_000_000_000,
        }
    }

    #[test]
    fn test_clock_consistency_converges() {
        // Simulate a piconet: a real base clock, packets at realistic slot
        // spacings on varying channels/content. Clock-consistency must resolve
        // the exact UAP where the vote-only method stays 6-way tied.
        let true_uap = 0x3cu8;
        let base_clk6: u32 = 41; // CLK[6:1] at t=0
        let mut tracker = PiconetTracker::new();
        let lap = 0x9e8b33u32;
        let mut slot: u64 = 0;
        let steps = [
            1u64, 2, 3, 5, 6, 1, 4, 2, 3, 1, 6, 2, 5, 1, 3, 2, 4, 1, 2, 3,
        ];
        let mut result = None;
        for (i, &st) in steps.iter().enumerate() {
            slot += st;
            let clk6 = ((base_clk6 + slot as u32) & 0x3f) as u8;
            let h10 = (i as u16 * 37 + 11) & 0x3ff; // varying content
            let raw = mk_header_at(true_uap, clk6, h10);
            let ts = mk_ts(slot * SLOT_NS);
            result = tracker.observe(lap, &raw, &ts);
        }
        // Header-only voting narrows to the true UAP plus its clock-shift alias;
        // only payload CRC (see the disambiguate test) can pick between them, so
        // the estimate here may be either, but the true UAP must be a leader.
        let survivors = tracker.candidate_uap_set(lap);
        assert!(
            survivors.contains(&true_uap),
            "true uap not a leader: {:?}",
            survivors
        );
        assert!(
            survivors.len() <= 2,
            "expected <=2 leaders, got {:?}",
            survivors
        );
        if let Some(u) = result {
            assert!(survivors.contains(&u), "estimate not among leaders");
        }
    }

    #[test]
    fn test_clock_consistency_accepts_out_of_order_timestamps() {
        let true_uap = 0x67;
        let base_clk6 = 23u8;
        let lap = 0x405060;
        let slots = [10u64, 4, 12, 7, 16, 2, 20, 14, 24, 18, 28, 22, 32];
        let mut tracker = PiconetTracker::new();

        for (index, slot) in slots.into_iter().enumerate() {
            let clk6 = base_clk6.wrapping_add(slot as u8) & 0x3f;
            let h10 = (index as u16 * 53 + 7) & 0x03ff;
            tracker.observe(lap, &mk_header_at(true_uap, clk6, h10), &mk_ts(slot * SLOT_NS));
        }

        let survivors = tracker.candidate_uap_set(lap);
        assert!(survivors.contains(&true_uap), "true UAP lost: {survivors:?}");
        assert!(survivors.len() <= 2, "too many survivors: {survivors:?}");
    }

    #[test]
    fn test_clock_delta_rounds_in_both_directions() {
        assert_eq!(clk6_from_delta(10, SLOT_NS as i64), 11);
        assert_eq!(clk6_from_delta(10, -(SLOT_NS as i64)), 9);
        assert_eq!(clk6_from_delta(0, -(SLOT_NS as i64)), 63);
    }

    #[test]
    fn test_known_uap_anchors_to_packet_clock_without_reannouncement() {
        let (lap, uap, phase) = (0x405060, 0x30, 19u8);
        let mut tracker = PiconetTracker::new();
        tracker.set_uap(lap, uap);

        for slot in [0u64, 1, 3, 6, 10, 15] {
            let clk6 = phase.wrapping_add(slot as u8) & 0x3f;
            let raw = mk_header_at(uap, clk6, (slot as u16 * 29 + 5) & 0x03ff);
            tracker.observe(lap, &raw, &mk_ts(slot * SLOT_NS));
        }

        assert_eq!(tracker.uap(lap), Some(uap));
        assert!(tracker.uap_verified(lap));
        assert_eq!(tracker.map[&lap].cands, vec![(phase, uap)]);
        assert!(!tracker.mark_announced(lap));
    }

    #[test]
    fn test_enrich_keeps_refining_unverified_uap() {
        let lap = 0x0a1b2c;
        let uap = 0x4a;
        let phase = 17u8;

        // Find a first header with a second clock phase that implies the same
        // UAP. This is the tentative state that used to make enrich stop
        // observing subsequent packets.
        let (alias, _) = (0u16..1024)
            .find_map(|h10| {
                let raw = mk_header_at(uap, phase, h10);
                let candidates = header_candidates(&raw);
                (0u8..64)
                    .find(|&candidate| candidate != phase && candidates[candidate as usize] == uap)
                    .map(|candidate| (candidate, raw))
            })
            .expect("a header with a same-UAP clock alias");

        let mut tracker = PiconetTracker::new();
        tracker.map.insert(
            lap,
            Piconet {
                anchor: mk_ts(0),
                cands: vec![(phase, uap), (alias, uap)],
                packets: 1,
                uap: Some(uap),
                uap_verified: false,
                crc_confirmations: 0,
                reanchors: 0,
                inconsistent_headers: 0,
                announced: false,
                clk_anchor: None,
            },
        );

        let next_phase = phase.wrapping_add(1) & 0x3f;
        let next_alias = alias.wrapping_add(1) & 0x3f;
        let raw_header = (0u16..1024)
            .map(|h10| mk_header_at(uap, next_phase, h10))
            .find(|raw| header_candidates(raw)[next_alias as usize] != uap)
            .expect("a later header that rejects the alias");
        let mut packet = ClassicBtPacket {
            lap,
            ac_errors: 0,
            sync_offset: 0,
            rssi_db: -20,
            noise_db: -80,
            freq: 2441,
            timestamp: mk_ts(SLOT_NS),
            raw_header,
            has_header: true,
            payload: Vec::new(),
            uap: None,
            uap_verified: false,
            nap: None,
            header: None,
            decoded_payload: Vec::new(),
            crc_ok: false,
            clkn: None,
        };

        enrich(&mut packet, &mut tracker);

        assert_eq!(packet.uap, Some(uap));
        assert!(packet.header.is_some());
        assert_eq!(tracker.map[&lap].cands.len(), 1);
    }

    #[test]
    fn test_enrich_requires_crc_from_two_packets() {
        let (lap, uap, phase) = (0x0a1b2cu32, 0x4au8, 17u8);
        let make_packet = |clk6: u8, timestamp: crate::Timespec| {
            let raw_header = mk_header_at(uap, clk6, 1 | (0x4u16 << 3));
            ClassicBtPacket {
                lap,
                ac_errors: 0,
                sync_offset: 0,
                rssi_db: -20,
                noise_db: -80,
                freq: 2441,
                timestamp,
                raw_header,
                has_header: true,
                payload: mk_dh1_payload(uap, clk6, 3, &[0x11, 0x22, 0x33]),
                uap: None,
                uap_verified: false,
                nap: None,
                header: None,
                decoded_payload: Vec::new(),
                crc_ok: false,
                clkn: None,
            }
        };

        let mut tracker = PiconetTracker::new();
        tracker.map.insert(
            lap,
            Piconet {
                anchor: mk_ts(0),
                cands: vec![(phase, uap)],
                packets: 1,
                uap: Some(uap),
                uap_verified: false,
                crc_confirmations: 0,
                reanchors: 0,
                inconsistent_headers: 0,
                announced: false,
                clk_anchor: None,
            },
        );

        let mut first = make_packet(phase, mk_ts(0));
        assert!(!enrich(&mut first, &mut tracker));
        assert!(!first.uap_verified);
        assert!(!first.crc_ok);
        assert!(first.decoded_payload.is_empty());
        assert_eq!(tracker.map[&lap].crc_confirmations, 1);

        let next_phase = phase.wrapping_add(1) & 0x3f;
        let mut second = make_packet(next_phase, mk_ts(SLOT_NS));
        assert!(enrich(&mut second, &mut tracker));
        assert!(second.uap_verified);
        assert!(second.crc_ok);
        assert!(!second.decoded_payload.is_empty());
        assert_eq!(tracker.map[&lap].crc_confirmations, 2);
    }

    #[test]
    fn test_edr_crc_confirmation_requires_later_clock_consistent_packet() {
        let (lap, uap, phase) = (0x0a1b2cu32, 0x67u8, 23u8);
        let mut tracker = PiconetTracker::new();
        let first_ts = mk_ts(0);

        assert!(!tracker.confirm_uap_at(lap, uap, phase, &first_ts));
        // Multiple demodulation variants from one burst are still one piece of
        // evidence and cannot verify themselves.
        assert!(!tracker.confirm_uap_at(lap, uap, phase, &first_ts));
        assert!(!tracker.uap_verified(lap));

        let second_ts = mk_ts(SLOT_NS);
        assert!(tracker.confirm_uap_at(lap, uap, phase.wrapping_add(1) & 0x3f, &second_ts,));
        assert!(tracker.uap_verified(lap));
    }

    #[test]
    fn test_tracker_tolerates_bounded_header_outliers() {
        let mut piconet = Piconet {
            anchor: mk_ts(0),
            cands: vec![(17, 0x4a), (29, 0x71)],
            packets: 1,
            uap: Some(0x4a),
            uap_verified: false,
            crc_confirmations: 0,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: false,
            clk_anchor: None,
        };
        let inconsistent = [0u8; 64];

        for slot in 1..8 {
            piconet.observe(&mk_ts(slot * SLOT_NS), &inconsistent);
            assert_eq!(piconet.cands, vec![(17, 0x4a), (29, 0x71)]);
            assert_eq!(piconet.reanchors, 0);
        }

        piconet.observe(&mk_ts(8 * SLOT_NS), &inconsistent);
        assert_eq!(piconet.reanchors, 1);
        assert_eq!(piconet.cands.len(), 64);
        assert_eq!(piconet.uap, None);
    }

    #[test]
    fn test_tracker_preserves_unique_phase_through_noisy_run() {
        let mut piconet = Piconet {
            anchor: mk_ts(0),
            cands: vec![(17, 0x4a)],
            packets: 1,
            uap: Some(0x4a),
            uap_verified: false,
            crc_confirmations: 0,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: false,
            clk_anchor: None,
        };
        let inconsistent = [0u8; 64];

        for slot in 1..64 {
            piconet.observe(&mk_ts(slot * SLOT_NS), &inconsistent);
            assert_eq!(piconet.cands, vec![(17, 0x4a)]);
        }
        assert_eq!(piconet.reanchors, 0);

        piconet.observe(&mk_ts(64 * SLOT_NS), &inconsistent);
        assert_eq!(piconet.reanchors, 1);
        assert_eq!(piconet.cands.len(), 64);
    }

    #[test]
    fn test_tracker_never_discards_crc_verified_uap_on_header_noise() {
        let mut piconet = Piconet {
            anchor: mk_ts(0),
            cands: vec![(17, 0x4a)],
            packets: 1,
            uap: Some(0x4a),
            uap_verified: true,
            crc_confirmations: 2,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: false,
            clk_anchor: None,
        };
        let inconsistent = [0u8; 64];

        for slot in 1..300 {
            piconet.observe(&mk_ts(slot * SLOT_NS), &inconsistent);
        }
        assert_eq!(piconet.cands, vec![(17, 0x4a)]);
        assert_eq!(piconet.uap, Some(0x4a));
        assert!(piconet.uap_verified);
        assert_eq!(piconet.reanchors, 0);
    }

    /// Build a raw header for an explicit 10-bit header value at a given clock.
    fn mk_header_at(uap: u8, clk6: u8, h10: u16) -> [u8; 7] {
        let hec = (0u16..256)
            .map(|h| h as u8)
            .find(|&h| uap_from_hec(h10, h) == uap)
            .unwrap();
        let mut bits18 = [0u8; 18];
        for (j, bit) in bits18.iter_mut().enumerate().take(10) {
            *bit = ((h10 >> j) & 1) as u8;
        }
        for j in 0..8 {
            bits18[10 + j] = (hec >> j) & 1;
        }
        let wh = whitening_bits(clk6, 18);
        for i in 0..18 {
            bits18[i] ^= wh[i];
        }
        let mut bits54 = [0u8; 54];
        for i in 0..18 {
            bits54[i * 3] = bits18[i];
            bits54[i * 3 + 1] = bits18[i];
            bits54[i * 3 + 2] = bits18[i];
        }
        let mut raw = [0u8; 7];
        for i in 0..54 {
            raw[i / 8] |= (bits54[i] & 1) << (i % 8);
        }
        raw
    }

    #[test]
    fn test_fec23_roundtrip_single_error() {
        // 10 data bits -> 15 coded; flip one bit; decode must recover data.
        let data: Vec<u8> = (0..10u8).map(|i| (i * 7 + 3) & 1).collect();
        let mut coded = fec23_encode_bits(&data);
        coded[6] ^= 1; // inject a single-bit error in the block
        let decoded = fec23_decode_bits(&coded, 10);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_crc_roundtrip() {
        // Build payload bits, append CRC seeded by uap, verify it checks out
        // and that a wrong uap fails.
        let uap = 0x8bu8;
        let payload: Vec<u8> = (0..40u8).map(|i| (i * 5 + 1) & 1).collect();
        let crc = btcrc(&payload, uap);
        // Recompute over payload+crc: the trailing CRC makes the register a
        // deterministic function; simplest check is recompute-and-compare.
        assert_eq!(btcrc(&payload, uap), crc);
        assert_ne!(btcrc(&payload, uap.wrapping_add(1)), crc);
    }

    #[test]
    fn test_payload_header_parse() {
        // multi-slot header: llid=2, flow=1, length=300
        let mut d = vec![0u8; 16];
        d[0] = 0;
        d[1] = 1; // llid = 0b10 = 2
        d[2] = 1; // flow
        let length: u16 = 300;
        for j in 0..10 {
            d[3 + j] = ((length >> j) & 1) as u8;
        }
        let h = parse_payload_header(&d, true).unwrap();
        assert_eq!(h.llid, 2);
        assert_eq!(h.flow, 1);
        assert_eq!(h.length, 300);
        assert!(h.multi_slot);
    }

    #[test]
    fn test_fhs_roundtrip() {
        // Place LAP/UAP/NAP/CLK into a 144-bit body and read them back.
        let (lap, uap, nap) = (0x9e8b33u32, 0x5du8, 0x1234u16);
        let cod = 0x0a010cu32;
        let (lt, clk) = (3u8, 0x3ab_cdefu32 & 0x3ff_ffff);
        let mut body = vec![0u8; 144];
        let mut put = |start: usize, len: usize, val: u32| {
            for j in 0..len {
                body[start + j] = ((val >> j) & 1) as u8;
            }
        };
        put(34, 24, lap);
        put(64, 8, uap as u32);
        put(72, 16, nap as u32);
        put(88, 24, cod);
        put(112, 3, lt as u32);
        put(115, 26, clk);
        let info = parse_fhs_body(&body).unwrap();
        assert_eq!(info.lap, lap);
        assert_eq!(info.uap, uap);
        assert_eq!(info.nap, nap);
        assert_eq!(info.class_of_device, cod);
        assert_eq!(info.lt_addr, lt);
        assert_eq!(info.clk27_2, clk);
        assert_eq!(info.bd_addr(), 0x1234_5d9e_8b33);
    }

    // DH1 payload: header(8) + body(len*8) + CRC(16), whitened after the 18
    // logical header bits. No FEC on DH packets.
    fn mk_dh1_payload(uap: u8, clk6: u8, length: u8, body: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        // payload header: llid=0b01, flow=1, length (5 bits)
        data.push(1);
        data.push(0); // llid bits (lsb-first) = 0b01
        data.push(1); // flow
        for j in 0..5 {
            data.push((length >> j) & 1);
        }
        for &byte in body.iter().take(length as usize) {
            for j in 0..8 {
                data.push((byte >> j) & 1);
            }
        }
        let crc = btcrc(&data, uap);
        for j in 0..16 {
            data.push(((crc >> j) & 1) as u8);
        }
        let wh = whitening_slice(clk6, 18, data.len());
        data.iter().zip(wh).map(|(&b, w)| b ^ w).collect()
    }

    fn mk_edr_payload(uap: u8, clk6: u8, body: &[u8]) -> Vec<u8> {
        let mut data = vec![0u8; 16];
        data[0] = 1; // LLID = continuation
        data[2] = 1; // FLOW = go
        for bit in 0..10 {
            data[3 + bit] = ((body.len() >> bit) & 1) as u8;
        }
        for &byte in body {
            for bit in 0..8 {
                data.push((byte >> bit) & 1);
            }
        }
        let crc = btcrc(&data, uap);
        for bit in 0..16 {
            data.push(((crc >> bit) & 1) as u8);
        }
        let whitening = whitening_slice(clk6, 18, data.len());
        data.iter()
            .zip(whitening)
            .map(|(&bit, mask)| bit ^ mask)
            .collect()
    }

    #[test]
    fn test_dh1_payload_crc() {
        let (uap, clk6) = (0x8b, 22u8);
        let body = [0x41u8, 0x42, 0x43];
        let raw = mk_dh1_payload(uap, clk6, 3, &body);
        match decode_payload(&raw, clk6, 0x4) {
            PayloadCheck::Decoded(hdr_body, rx_crc) => {
                assert_eq!(btcrc(&hdr_body, uap), rx_crc);
                assert_ne!(btcrc(&hdr_body, uap.wrapping_add(1)), rx_crc);
            }
            _ => panic!("expected a decoded payload"),
        }
    }

    #[test]
    fn test_edr_payload_crc_and_dewhitening() {
        let (uap, clk6) = (0x4au8, 31u8);
        let body = [0x11, 0x22, 0x33, 0x44, 0x55];
        let raw = mk_edr_payload(uap, clk6, &body);
        let decoded = decode_edr_payload_bits(&raw, uap, clk6).expect("valid EDR CRC");

        assert_eq!(decoded.len(), 16 + body.len() * 8);
        let mut decoded_body = Vec::new();
        for byte_bits in decoded[16..].chunks_exact(8) {
            let mut byte = 0u8;
            for (bit, &value) in byte_bits.iter().enumerate() {
                byte |= (value & 1) << bit;
            }
            decoded_body.push(byte);
        }
        assert_eq!(decoded_body, body);
        assert!(decode_edr_payload_bits(&raw, uap ^ 0xff, clk6).is_none());
    }

    #[test]
    fn test_bad_payload_keeps_header_candidates() {
        let (uap, clk6) = (0x3cu8, 17u8);
        let ts = mk_ts(0);
        let raw_header = mk_header_at(uap, clk6, 1 | (0x4u16 << 3));
        let mut pn = Piconet {
            anchor: ts.clone(),
            cands: vec![(clk6, uap)],
            packets: 1,
            uap: Some(uap),
            uap_verified: true,
            crc_confirmations: 2,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: false,
            clk_anchor: None,
        };

        // Whitening still matches, but the CRC is seeded with another UAP.
        let payload = mk_dh1_payload(uap ^ 0xff, clk6, 3, &[0x11, 0x22, 0x33]);
        pn.disambiguate(&ts, &raw_header, &payload);

        assert_eq!(pn.cands, vec![(clk6, uap)]);
        assert_eq!(pn.uap, Some(uap));
    }

    #[test]
    fn test_uap_requires_two_crc_confirmations() {
        let (uap, clk6) = (0x4au8, 17u8);
        let ts = mk_ts(0);
        let raw_header = mk_header_at(uap, clk6, 1 | (0x4u16 << 3));
        let payload = mk_dh1_payload(uap, clk6, 3, &[0x11, 0x22, 0x33]);
        let mut pn = Piconet {
            anchor: ts.clone(),
            cands: vec![(clk6, uap)],
            packets: 1,
            uap: Some(uap),
            uap_verified: false,
            crc_confirmations: 0,
            reanchors: 0,
            inconsistent_headers: 0,
            announced: false,
            clk_anchor: None,
        };

        pn.disambiguate(&ts, &raw_header, &payload);
        assert!(!pn.uap_verified);
        assert_eq!(pn.crc_confirmations, 1);
        pn.disambiguate(&ts, &raw_header, &payload);
        assert!(pn.uap_verified);
        assert_eq!(pn.crc_confirmations, 2);
    }

    #[test]
    fn test_disambiguate_preserves_true_candidate() {
        let true_uap = 0x3cu8;
        let base_clk6: u32 = 41;
        let lap = 0x9e8b33u32;
        let mut tracker = PiconetTracker::new();
        let steps = [
            1u64, 2, 3, 5, 6, 1, 4, 2, 3, 1, 6, 2, 5, 1, 3, 2, 4, 1, 2, 3,
        ];
        let body = [0x11u8, 0x22, 0x33, 0x44];
        let mut slot: u64 = 0;
        for (i, &st) in steps.iter().enumerate() {
            slot += st;
            let clk6 = ((base_clk6 + slot as u32) & 0x3f) as u8;
            // DH1 header (type 0x4), varying lt_addr/flags for realism.
            let h10: u16 = (i as u16 & 0x7)              // lt_addr
                | (0x4u16 << 3)                          // type = DH1
                | (((i as u16 >> 1) & 1) << 7); // flow
            let raw_hdr = mk_header_at(true_uap, clk6, h10);
            let ts = crate::Timespec {
                tv_sec: (slot * SLOT_NS) / 1_000_000_000,
                tv_nsec: (slot * SLOT_NS) % 1_000_000_000,
            };
            let payload = mk_dh1_payload(true_uap, clk6, 4, &body);
            tracker.feed(lap, &raw_hdr, Some(&payload), &ts);
        }
        let survivors = tracker.candidate_uap_set(lap);
        assert!(
            survivors.contains(&true_uap),
            "true UAP missing from {:?}",
            survivors
        );
        assert!(
            survivors.len() <= 2,
            "too many UAP aliases: {:?}",
            survivors
        );
    }

    // Vectors lifted from libbtbb's own tests (tests/test_header.c,
    // tests/test_fec23.c). These are interoperability facts and confirm the
    // clean-room math matches real Bluetooth.
    #[test]
    fn test_hec_vectors_libbtbb() {
        let vecs: [(u8, u16, u8); 20] = [
            (0x00, 0x123, 0xe1),
            (0x47, 0x123, 0x06),
            (0x00, 0x124, 0x32),
            (0x47, 0x124, 0xd5),
            (0x00, 0x125, 0x5a),
            (0x47, 0x125, 0xbd),
            (0x00, 0x126, 0xe2),
            (0x47, 0x126, 0x05),
            (0x00, 0x127, 0x8a),
            (0x47, 0x127, 0x6d),
            (0x00, 0x11b, 0x9e),
            (0x47, 0x11b, 0x79),
            (0x00, 0x11c, 0x4d),
            (0x47, 0x11c, 0xaa),
            (0x00, 0x11d, 0x25),
            (0x47, 0x11d, 0xc2),
            (0x00, 0x11e, 0x9d),
            (0x47, 0x11e, 0x7a),
            (0x00, 0x11f, 0xf5),
            (0x47, 0x11f, 0x12),
        ];
        for (uap, data, hec) in vecs {
            assert_eq!(
                uap_from_hec(data, hec),
                uap,
                "data={:#x} hec={:#x}",
                data,
                hec
            );
        }
    }

    #[test]
    fn test_whitening_vectors_libbtbb() {
        let cases: [(u8, [u8; 18]); 4] = [
            (0, [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]),
            (1, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]),
            (17, [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]),
            (63, [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1]),
        ];
        for (clock, expected) in cases {
            assert_eq!(whitening_bits(clock, expected.len()), expected);
        }
    }

    #[test]
    fn test_fec23_vectors_libbtbb() {
        // 15-bit inputs (data 0..9, parity 10..14) -> 10 data bits out.
        let cases: [([u8; 15], [u8; 10]); 12] = [
            (
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            ),
            (
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ),
            (
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ),
            (
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ),
            // single-error rows: parity implies the data bit, corrected on decode
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];
        for (input, expected) in cases {
            let out = fec23_decode_bits(&input, 10);
            assert_eq!(out.as_slice(), &expected[..], "input {:?}", input);
        }
    }

    // Generate a valid 64-bit sync word for a LAP by solving for the 34 BCH
    // parity bits that zero the syndrome (uses the existing decode tables +
    // linearity; no external generator). Returns the sync word in air/host bit
    // order (bit i = symbol i).
    fn gen_syncword(lap: u32, barker6: u64) -> Option<u64> {
        let base: u64 = ((lap as u64 & 0xffffff) << 34) | ((barker6 & 0x3f) << 58);
        let s0 = gen_syndrome(base ^ PN);
        // gen_syndrome(P) = P_low32 ^ TABLE4[b], b = bits 32..33 of P.
        for b in 0u64..4 {
            let t = SW_CHECK_TABLE4[b as usize];
            if (t >> 32) & 3 == (s0 >> 32) & 3 {
                let p_low32 = (s0 ^ t) & 0xffff_ffff;
                let p = (b << 32) | p_low32;
                let syncword = base | p;
                if gen_syndrome(syncword ^ PN) == 0 {
                    return Some(syncword);
                }
            }
        }
        None
    }

    #[test]
    fn test_gen_syncword_roundtrips_through_find_ac() {
        let lap = 0x0A1B2Cu32;
        let sm = SyndromeMap::new(1);
        // Find a barker that find_ac accepts cleanly.
        let mut found = None;
        for barker in 0u64..64 {
            if let Some(sw) = gen_syncword(lap, barker) {
                // Lay the 64 sync bits into a stream (LSB-first), pad so the
                // sliding window has room.
                let mut stream = vec![0u8; 200];
                for (i, bit) in stream.iter_mut().enumerate().take(64) {
                    *bit = ((sw >> i) & 1) as u8;
                }
                if let Some((got_lap, off, errs)) = find_ac(&stream, 1, &sm) {
                    if got_lap == lap && off == 0 && errs == 0 {
                        found = Some((barker, sw));
                        break;
                    }
                }
            }
        }
        assert!(
            found.is_some(),
            "no barker produced a clean access code for {:#x}",
            lap
        );
    }

    fn mk_fhs_payload(lap: u32, uap: u8, nap: u16, clk6: u8) -> Vec<u8> {
        let mut body = vec![0u8; 144];
        {
            let mut put = |start: usize, len: usize, val: u32| {
                for j in 0..len {
                    body[start + j] = ((val >> j) & 1) as u8;
                }
            };
            put(34, 24, lap);
            put(64, 8, uap as u32);
            put(72, 16, nap as u32);
            put(88, 24, 0x5a0102); // class of device
            put(112, 3, 3); // lt_addr
            put(115, 26, 0x003a_bcde); // clk27-2
        }
        let crc = btcrc(&body, uap);
        let mut info = body;
        for j in 0..16 {
            info.push(((crc >> j) & 1) as u8);
        }
        let wh = whitening_slice(clk6, 18, info.len());
        let whitened: Vec<u8> = info.iter().zip(wh).map(|(&b, w)| b ^ w).collect();
        fec23_encode_bits(&whitened) // 160 -> 240
    }

    fn working_syncword(lap: u32, sm: &SyndromeMap) -> u64 {
        (0u64..64)
            .find_map(|b| {
                gen_syncword(lap, b).and_then(|sw| {
                    let mut st = vec![0u8; 200];
                    for (i, bit) in st.iter_mut().enumerate().take(64) {
                        *bit = ((sw >> i) & 1) as u8;
                    }
                    find_ac(&st, 1, sm)
                        .filter(|&(l, o, e)| l == lap && o == 0 && e == 0)
                        .map(|_| sw)
                })
            })
            .expect("a barker yields a clean access code")
    }

    fn append_sync_and_trailer(bits: &mut Vec<u8>, syncword: u64) {
        for i in 0..64 {
            bits.push(((syncword >> i) & 1) as u8);
        }
        let mut previous = ((syncword >> 63) & 1) as u8;
        for _ in 0..4 {
            previous ^= 1;
            bits.push(previous);
        }
    }

    // End-to-end through the real detect() + enrich() path, using a synthetic
    // (fake) address: an FHS packet must yield the full BD_ADDR in one shot.
    #[test]
    fn test_synthetic_fhs_recovers_full_bd_addr() {
        let sm = SyndromeMap::new(1);
        let (lap, uap, nap) = (0x0A1B2Cu32, 0x5Du8, 0x1234u16);
        let sw = working_syncword(lap, &sm);
        let clk6 = 25u8;
        let raw_header = mk_header_at(uap, clk6, 1 | (0x2u16 << 3)); // FHS type
        let payload = mk_fhs_payload(lap, uap, nap, clk6);

        let mut bits = Vec::new();
        append_sync_and_trailer(&mut bits, sw);
        for i in 0..54 {
            bits.push((raw_header[i / 8] >> (i % 8)) & 1);
        }
        bits.extend_from_slice(&payload);
        bits.resize(bits.len() + 64, 0);

        let ts = crate::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let mut pkt = detect(&bits, 2441, -40, -90, ts, &sm).expect("detects a BT packet");
        assert_eq!(pkt.lap, lap);

        let mut tracker = PiconetTracker::new();
        assert!(
            enrich(&mut pkt, &mut tracker),
            "should resolve on first FHS"
        );
        assert_eq!(pkt.uap, Some(uap));
        assert!(pkt.uap_verified);
        assert_eq!(pkt.nap, Some(nap));
        let header = pkt
            .header
            .expect("FHS header decoded at its verified clock");
        assert_eq!(header.pkt_type, 0x2);
    }

    #[test]
    fn test_decoded_payload_surfaced() {
        let sm = SyndromeMap::new(1);
        let (lap, uap) = (0x0A1B2Cu32, 0x8Bu8);
        let sw = working_syncword(lap, &sm);
        let clk6 = 22u8;
        let body = [0x41u8, 0x42, 0x43, 0x44];
        let raw_header = mk_header_at(uap, clk6, 1 | (0x4u16 << 3)); // DH1
        let payload = mk_dh1_payload(uap, clk6, 4, &body);

        let mut bits = Vec::new();
        append_sync_and_trailer(&mut bits, sw);
        for i in 0..54 {
            bits.push((raw_header[i / 8] >> (i % 8)) & 1);
        }
        bits.extend_from_slice(&payload);
        bits.resize(bits.len() + 64, 0);

        let ts = crate::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let mut pkt = detect(&bits, 2441, -40, -90, ts, &sm).expect("detects");
        // Preseed the UAP (as an FHS or prior convergence would have), then enrich.
        let mut tracker = PiconetTracker::new();
        tracker.set_uap(lap, uap);
        enrich(&mut pkt, &mut tracker);

        assert_eq!(pkt.uap, Some(uap));
        assert!(pkt.crc_ok, "payload CRC should validate");
        let header = pkt
            .header
            .expect("payload CRC selects a consistent header clock");
        assert_eq!(header.pkt_type, 0x4);
        // payload = 1 header byte + 4 body bytes
        assert_eq!(pkt.decoded_payload.len(), 5);
        assert_eq!(&pkt.decoded_payload[1..5], &body[..]);
    }

    #[test]
    fn test_tracked_clock_controls_header_decode() {
        let (lap, uap, clk6) = (0x0A1B2Cu32, 0x8Bu8, 22u8);
        let ts = mk_ts(10 * SLOT_NS);
        let raw_header = mk_header_at(uap, clk6, 1 | (0x4u16 << 3));
        let mut tracker = PiconetTracker::new();
        tracker.set_uap_at(lap, uap, clk6, &ts);

        let header = tracker
            .decode_tracked_header(lap, &raw_header, &ts)
            .expect("tracked clock validates the header");
        assert_eq!(header.clk6, clk6);
        assert_eq!(header.lt_addr, 1);
        assert_eq!(header.pkt_type, 0x4);
    }

    #[test]
    fn test_clkn_from_fhs_advances() {
        let mut t = PiconetTracker::new();
        let lap = 0x0A1B2Cu32;
        t.set_uap(lap, 0x5d);
        // FHS at t=0 with CLK27-2 = 0x100000
        let clk27_2 = 0x100000u32;
        let t0 = crate::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        t.set_clock_anchor(lap, clk27_2, &t0);
        let base = (clk27_2 << 2) & 0x0fff_ffff;
        assert_eq!(t.clkn_at(lap, &t0), Some(base));
        // 1 second later = 3_200_000 ticks (1e9 / 312500)
        let t1 = crate::Timespec {
            tv_sec: 1,
            tv_nsec: 0,
        };
        // CLK ticks every 312.5us -> 3200 ticks per second
        let expected = base.wrapping_add(3_200) & 0x0fff_ffff;
        assert_eq!(t.clkn_at(lap, &t1), Some(expected));
    }
}
