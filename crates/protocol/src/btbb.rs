// Copyright 2025-2026 CEMAXECUTER LLC

use std::collections::HashMap;

const MAX_BARKER_ERRORS: u8 = 1;
const DEFAULT_AC: u64 = 0xcc7b7268ff614e1b;
const PN: u64 = 0x83848D96BBCC54FC;

static BARKER_DISTANCE: [u8; 128] = [
    3,3,3,2,3,2,2,1,2,3,3,3,3,3,3,2,2,3,3,3,3,3,3,2,1,2,2,3,2,3,3,3,
    3,2,2,1,2,1,1,0,3,3,3,2,3,2,2,1,3,3,3,2,3,2,2,1,2,3,3,3,3,3,3,2,
    2,3,3,3,3,3,3,2,1,2,2,3,2,3,3,3,1,2,2,3,2,3,3,3,0,1,1,2,1,2,2,3,
    3,3,3,2,3,2,2,1,2,3,3,3,3,3,3,2,2,3,3,3,3,3,3,2,1,2,2,3,2,3,3,3,
];

static BARKER_CORRECT: [u64; 128] = [
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0xb000000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0xb000000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0xb000000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0xb000000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0xb000000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
    0xb000000000000000, 0x4e00000000000000, 0x4e00000000000000, 0x4e00000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0xb000000000000000,
    0xb000000000000000, 0xb000000000000000, 0xb000000000000000, 0x4e00000000000000,
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
    pub rssi_db: i32,
    pub noise_db: i32,
    pub freq: u32,
    pub timestamp: crate::Timespec,
    pub raw_header: [u8; 7],
    pub has_header: bool,
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
        rssi_db: rssi,
        noise_db: noise,
        freq,
        timestamp,
        raw_header: [0; 7],
        has_header: false,
    };

    // Extract 54 raw header bits after the 64-bit sync word
    let header_start = ac_offset + 64;
    if header_start + 54 <= bits.len() {
        pkt.has_header = true;
        for i in 0..54 {
            pkt.raw_header[i / 8] |= (bits[header_start + i] & 1) << (i % 8);
        }
    }

    Some(pkt)
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
}
