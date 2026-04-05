// Copyright 2026 CEMAXECUTER LLC
//
// BLE PDU fuzzer: generates malformed BLE packets to test device robustness.
//
// Mutation strategies:
// - Bit flips (single, double, nibble, byte)
// - Boundary values (0x00, 0xFF, max length)
// - Length field corruption (oversized, undersized, zero)
// - PDU type enumeration (all 16 types)
// - Truncation (progressively shorter packets)
// - Extension (oversized payloads)
// - Field-specific: corrupt AA, CRC, header flags
//
// Output: raw PDU bytes ready for whitening + CRC + GFSK modulation.

use std::collections::VecDeque;

/// Fuzz strategy identifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FuzzStrategy {
    /// Flip single bits across the PDU
    BitFlip,
    /// Set bytes to boundary values (0x00, 0xFF, 0x7F, 0x80)
    BoundaryValues,
    /// Corrupt the length field
    LengthCorrupt,
    /// Try all 16 PDU types
    PduTypeEnum,
    /// Progressively truncate the packet
    Truncation,
    /// Extend payload beyond declared length
    Extension,
    /// Random byte mutations
    RandomBytes,
    /// Oversized advertisement data
    OversizedAdv,
}

/// A single fuzz test case.
#[derive(Debug, Clone)]
pub struct FuzzCase {
    pub strategy: FuzzStrategy,
    pub description: String,
    /// Raw PDU bytes (header + payload, before whitening)
    pub pdu: Vec<u8>,
    /// Test case number within the strategy
    pub index: u32,
}

/// BLE PDU fuzzer.
pub struct BleFuzzer {
    /// Base PDU to mutate (typically a valid advertising PDU)
    base_pdu: Vec<u8>,
    /// Queue of pending test cases
    queue: VecDeque<FuzzCase>,
    /// Total cases generated
    total_generated: u32,
    /// RNG state (simple LCG for reproducibility)
    rng_state: u64,
}

impl BleFuzzer {
    /// Create a new fuzzer from a base PDU.
    /// The base PDU should be a valid BLE PDU (header + payload bytes).
    pub fn new(base_pdu: Vec<u8>) -> Self {
        let mut fuzzer = Self {
            base_pdu,
            queue: VecDeque::new(),
            total_generated: 0,
            rng_state: 0xDEADBEEFCAFE,
        };
        fuzzer.generate_all();
        fuzzer
    }

    /// Create a fuzzer with a default ADV_IND advertising PDU.
    /// Advertises from MAC 00:11:22:33:44:55 with flags + name "FUZZ".
    pub fn new_default_adv() -> Self {
        // ADV_IND PDU: type=0, TxAdd=0, length=15
        // AdvA(6) + AD: Flags(3) + Name(6)
        let pdu = vec![
            0x00,       // header: PDU type 0 (ADV_IND), TxAdd=0, RxAdd=0
            15,         // length
            // AdvA (6 bytes)
            0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
            // AD structures
            0x02, 0x01, 0x06,   // Flags: LE General Discoverable + BR/EDR Not Supported
            0x05, 0x09, b'F', b'U', b'Z', b'Z', // Complete Local Name: "FUZZ"
        ];
        Self::new(pdu)
    }

    /// Set a custom RNG seed for reproducible fuzzing.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = seed;
        self.queue.clear();
        self.total_generated = 0;
        self.generate_all();
    }

    /// Get the next fuzz case, or None if all cases exhausted.
    pub fn next(&mut self) -> Option<FuzzCase> {
        self.queue.pop_front()
    }

    /// Number of remaining test cases.
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }

    /// Total test cases generated.
    pub fn total(&self) -> u32 {
        self.total_generated
    }

    fn generate_all(&mut self) {
        self.gen_bit_flips();
        self.gen_boundary_values();
        self.gen_length_corrupt();
        self.gen_pdu_type_enum();
        self.gen_truncation();
        self.gen_extension();
        self.gen_random_bytes(50);
        self.gen_oversized_adv();
    }

    fn push_case(&mut self, strategy: FuzzStrategy, desc: String, pdu: Vec<u8>) {
        self.total_generated += 1;
        self.queue.push_back(FuzzCase {
            strategy,
            description: desc,
            pdu,
            index: self.total_generated,
        });
    }

    /// Flip each bit in the PDU one at a time.
    fn gen_bit_flips(&mut self) {
        for byte_idx in 0..self.base_pdu.len() {
            for bit in 0..8u8 {
                let mut pdu = self.base_pdu.clone();
                pdu[byte_idx] ^= 1 << bit;
                self.push_case(
                    FuzzStrategy::BitFlip,
                    format!("flip byte[{}] bit {}", byte_idx, bit),
                    pdu,
                );
            }
        }
    }

    /// Set each byte to boundary values.
    fn gen_boundary_values(&mut self) {
        let boundaries = [0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF];
        for byte_idx in 0..self.base_pdu.len() {
            for &val in &boundaries {
                if self.base_pdu[byte_idx] == val {
                    continue; // skip if already this value
                }
                let mut pdu = self.base_pdu.clone();
                pdu[byte_idx] = val;
                self.push_case(
                    FuzzStrategy::BoundaryValues,
                    format!("byte[{}] = 0x{:02X}", byte_idx, val),
                    pdu,
                );
            }
        }
    }

    /// Corrupt the length field (byte 1 in BLE PDU header).
    fn gen_length_corrupt(&mut self) {
        if self.base_pdu.len() < 2 {
            return;
        }
        let lengths = [0, 1, 2, 37, 38, 39, 127, 128, 255];
        for &len in &lengths {
            let mut pdu = self.base_pdu.clone();
            pdu[1] = len;
            self.push_case(
                FuzzStrategy::LengthCorrupt,
                format!("length={}", len),
                pdu,
            );
        }
    }

    /// Enumerate all 16 PDU types.
    fn gen_pdu_type_enum(&mut self) {
        if self.base_pdu.is_empty() {
            return;
        }
        for pdu_type in 0..16u8 {
            let mut pdu = self.base_pdu.clone();
            pdu[0] = (pdu[0] & 0xF0) | (pdu_type & 0x0F);
            self.push_case(
                FuzzStrategy::PduTypeEnum,
                format!("pdu_type={}", pdu_type),
                pdu,
            );
        }
    }

    /// Progressively truncate the packet.
    fn gen_truncation(&mut self) {
        for trunc_len in 1..self.base_pdu.len() {
            let pdu = self.base_pdu[..trunc_len].to_vec();
            self.push_case(
                FuzzStrategy::Truncation,
                format!("truncate to {} bytes", trunc_len),
                pdu,
            );
        }
    }

    /// Extend payload beyond the declared length.
    fn gen_extension(&mut self) {
        let extra_sizes = [1, 10, 50, 100, 200];
        for &extra in &extra_sizes {
            let mut pdu = self.base_pdu.clone();
            pdu.extend(vec![0xAA; extra]);
            self.push_case(
                FuzzStrategy::Extension,
                format!("extend +{} bytes (0xAA fill)", extra),
                pdu,
            );
        }
    }

    /// Generate random byte mutations.
    fn gen_random_bytes(&mut self, count: usize) {
        for _ in 0..count {
            let mut pdu = self.base_pdu.clone();
            if pdu.is_empty() {
                continue;
            }
            // Mutate 1-4 random positions
            let n_mutations = (self.next_rng() % 4 + 1) as usize;
            for _ in 0..n_mutations {
                let pos = (self.next_rng() as usize) % pdu.len();
                let val = (self.next_rng() & 0xFF) as u8;
                pdu[pos] = val;
            }
            self.push_case(
                FuzzStrategy::RandomBytes,
                format!("{} random byte mutations", n_mutations),
                pdu,
            );
        }
    }

    /// Generate oversized advertisement data.
    fn gen_oversized_adv(&mut self) {
        // BLE 4.x max adv data: 31 bytes. BLE 5 extended: 254 bytes.
        // Generate packets that exceed these limits.
        for &size in &[32, 64, 128, 254, 255] {
            let mut pdu = vec![0x00, size as u8]; // ADV_IND, length
            // AdvA (6 bytes)
            pdu.extend_from_slice(&[0x55, 0x44, 0x33, 0x22, 0x11, 0x00]);
            // Fill with repeated AD structure
            while pdu.len() < 2 + size {
                pdu.push(0x41); // 'A'
            }
            pdu.truncate(2 + size);
            self.push_case(
                FuzzStrategy::OversizedAdv,
                format!("oversized adv data: {} bytes", size),
                pdu,
            );
        }
    }

    fn next_rng(&mut self) -> u64 {
        // Simple LCG for reproducible fuzzing
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng_state >> 33
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_adv_fuzzer() {
        let mut fuzzer = BleFuzzer::new_default_adv();
        assert!(fuzzer.total() > 0);
        assert!(fuzzer.remaining() > 0);

        // Should generate hundreds of test cases
        assert!(fuzzer.total() > 100);
    }

    #[test]
    fn test_fuzzer_exhaustion() {
        let pdu = vec![0x00, 0x02, 0x01, 0x06]; // tiny PDU
        let mut fuzzer = BleFuzzer::new(pdu);
        let total = fuzzer.total();

        let mut count = 0u32;
        while let Some(_case) = fuzzer.next() {
            count += 1;
        }
        assert_eq!(count, total);
        assert_eq!(fuzzer.remaining(), 0);
    }

    #[test]
    fn test_bit_flip_count() {
        let pdu = vec![0x00, 0x02, 0x01, 0x06]; // 4 bytes
        let fuzzer = BleFuzzer::new(pdu);
        // Bit flips: 4 bytes * 8 bits = 32 cases
        // Plus other strategies
        assert!(fuzzer.total() >= 32);
    }

    #[test]
    fn test_reproducible_rng() {
        let mut f1 = BleFuzzer::new_default_adv();
        f1.set_seed(42);
        let mut f2 = BleFuzzer::new_default_adv();
        f2.set_seed(42);

        // Same seed should produce same sequence
        for _ in 0..10 {
            let c1 = f1.next().unwrap();
            let c2 = f2.next().unwrap();
            assert_eq!(c1.pdu, c2.pdu);
            assert_eq!(c1.strategy, c2.strategy);
        }
    }

    #[test]
    fn test_pdu_type_enum() {
        let pdu = vec![0x00, 0x06, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00];
        let fuzzer = BleFuzzer::new(pdu);

        // Collect all PDU type enum cases
        let type_cases: Vec<_> = fuzzer
            .queue
            .iter()
            .filter(|c| c.strategy == FuzzStrategy::PduTypeEnum)
            .collect();
        assert_eq!(type_cases.len(), 16); // all 16 types
    }

    #[test]
    fn test_length_corrupt() {
        let pdu = vec![0x00, 0x06, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00];
        let fuzzer = BleFuzzer::new(pdu);

        let len_cases: Vec<_> = fuzzer
            .queue
            .iter()
            .filter(|c| c.strategy == FuzzStrategy::LengthCorrupt)
            .collect();
        assert!(len_cases.len() >= 8); // various length values
    }
}
