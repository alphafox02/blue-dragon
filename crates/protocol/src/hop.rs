// Copyright 2025-2026 CEMAXECUTER LLC

//! BR/EDR channel hopping: the basic hop selection kernel for the connection
//! state (79 channels, no adaptive hopping), Core Specification Vol 2 Part B
//! section 2.6.
//!
//! The channel for each master clock value follows from the lower 28 bits of
//! the master's address (LAP and the low four bits of the UAP) and clock
//! bits CLK27-1: the address and clock are combined by an adder, an XOR, a
//! five-bit butterfly permutation driven by fourteen control bits, and a
//! second adder modulo 79, whose output indexes a register bank holding the
//! even channels followed by the odd ones.

/// Butterfly stages of PERM5: control bit P_i swaps bits (FIRST[i], SECOND[i])
/// of the 5-bit input; P13 acts first and P0 last.
const FIRST: [u8; 14] = [0, 2, 1, 3, 0, 1, 0, 3, 1, 0, 2, 1, 0, 1];
const SECOND: [u8; 14] = [1, 3, 2, 4, 4, 3, 2, 4, 4, 3, 4, 3, 3, 2];

pub const CHANNELS: u32 = 79;

/// Hop selection for one piconet (one master address).
#[derive(Debug, Clone)]
pub struct HopKernel {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
}

impl HopKernel {
    /// The kernel for a master with this LAP and UAP.
    pub fn new(lap: u32, uap: u8) -> Self {
        let addr = ((uap as u32 & 0x0f) << 24) | (lap & 0x00ff_ffff);
        let bit = |i: u32| (addr >> i) & 1;
        Self {
            a: (addr >> 23) & 0x1f,
            b: (addr >> 19) & 0x0f,
            // A8, A6, A4, A2, A0
            c: (bit(8) << 4) | (bit(6) << 3) | (bit(4) << 2) | (bit(2) << 1) | bit(0),
            d: (addr >> 10) & 0x1ff,
            // A13, A11, A9, A7, A5, A3, A1
            e: (bit(13) << 6)
                | (bit(11) << 5)
                | (bit(9) << 4)
                | (bit(7) << 3)
                | (bit(5) << 2)
                | (bit(3) << 1)
                | bit(1),
        }
    }

    /// The channel (0..78, i.e. 2402 + n MHz) for master clock `clk`
    /// (CLK27-0; bit 0 is ignored, as both halves of a slot share it).
    pub fn channel(&self, clk: u32) -> u8 {
        let x = (clk >> 2) & 0x1f;
        let y1 = (clk >> 1) & 1;
        let y2 = y1 << 5;
        let a = self.a ^ ((clk >> 21) & 0x1f);
        let c = self.c ^ ((clk >> 16) & 0x1f);
        let d = self.d ^ ((clk >> 7) & 0x1ff);
        let f = (16 * ((clk >> 7) & 0x1f_ffff)) % CHANNELS;
        let z = ((x + a) % 32) ^ self.b;
        // P13-9 are C XOR Y1 (Y1 repeated over five bits), P8-0 are D.
        let p = (((c ^ (y1 * 0x1f)) & 0x1f) << 9) | d;
        let z = perm5(z, p);
        let index = (z + self.e + f + y2) % CHANNELS;
        bank(index)
    }
}

/// The five-bit butterfly permutation with control word `p` (P13-0).
fn perm5(z: u32, p: u32) -> u32 {
    let mut z = z;
    for i in (0..14).rev() {
        if (p >> i) & 1 == 1 {
            let (s, t) = (FIRST[i] as u32, SECOND[i] as u32);
            if ((z >> s) ^ (z >> t)) & 1 == 1 {
                z ^= (1 << s) | (1 << t);
            }
        }
    }
    z
}

/// The register bank: the even channels in order, then the odd ones.
fn bank(index: u32) -> u8 {
    ((2 * index) % CHANNELS) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_holds_even_then_odd_channels() {
        assert_eq!(bank(0), 0);
        assert_eq!(bank(39), 78);
        assert_eq!(bank(40), 1);
        assert_eq!(bank(78), 77);
    }

    #[test]
    fn perm5_with_no_control_bits_is_identity() {
        for z in 0..32 {
            assert_eq!(perm5(z, 0), z);
        }
    }

    #[test]
    fn perm5_is_a_permutation_of_bits() {
        // Swapping bits never changes how many are set.
        for z in 0..32u32 {
            for p in [0x0001, 0x2000, 0x1555, 0x2aaa, 0x3fff] {
                assert_eq!(perm5(z, p).count_ones(), z.count_ones());
            }
        }
    }

    #[test]
    fn hops_visit_every_channel() {
        // Over a stretch of clock the kernel visits all 79 channels.
        let k = HopKernel::new(0x9e8b33, 0x00);
        let mut seen = [false; 79];
        for clk in (0..0x4000u32).step_by(2) {
            seen[k.channel(clk) as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// (LAP, UAP, CLK, channel) computed independently by libbtbb's kernel
    /// (the Ubertooth project's implementation) for pseudo-random inputs.
    const REFERENCE: [(u32, u8, u32, u8); 48] = [
        (0x3dc167, 0x27, 0xacca384, 61),
        (0xdaa96f, 0x1c, 0x7d5ac5e, 56),
        (0xd1dcf1, 0xad, 0x41b4eea, 57),
        (0xfe533d, 0xc4, 0x2c375f4, 56),
        (0x61eb2e, 0x36, 0xca26d26, 59),
        (0xe7099a, 0x1b, 0x76d98d8, 12),
        (0x4fde85, 0x27, 0x3fff786, 73),
        (0xa1900e, 0x2e, 0xfac1a9c, 8),
        (0xa1d620, 0xc0, 0xb559cde, 63),
        (0xeb4de1, 0x98, 0x040544c, 48),
        (0x42b7e6, 0x3f, 0x98f157a, 37),
        (0xf7301b, 0x20, 0x83e567a, 28),
        (0x2dfa7e, 0xfa, 0xa55d72e, 41),
        (0xbb7b85, 0x58, 0xf8b2d36, 35),
        (0xad1f52, 0x88, 0xb472948, 44),
        (0x010ea5, 0x3f, 0x8c90410, 50),
        (0x995c88, 0xd8, 0x6599896, 58),
        (0xa893c4, 0xdf, 0x0422a18, 11),
        (0xfff70a, 0x57, 0x8f6cb6c, 19),
        (0xf622ea, 0xef, 0xcdbf5de, 15),
        (0x93cc7f, 0x0e, 0x326b598, 26),
        (0x2fd3df, 0x71, 0x7d28972, 52),
        (0x028d4e, 0xc0, 0x7412c68, 58),
        (0x77c02b, 0x53, 0x1f67666, 65),
        (0x9026a1, 0x0f, 0x6f2dab0, 47),
        (0x85f516, 0x11, 0xf5d41c6, 12),
        (0x46dc5f, 0x97, 0xbfa94bc, 40),
        (0xc2d5a8, 0x51, 0x2edc824, 75),
        (0x96932f, 0x10, 0xfdc0c60, 2),
        (0x3e6caa, 0x84, 0xfe98190, 41),
        (0x69ca7b, 0x6f, 0x3dce4ea, 55),
        (0x1d2ea2, 0x8b, 0xe8ba59a, 12),
        (0x82c669, 0x05, 0x7f92728, 64),
        (0x4eabd9, 0x4e, 0x00f2f52, 15),
        (0x5f69e3, 0x9e, 0x214156e, 33),
        (0xbeb257, 0x65, 0x3e4c148, 21),
        (0xee4090, 0xa2, 0x4825f8a, 43),
        (0xad5fe4, 0xb3, 0x999698c, 60),
        (0x0b38d7, 0x36, 0x222b6ca, 8),
        (0x76a408, 0x05, 0x6e745b0, 42),
        (0x068de2, 0x5a, 0x491c202, 68),
        (0xcdb20b, 0xb6, 0x5a106c0, 12),
        (0x646198, 0x0b, 0xc9e717e, 30),
        (0x4ce2f5, 0x4d, 0x7a1554e, 51),
        (0x8f86a0, 0x62, 0xbfbb312, 65),
        (0x36878f, 0x1c, 0x59f156a, 18),
        (0x65fa64, 0xb3, 0x744860c, 73),
        (0x5f2a5f, 0xe2, 0xebf8aa4, 69),
    ];

    #[test]
    fn matches_an_independent_implementation() {
        for &(lap, uap, clk, channel) in REFERENCE.iter() {
            assert_eq!(HopKernel::new(lap, uap).channel(clk), channel, "LAP {:06x} UAP {:02x} CLK {:07x}", lap, uap, clk);
        }
    }
}
