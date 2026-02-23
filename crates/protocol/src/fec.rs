// Copyright 2025-2026 CEMAXECUTER LLC

/// Forward error correction for BLE LE Coded PHY.
///
/// Convolutional code per Bluetooth Core Spec v5.4, Vol 6, Part B, Section 3.3:
///   Rate 1/2, constraint length K=4 (3 memory bits, 8 trellis states)
///   g0(D) = 1 + D + D^2 + D^3  (octal 17, 0b1111 = 15)
///   g1(D) = 1 + D^2 + D^3       (octal 15, 0b1011 in register layout)
///
/// Pattern mapping: S=2 maps coded bit 0->"11", 1->"01".
///                  S=8 maps coded bit 0->"11110000", 1->"00001111".

/// Number of memory bits in the shift register.
const FEC_MEM: usize = 3;
/// Number of trellis states (2^FEC_MEM).
const FEC_STATES: usize = 1 << FEC_MEM;
/// g0 polynomial: 1+D+D^2+D^3 = 0b1111
const G0: u8 = 0b1111;
/// g1 polynomial: 1+D^2+D^3 (octal 15)
/// Register [D^0, D^1, D^2, D^3] = [bit3, bit2, bit1, bit0]
/// Taps D^0(bit3) + D^2(bit1) + D^3(bit0) = 0b1011
const G1: u8 = 0b1011;

/// Compute parity of a byte (XOR of all bits).
#[inline]
fn parity(x: u8) -> u8 {
    let mut p = x;
    p ^= p >> 4;
    p ^= p >> 2;
    p ^= p >> 1;
    p & 1
}

/// Convolutional encode: rate 1/2, K=4, g0=17o, g1=15o.
/// Takes data bits, produces coded bits (2x length + 2*FEC_MEM flush bits).
/// Used for testing and for building the coded advertising template.
pub fn conv_encode(data: &[u8]) -> Vec<u8> {
    let mut coded = Vec::with_capacity(data.len() * 2 + FEC_MEM * 2);
    let mut state: u8 = 0; // 3-bit shift register

    for &bit in data {
        let input = bit & 1;
        // reg = [input, state] where state has FEC_MEM bits
        let reg = (input << FEC_MEM) | state;
        let c0 = parity(reg & G0);
        let c1 = parity(reg & G1);
        coded.push(c0);
        coded.push(c1);
        // Shift: new state = [input at MSB, old state shifted right by 1]
        state = ((input << (FEC_MEM - 1)) | (state >> 1)) & ((1 << FEC_MEM) - 1);
    }

    // Flush: FEC_MEM zero bits to terminate trellis to state 0
    for _ in 0..FEC_MEM {
        let reg = state; // input = 0
        let c0 = parity(reg & G0);
        let c1 = parity(reg & G1);
        coded.push(c0);
        coded.push(c1);
        state = (state >> 1) & ((1 << FEC_MEM) - 1);
    }

    coded
}

/// Pattern map at S=2: coded bit 0 -> [1,1], coded bit 1 -> [0,1]
pub fn pattern_map_s2(coded: &[u8]) -> Vec<u8> {
    let mut symbols = Vec::with_capacity(coded.len() * 2);
    for &bit in coded {
        if bit == 0 {
            symbols.push(1);
            symbols.push(1);
        } else {
            symbols.push(0);
            symbols.push(1);
        }
    }
    symbols
}

/// Pattern map at S=8: coded bit 0 -> [1,1,1,1,0,0,0,0], coded bit 1 -> [0,0,0,0,1,1,1,1]
pub fn pattern_map_s8(coded: &[u8]) -> Vec<u8> {
    let mut symbols = Vec::with_capacity(coded.len() * 8);
    for &bit in coded {
        if bit == 0 {
            symbols.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0]);
        } else {
            symbols.extend_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
        }
    }
    symbols
}

/// Soft pattern demap at S=2.
/// Takes soft demod values (positive = 1, negative = 0).
/// Returns soft coded bits: positive = more likely 0, negative = more likely 1.
///
/// S=2 mapping: coded 0 -> [+1, +1], coded 1 -> [-1, +1]
/// The first symbol differentiates: if positive -> coded 0, if negative -> coded 1.
pub fn pattern_demap_s2(symbols: &[f32]) -> Vec<f32> {
    let n = symbols.len() / 2;
    let mut soft_coded = Vec::with_capacity(n);
    for i in 0..n {
        // S=2: pattern 0 = [+1, +1], pattern 1 = [-1, +1]
        // Metric = 2*s[0] (positive = coded 0)
        soft_coded.push(symbols[i * 2]);
    }
    soft_coded
}

/// Soft pattern demap at S=8.
/// Takes soft demod values, returns soft coded bits.
///
/// S=8 mapping: coded 0 -> [+1,+1,+1,+1,-1,-1,-1,-1]
///              coded 1 -> [-1,-1,-1,-1,+1,+1,+1,+1]
/// Metric: sum of first 4 minus sum of last 4 (positive = coded 0).
pub fn pattern_demap_s8(symbols: &[f32]) -> Vec<f32> {
    let n = symbols.len() / 8;
    let mut soft_coded = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 8;
        let first_half: f32 = symbols[base..base + 4].iter().sum();
        let second_half: f32 = symbols[base + 4..base + 8].iter().sum();
        soft_coded.push(first_half - second_half);
    }
    soft_coded
}

/// Pre-computed branch info for Viterbi decoder.
#[derive(Clone, Copy)]
struct Branch {
    next_state: u8,
    c0: u8,
    c1: u8,
}

/// Build the branch table for the K=4 convolutional code.
/// Returns branches[state][input] for all 8 states x 2 inputs.
fn build_branch_table() -> [[Branch; 2]; FEC_STATES] {
    let mut branches = [[Branch { next_state: 0, c0: 0, c1: 0 }; 2]; FEC_STATES];
    let state_mask = (1u8 << FEC_MEM) - 1;

    for state in 0u8..(FEC_STATES as u8) {
        for input in 0u8..2 {
            let reg = (input << FEC_MEM) | state;
            let c0 = parity(reg & G0);
            let c1 = parity(reg & G1);
            let next_state = ((input << (FEC_MEM - 1)) | (state >> 1)) & state_mask;
            branches[state as usize][input as usize] = Branch { next_state, c0, c1 };
        }
    }
    branches
}

/// Viterbi decoder for rate 1/2, K=4 convolutional code (BLE Coded PHY).
/// 8 states, generators g0=17o (0b1111), g1=15o (0b1011 in register layout).
///
/// Takes soft coded bits (pairs of soft values for the two coded outputs).
/// Returns decoded data bits.
///
/// The soft_coded input has 2*N values (N pairs). Each pair (c0, c1) represents
/// the soft-decision coded output. Positive = more likely 0.
pub fn viterbi_decode(soft_coded: &[f32]) -> Vec<u8> {
    let n_pairs = soft_coded.len() / 2;
    if n_pairs == 0 {
        return Vec::new();
    }

    let branches = build_branch_table();

    // Path metrics
    let mut pm = [f32::NEG_INFINITY; FEC_STATES];
    pm[0] = 0.0; // start in state 0

    // Traceback: for each step, store (prev_state << 1 | input) per state.
    // Use u8 since we have 8 states (3-bit) + 1 bit input = 4 bits needed.
    let mut traceback = vec![[0u8; FEC_STATES]; n_pairs];

    let mut pm_new;

    for step in 0..n_pairs {
        let s0 = soft_coded[step * 2];     // soft c0 (positive = 0)
        let s1 = soft_coded[step * 2 + 1]; // soft c1 (positive = 0)

        pm_new = [f32::NEG_INFINITY; FEC_STATES];
        let mut tb_step = [0u8; FEC_STATES];

        for state in 0..(FEC_STATES as u8) {
            if pm[state as usize] == f32::NEG_INFINITY {
                continue;
            }
            for input in 0u8..2 {
                let br = &branches[state as usize][input as usize];
                // Branch metric: correlation of soft values with expected outputs
                let bm0 = if br.c0 == 0 { s0 } else { -s0 };
                let bm1 = if br.c1 == 0 { s1 } else { -s1 };
                let metric = pm[state as usize] + bm0 + bm1;

                let ns = br.next_state as usize;
                if metric > pm_new[ns] {
                    pm_new[ns] = metric;
                    tb_step[ns] = (state << 1) | input;
                }
            }
        }

        pm = pm_new;
        traceback[step] = tb_step;
    }

    // Find best final state
    let mut best_state: u8 = 0;
    let mut best_metric = f32::NEG_INFINITY;
    for s in 0..(FEC_STATES as u8) {
        if pm[s as usize] > best_metric {
            best_metric = pm[s as usize];
            best_state = s;
        }
    }

    // Traceback
    let mut decoded = vec![0u8; n_pairs];
    let mut state = best_state;
    for step in (0..n_pairs).rev() {
        let entry = traceback[step][state as usize];
        let input = entry & 1;
        let prev_state = entry >> 1;
        decoded[step] = input;
        state = prev_state;
    }

    decoded
}

/// Viterbi decoder with forced termination to state 0.
/// Use for FEC Block 1 (LE Coded PHY) where TERM1 guarantees the encoder
/// ends in state 0.  Returns (decoded_bits, best_final_state, path_metric_state0).
/// If the final state 0 has NEG_INFINITY metric, the signal is not a genuine
/// terminated block (i.e., false positive from WiFi/interference).
pub fn viterbi_decode_terminated(soft_coded: &[f32]) -> (Vec<u8>, u8, f32) {
    let n_pairs = soft_coded.len() / 2;
    if n_pairs == 0 {
        return (Vec::new(), 0, f32::NEG_INFINITY);
    }

    let branches = build_branch_table();

    let mut pm = [f32::NEG_INFINITY; FEC_STATES];
    pm[0] = 0.0;

    let mut traceback = vec![[0u8; FEC_STATES]; n_pairs];
    let mut pm_new;

    for step in 0..n_pairs {
        let s0 = soft_coded[step * 2];
        let s1 = soft_coded[step * 2 + 1];

        pm_new = [f32::NEG_INFINITY; FEC_STATES];
        let mut tb_step = [0u8; FEC_STATES];

        for state in 0..(FEC_STATES as u8) {
            if pm[state as usize] == f32::NEG_INFINITY {
                continue;
            }
            for input in 0u8..2 {
                let br = &branches[state as usize][input as usize];
                let bm0 = if br.c0 == 0 { s0 } else { -s0 };
                let bm1 = if br.c1 == 0 { s1 } else { -s1 };
                let metric = pm[state as usize] + bm0 + bm1;

                let ns = br.next_state as usize;
                if metric > pm_new[ns] {
                    pm_new[ns] = metric;
                    tb_step[ns] = (state << 1) | input;
                }
            }
        }

        pm = pm_new;
        traceback[step] = tb_step;
    }

    // Find best state for reporting, and state 0 metric for termination check
    let mut best_state: u8 = 0;
    let mut best_metric = f32::NEG_INFINITY;
    for s in 0..(FEC_STATES as u8) {
        if pm[s as usize] > best_metric {
            best_metric = pm[s as usize];
            best_state = s;
        }
    }
    let state0_metric = pm[0];

    // Force traceback from state 0 (terminated)
    let mut decoded = vec![0u8; n_pairs];
    let mut state: u8 = 0;
    for step in (0..n_pairs).rev() {
        let entry = traceback[step][state as usize];
        let input = entry & 1;
        let prev_state = entry >> 1;
        decoded[step] = input;
        state = prev_state;
    }

    (decoded, best_state, state0_metric)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_encode_zeros() {
        let data = vec![0u8; 8];
        let coded = conv_encode(&data);
        // All-zero input with all-zero state should produce all-zero output
        assert_eq!(coded.len(), 8 * 2 + FEC_MEM * 2);
        for &bit in &coded {
            assert_eq!(bit, 0, "expected all zeros for all-zero input");
        }
    }

    #[test]
    fn test_conv_encode_impulse() {
        // Single 1 followed by zeros: tests the impulse response
        let data = vec![1, 0, 0, 0, 0, 0];
        let coded = conv_encode(&data);
        // g0 = 0b1111: impulse response = 1,1,1,1 (all 4 taps)
        // g1 = 0b1011: impulse response = 1,0,1,1 (taps D^0,D^2,D^3 -- skip D^1)
        // c0: 1, 1, 1, 1, 0, 0, ...
        // c1: 1, 0, 1, 1, 0, 0, ...
        // Interleaved: (1,1), (1,0), (1,1), (1,1), (0,0), (0,0), ...
        assert_eq!(coded[0], 1); // c0[0]
        assert_eq!(coded[1], 1); // c1[0]
        assert_eq!(coded[2], 1); // c0[1]
        assert_eq!(coded[3], 0); // c1[1]
        assert_eq!(coded[4], 1); // c0[2]
        assert_eq!(coded[5], 1); // c1[2]
        assert_eq!(coded[6], 1); // c0[3]
        assert_eq!(coded[7], 1); // c1[3]
        assert_eq!(coded[8], 0); // c0[4]
        assert_eq!(coded[9], 0); // c1[4]
    }

    #[test]
    fn test_conv_roundtrip() {
        // Encode, pattern map at S=8, demap, Viterbi decode, should match
        let data = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1];
        let coded = conv_encode(&data);
        let symbols = pattern_map_s8(&coded);

        let soft_symbols: Vec<f32> = symbols.iter().map(|&s| if s == 1 { 1.0 } else { -1.0 }).collect();
        let soft_coded = pattern_demap_s8(&soft_symbols);
        let decoded = viterbi_decode(&soft_coded);

        assert!(decoded.len() >= data.len(), "decoded too short: {} < {}", decoded.len(), data.len());
        assert_eq!(&decoded[..data.len()], &data[..], "Viterbi roundtrip failed");
    }

    #[test]
    fn test_conv_roundtrip_s2() {
        let data = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let coded = conv_encode(&data);
        let symbols = pattern_map_s2(&coded);

        let soft_symbols: Vec<f32> = symbols.iter().map(|&s| if s == 1 { 1.0 } else { -1.0 }).collect();
        let soft_coded = pattern_demap_s2(&soft_symbols);
        let decoded = viterbi_decode(&soft_coded);

        assert_eq!(&decoded[..data.len()], &data[..], "S=2 roundtrip failed");
    }

    #[test]
    fn test_viterbi_with_noise() {
        // Test error correction: introduce errors in the coded stream
        let data = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0];
        let coded = conv_encode(&data);
        let symbols = pattern_map_s8(&coded);

        let mut soft_symbols: Vec<f32> = symbols.iter().map(|&s| if s == 1 { 1.0 } else { -1.0 }).collect();
        // Corrupt ~10% of symbols (soft values reduced, not fully flipped)
        for i in (0..soft_symbols.len()).step_by(11) {
            soft_symbols[i] *= -0.3;
        }

        let soft_coded = pattern_demap_s8(&soft_symbols);
        let decoded = viterbi_decode(&soft_coded);
        assert_eq!(&decoded[..data.len()], &data[..], "Viterbi should correct mild noise");
    }

    #[test]
    fn test_viterbi_ble_advertising_aa() {
        // Test with the BLE advertising Access Address (0x8E89BED6)
        // AA bits: LSB first within each byte
        let aa: u32 = 0x8E89BED6;
        let mut aa_bits = Vec::with_capacity(32);
        for i in 0..32 {
            aa_bits.push(((aa >> i) & 1) as u8);
        }

        let coded = conv_encode(&aa_bits);
        let symbols = pattern_map_s8(&coded);
        let soft_symbols: Vec<f32> = symbols.iter().map(|&s| if s == 1 { 1.0 } else { -1.0 }).collect();
        let soft_coded = pattern_demap_s8(&soft_symbols);
        let decoded = viterbi_decode(&soft_coded);

        // Reconstruct AA from decoded bits
        let mut decoded_aa: u32 = 0;
        for i in 0..32 {
            decoded_aa |= (decoded[i] as u32) << i;
        }
        assert_eq!(decoded_aa, aa, "AA roundtrip failed: got 0x{:08X}", decoded_aa);
    }

    #[test]
    fn test_pattern_map_s8_values() {
        let coded = vec![0, 1];
        let symbols = pattern_map_s8(&coded);
        assert_eq!(symbols.len(), 16);
        assert_eq!(&symbols[..8], &[1, 1, 1, 1, 0, 0, 0, 0]); // coded 0
        assert_eq!(&symbols[8..], &[0, 0, 0, 0, 1, 1, 1, 1]); // coded 1
    }

    #[test]
    fn test_pattern_map_s2_values() {
        let coded = vec![0, 1];
        let symbols = pattern_map_s2(&coded);
        assert_eq!(symbols.len(), 4);
        assert_eq!(&symbols[..2], &[1, 1]); // coded 0
        assert_eq!(&symbols[2..], &[0, 1]); // coded 1
    }
}
