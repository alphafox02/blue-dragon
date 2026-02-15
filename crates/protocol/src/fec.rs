/// Forward error correction for BLE LE Coded PHY.
///
/// Convolutional code: rate 1/2, constraint length K=3,
/// generators g0=7 (0b111), g1=5 (0b101).
/// 4 trellis states, Viterbi decoder runs in microseconds.
///
/// Pattern mapping: S=2 maps coded bit 0->"11", 1->"01".
///                  S=8 maps coded bit 0->"11110000", 1->"00001111".

/// Convolutional encode: rate 1/2, K=3, g0=7, g1=5.
/// Takes data bits, produces coded bits (2x length + 2*3 flush bits).
/// Used for testing -- the live path only decodes.
pub fn conv_encode(data: &[u8]) -> Vec<u8> {
    let mut coded = Vec::with_capacity(data.len() * 2 + 6);
    let mut state: u8 = 0; // 2-bit shift register

    for &bit in data {
        let input = bit & 1;
        let reg = (input << 2) | state;
        // g0 = 0b111: XOR of bits 2,1,0
        let c0 = ((reg >> 2) ^ (reg >> 1) ^ reg) & 1;
        // g1 = 0b101: XOR of bits 2,0
        let c1 = ((reg >> 2) ^ reg) & 1;
        coded.push(c0);
        coded.push(c1);
        state = ((input << 1) | (state >> 1)) & 0x03;
    }

    // Flush: 3 zero bits to terminate trellis
    for _ in 0..3 {
        let reg = state; // input = 0
        let c0 = ((reg >> 2) ^ (reg >> 1) ^ reg) & 1;
        let c1 = ((reg >> 2) ^ reg) & 1;
        coded.push(c0);
        coded.push(c1);
        state = (state >> 1) & 0x03;
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
/// The first symbol differentiates: if positive -> coded 0, if negative -> coded 1
/// Average both for more reliability.
pub fn pattern_demap_s2(symbols: &[f32]) -> Vec<f32> {
    let n = symbols.len() / 2;
    let mut soft_coded = Vec::with_capacity(n);
    for i in 0..n {
        // S=2: pattern 0 = [+1, +1], pattern 1 = [-1, +1]
        // Correlation with pattern 0 = s[0] + s[1]
        // Correlation with pattern 1 = -s[0] + s[1]
        // Metric = corr_0 - corr_1 = 2*s[0]
        // Positive means coded 0, negative means coded 1
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

/// Viterbi decoder for rate 1/2, K=3 convolutional code.
/// 4 states (00, 01, 10, 11), generators g0=7, g1=5.
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

    // Branch metric table: for each state (2 bits) and input (1 bit),
    // pre-compute expected outputs (c0, c1).
    // state = previous 2 bits of shift register
    // input = new data bit
    // reg = [input, state_bit1, state_bit0]
    #[derive(Clone, Copy)]
    struct Branch {
        next_state: u8,
        c0: u8,
        c1: u8,
    }

    // Build branch table: branches[state][input]
    let mut branches = [[Branch { next_state: 0, c0: 0, c1: 0 }; 2]; 4];
    for state in 0u8..4 {
        for input in 0u8..2 {
            let reg = (input << 2) | state;
            let c0 = ((reg >> 2) ^ (reg >> 1) ^ reg) & 1;
            let c1 = ((reg >> 2) ^ reg) & 1;
            let next_state = ((input << 1) | (state >> 1)) & 0x03;
            branches[state as usize][input as usize] = Branch {
                next_state,
                c0,
                c1,
            };
        }
    }

    // Path metrics (use f32 for simplicity; 4 states is tiny)
    let mut pm = [f32::NEG_INFINITY; 4];
    pm[0] = 0.0; // start in state 0

    // Traceback storage: for each step, store the chosen input for each state
    let mut traceback = vec![[0u8; 4]; n_pairs];

    let mut pm_new;

    for step in 0..n_pairs {
        let s0 = soft_coded[step * 2];     // soft c0 (positive = 0)
        let s1 = soft_coded[step * 2 + 1]; // soft c1 (positive = 0)

        pm_new = [f32::NEG_INFINITY; 4];
        let mut tb_step = [0u8; 4];

        for state in 0u8..4 {
            if pm[state as usize] == f32::NEG_INFINITY {
                continue;
            }
            for input in 0u8..2 {
                let br = &branches[state as usize][input as usize];
                // Branch metric: correlation of soft values with expected outputs
                // expected c0: 0 -> positive metric, 1 -> negative metric
                let bm0 = if br.c0 == 0 { s0 } else { -s0 };
                let bm1 = if br.c1 == 0 { s1 } else { -s1 };
                let metric = pm[state as usize] + bm0 + bm1;

                let ns = br.next_state as usize;
                if metric > pm_new[ns] {
                    pm_new[ns] = metric;
                    tb_step[ns] = (state << 1) | input; // encode prev_state + input
                }
            }
        }

        pm = pm_new;
        traceback[step] = tb_step;
    }

    // Find best final state
    let mut best_state: u8 = 0;
    let mut best_metric = f32::NEG_INFINITY;
    for s in 0u8..4 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_encode_zeros() {
        let data = vec![0u8; 8];
        let coded = conv_encode(&data);
        // All-zero input should produce all-zero output (both generators produce 0 for zero state)
        assert_eq!(coded.len(), 8 * 2 + 6); // 16 data + 6 flush
        for &bit in &coded {
            assert_eq!(bit, 0, "expected all zeros for all-zero input");
        }
    }

    #[test]
    fn test_conv_roundtrip() {
        // Encode some data, pattern map at S=8, demap, Viterbi decode, should match
        let data = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1];
        let coded = conv_encode(&data);
        let symbols = pattern_map_s8(&coded);

        // Convert hard symbols to soft values
        let soft_symbols: Vec<f32> = symbols.iter().map(|&s| if s == 1 { 1.0 } else { -1.0 }).collect();
        let soft_coded = pattern_demap_s8(&soft_symbols);

        // Interleave soft_coded as pairs for Viterbi
        // conv_encode produces alternating c0, c1 already
        let decoded = viterbi_decode(&soft_coded);

        // decoded includes flush bits at the end, trim to data length
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

        // Add noise: flip some symbols
        let mut soft_symbols: Vec<f32> = symbols.iter().map(|&s| if s == 1 { 1.0 } else { -1.0 }).collect();
        // Corrupt ~10% of symbols (soft values reduced, not fully flipped)
        for i in (0..soft_symbols.len()).step_by(11) {
            soft_symbols[i] *= -0.3; // weaken, not fully flip
        }

        let soft_coded = pattern_demap_s8(&soft_symbols);
        let decoded = viterbi_decode(&soft_coded);
        assert_eq!(&decoded[..data.len()], &data[..], "Viterbi should correct mild noise");
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
