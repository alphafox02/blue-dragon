// Copyright 2025-2026 CEMAXECUTER LLC
//
// Enhanced Data Rate (EDR) payload demodulation. EDR keeps the Basic Rate
// (GFSK) access code + header, then switches the payload to differential
// phase-shift keying: pi/4-DQPSK for 2 Mbps (2-DHx) and 8DPSK for 3 Mbps
// (3-DHx). This module turns the complex payload samples into bits; whitening,
// CRC and packet framing are handled in the protocol crate, same as Basic Rate.

use num_complex::Complex32;
use std::f32::consts::PI;

// Per-symbol phase changes, indexed by symbol value. pi/4-DQPSK carries 2 bits
// on the odd multiples of pi/4; 8DPSK carries 3 bits on all multiples of pi/4.
// Indexed by the LSB-first packed value of the on-air bits. Bluetooth's
// constellation is Gray-coded, so this is not monotonically phase ordered.
const DQPSK_DPHI: [f32; 4] = [PI / 4.0, -PI / 4.0, 3.0 * PI / 4.0, -3.0 * PI / 4.0];
const DPSK8_DPHI: [f32; 8] = [
    0.0,
    -PI / 4.0,
    3.0 * PI / 4.0,
    PI,
    PI / 4.0,
    -PI / 2.0,
    PI / 2.0,
    -3.0 * PI / 4.0,
];

// Differential phases after the arbitrary EDR reference symbol. This sequence
// is common to pi/4-DQPSK and 8DPSK packets.
const SYNC_DPHI: [f32; 10] = [
    3.0 * PI / 4.0,
    -3.0 * PI / 4.0,
    3.0 * PI / 4.0,
    -3.0 * PI / 4.0,
    3.0 * PI / 4.0,
    -3.0 * PI / 4.0,
    -3.0 * PI / 4.0,
    3.0 * PI / 4.0,
    3.0 * PI / 4.0,
    3.0 * PI / 4.0,
];

/// Wrap an angle to (-pi, pi].
fn wrap(a: f32) -> f32 {
    let mut x = a;
    while x > PI {
        x -= 2.0 * PI;
    }
    while x <= -PI {
        x += 2.0 * PI;
    }
    x
}

/// Nearest constellation symbol to a measured phase change.
fn nearest(dphi: f32, table: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for (i, &t) in table.iter().enumerate() {
        let d = wrap(dphi - t).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn differential_phase(
    samples: &[Complex32],
    previous_start: usize,
    current_start: usize,
    sps: usize,
) -> Option<f32> {
    if previous_start + sps > samples.len() || current_start + sps > samples.len() {
        return None;
    }
    let correlation = (0..sps).fold(Complex32::new(0.0, 0.0), |sum, offset| {
        sum + samples[current_start + offset] * samples[previous_start + offset].conj()
    });
    (correlation.norm_sqr() > f32::EPSILON).then(|| correlation.arg())
}

/// Differential DPSK demod. `samples` are the payload, `sps` samples per symbol;
/// `bits_per_symbol` is 2 (pi/4-DQPSK) or 3 (8DPSK). The first symbol is the
/// phase reference. Returns payload bits, LSB-first within each symbol.
pub fn demod_dpsk(samples: &[Complex32], sps: usize, bits_per_symbol: usize) -> Vec<u8> {
    let table: &[f32] = if bits_per_symbol == 3 {
        &DPSK8_DPHI
    } else {
        &DQPSK_DPHI
    };
    let n = if sps == 0 { 0 } else { samples.len() / sps };
    let mut bits = Vec::with_capacity(n.saturating_sub(1) * bits_per_symbol);
    if n < 2 {
        return bits;
    }
    for k in 1..n {
        let dphi = match differential_phase(samples, (k - 1) * sps, k * sps, sps) {
            Some(phase) => phase,
            None => continue,
        };
        let sym = nearest(dphi, table);
        for b in 0..bits_per_symbol {
            bits.push(((sym >> b) & 1) as u8);
        }
    }
    bits
}

fn demod_dpsk_point(
    samples: &[Complex32],
    sps: usize,
    bits_per_symbol: usize,
    sample_offset: usize,
) -> Vec<u8> {
    if sps == 0 || sample_offset >= samples.len() {
        return Vec::new();
    }
    let table: &[f32] = if bits_per_symbol == 3 {
        &DPSK8_DPHI
    } else {
        &DQPSK_DPHI
    };
    let n = (samples.len() - sample_offset) / sps;
    let mut bits = Vec::with_capacity(n.saturating_sub(1) * bits_per_symbol);
    for symbol in 1..n {
        let previous = samples[sample_offset + (symbol - 1) * sps];
        let current = samples[sample_offset + symbol * sps];
        let value = nearest((current * previous.conj()).arg(), table);
        for bit in 0..bits_per_symbol {
            bits.push(((value >> bit) & 1) as u8);
        }
    }
    bits
}

/// Correct a constant carrier-frequency offset (rad/symbol) before demod. EDR is
/// phase-sensitive, so residual CFO rotates every differential angle by a fixed
/// amount; removing it recentres the constellation.
pub fn derotate(samples: &mut [Complex32], radians_per_sample: f32) {
    for (i, s) in samples.iter_mut().enumerate() {
        let ph = -radians_per_sample * i as f32;
        *s *= Complex32::new(ph.cos(), ph.sin());
    }
}

/// Remove the Basic Rate CFO estimate and resample channel IQ to exactly two
/// samples per Bluetooth symbol, matching the FSK detector's timing domain.
pub fn prepare_iq(
    samples: &[Complex32],
    radians_per_sample: f32,
    resample_ratio: f64,
) -> Vec<Complex32> {
    let mut corrected = samples.to_vec();
    derotate(&mut corrected, radians_per_sample);
    if (resample_ratio - 1.0).abs() <= 0.001 {
        return corrected;
    }

    let out_len = (corrected.len() as f64 * resample_ratio) as usize;
    let step = 1.0 / resample_ratio;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let source = i as f64 * step;
        let index = source as usize;
        let fraction = (source - index as f64) as f32;
        if index + 1 < corrected.len() {
            output.push(corrected[index] * (1.0 - fraction) + corrected[index + 1] * fraction);
        } else if index < corrected.len() {
            output.push(corrected[index]);
        }
    }
    output
}

/// Apply the Bluetooth EDR square-root raised-cosine receive filter. The
/// centered convolution preserves sample indices so callers can use the GFSK
/// access-code timing as the initial synchronization estimate.
pub fn rrc_matched_filter(
    samples: &[Complex32],
    sps: usize,
    rolloff: f32,
    span_symbols: usize,
) -> Vec<Complex32> {
    if samples.is_empty() || sps == 0 || !(0.0..=1.0).contains(&rolloff) {
        return Vec::new();
    }
    let half = span_symbols * sps / 2;
    let mut taps = Vec::with_capacity(2 * half + 1);
    for index in 0..=2 * half {
        let t = (index as isize - half as isize) as f32 / sps as f32;
        let value = if t.abs() < 1e-6 {
            1.0 + rolloff * (4.0 / PI - 1.0)
        } else if rolloff > 0.0 && (t.abs() - 1.0 / (4.0 * rolloff)).abs() < 1e-5 {
            let angle = PI / (4.0 * rolloff);
            rolloff / 2.0f32.sqrt()
                * ((1.0 + 2.0 / PI) * angle.sin() + (1.0 - 2.0 / PI) * angle.cos())
        } else {
            let numerator = (PI * t * (1.0 - rolloff)).sin()
                + 4.0 * rolloff * t * (PI * t * (1.0 + rolloff)).cos();
            let denominator = PI * t * (1.0 - (4.0 * rolloff * t).powi(2));
            numerator / denominator
        };
        taps.push(value);
    }
    let energy = taps.iter().map(|tap| tap * tap).sum::<f32>().sqrt();
    for tap in &mut taps {
        *tap /= energy;
    }

    let mut output = vec![Complex32::new(0.0, 0.0); samples.len()];
    for (out_index, output_sample) in output.iter_mut().enumerate() {
        let tap_start = half.saturating_sub(out_index);
        let tap_end = taps.len().min(samples.len() + half - out_index);
        for (tap_index, &tap) in taps.iter().enumerate().take(tap_end).skip(tap_start) {
            let sample_index = out_index + tap_index - half;
            *output_sample += samples[sample_index] * tap;
        }
    }
    output
}

/// Validate the fixed EDR synchronization sequence, remove its residual phase
/// drift, and demodulate the payload. `sync_reference` is the estimated sample
/// center of the arbitrary reference symbol at the start of the 11-symbol sync.
pub fn demod_payload(
    samples: &[Complex32],
    sync_reference: usize,
    sps: usize,
    bits_per_symbol: usize,
) -> Option<Vec<u8>> {
    demod_payload_variants(samples, sync_reference, sps, bits_per_symbol)
        .into_iter()
        .next()
}

/// Synchronize and return bounded demodulation variants for CRC selection.
/// At the two-sample minimum, integrated and point decisions have different
/// failure modes after channel filtering; a valid payload CRC chooses safely.
pub fn demod_payload_variants(
    samples: &[Complex32],
    sync_reference: usize,
    sps: usize,
    bits_per_symbol: usize,
) -> Vec<Vec<u8>> {
    demod_payload_variants_with_diagnostic(samples, sync_reference, sps, bits_per_symbol).0
}

#[derive(Debug, Clone, Copy)]
pub struct SyncDiagnostic {
    pub score: f32,
    pub offset: isize,
    pub conjugated: bool,
}

/// Demodulate bounded timing/orientation variants and report the best fixed
/// sync score, including when it is too poor to admit any payload variant.
pub fn demod_payload_variants_with_diagnostic(
    samples: &[Complex32],
    sync_reference: usize,
    sps: usize,
    bits_per_symbol: usize,
) -> (Vec<Vec<u8>>, Option<SyncDiagnostic>) {
    if sps == 0 || !matches!(bits_per_symbol, 2 | 3) {
        return (Vec::new(), None);
    }

    const MAX_SYNC_SCORE: f32 = 0.35;
    const MAX_SYNC_PEAKS: usize = 4;
    let search_radius = 8 * sps;
    let mut variants = Vec::new();
    let mut diagnostic: Option<SyncDiagnostic> = None;

    // A channelizer or IQ source can reverse spectral orientation. GFSK can
    // still correlate after a bit inversion, while differential PSK changes
    // phase sign, so check both orientations and let the payload CRC decide.
    for conjugate in [false, true] {
        let oriented;
        let samples = if conjugate {
            oriented = samples.iter().map(Complex32::conj).collect::<Vec<_>>();
            oriented.as_slice()
        } else {
            samples
        };
        let mut peaks: Vec<(f32, usize, f32)> = Vec::new();
        for delta in -(search_radius as isize)..=(search_radius as isize) {
            let start = sync_reference as isize + delta;
            if start < 0 {
                continue;
            }
            let start = start as usize;
            if start + SYNC_DPHI.len() * sps >= samples.len() {
                continue;
            }

            let mut errors = [0.0f32; SYNC_DPHI.len()];
            let mut sum_sin = 0.0f32;
            let mut sum_cos = 0.0f32;
            let mut valid = true;
            for (k, &expected) in SYNC_DPHI.iter().enumerate() {
                let Some(measured) =
                    differential_phase(samples, start + k * sps, start + (k + 1) * sps, sps)
                else {
                    valid = false;
                    break;
                };
                let error = wrap(measured - expected);
                errors[k] = error;
                sum_sin += error.sin();
                sum_cos += error.cos();
            }
            if !valid {
                continue;
            }
            let residual = sum_sin.atan2(sum_cos);
            let score = errors
                .iter()
                .map(|&error| {
                    let centered = wrap(error - residual);
                    centered * centered
                })
                .sum::<f32>()
                / SYNC_DPHI.len() as f32;
            if diagnostic.is_none_or(|best| score < best.score) {
                diagnostic = Some(SyncDiagnostic {
                    score,
                    offset: start as isize - sync_reference as isize,
                    conjugated: conjugate,
                });
            }
            if score <= MAX_SYNC_SCORE {
                peaks.push((score, start, residual));
            }
        }
        peaks.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        for &(_, start, residual) in peaks.iter().take(MAX_SYNC_PEAKS) {
            let payload_reference = start + SYNC_DPHI.len() * sps;
            let mut payload = samples[payload_reference..].to_vec();
            derotate(&mut payload, residual / sps as f32);
            let integrated = demod_dpsk(&payload, sps, bits_per_symbol);
            if !variants.contains(&integrated) {
                variants.push(integrated);
            }
            for offset in 0..sps {
                let point = demod_dpsk_point(&payload, sps, bits_per_symbol, offset);
                if !variants.contains(&point) {
                    variants.push(point);
                }
            }
        }
    }
    (variants, diagnostic)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples_for_phase(dphi: f32) -> [Complex32; 2] {
        [
            Complex32::new(1.0, 0.0),
            Complex32::new(dphi.cos(), dphi.sin()),
        ]
    }

    // Encode bits -> DPSK IQ at `sps` samples/symbol (constant phase across the
    // symbol), mirroring the demod tables so the round-trip is exact.
    fn mod_dpsk(bits: &[u8], sps: usize, bps: usize) -> Vec<Complex32> {
        let table: &[f32] = if bps == 3 { &DPSK8_DPHI } else { &DQPSK_DPHI };
        let mut out = Vec::new();
        let mut phase = 0.0f32;
        // reference symbol
        for _ in 0..sps {
            out.push(Complex32::new(phase.cos(), phase.sin()));
        }
        for chunk in bits.chunks(bps) {
            if chunk.len() < bps {
                break;
            }
            let mut sym = 0usize;
            for (b, &bit) in chunk.iter().enumerate() {
                sym |= (bit as usize & 1) << b;
            }
            phase = wrap(phase + table[sym]);
            for _ in 0..sps {
                out.push(Complex32::new(phase.cos(), phase.sin()));
            }
        }
        out
    }

    fn mod_edr_with_sync(bits: &[u8], sps: usize, bps: usize, cfo: f32) -> Vec<Complex32> {
        let table: &[f32] = if bps == 3 { &DPSK8_DPHI } else { &DQPSK_DPHI };
        let mut phases = Vec::new();
        let mut phase = 0.37f32;
        phases.push(phase);
        for &change in &SYNC_DPHI {
            phase = wrap(phase + change);
            phases.push(phase);
        }
        for chunk in bits.chunks_exact(bps) {
            let mut symbol = 0usize;
            for (bit_index, &bit) in chunk.iter().enumerate() {
                symbol |= (bit as usize & 1) << bit_index;
            }
            phase = wrap(phase + table[symbol]);
            phases.push(phase);
        }

        let mut samples = Vec::new();
        for phase in phases {
            for _ in 0..sps {
                let n = samples.len() as f32;
                samples.push(Complex32::from_polar(1.0, phase + cfo * n));
            }
        }
        samples
    }

    #[test]
    fn test_dqpsk_roundtrip() {
        let bits: Vec<u8> = (0..200).map(|i| ((i * 7 + 3) & 1) as u8).collect();
        let iq = mod_dpsk(&bits, 2, 2);
        let out = demod_dpsk(&iq, 2, 2);
        assert_eq!(out, bits);
    }

    #[test]
    fn test_dqpsk_bluetooth_gray_mapping() {
        let vectors = [
            ([0, 0], PI / 4.0),
            ([0, 1], 3.0 * PI / 4.0),
            ([1, 1], -3.0 * PI / 4.0),
            ([1, 0], -PI / 4.0),
        ];
        for (bits, phase) in vectors {
            assert_eq!(demod_dpsk(&samples_for_phase(phase), 1, 2), bits);
        }
    }

    #[test]
    fn test_8dpsk_roundtrip() {
        let bits: Vec<u8> = (0..300).map(|i| ((i * 5 + 1) & 1) as u8).collect();
        let iq = mod_dpsk(&bits, 2, 3);
        let out = demod_dpsk(&iq, 2, 3);
        // 300 bits -> 100 symbols exactly
        assert_eq!(out.len(), bits.len());
        assert_eq!(out, bits);
    }

    #[test]
    fn test_8dpsk_bluetooth_gray_mapping() {
        let vectors = [
            ([0, 0, 0], 0.0),
            ([0, 0, 1], PI / 4.0),
            ([0, 1, 1], PI / 2.0),
            ([0, 1, 0], 3.0 * PI / 4.0),
            ([1, 1, 0], PI),
            ([1, 1, 1], -3.0 * PI / 4.0),
            ([1, 0, 1], -PI / 2.0),
            ([1, 0, 0], -PI / 4.0),
        ];
        for (bits, phase) in vectors {
            assert_eq!(demod_dpsk(&samples_for_phase(phase), 1, 3), bits);
        }
    }

    #[test]
    fn test_dqpsk_survives_cfo_after_derotate() {
        let bits: Vec<u8> = (0..120).map(|i| (i & 1) as u8).collect();
        let mut iq = mod_dpsk(&bits, 2, 2);
        // inject a carrier offset, then remove it
        let cfo = 0.05f32;
        for (i, s) in iq.iter_mut().enumerate() {
            let ph = cfo * i as f32;
            *s *= Complex32::new(ph.cos(), ph.sin());
        }
        derotate(&mut iq, cfo);
        assert_eq!(demod_dpsk(&iq, 2, 2), bits);
    }

    #[test]
    fn test_sync_guided_payload_demod() {
        for bps in [2usize, 3] {
            let bit_count = 120 / bps * bps;
            let bits: Vec<u8> = (0..bit_count).map(|i| ((i * 5 + 1) & 1) as u8).collect();
            let cfo = 0.04f32;
            let iq = mod_edr_with_sync(&bits, 2, bps, cfo);
            let prepared = prepare_iq(&iq, cfo, 1.0);
            let decoded = demod_payload(&prepared, 0, 2, bps).expect("valid sync");
            assert_eq!(&decoded[..bits.len()], bits.as_slice());
        }
    }

    #[test]
    fn test_sync_search_handles_timing_error_and_spectrum_inversion() {
        let bits: Vec<u8> = (0..120).map(|i| ((i * 5 + 1) & 1) as u8).collect();
        let mut iq = vec![Complex32::new(0.0, 0.0); 13];
        iq.extend(mod_edr_with_sync(&bits, 2, 2, 0.03));
        for sample in &mut iq {
            *sample = sample.conj();
        }

        let variants = demod_payload_variants(&iq, 3, 2, 2);
        assert!(variants.iter().any(|decoded| decoded.starts_with(&bits)));
    }

    #[test]
    fn test_sync_rejects_unmodulated_samples() {
        let samples = vec![Complex32::new(1.0, 0.0); 128];
        assert!(demod_payload(&samples, 0, 2, 2).is_none());
    }

    #[test]
    fn test_rrc_filter_preserves_length_and_constant_phase() {
        let samples = vec![Complex32::from_polar(1.0, 0.7); 128];
        let filtered = rrc_matched_filter(&samples, 2, 0.4, 6);
        assert_eq!(filtered.len(), samples.len());
        for sample in &filtered[16..112] {
            assert!(wrap(sample.arg() - 0.7).abs() < 1e-5);
        }
    }
}
