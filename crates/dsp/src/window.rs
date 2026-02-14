use std::f64::consts::PI;

/// Modified Bessel function of the first kind, order 0 (for Kaiser window)
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_sq_over_4 = x * x / 4.0;
    for k in 1..=30 {
        term *= x_sq_over_4 / (k * k) as f64;
        sum += term;
        if term < sum * 1e-12 {
            break;
        }
    }
    sum
}

/// Generate Kaiser window coefficients
///
/// - `n`: window length
/// - `beta`: shape parameter (higher = narrower mainlobe, lower sidelobes)
///   Typical: 6.0-8.0 for channelizer applications
pub fn kaiser(n: usize, beta: f64) -> Vec<f64> {
    let mut w = Vec::with_capacity(n);
    let n_f = n as f64;
    let denom = bessel_i0(beta);

    for i in 0..n {
        let x = 2.0 * i as f64 / (n_f - 1.0) - 1.0;
        let arg = beta * (1.0 - x * x).max(0.0).sqrt();
        w.push(bessel_i0(arg) / denom);
    }
    w
}

/// Generate prototype lowpass filter coefficients for the PFB channelizer.
///
/// The filter is a windowed-sinc FIR with a Kaiser window, designed for M channels
/// with `taps_per_channel` taps per polyphase branch.
///
/// Returns filter coefficients as int16 (Q15 fixed-point) for SIMD dot product.
pub fn pfb_prototype(num_channels: usize, taps_per_channel: usize, beta: f64) -> Vec<i16> {
    let total_taps = num_channels * taps_per_channel;

    // Windowed-sinc lowpass at cutoff = 1/(2*M)
    let fc = 1.0 / (2.0 * num_channels as f64);
    let window = kaiser(total_taps, beta);

    let mut h = Vec::with_capacity(total_taps);
    let half = (total_taps as f64 - 1.0) / 2.0;

    for i in 0..total_taps {
        let n = i as f64 - half;
        let sinc = if n.abs() < 1e-12 {
            2.0 * fc
        } else {
            (2.0 * PI * fc * n).sin() / (PI * n)
        };
        h.push(sinc * window[i]);
    }

    // Normalize so that sum of taps = 1
    let sum: f64 = h.iter().sum();
    for val in h.iter_mut() {
        *val /= sum;
    }

    // Convert to Q15 fixed-point (int16)
    // Scale by 32767 * num_channels to maintain energy after polyphase decomposition
    let scale = 32767.0 * num_channels as f64;
    h.iter()
        .map(|&val| (val * scale).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaiser_window() {
        let w = kaiser(64, 7.0);
        assert_eq!(w.len(), 64);
        // Should be symmetric
        for i in 0..32 {
            assert!(
                (w[i] - w[63 - i]).abs() < 1e-10,
                "asymmetry at index {}: {} != {}",
                i,
                w[i],
                w[63 - i]
            );
        }
        // Peak at center
        assert!(w[31] > 0.99);
        // Edges should be small
        assert!(w[0] < 0.1);
    }

    #[test]
    fn test_pfb_prototype() {
        let coeffs = pfb_prototype(96, 12, 7.0);
        assert_eq!(coeffs.len(), 96 * 12);
        // Should be non-trivial
        assert!(coeffs.iter().any(|&c| c != 0));
    }
}
