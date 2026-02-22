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

/// Generate prototype lowpass filter for the PFB type 2 channelizer.
///
/// Matches C code's `liquid_firdes_kaiser(2*M*m+1, 0.75/M, 60.0, 0.0, h)`.
///
/// - `num_channels`: M (number of output channels)
/// - `semi_len`: m (filter semi-length, C code uses m=4)
///
/// Returns float coefficients of length 2*M*m+1.
pub fn pfb_prototype_float(num_channels: usize, semi_len: usize) -> Vec<f32> {
    let h_len = 2 * num_channels * semi_len + 1;
    let fc = 0.75 / num_channels as f64; // normalized cutoff (C: lp_cutoff=0.75)
    let as_db = 60.0_f64; // stopband attenuation

    // Kaiser beta from stopband attenuation (matching liquid-dsp formula)
    let beta = if as_db > 50.0 {
        0.1102 * (as_db - 8.7)
    } else if as_db > 21.0 {
        0.5842 * (as_db - 21.0).powf(0.4) + 0.07886 * (as_db - 21.0)
    } else {
        0.0
    };

    let win = kaiser(h_len, beta);
    let half = (h_len as f64 - 1.0) / 2.0;
    let mut h = Vec::with_capacity(h_len);

    for n in 0..h_len {
        let t = n as f64 - half;
        let sinc_val = if t.abs() < 1e-12 {
            1.0
        } else {
            let x = 2.0 * fc * t;
            (PI * x).sin() / (PI * x)
        };
        h.push((sinc_val * win[n]) as f32);
    }

    h
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
    fn test_pfb_prototype_float() {
        let m = 40;
        let semi_len = 4;
        let h = pfb_prototype_float(m, semi_len);
        assert_eq!(h.len(), 2 * m * semi_len + 1);
        // Peak should be at center
        let center = h.len() / 2;
        let peak = h.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            (h[center].abs() - peak).abs() < 0.01,
            "peak not at center: center={}, peak={}",
            h[center],
            peak
        );
        // Should be symmetric
        for i in 0..h.len() / 2 {
            assert!(
                (h[i] - h[h.len() - 1 - i]).abs() < 1e-5,
                "asymmetry at {}: {} != {}",
                i,
                h[i],
                h[h.len() - 1 - i]
            );
        }
    }
}
