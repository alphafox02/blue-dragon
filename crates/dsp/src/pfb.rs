use num_complex::Complex32;

/// Polyphase Filter Bank Channelizer (analysis, type 2)
///
/// Takes wideband input samples at rate M*fs and produces M narrowband channels
/// each at rate fs, using a polyphase decomposition of a prototype lowpass filter
/// followed by an M-point FFT.
///
/// The filter coefficients are stored as int16 (Q15 fixed-point) for performance.
/// The input samples are also in int16 format, matching the SDR native format.
/// The dot products are computed in int32 to avoid overflow, then converted to float
/// for the FFT stage.
pub struct PfbChannelizer {
    num_channels: usize,
    taps_per_channel: usize,
    /// Filter coefficients: [num_channels][taps_per_channel], int16 Q15 format
    coeffs: Vec<Vec<i16>>,
    /// Delay line: [num_channels][taps_per_channel], complex int16 pairs (I, Q)
    delay_i: Vec<Vec<i16>>,
    delay_q: Vec<Vec<i16>>,
    /// Current write position in delay line (circular buffer)
    delay_pos: usize,
}

impl PfbChannelizer {
    /// Create a new PFB channelizer.
    ///
    /// - `num_channels`: number of output channels (M)
    /// - `prototype`: prototype lowpass filter coefficients (length = M * taps_per_channel)
    pub fn new(num_channels: usize, prototype: &[i16]) -> Self {
        assert!(
            prototype.len() % num_channels == 0,
            "prototype length must be divisible by num_channels"
        );
        let taps_per_channel = prototype.len() / num_channels;

        // Decompose prototype into polyphase branches
        // Branch k gets taps at indices k, k+M, k+2M, ...
        let mut coeffs = vec![vec![0i16; taps_per_channel]; num_channels];
        for k in 0..num_channels {
            for t in 0..taps_per_channel {
                coeffs[k][t] = prototype[k + t * num_channels];
            }
        }

        let delay_i = vec![vec![0i16; taps_per_channel]; num_channels];
        let delay_q = vec![vec![0i16; taps_per_channel]; num_channels];

        Self {
            num_channels,
            taps_per_channel,
            coeffs,
            delay_i,
            delay_q,
            delay_pos: 0,
        }
    }

    /// Push M new int16 IQ samples into the delay line.
    /// `samples` must be exactly num_channels*2 int16 values (I,Q interleaved).
    pub fn push_samples(&mut self, samples: &[i16]) {
        assert_eq!(
            samples.len(),
            self.num_channels * 2,
            "expected {} samples (M*2), got {}",
            self.num_channels * 2,
            samples.len()
        );

        let pos = self.delay_pos;
        for k in 0..self.num_channels {
            self.delay_i[k][pos] = samples[k * 2];
            self.delay_q[k][pos] = samples[k * 2 + 1];
        }
        self.delay_pos = (self.delay_pos + 1) % self.taps_per_channel;
    }

    /// Compute the polyphase filter outputs (one per channel).
    /// Returns M complex f32 values ready for the FFT stage.
    ///
    /// This is the inner dot product: for each channel k,
    /// output[k] = sum_{t=0}^{T-1} coeff[k][t] * delay[k][(pos-t) mod T]
    pub fn filter_outputs(&self) -> Vec<Complex32> {
        let mut outputs = Vec::with_capacity(self.num_channels);

        for k in 0..self.num_channels {
            let (sum_i, sum_q) = self.dot_product_scalar(k);
            // Convert from Q15*Q15 = Q30 to float
            // Division by 2^30 normalizes the fixed-point product
            let scale = 1.0 / (1u64 << 30) as f32;
            outputs.push(Complex32::new(sum_i as f32 * scale, sum_q as f32 * scale));
        }

        outputs
    }

    /// Scalar dot product for channel k
    fn dot_product_scalar(&self, k: usize) -> (i64, i64) {
        let mut sum_i: i64 = 0;
        let mut sum_q: i64 = 0;
        let t = self.taps_per_channel;

        for tap in 0..t {
            let delay_idx = (self.delay_pos + t - 1 - tap) % t;
            let coeff = self.coeffs[k][tap] as i64;
            sum_i += coeff * self.delay_i[k][delay_idx] as i64;
            sum_q += coeff * self.delay_q[k][delay_idx] as i64;
        }

        (sum_i, sum_q)
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn taps_per_channel(&self) -> usize {
        self.taps_per_channel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pfb_construction() {
        let m = 8;
        let taps = 4;
        let proto = vec![1i16; m * taps];
        let pfb = PfbChannelizer::new(m, &proto);
        assert_eq!(pfb.num_channels(), m);
        assert_eq!(pfb.taps_per_channel(), taps);
    }

    #[test]
    fn test_pfb_filter_outputs() {
        let m = 4;
        let taps = 2;
        // Simple prototype: all ones
        let proto = vec![100i16; m * taps];
        let mut pfb = PfbChannelizer::new(m, &proto);

        // Push a block of samples (all 1000+0j)
        let samples: Vec<i16> = (0..m * 2)
            .map(|i| if i % 2 == 0 { 1000 } else { 0 })
            .collect();
        pfb.push_samples(&samples);

        let outputs = pfb.filter_outputs();
        assert_eq!(outputs.len(), m);
        // With all-ones filter and constant input, outputs should be non-zero
        assert!(outputs.iter().any(|o| o.norm() > 0.0));
    }
}
