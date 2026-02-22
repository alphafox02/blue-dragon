use num_complex::Complex32;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;

/// Batched IFFT processor using rustfft.
///
/// Performs M-point inverse FFTs for the PFB channelizer output.
/// The C code uses FFTW_BACKWARD (inverse FFT) for the PFB analysis channelizer.
/// Using forward FFT instead would mirror the frequency bins, mapping signals
/// to conjugate channels and breaking channel-dependent operations like BLE whitening.
pub struct BatchFft {
    fft: Arc<dyn Fft<f32>>,
    size: usize,
    scratch: Vec<Complex32>,
}

impl BatchFft {
    /// Create a new IFFT processor for a given size (number of channels).
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(size);
        let scratch_len = fft.get_inplace_scratch_len();
        Self {
            fft,
            size,
            scratch: vec![Complex32::new(0.0, 0.0); scratch_len],
        }
    }

    /// Perform an in-place FFT on a buffer of exactly `size` complex samples.
    /// The buffer is modified in place with the FFT result.
    pub fn process(&mut self, buffer: &mut [Complex32]) {
        assert_eq!(
            buffer.len(),
            self.size,
            "buffer length {} != FFT size {}",
            buffer.len(),
            self.size
        );
        self.fft.process_with_scratch(buffer, &mut self.scratch);
    }

    /// Perform FFT and return the result as a new vector.
    pub fn process_copy(&mut self, input: &[Complex32]) -> Vec<Complex32> {
        let mut buffer = input.to_vec();
        self.process(&mut buffer);
        buffer
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ifft_dc() {
        let size = 64;
        let mut fft = BatchFft::new(size);

        // DC input (all ones) should give energy only in bin 0 after IFFT
        let mut input = vec![Complex32::new(1.0, 0.0); size];
        fft.process(&mut input);

        // Bin 0 should have magnitude = size (IFFT without normalization)
        assert!((input[0].norm() - size as f32).abs() < 0.01);
        // Other bins should be ~0
        for &val in &input[1..] {
            assert!(val.norm() < 0.01, "non-zero energy in non-DC bin: {}", val.norm());
        }
    }

    #[test]
    fn test_ifft_single_tone() {
        let size = 64;
        let mut fft = BatchFft::new(size);

        // Single tone at bin 4: IFFT maps positive freq input to same bin
        let bin = 4;
        let mut input: Vec<Complex32> = (0..size)
            .map(|n| {
                let phase = 2.0 * std::f32::consts::PI * bin as f32 * n as f32 / size as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect();

        fft.process(&mut input);

        // For IFFT, a signal at +freq in time domain maps to the same bin
        let mut max_bin = 0;
        let mut max_mag = 0.0f32;
        for (i, val) in input.iter().enumerate() {
            if val.norm() > max_mag {
                max_mag = val.norm();
                max_bin = i;
            }
        }
        // IFFT conjugates the twiddle factors, so positive freq tone maps to bin (size-bin)
        let expected_bin = (size - bin) % size;
        assert!(
            max_bin == bin || max_bin == expected_bin,
            "expected peak at bin {} or {}, got bin {}",
            bin, expected_bin, max_bin,
        );
    }
}
