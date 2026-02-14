use num_complex::Complex32;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;

/// Batched FFT processor using rustfft.
///
/// Performs M-point FFTs for the PFB channelizer output.
/// Pre-plans the FFT for a given size and reuses it for every call.
pub struct BatchFft {
    fft: Arc<dyn Fft<f32>>,
    size: usize,
    scratch: Vec<Complex32>,
}

impl BatchFft {
    /// Create a new FFT processor for a given size (number of channels).
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
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
    fn test_fft_dc() {
        let size = 64;
        let mut fft = BatchFft::new(size);

        // DC input (all ones) should give energy only in bin 0
        let mut input = vec![Complex32::new(1.0, 0.0); size];
        fft.process(&mut input);

        // Bin 0 should have magnitude = size
        assert!((input[0].norm() - size as f32).abs() < 0.01);
        // Other bins should be ~0
        for &val in &input[1..] {
            assert!(val.norm() < 0.01, "non-zero energy in non-DC bin: {}", val.norm());
        }
    }

    #[test]
    fn test_fft_single_tone() {
        let size = 64;
        let mut fft = BatchFft::new(size);

        // Single tone at bin 4
        let bin = 4;
        let mut input: Vec<Complex32> = (0..size)
            .map(|n| {
                let phase = 2.0 * std::f32::consts::PI * bin as f32 * n as f32 / size as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect();

        fft.process(&mut input);

        // Bin 4 should have most energy
        let mut max_bin = 0;
        let mut max_mag = 0.0f32;
        for (i, val) in input.iter().enumerate() {
            if val.norm() > max_mag {
                max_mag = val.norm();
                max_bin = i;
            }
        }
        assert_eq!(max_bin, bin, "expected peak at bin {}, got bin {}", bin, max_bin);
    }
}
