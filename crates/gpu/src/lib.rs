// Copyright 2025-2026 CEMAXECUTER LLC

//! GPU-accelerated PFB channelizer + FFT using OpenCL + VkFFT.
//!
//! Offloads the polyphase filterbank and inverse FFT to the GPU,
//! eliminating the CPU bottleneck that causes USRP overflows at
//! high channel counts.

use num_complex::Complex32;
use std::slice;

mod ffi {
    use std::os::raw::{c_float, c_int, c_uint};

    extern "C" {
        pub fn gpu_pfb_init(
            m: c_uint,
            h_sub_len: c_uint,
            h_sub: *const i16,
            batch_size: c_uint,
        ) -> c_int;

        pub fn gpu_pfb_get_buffer() -> *mut i16;
        pub fn gpu_pfb_buffer_len() -> c_uint;
        pub fn gpu_pfb_submit() -> *mut c_float;
        pub fn gpu_pfb_flush() -> *mut c_float;
        pub fn gpu_pfb_result_len() -> c_uint;
        pub fn gpu_pfb_batch_size() -> c_uint;
        pub fn gpu_pfb_num_channels() -> c_uint;
        pub fn gpu_pfb_deinit();
    }
}

/// GPU-accelerated polyphase filterbank channelizer + inverse FFT.
///
/// Double-buffered: while the GPU processes one batch, the host fills the next.
/// Each batch produces `batch_size * num_channels` complex samples.
pub struct GpuChannelizer {
    num_channels: usize,
    batch_size: usize,
    buffer_len: usize,
    result_len: usize,
}

impl GpuChannelizer {
    /// Initialize the GPU channelizer.
    ///
    /// - `num_channels`: M (number of PFB output channels, e.g. 96)
    /// - `semi_len`: m (prototype filter semi-length, typically 4)
    /// - `prototype`: float prototype filter coefficients
    /// - `batch_size`: number of PFB time steps per GPU batch (e.g. 4096)
    pub fn new(
        num_channels: usize,
        semi_len: usize,
        prototype: &[f32],
        batch_size: usize,
    ) -> Result<Self, String> {
        let h_sub_len = 2 * semi_len;
        let h_used = 2 * num_channels * semi_len;

        if prototype.len() < h_used {
            return Err(format!(
                "prototype length {} < required {} (2*M*m)",
                prototype.len(),
                h_used
            ));
        }

        // Decompose prototype into polyphase sub-filters (same as CPU PfbChannelizer)
        let h_int: Vec<i16> = prototype[..h_used]
            .iter()
            .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect();

        let mut h_sub = vec![0i16; num_channels * h_sub_len];
        for i in 0..num_channels {
            for n in 0..h_sub_len {
                let src_idx = i + n * num_channels;
                if src_idx < h_int.len() {
                    h_sub[i * h_sub_len + h_sub_len - n - 1] = h_int[src_idx];
                }
            }
        }

        let ret = unsafe {
            ffi::gpu_pfb_init(
                num_channels as u32,
                h_sub_len as u32,
                h_sub.as_ptr(),
                batch_size as u32,
            )
        };

        if ret != 0 {
            return Err("GPU initialization failed (check stderr for details)".to_string());
        }

        let buffer_len = unsafe { ffi::gpu_pfb_buffer_len() } as usize;
        let result_len = unsafe { ffi::gpu_pfb_result_len() } as usize;

        Ok(Self {
            num_channels,
            batch_size,
            buffer_len,
            result_len,
        })
    }

    /// Get a mutable slice to the raw int16 buffer for the current batch.
    ///
    /// Fill this with `batch_size * M` int16 IQ samples (M/2 complex samples per step,
    /// 2 int16 values per complex sample, M int16 values per step). The pre-roll region
    /// is already filled from the previous batch.
    pub fn raw_buffer(&self) -> &mut [i16] {
        unsafe {
            let ptr = ffi::gpu_pfb_get_buffer();
            slice::from_raw_parts_mut(ptr, self.buffer_len)
        }
    }

    /// Submit the filled buffer to the GPU and get the previous batch's result.
    ///
    /// Returns `None` on the first call (no previous result available).
    /// On subsequent calls, returns the FFT output as interleaved float complex:
    /// `result[t * M * 2 + ch * 2]` = real, `result[t * M * 2 + ch * 2 + 1]` = imag.
    ///
    /// The returned slice is valid until the next call to `submit()`.
    pub fn submit(&self) -> Option<&[f32]> {
        let ptr = unsafe { ffi::gpu_pfb_submit() };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(ptr, self.result_len) })
        }
    }

    /// Get the final batch's result after the last `submit()`.
    pub fn flush(&self) -> Option<&[f32]> {
        let ptr = unsafe { ffi::gpu_pfb_flush() };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(ptr, self.result_len) })
        }
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer_len
    }
}

impl Drop for GpuChannelizer {
    fn drop(&mut self) {
        unsafe { ffi::gpu_pfb_deinit() };
    }
}

/// Process a GPU result batch, calling `f(time_step, fft_bin, sample)` for each output.
#[inline]
pub fn iter_gpu_result(
    result: &[f32],
    num_channels: usize,
    batch_size: usize,
    fft_scale: f32,
    mut f: impl FnMut(usize, usize, Complex32),
) {
    for t in 0..batch_size {
        let base = t * num_channels * 2;
        for ch in 0..num_channels {
            let idx = base + ch * 2;
            let sample = Complex32::new(
                result[idx] * fft_scale,
                result[idx + 1] * fft_scale,
            );
            f(t, ch, sample);
        }
    }
}
