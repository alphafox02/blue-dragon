// Copyright 2025-2026 CEMAXECUTER LLC

use num_complex::Complex32;

// ── Delay line: flat buffer with power-of-2 wrap ─────────────────────────────
//
// Matches C window.c: contiguous data from read_index enables SIMD loads.
// Every `n` pushes, memmove copies the tail (len-1 elements) back to front.

struct DelayLine {
    r: Vec<i16>,
    i: Vec<i16>,
    read_index: usize,
    mask: usize,
    len: usize,
    n: usize,
}

impl DelayLine {
    fn new(len: usize) -> Self {
        // Next power of 2 strictly > len (matches C: m = floor(log2(len)) + 1)
        let m = if len <= 1 {
            1
        } else {
            (len as f64).log2().floor() as u32 + 1
        };
        let n = 1usize << m;
        let mask = n - 1;
        let num_allocated = n + len - 1;

        Self {
            r: vec![0i16; num_allocated],
            i: vec![0i16; num_allocated],
            read_index: 0,
            mask,
            len,
            n,
        }
    }

    #[inline(always)]
    fn push(&mut self, real: i16, imag: i16) {
        self.read_index = self.read_index.wrapping_add(1) & self.mask;
        if self.read_index == 0 {
            let copy_len = self.len - 1;
            self.r.copy_within(self.n..self.n + copy_len, 0);
            self.i.copy_within(self.n..self.n + copy_len, 0);
        }
        let pos = self.read_index + self.len - 1;
        self.r[pos] = real;
        self.i[pos] = imag;
    }

    #[inline]
    fn dot_product_scalar(&self, coeffs: &[i16]) -> (i32, i32) {
        let base = self.read_index;
        let mut sum_r: i32 = 0;
        let mut sum_i: i32 = 0;
        for j in 0..self.len {
            sum_r += self.r[base + j] as i32 * coeffs[j] as i32;
            sum_i += self.i[base + j] as i32 * coeffs[j] as i32;
        }
        (sum_r, sum_i)
    }
}

// ── SIMD dispatch ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum SimdLevel {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

fn detect_simd() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            eprintln!("SIMD: AVX2");
            return SimdLevel::Avx2;
        }
        eprintln!("SIMD: SSE2");
        return SimdLevel::Sse2;
    }
    #[cfg(target_arch = "aarch64")]
    {
        eprintln!("SIMD: NEON");
        return SimdLevel::Neon;
    }
    #[allow(unreachable_code)]
    {
        eprintln!("SIMD: scalar");
        SimdLevel::Scalar
    }
}

// ── x86_64 SIMD dot products ────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2 dot product using _mm_madd_epi16 (8 i16 elements per iteration).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn dotprod_sse2(
    r: *const i16,
    i_data: *const i16,
    coeffs: *const i16,
    len: usize,
) -> (i32, i32) {
    let mut r_acc = _mm_setzero_si128();
    let mut i_acc = _mm_setzero_si128();
    let mut j = 0usize;

    // Process 8 elements at a time
    while j + 8 <= len {
        let r_vec = _mm_loadu_si128(r.add(j) as *const __m128i);
        let i_vec = _mm_loadu_si128(i_data.add(j) as *const __m128i);
        let b_vec = _mm_loadu_si128(coeffs.add(j) as *const __m128i);

        r_acc = _mm_add_epi32(r_acc, _mm_madd_epi16(r_vec, b_vec));
        i_acc = _mm_add_epi32(i_acc, _mm_madd_epi16(i_vec, b_vec));
        j += 8;
    }

    // Remaining 4 elements (loadl_epi64 zero-extends upper half)
    if j + 4 <= len {
        let r_vec = _mm_loadl_epi64(r.add(j) as *const __m128i);
        let i_vec = _mm_loadl_epi64(i_data.add(j) as *const __m128i);
        let b_vec = _mm_loadl_epi64(coeffs.add(j) as *const __m128i);

        r_acc = _mm_add_epi32(r_acc, _mm_madd_epi16(r_vec, b_vec));
        i_acc = _mm_add_epi32(i_acc, _mm_madd_epi16(i_vec, b_vec));
        j += 4;
    }

    // Scalar tail (0-3 elements)
    let mut r_scalar: i32 = 0;
    let mut i_scalar: i32 = 0;
    while j < len {
        r_scalar += *r.add(j) as i32 * *coeffs.add(j) as i32;
        i_scalar += *i_data.add(j) as i32 * *coeffs.add(j) as i32;
        j += 1;
    }

    // Horizontal sum of 4 i32 lanes
    r_acc = _mm_add_epi32(r_acc, _mm_shuffle_epi32(r_acc, 0b00_01_10_11));
    r_acc = _mm_add_epi32(r_acc, _mm_shuffle_epi32(r_acc, 0b10_11_00_01));
    i_acc = _mm_add_epi32(i_acc, _mm_shuffle_epi32(i_acc, 0b00_01_10_11));
    i_acc = _mm_add_epi32(i_acc, _mm_shuffle_epi32(i_acc, 0b10_11_00_01));

    (_mm_cvtsi128_si32(r_acc) + r_scalar, _mm_cvtsi128_si32(i_acc) + i_scalar)
}

/// AVX2 dot product using _mm256_madd_epi16 (16 i16 elements per iteration),
/// with SSE2 fallback for remainders.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_avx2(
    r: *const i16,
    i_data: *const i16,
    coeffs: *const i16,
    len: usize,
) -> (i32, i32) {
    let mut r_acc256 = _mm256_setzero_si256();
    let mut i_acc256 = _mm256_setzero_si256();
    let mut j = 0usize;

    // Process 16 elements at a time
    while j + 16 <= len {
        let r_vec = _mm256_loadu_si256(r.add(j) as *const __m256i);
        let i_vec = _mm256_loadu_si256(i_data.add(j) as *const __m256i);
        let b_vec = _mm256_loadu_si256(coeffs.add(j) as *const __m256i);

        r_acc256 = _mm256_add_epi32(r_acc256, _mm256_madd_epi16(r_vec, b_vec));
        i_acc256 = _mm256_add_epi32(i_acc256, _mm256_madd_epi16(i_vec, b_vec));
        j += 16;
    }

    // Reduce 256 -> 128
    let mut r_acc = _mm_add_epi32(
        _mm256_castsi256_si128(r_acc256),
        _mm256_extracti128_si256(r_acc256, 1),
    );
    let mut i_acc = _mm_add_epi32(
        _mm256_castsi256_si128(i_acc256),
        _mm256_extracti128_si256(i_acc256, 1),
    );

    // Remaining 8 elements (SSE2)
    if j + 8 <= len {
        let r_vec = _mm_loadu_si128(r.add(j) as *const __m128i);
        let i_vec = _mm_loadu_si128(i_data.add(j) as *const __m128i);
        let b_vec = _mm_loadu_si128(coeffs.add(j) as *const __m128i);

        r_acc = _mm_add_epi32(r_acc, _mm_madd_epi16(r_vec, b_vec));
        i_acc = _mm_add_epi32(i_acc, _mm_madd_epi16(i_vec, b_vec));
        j += 8;
    }

    // Remaining 4 elements
    if j + 4 <= len {
        let r_vec = _mm_loadl_epi64(r.add(j) as *const __m128i);
        let i_vec = _mm_loadl_epi64(i_data.add(j) as *const __m128i);
        let b_vec = _mm_loadl_epi64(coeffs.add(j) as *const __m128i);

        r_acc = _mm_add_epi32(r_acc, _mm_madd_epi16(r_vec, b_vec));
        i_acc = _mm_add_epi32(i_acc, _mm_madd_epi16(i_vec, b_vec));
        j += 4;
    }

    // Scalar tail
    let mut r_scalar: i32 = 0;
    let mut i_scalar: i32 = 0;
    while j < len {
        r_scalar += *r.add(j) as i32 * *coeffs.add(j) as i32;
        i_scalar += *i_data.add(j) as i32 * *coeffs.add(j) as i32;
        j += 1;
    }

    // Horizontal sum
    r_acc = _mm_add_epi32(r_acc, _mm_shuffle_epi32(r_acc, 0b00_01_10_11));
    r_acc = _mm_add_epi32(r_acc, _mm_shuffle_epi32(r_acc, 0b10_11_00_01));
    i_acc = _mm_add_epi32(i_acc, _mm_shuffle_epi32(i_acc, 0b00_01_10_11));
    i_acc = _mm_add_epi32(i_acc, _mm_shuffle_epi32(i_acc, 0b10_11_00_01));

    (_mm_cvtsi128_si32(r_acc) + r_scalar, _mm_cvtsi128_si32(i_acc) + i_scalar)
}

// ── aarch64 NEON dot product ────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
unsafe fn dotprod_neon(
    r: *const i16,
    i_data: *const i16,
    coeffs: *const i16,
    len: usize,
) -> (i32, i32) {
    let mut r_acc = vdupq_n_s32(0);
    let mut i_acc = vdupq_n_s32(0);
    let mut j = 0usize;

    while j + 4 <= len {
        let r_vec = vld1_s16(r.add(j));
        let i_vec = vld1_s16(i_data.add(j));
        let b_vec = vld1_s16(coeffs.add(j));

        r_acc = vmlal_s16(r_acc, r_vec, b_vec);
        i_acc = vmlal_s16(i_acc, i_vec, b_vec);
        j += 4;
    }

    let mut r_sum = vaddvq_s32(r_acc);
    let mut i_sum = vaddvq_s32(i_acc);

    while j < len {
        r_sum += *r.add(j) as i32 * *coeffs.add(j) as i32;
        i_sum += *i_data.add(j) as i32 * *coeffs.add(j) as i32;
        j += 1;
    }

    (r_sum, i_sum)
}

// ── Dispatch helper ─────────────────────────────────────────────────────────

#[inline(always)]
fn dot_product_dispatch(
    simd: SimdLevel,
    w: &DelayLine,
    coeffs_ptr: *const i16,
    len: usize,
) -> (i32, i32) {
    match simd {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe {
            dotprod_avx2(
                w.r.as_ptr().add(w.read_index),
                w.i.as_ptr().add(w.read_index),
                coeffs_ptr,
                len,
            )
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse2 => unsafe {
            dotprod_sse2(
                w.r.as_ptr().add(w.read_index),
                w.i.as_ptr().add(w.read_index),
                coeffs_ptr,
                len,
            )
        },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe {
            dotprod_neon(
                w.r.as_ptr().add(w.read_index),
                w.i.as_ptr().add(w.read_index),
                coeffs_ptr,
                len,
            )
        },
        SimdLevel::Scalar => w.dot_product_scalar(unsafe {
            std::slice::from_raw_parts(coeffs_ptr, len)
        }),
    }
}

// ── Polyphase Filter Bank Analysis Channelizer, Type 2 ──────────────────────

/// Polyphase Filter Bank Analysis Channelizer, Type 2
///
/// Matches the C pfbch2 implementation:
/// - Takes M/2 new complex samples per call (as M interleaved int16 values)
/// - Produces M complex output samples per call
/// - Alternates between two coefficient offsets (flag)
/// - Coefficients stored in reversed order per sub-filter
/// - Int16 fixed-point arithmetic with >> 16 normalization
/// - SIMD-accelerated dot product (SSE2/AVX2 on x86_64, NEON on aarch64)
pub struct PfbChannelizer {
    m: usize,
    m2: usize,
    h_sub_len: usize,
    h_sub: Vec<i16>,
    windows: Vec<DelayLine>,
    flag: bool,
    simd: SimdLevel,
}

impl PfbChannelizer {
    /// Create a new Type 2 PFB channelizer.
    ///
    /// - `num_channels`: M, number of output channels
    /// - `semi_len`: m, prototype filter semi-length (C code uses m=4)
    /// - `prototype`: float prototype filter coefficients (length >= 2*M*m)
    pub fn new(num_channels: usize, semi_len: usize, prototype: &[f32]) -> Self {
        let h_used = 2 * num_channels * semi_len;
        assert!(
            prototype.len() >= h_used,
            "prototype length {} < required {} (2*M*m)",
            prototype.len(),
            h_used
        );

        let h_sub_len = 2 * semi_len;

        // Convert float prototype to int16 (matching C: roundf(h[i] * 32768.f))
        let h_int: Vec<i16> = prototype[..h_used]
            .iter()
            .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect();

        // Decompose into sub-filters with reversed coefficient order
        // C: h_sub[i * h_sub_len + h_sub_len - n - 1] = h[i + n * M]
        let mut h_sub = vec![0i16; num_channels * h_sub_len];
        for i in 0..num_channels {
            for n in 0..h_sub_len {
                let src_idx = i + n * num_channels;
                if src_idx < h_int.len() {
                    h_sub[i * h_sub_len + h_sub_len - n - 1] = h_int[src_idx];
                }
            }
        }

        let windows: Vec<DelayLine> = (0..num_channels)
            .map(|_| DelayLine::new(h_sub_len))
            .collect();

        let simd = detect_simd();

        Self {
            m: num_channels,
            m2: num_channels / 2,
            h_sub_len,
            h_sub,
            windows,
            flag: false,
            simd,
        }
    }

    /// Process M/2 new complex int16 samples. Writes M Complex32 outputs into `output`.
    ///
    /// `input`: M int16 values (M/2 interleaved complex samples: I0,Q0,I1,Q1,...)
    /// `output`: pre-allocated buffer of M Complex32 values
    pub fn execute_into(&mut self, input: &[i16], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), self.m);
        debug_assert_eq!(output.len(), self.m);

        // Push M/2 samples into the appropriate windows
        let base_index = if self.flag { self.m } else { self.m2 };
        for i in 0..self.m2 {
            let r = input[2 * i];
            let q = input[2 * i + 1];
            self.windows[base_index - i - 1].push(r, q);
        }

        // Compute dot products with alternating coefficient offset
        let coeff_offset = if self.flag { self.m2 } else { 0 };
        let simd = self.simd;
        let h_sub_len = self.h_sub_len;

        for i in 0..self.m {
            let cur_offset = (coeff_offset + i) % self.m;
            let coeff_start = cur_offset * h_sub_len;

            let (sum_r, sum_i) = dot_product_dispatch(
                simd,
                &self.windows[i],
                unsafe { self.h_sub.as_ptr().add(coeff_start) },
                h_sub_len,
            );

            output[i] = Complex32::new(
                (sum_r >> 16) as f32 / 32768.0,
                (sum_i >> 16) as f32 / 32768.0,
            );
        }

        self.flag = !self.flag;
    }

    /// Process M/2 new complex int16 samples, returning a new Vec.
    /// Prefer `execute_into` in hot paths to avoid allocation.
    pub fn execute(&mut self, input: &[i16]) -> Vec<Complex32> {
        let mut output = vec![Complex32::new(0.0, 0.0); self.m];
        self.execute_into(input, &mut output);
        output
    }

    pub fn num_channels(&self) -> usize {
        self.m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pfb_construction() {
        let m = 8;
        let semi_len = 4;
        let proto = vec![0.01f32; 2 * m * semi_len + 1];
        let pfb = PfbChannelizer::new(m, semi_len, &proto);
        assert_eq!(pfb.num_channels(), m);
    }

    #[test]
    fn test_pfb_execute_returns_m_outputs() {
        let m = 8;
        let semi_len = 4;
        let proto = vec![0.01f32; 2 * m * semi_len + 1];
        let mut pfb = PfbChannelizer::new(m, semi_len, &proto);

        let input = vec![1000i16; m];
        let outputs = pfb.execute(&input);
        assert_eq!(outputs.len(), m);
    }

    #[test]
    fn test_pfb_execute_into_matches_execute() {
        let m = 16;
        let semi_len = 4;
        let proto: Vec<f32> = (0..2 * m * semi_len + 1)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let mut pfb1 = PfbChannelizer::new(m, semi_len, &proto);
        let mut pfb2 = PfbChannelizer::new(m, semi_len, &proto);

        let input: Vec<i16> = (0..m as i16).map(|i| i * 100).collect();
        let mut output = vec![Complex32::new(0.0, 0.0); m];

        for _ in 0..10 {
            let vec_out = pfb1.execute(&input);
            pfb2.execute_into(&input, &mut output);

            for (a, b) in vec_out.iter().zip(output.iter()) {
                assert!(
                    (a.re - b.re).abs() < 1e-6 && (a.im - b.im).abs() < 1e-6,
                    "mismatch: {:?} vs {:?}",
                    a,
                    b,
                );
            }
        }
    }

    #[test]
    fn test_pfb_type2_alternation() {
        let m = 8;
        let semi_len = 2;
        let proto = vec![0.1f32; 2 * m * semi_len + 1];
        let mut pfb = PfbChannelizer::new(m, semi_len, &proto);

        let input = vec![1000i16; m];
        let out1 = pfb.execute(&input);
        let out2 = pfb.execute(&input);

        let sum1: f32 = out1.iter().map(|c| c.norm()).sum();
        let sum2: f32 = out2.iter().map(|c| c.norm()).sum();
        assert!(sum1 > 0.0 || sum2 > 0.0, "both outputs are zero");
    }

    #[test]
    fn test_delay_line_contiguous() {
        let len = 8;
        let mut dl = DelayLine::new(len);

        // Push enough to trigger at least one memmove
        for k in 0..50 {
            dl.push(k as i16, -(k as i16));
        }

        // Verify the window is contiguous and correct
        let base = dl.read_index;
        for j in 0..len {
            let _r = dl.r[base + j]; // should not panic
            let _i = dl.i[base + j];
        }
    }
}
