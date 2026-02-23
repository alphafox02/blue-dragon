use num_complex::Complex32;

const MEDIAN_SYMBOLS: usize = 64;
const MAX_FREQ_OFFSET: f32 = 0.4;

/// Result of FSK demodulation
pub struct FskResult {
    pub demod: Vec<f32>,
    pub bits: Vec<u8>,
    pub silence: usize,
    pub cfo: f32,
    pub deviation: f32,
}

/// FSK demodulator: replaces liquid-dsp's freqdem
pub struct FskDemod {
    prev_sample: Complex32,
    pos_points: Vec<f32>,
    neg_points: Vec<f32>,
    sps: usize,
}

impl FskDemod {
    pub fn new(sps: usize) -> Self {
        let median_size = sps * MEDIAN_SYMBOLS;
        Self {
            prev_sample: Complex32::new(0.0, 0.0),
            pos_points: Vec::with_capacity(median_size),
            neg_points: Vec::with_capacity(median_size),
            sps,
        }
    }

    fn median_size(&self) -> usize {
        self.sps * MEDIAN_SYMBOLS
    }

    /// Reset demodulator state
    pub fn reset(&mut self) {
        self.prev_sample = Complex32::new(0.0, 0.0);
    }

    /// FM frequency discriminator: arg(y[n] * conj(y[n-1])) / (2*pi*kf)
    /// This replaces liquid-dsp's freqdem_demodulate_block.
    /// The kf=0.5 normalization matches liquid-dsp's freqdem_create(0.5f),
    /// producing output in roughly [-1, 1] range for BLE signals.
    fn freq_discriminate(&mut self, burst: &[Complex32]) -> Vec<f32> {
        // kf = 0.5 → normalization = 1/(2*pi*0.5) = 1/pi
        const KF_NORM: f32 = std::f32::consts::FRAC_1_PI;
        let mut demod = Vec::with_capacity(burst.len());
        for &sample in burst {
            let product = sample * self.prev_sample.conj();
            demod.push(product.arg() * KF_NORM);
            self.prev_sample = sample;
        }
        demod
    }

    /// Carrier frequency offset correction using median of positive/negative points.
    /// Returns (cfo, deviation) or None if the burst is invalid.
    fn cfo_median(&mut self, demod: &[f32]) -> Option<(f32, f32)> {
        self.pos_points.clear();
        self.neg_points.clear();

        let end = 8 + self.median_size();
        if end > demod.len() {
            return None;
        }

        for &val in &demod[8..end] {
            if val.abs() > MAX_FREQ_OFFSET {
                return None;
            }
            if val > 0.0 {
                self.pos_points.push(val);
            } else {
                self.neg_points.push(val);
            }
        }

        if self.pos_points.len() < MEDIAN_SYMBOLS / 4
            || self.neg_points.len() < MEDIAN_SYMBOLS / 4
        {
            return None;
        }

        self.pos_points.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        self.neg_points.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let midpoint = (self.pos_points[self.pos_points.len() * 3 / 4]
            + self.neg_points[self.neg_points.len() / 4])
            / 2.0;
        let deviation = self.pos_points[self.pos_points.len() * 3 / 4] - midpoint;

        Some((midpoint, deviation))
    }

    /// Silence detection: EWMA of |demod|, returns first index where signal exceeds threshold
    fn silence_skip(demod: &[f32]) -> usize {
        const ALPHA: f32 = 0.8;
        let mut ewma: f32 = 0.0;

        for (i, &val) in demod.iter().enumerate() {
            ewma = ALPHA * val.abs() + (1.0 - ALPHA) * ewma;
            if ewma > 0.5 {
                return if i > 0 { i - 1 } else { 0 };
            }
        }
        0
    }

    /// Raw FM discriminator only (no CFO check, no normalization).
    /// Used by LE Coded decoder as a fallback when full demod fails,
    /// since coded preamble correlation works on unnormalized FM output.
    pub fn fm_discriminate_raw(&mut self, burst: &[Complex32]) -> Vec<f32> {
        self.reset();
        self.freq_discriminate(burst)
    }

    /// Full FSK demodulation pipeline:
    /// 1. FM demodulate
    /// 2. CFO correction (median-based)
    /// 3. Normalize to roughly [-1, 1]
    /// 4. Silence detection
    /// 5. Bit slicing (one bit per symbol, sample at center)
    pub fn demodulate(&mut self, burst: &[Complex32]) -> Option<FskResult> {
        self.reset();

        if burst.len() < 8 + self.median_size() {
            return None;
        }

        let mut demod = self.freq_discriminate(burst);

        let (cfo, deviation) = self.cfo_median(&demod)?;

        // CFO correction and normalization
        for val in demod.iter_mut() {
            *val -= cfo;
            *val /= deviation;
        }
        // Clamp first sample (tends to be wild)
        if demod[0].abs() > 1.5 {
            demod[0] = 0.0;
        }

        let silence_offset = Self::silence_skip(&demod);

        // Bit slicing: sample every sps samples starting at offset+1
        let mut bits = Vec::with_capacity((demod.len() - silence_offset) / self.sps);
        let mut i = silence_offset + 1;
        while i < demod.len() {
            bits.push(if demod[i] > 0.0 { 1 } else { 0 });
            i += self.sps;
        }

        Some(FskResult {
            demod,
            bits,
            silence: silence_offset,
            cfo,
            deviation,
        })
    }
}

/// Re-slice an existing demod signal at a different samples-per-symbol rate.
/// Used to try LE 2M decoding (SPS=1) from the same burst that was already
/// FM-demodulated and CFO-corrected at 2 Msps.
pub fn reslice(demod: &[f32], silence_offset: usize, sps: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity((demod.len() - silence_offset) / sps);
    let mut i = silence_offset + 1;
    while i < demod.len() {
        bits.push(if demod[i] > 0.0 { 1 } else { 0 });
        i += sps;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freq_discriminator_dc() {
        let mut demod = FskDemod::new(2);
        // Constant frequency signal should give constant output
        let freq = 0.1f32;
        let burst: Vec<Complex32> = (0..256)
            .map(|i| Complex32::from_polar(1.0, freq * i as f32))
            .collect();
        let result = demod.freq_discriminate(&burst);
        // After the first sample, all values should be ~freq/PI (normalized by kf=0.5)
        let expected = freq * std::f32::consts::FRAC_1_PI;
        for &val in &result[1..] {
            assert!((val - expected).abs() < 0.01, "got {}, expected {}", val, expected);
        }
    }

    #[test]
    fn test_silence_skip() {
        // Signal that starts quiet then gets loud
        let mut demod = vec![0.0f32; 100];
        for i in 50..100 {
            demod[i] = 1.0;
        }
        let offset = FskDemod::silence_skip(&demod);
        assert!(offset > 40 && offset < 55, "silence_skip returned {}", offset);
    }
}
