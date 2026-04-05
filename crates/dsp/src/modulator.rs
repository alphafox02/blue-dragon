// Copyright 2026 CEMAXECUTER LLC
//
// BLE GFSK modulator: generates IQ waveforms from BLE packet bit streams.
//
// Parameters per BT Core Spec Vol 6 Part B Section 3:
//   LE 1M: 1 Msym/s, h=0.5, BT=0.5, Gaussian filter
//   LE 2M: 2 Msym/s, h=0.5, BT=0.5, Gaussian filter
//
// Output: interleaved I,Q as f32 at the specified samples-per-symbol rate.

use std::f64::consts::PI;

/// BLE PHY type for modulation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModPhy {
    /// LE 1M: 1 Msym/s
    Le1M,
    /// LE 2M: 2 Msym/s
    Le2M,
}

/// GFSK modulator for BLE packet generation.
pub struct GfskModulator {
    /// Samples per symbol
    sps: usize,
    /// Gaussian filter coefficients (length = filter_span * sps)
    gauss_filter: Vec<f64>,
    /// Modulation index h
    h: f64,
}

impl GfskModulator {
    /// Create a new GFSK modulator.
    /// `sps`: samples per symbol (typically 2 or 4)
    pub fn new(sps: usize) -> Self {
        let bt = 0.5; // BLE specification: BT=0.5
        let h = 0.5;  // BLE specification: h=0.5
        let filter_span = 3; // 3 symbol periods

        let gauss_filter = gaussian_filter(bt, sps, filter_span);

        Self {
            sps,
            gauss_filter,
            h,
        }
    }

    /// Modulate a BLE packet into IQ samples.
    ///
    /// `bits`: raw bit stream (preamble + AA + PDU + CRC, after whitening)
    /// Returns interleaved I,Q pairs as f32 (suitable for SDR TX or WHAD injection).
    pub fn modulate(&self, bits: &[u8]) -> Vec<f32> {
        if bits.is_empty() {
            return Vec::new();
        }

        // Convert bits to NRZ: 0 -> -1.0, 1 -> +1.0
        let nrz: Vec<f64> = bits.iter().map(|&b| if b != 0 { 1.0 } else { -1.0 }).collect();

        // Upsample to SPS rate (zero-insert + filter)
        let upsampled = self.upsample_and_filter(&nrz);

        // Integrate frequency to get instantaneous phase
        // phase[n] = pi * h * sum(freq[0..n]) / sps
        let phase_inc = PI * self.h / self.sps as f64;
        let mut phase = 0.0f64;
        let mut iq = Vec::with_capacity(upsampled.len() * 2);

        for &freq in &upsampled {
            phase += freq * phase_inc;
            // Wrap phase to [-pi, pi] for numerical stability
            if phase > PI { phase -= 2.0 * PI; }
            if phase < -PI { phase += 2.0 * PI; }
            iq.push(phase.cos() as f32);
            iq.push(phase.sin() as f32);
        }

        iq
    }

    /// Modulate and return as i16 IQ (for SDR TX backends that use i16).
    pub fn modulate_i16(&self, bits: &[u8]) -> Vec<i16> {
        let iq_f32 = self.modulate(bits);
        iq_f32
            .iter()
            .map(|&s| (s * 32000.0).clamp(-32768.0, 32767.0) as i16)
            .collect()
    }

    fn upsample_and_filter(&self, nrz: &[f64]) -> Vec<f64> {
        let n_out = nrz.len() * self.sps;
        let mut upsampled = vec![0.0f64; n_out];

        // Zero-insert upsample
        for (i, &val) in nrz.iter().enumerate() {
            upsampled[i * self.sps] = val;
        }

        // Convolve with Gaussian filter
        convolve(&upsampled, &self.gauss_filter)
    }
}

/// Generate Gaussian filter coefficients for GFSK.
/// BT: bandwidth-time product (0.5 for BLE)
/// sps: samples per symbol
/// span: filter span in symbols
fn gaussian_filter(bt: f64, sps: usize, span: usize) -> Vec<f64> {
    let len = span * sps + 1;
    let alpha = (2.0 * PI * bt) / (2.0 * 2.0f64.ln()).sqrt();
    let mid = (len / 2) as f64;

    let mut filter: Vec<f64> = (0..len)
        .map(|i| {
            let t = (i as f64 - mid) / sps as f64;
            // Gaussian pulse: g(t) = alpha * exp(-alpha^2 * t^2 / 2)
            alpha * (-0.5 * alpha * alpha * t * t).exp()
        })
        .collect();

    // Normalize so sum of absolute values = sps (unit energy per symbol)
    let sum: f64 = filter.iter().map(|x| x.abs()).sum();
    if sum > 0.0 {
        let scale = sps as f64 / sum;
        for v in &mut filter {
            *v *= scale;
        }
    }

    filter
}

/// Simple convolution (output same length as input, centered).
fn convolve(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let m = filter.len();
    let half = m / 2;
    let mut out = vec![0.0f64; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            let idx = i as isize + j as isize - half as isize;
            if idx >= 0 && (idx as usize) < n {
                sum += signal[idx as usize] * filter[j];
            }
        }
        out[i] = sum;
    }

    out
}

/// Build a complete BLE packet bit stream ready for modulation.
///
/// Includes: preamble + access address + PDU + CRC.
/// PDU must already be whitened.
/// Returns the raw bit stream (LSB-first per byte, as BLE specifies).
pub fn build_ble_bitstream(
    phy: ModPhy,
    access_address: u32,
    pdu_whitened: &[u8],
    crc: u32,
) -> Vec<u8> {
    let mut bits = Vec::with_capacity(8 + 32 + pdu_whitened.len() * 8 + 24 + 16);

    // Preamble
    match phy {
        ModPhy::Le1M => {
            // 1-byte preamble: depends on LSB of AA
            let aa_lsb = access_address & 1;
            if aa_lsb == 0 {
                push_byte_bits(&mut bits, 0xAA); // 10101010
            } else {
                push_byte_bits(&mut bits, 0x55); // 01010101
            }
        }
        ModPhy::Le2M => {
            // 2-byte preamble: depends on LSB of AA
            let aa_lsb = access_address & 1;
            if aa_lsb == 0 {
                push_byte_bits(&mut bits, 0xAA);
                push_byte_bits(&mut bits, 0xAA);
            } else {
                push_byte_bits(&mut bits, 0x55);
                push_byte_bits(&mut bits, 0x55);
            }
        }
    }

    // Access Address (4 bytes, LSB first)
    for i in 0..4 {
        push_byte_bits(&mut bits, ((access_address >> (i * 8)) & 0xFF) as u8);
    }

    // PDU (already whitened)
    for &byte in pdu_whitened {
        push_byte_bits(&mut bits, byte);
    }

    // CRC (3 bytes, LSB first)
    for i in 0..3 {
        push_byte_bits(&mut bits, ((crc >> (i * 8)) & 0xFF) as u8);
    }

    bits
}

/// Push 8 bits from a byte, LSB first (BLE bit ordering).
fn push_byte_bits(bits: &mut Vec<u8>, byte: u8) {
    for i in 0..8 {
        bits.push((byte >> i) & 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_filter() {
        let filter = gaussian_filter(0.5, 4, 3);
        assert_eq!(filter.len(), 13); // 3*4 + 1
        // Filter should be symmetric
        let mid = filter.len() / 2;
        for i in 0..mid {
            assert!((filter[i] - filter[filter.len() - 1 - i]).abs() < 1e-10);
        }
        // Peak at center
        assert!(filter[mid] > filter[0]);
    }

    #[test]
    fn test_modulate_output_length() {
        let mod_ = GfskModulator::new(2);
        // 8 bits at SPS=2 = 16 IQ samples = 32 f32 values
        let bits = vec![1u8, 0, 1, 0, 1, 0, 1, 0];
        let iq = mod_.modulate(&bits);
        assert_eq!(iq.len(), 32); // 16 samples * 2 (I+Q)
    }

    #[test]
    fn test_modulate_amplitude() {
        let mod_ = GfskModulator::new(4);
        let bits = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let iq = mod_.modulate(&bits);
        // All IQ samples should be unit magnitude (pure FM)
        for chunk in iq.chunks_exact(2) {
            let mag = (chunk[0] * chunk[0] + chunk[1] * chunk[1]).sqrt();
            assert!((mag - 1.0).abs() < 0.01, "magnitude {} != 1.0", mag);
        }
    }

    #[test]
    fn test_modulate_i16() {
        let mod_ = GfskModulator::new(2);
        let bits = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let iq = mod_.modulate_i16(&bits);
        assert_eq!(iq.len(), 32);
        // Values should be in i16 range
        for &v in &iq {
            assert!(v >= -32768 && v <= 32767);
        }
    }

    #[test]
    fn test_build_ble_bitstream_le1m() {
        let aa = 0x8E89BED6u32; // advertising AA
        let pdu = vec![0x00, 0x06, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06]; // 8 bytes
        let crc = 0x123456u32;

        let bits = build_ble_bitstream(ModPhy::Le1M, aa, &pdu, crc);

        // Length: 8 (preamble) + 32 (AA) + 64 (PDU) + 24 (CRC) = 128 bits
        assert_eq!(bits.len(), 128);

        // Preamble: AA LSB = 0, so preamble = 0xAA = 01010101 (LSB first)
        assert_eq!(&bits[0..8], &[0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_build_ble_bitstream_le2m() {
        let aa = 0x8E89BED6u32;
        let pdu = vec![0x00, 0x06];
        let crc = 0x000000u32;

        let bits = build_ble_bitstream(ModPhy::Le2M, aa, &pdu, crc);

        // Length: 16 (preamble) + 32 (AA) + 16 (PDU) + 24 (CRC) = 88 bits
        assert_eq!(bits.len(), 88);
    }

    #[test]
    fn test_push_byte_bits_lsb_first() {
        let mut bits = Vec::new();
        push_byte_bits(&mut bits, 0xD6); // 11010110 -> LSB first: 0,1,1,0,1,0,1,1
        assert_eq!(bits, vec![0, 1, 1, 0, 1, 0, 1, 1]);
    }
}
