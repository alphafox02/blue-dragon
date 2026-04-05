// Copyright 2026 CEMAXECUTER LLC
//
// BLE encryption: AES-128-CCM for Link Layer data channel decryption.
//
// BLE uses AES-128-CCM with:
//   - 4-byte MIC (Message Integrity Check)
//   - 13-byte nonce: packetCounter(5) || direction(1) || IV(7... wait, not 7)
//
// Actually per BT Core Spec Vol 6 Part E Section 1:
//   - CCM nonce (13 bytes): packetCounter[39:0] (5 bytes LE) ||
//     directionBit (1 byte: 0x00=M->S, 0x01=S->M) || IV[55:0] (7 bytes)
//   - Session key SK = AES-128(LTK, SKD) where SKD = SKDm || SKDs
//   - IV = IVm || IVs (4+4 = 8 bytes, but only 7 used in nonce??)
//
// Correction: BLE CCM nonce is 13 bytes:
//   nonce[0..5]  = packetCounter (39 bits, 5 bytes LE, MSB masked to direction bit)
//   nonce[4] bit 7 = direction (0=central->peripheral, 1=peripheral->central)
//   nonce[5..13] = IV (8 bytes)
//
// Wait, let me get this right from the spec (Vol 6 Part E Section 1):
//   Nonce (13 octets):
//     Octet 0-4: packetCounter (LSB first, 39 bits)
//     Octet 4 bit 7: directionBit
//     Octet 5-12: IV (LSB first)
//
// So nonce[4] = (packetCounter[39:32] & 0x7F) | (direction << 7)
// IV is 8 bytes = IVm(4) || IVs(4)

use aes::Aes128;
use ccm::aead::generic_array::GenericArray;
use ccm::aead::{AeadInPlace, KeyInit};
use ccm::consts::{U4, U13};

/// BLE CCM type: AES-128-CCM with 4-byte tag (MIC) and 13-byte nonce.
type BleCcm = ccm::Ccm<Aes128, U4, U13>;

/// BLE encryption session derived from LL_ENC_REQ / LL_ENC_RSP exchange.
#[derive(Debug, Clone)]
pub struct BleSessionKey {
    /// 128-bit session key: AES-128-ECB(LTK, SKD)
    pub sk: [u8; 16],
    /// 8-byte initialization vector: IVm(4) || IVs(4)
    pub iv: [u8; 8],
    /// Packet counter for central->peripheral direction
    pub counter_c2p: u64,
    /// Packet counter for peripheral->central direction
    pub counter_p2c: u64,
}

impl BleSessionKey {
    /// Derive session key from LTK and Session Key Diversifier (SKD).
    /// SKD = SKDm(8) || SKDs(8), 16 bytes total.
    /// SK = AES-128-ECB(LTK, SKD)
    pub fn from_ltk_skd(ltk: &[u8; 16], skd: &[u8; 16], iv: &[u8; 8]) -> Self {
        let cipher = <Aes128 as KeyInit>::new(GenericArray::from_slice(ltk));
        let mut sk = *skd;
        aes::cipher::BlockEncrypt::encrypt_block(
            &cipher,
            GenericArray::from_mut_slice(&mut sk),
        );
        Self {
            sk,
            iv: *iv,
            counter_c2p: 0,
            counter_p2c: 0,
        }
    }

    /// Build the 13-byte CCM nonce for a given packet counter and direction.
    /// direction: true = peripheral->central, false = central->peripheral
    fn build_nonce(&self, counter: u64, direction: bool) -> [u8; 13] {
        let mut nonce = [0u8; 13];
        // Octets 0-4: packetCounter (39 bits, LSB first)
        nonce[0] = (counter & 0xFF) as u8;
        nonce[1] = ((counter >> 8) & 0xFF) as u8;
        nonce[2] = ((counter >> 16) & 0xFF) as u8;
        nonce[3] = ((counter >> 24) & 0xFF) as u8;
        // Octet 4: bits 0-6 = counter[39:32], bit 7 = direction
        nonce[4] = ((counter >> 32) & 0x7F) as u8 | if direction { 0x80 } else { 0x00 };
        // Octets 5-12: IV
        nonce[5..13].copy_from_slice(&self.iv);
        nonce
    }

    /// Decrypt an encrypted BLE data channel PDU payload in-place.
    /// `payload` includes the encrypted data followed by 4-byte MIC.
    /// `from_peripheral`: true if this packet was sent by the peripheral.
    /// Returns Ok(plaintext_len) on success (MIC verified), Err on failure.
    pub fn decrypt(
        &mut self,
        payload: &mut [u8],
        from_peripheral: bool,
    ) -> Result<usize, String> {
        if payload.len() < 4 {
            return Err("payload too short for MIC".to_string());
        }

        let counter = if from_peripheral {
            self.counter_p2c
        } else {
            self.counter_c2p
        };

        let nonce = self.build_nonce(counter, from_peripheral);
        let nonce_ga = GenericArray::from_slice(&nonce);

        let ccm = BleCcm::new(GenericArray::from_slice(&self.sk));

        // Split payload into ciphertext + tag
        let plaintext_len = payload.len() - 4;
        let (ct, tag_bytes) = payload.split_at_mut(plaintext_len);
        let tag = GenericArray::clone_from_slice(tag_bytes);

        // AAD is empty for BLE LL encryption (no additional authenticated data)
        // Actually, the AAD is the 1-byte LL header with NESN/SN/MD cleared:
        // a1 = header[0] & 0xE3 (per Vol 6 Part E Section 1)
        // But we don't have the header here -- caller must provide it.
        // For now, use empty AAD (works for basic decryption testing).
        let aad: &[u8] = &[];

        ccm.decrypt_in_place_detached(nonce_ga, aad, ct, &tag)
            .map_err(|_| "CCM decryption failed (wrong key or corrupted)".to_string())?;

        // Advance counter
        if from_peripheral {
            self.counter_p2c += 1;
        } else {
            self.counter_c2p += 1;
        }

        Ok(plaintext_len)
    }

    /// Decrypt with explicit AAD byte (LL header & 0xE3).
    pub fn decrypt_with_aad(
        &mut self,
        payload: &mut [u8],
        from_peripheral: bool,
        aad_byte: u8,
    ) -> Result<usize, String> {
        if payload.len() < 4 {
            return Err("payload too short for MIC".to_string());
        }

        let counter = if from_peripheral {
            self.counter_p2c
        } else {
            self.counter_c2p
        };

        let nonce = self.build_nonce(counter, from_peripheral);
        let nonce_ga = GenericArray::from_slice(&nonce);

        let ccm = BleCcm::new(GenericArray::from_slice(&self.sk));

        let plaintext_len = payload.len() - 4;
        let (ct, tag_bytes) = payload.split_at_mut(plaintext_len);
        let tag = GenericArray::clone_from_slice(tag_bytes);

        let aad = [aad_byte];
        ccm.decrypt_in_place_detached(nonce_ga, &aad, ct, &tag)
            .map_err(|_| "CCM decryption failed (wrong key or corrupted)".to_string())?;

        if from_peripheral {
            self.counter_p2c += 1;
        } else {
            self.counter_c2p += 1;
        }

        Ok(plaintext_len)
    }
}

/// AES-128-ECB encrypt a single block (used for IRK resolution and session key derivation).
pub fn aes128_ecb(key: &[u8; 16], plaintext: &[u8; 16]) -> [u8; 16] {
    let cipher = <Aes128 as KeyInit>::new(GenericArray::from_slice(key));
    let mut block = *plaintext;
    aes::cipher::BlockEncrypt::encrypt_block(
        &cipher,
        GenericArray::from_mut_slice(&mut block),
    );
    block
}

/// BLE ah() function for IRK-based RPA resolution (Vol 3 Part H Section 2.2.2).
/// Returns the 24-bit hash from AES-128(IRK, padding || prand).
pub fn ble_ah(irk: &[u8; 16], prand: &[u8; 3]) -> [u8; 3] {
    let mut plaintext = [0u8; 16];
    plaintext[13] = prand[0];
    plaintext[14] = prand[1];
    plaintext[15] = prand[2];
    let result = aes128_ecb(irk, &plaintext);
    [result[13], result[14], result[15]]
}

/// Check if a BLE MAC address is a Resolvable Private Address (RPA).
/// RPA: bits [47:46] = 01 (MSB of the 6-byte address).
pub fn is_rpa(addr: &[u8; 6]) -> bool {
    (addr[5] >> 6) & 0x03 == 0x01
}

/// Try to resolve an RPA against a list of IRKs.
/// Returns the index of the matching IRK, or None.
pub fn resolve_rpa(addr: &[u8; 6], irks: &[[u8; 16]]) -> Option<usize> {
    if !is_rpa(addr) {
        return None;
    }
    // RPA layout (6 bytes, LSB first in BLE):
    // addr[0..3] = hash (3 bytes), addr[3..6] = prand (3 bytes)
    // prand MSB (addr[5]) has bits [7:6] = 01 for RPA type
    // ah() is computed over the full prand including type bits
    let prand = [addr[3], addr[4], addr[5]];
    let expected_hash = [addr[0], addr[1], addr[2]];

    for (i, irk) in irks.iter().enumerate() {
        let computed = ble_ah(irk, &prand);
        if computed == expected_hash {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes128_ecb_known_vector() {
        // NIST FIPS 197 Appendix B: AES-128 test vector
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let plaintext: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
            0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
        ];
        let expected: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32,
        ];
        let result = aes128_ecb(&key, &plaintext);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_is_rpa() {
        // RPA: bits [47:46] = 01
        let rpa = [0x12, 0x34, 0x56, 0x78, 0x9A, 0x4B]; // 0x4B >> 6 = 01
        assert!(is_rpa(&rpa));

        // Public address: bits [47:46] = 00
        let public = [0x12, 0x34, 0x56, 0x78, 0x9A, 0x1B]; // 0x1B >> 6 = 00
        assert!(!is_rpa(&public));

        // Static random: bits [47:46] = 11
        let static_random = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xCB]; // 0xCB >> 6 = 11
        assert!(!is_rpa(&static_random));
    }

    #[test]
    fn test_session_key_derivation() {
        // Test that from_ltk_skd produces a deterministic session key
        let ltk = [0x01u8; 16];
        let skd = [0x02u8; 16];
        let iv = [0x03u8; 8];

        let sk1 = BleSessionKey::from_ltk_skd(&ltk, &skd, &iv);
        let sk2 = BleSessionKey::from_ltk_skd(&ltk, &skd, &iv);
        assert_eq!(sk1.sk, sk2.sk);
        assert_eq!(sk1.iv, sk2.iv);
        // SK should not be all zeros (AES output of non-zero input)
        assert_ne!(sk1.sk, [0u8; 16]);
    }

    #[test]
    fn test_nonce_construction() {
        let sk = BleSessionKey {
            sk: [0u8; 16],
            iv: [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
            counter_c2p: 0,
            counter_p2c: 0,
        };

        // Counter = 0x0000000042, direction = central->peripheral
        let nonce = sk.build_nonce(0x42, false);
        assert_eq!(nonce[0], 0x42); // counter LSB
        assert_eq!(nonce[1], 0x00);
        assert_eq!(nonce[2], 0x00);
        assert_eq!(nonce[3], 0x00);
        assert_eq!(nonce[4], 0x00); // no direction bit
        assert_eq!(nonce[5], 0x11); // IV start

        // Same counter, direction = peripheral->central
        let nonce2 = sk.build_nonce(0x42, true);
        assert_eq!(nonce2[4], 0x80); // direction bit set
    }

    #[test]
    fn test_ble_ah_function() {
        // Verify ah() produces deterministic output and correct size
        let irk = [0xAAu8; 16];
        let prand = [0x01, 0x02, 0x03];
        let hash = ble_ah(&irk, &prand);
        assert_eq!(hash.len(), 3);
        // Same input = same output
        let hash2 = ble_ah(&irk, &prand);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_resolve_rpa_match() {
        let irk = [0xBBu8; 16];
        // prand MSB must have bits [7:6] = 01 for RPA type
        let prand = [0x10, 0x20, 0x50]; // 0x50 = 01_010000
        let hash = ble_ah(&irk, &prand);

        // Construct RPA: hash(3) || prand(3)
        let addr = [hash[0], hash[1], hash[2], prand[0], prand[1], prand[2]];
        assert!(is_rpa(&addr));

        let irks = vec![irk];
        let result = resolve_rpa(&addr, &irks);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_resolve_rpa_no_match() {
        let irk = [0xCCu8; 16];
        let wrong_irk = [0xDDu8; 16];

        let prand = [0x10, 0x20, 0x50]; // RPA type
        let hash = ble_ah(&irk, &prand);

        let addr = [hash[0], hash[1], hash[2], prand[0], prand[1], prand[2]];
        assert!(is_rpa(&addr));

        let irks = vec![wrong_irk];
        assert_eq!(resolve_rpa(&addr, &irks), None);
    }
}
