// Copyright 2026 CEMAXECUTER LLC
//
// BLE Security Manager Protocol (SMP) parser.
//
// Parses SMP exchanges from data channel L2CAP PDUs to detect pairing
// security level, extract observable key material, and flag weak pairing.
//
// Reference: BT Core Spec Vol 3 Part H (Security Manager Specification)

// Note: Serialize/Deserialize will be added when serde is added to protocol crate deps.
// For now, types are used directly in the pipeline without serialization.

// SMP command opcodes (Vol 3 Part H, Section 3.3)
const SMP_PAIRING_REQUEST: u8 = 0x01;
const SMP_PAIRING_RESPONSE: u8 = 0x02;
const SMP_PAIRING_CONFIRM: u8 = 0x03;
const SMP_PAIRING_RANDOM: u8 = 0x04;
const SMP_PAIRING_FAILED: u8 = 0x05;
const SMP_ENCRYPTION_INFORMATION: u8 = 0x06;
const SMP_CENTRAL_IDENTIFICATION: u8 = 0x07;
const SMP_IDENTITY_INFORMATION: u8 = 0x08;
const SMP_IDENTITY_ADDRESS_INFORMATION: u8 = 0x09;
const SMP_SIGNING_INFORMATION: u8 = 0x0A;
const SMP_SECURITY_REQUEST: u8 = 0x0B;
const SMP_PAIRING_PUBLIC_KEY: u8 = 0x0C;
const SMP_PAIRING_DHKEY_CHECK: u8 = 0x0D;

// L2CAP CID for SMP (fixed channel)
const L2CAP_CID_SMP: u16 = 0x0006;

// IO Capability values (Vol 3 Part H, Section 3.5.1)
const IO_DISPLAY_ONLY: u8 = 0x00;
const IO_DISPLAY_YESNO: u8 = 0x01;
const IO_KEYBOARD_ONLY: u8 = 0x02;
const IO_NO_INPUT_NO_OUTPUT: u8 = 0x03;
const IO_KEYBOARD_DISPLAY: u8 = 0x04;

// AuthReq flag bits (Vol 3 Part H, Section 3.5.1)
const AUTH_BONDING: u8 = 0x01;
const AUTH_MITM: u8 = 0x04;
const AUTH_SC: u8 = 0x08;

/// Pairing method determined from IO capabilities and flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairingMethod {
    /// No MITM protection -- passively eavesdroppable
    JustWorks,
    /// 6-digit passkey entry (one side displays, other enters)
    PasskeyEntry,
    /// Numeric comparison (both sides display, user confirms match)
    NumericComparison,
    /// Out-of-band data exchange
    OutOfBand,
    /// Unknown / not yet determined
    Unknown,
}

/// Security level of the pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// LE Legacy Pairing (no ECDH)
    Legacy,
    /// LE Secure Connections (ECDH P-256)
    SecureConnections,
    /// Not yet determined
    Unknown,
}

/// Pairing features from Pairing Request/Response PDUs.
#[derive(Debug, Clone)]
pub struct PairingFeatures {
    pub io_capability: u8,
    pub oob_data_flag: u8,
    pub auth_req: u8,
    pub max_key_size: u8,
    pub init_key_dist: u8,
    pub resp_key_dist: u8,
}

impl PairingFeatures {
    fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 6 {
            return None;
        }
        Some(Self {
            io_capability: data[0],
            oob_data_flag: data[1],
            auth_req: data[2],
            max_key_size: data[3],
            init_key_dist: data[4],
            resp_key_dist: data[5],
        })
    }

    pub fn has_mitm(&self) -> bool {
        self.auth_req & AUTH_MITM != 0
    }

    pub fn has_secure_connections(&self) -> bool {
        self.auth_req & AUTH_SC != 0
    }

    pub fn has_bonding(&self) -> bool {
        self.auth_req & AUTH_BONDING != 0
    }

    pub fn io_capability_str(&self) -> &'static str {
        match self.io_capability {
            IO_DISPLAY_ONLY => "DisplayOnly",
            IO_DISPLAY_YESNO => "DisplayYesNo",
            IO_KEYBOARD_ONLY => "KeyboardOnly",
            IO_NO_INPUT_NO_OUTPUT => "NoInputNoOutput",
            IO_KEYBOARD_DISPLAY => "KeyboardDisplay",
            _ => "Unknown",
        }
    }
}

/// SMP pairing event emitted during parsing.
#[derive(Debug, Clone)]
pub enum SmpEvent {
    /// Pairing initiated (SecurityRequest or PairingRequest seen)
    PairingStarted {
        aa: u32,
    },
    /// Pairing features exchanged (both Request and Response seen)
    FeaturesExchanged {
        aa: u32,
        initiator: PairingFeatures,
        responder: PairingFeatures,
        method: PairingMethod,
        security: SecurityLevel,
    },
    /// Pairing confirm value observed (Legacy pairing phase 2)
    PairingConfirm {
        aa: u32,
        from_initiator: bool,
        confirm: [u8; 16],
    },
    /// Pairing random value observed (Legacy pairing phase 2)
    PairingRandom {
        aa: u32,
        from_initiator: bool,
        random: [u8; 16],
    },
    /// SC public key exchanged
    PublicKey {
        aa: u32,
        from_initiator: bool,
    },
    /// LTK distributed (EncryptionInformation)
    LtkDistributed {
        aa: u32,
        ltk: [u8; 16],
    },
    /// IRK distributed (IdentityInformation)
    IrkDistributed {
        aa: u32,
        irk: [u8; 16],
    },
    /// Identity address distributed
    IdentityAddress {
        aa: u32,
        addr_type: u8,
        addr: [u8; 6],
    },
    /// CSRK distributed (SigningInformation)
    CsrkDistributed {
        aa: u32,
        csrk: [u8; 16],
    },
    /// Pairing failed
    PairingFailed {
        aa: u32,
        reason: u8,
    },
    /// Weak pairing detected (security alert)
    WeakPairing {
        aa: u32,
        reason: String,
    },
}

/// Per-connection SMP state tracker.
#[derive(Debug, Clone)]
struct SmpState {
    aa: u32,
    init_features: Option<PairingFeatures>,
    resp_features: Option<PairingFeatures>,
    /// Track which side sent confirm/random (for key derivation)
    init_confirm: Option<[u8; 16]>,
    resp_confirm: Option<[u8; 16]>,
    init_random: Option<[u8; 16]>,
    resp_random: Option<[u8; 16]>,
    method: PairingMethod,
    security: SecurityLevel,
    /// Direction tracking: first SMP PDU sender is initiator
    initiator_is_central: Option<bool>,
}

impl SmpState {
    fn new(aa: u32) -> Self {
        Self {
            aa,
            init_features: None,
            resp_features: None,
            init_confirm: None,
            resp_confirm: None,
            init_random: None,
            resp_random: None,
            method: PairingMethod::Unknown,
            security: SecurityLevel::Unknown,
            initiator_is_central: None,
        }
    }
}

/// SMP protocol parser. Tracks pairing state per connection.
pub struct SmpParser {
    /// Active pairing sessions keyed by connection access address
    sessions: std::collections::HashMap<u32, SmpState>,
}

impl SmpParser {
    pub fn new() -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
        }
    }

    /// Parse a data channel PDU for SMP content.
    /// `aa` is the connection access address.
    /// `l2cap_payload` is the L2CAP payload (after the 4-byte L2CAP header).
    /// `from_central` indicates direction (true = central->peripheral).
    /// Returns any SMP events detected.
    pub fn parse_l2cap(
        &mut self,
        aa: u32,
        l2cap_cid: u16,
        l2cap_payload: &[u8],
        from_central: bool,
    ) -> Vec<SmpEvent> {
        if l2cap_cid != L2CAP_CID_SMP || l2cap_payload.is_empty() {
            return Vec::new();
        }

        let opcode = l2cap_payload[0];
        let smp_data = &l2cap_payload[1..];

        let mut events = Vec::new();

        match opcode {
            SMP_PAIRING_REQUEST => {
                if let Some(features) = PairingFeatures::from_bytes(smp_data) {
                    let state = self.sessions.entry(aa).or_insert_with(|| SmpState::new(aa));
                    state.init_features = Some(features.clone());
                    state.initiator_is_central = Some(from_central);
                    events.push(SmpEvent::PairingStarted { aa });
                    self.check_features(aa, &mut events);
                }
            }
            SMP_PAIRING_RESPONSE => {
                if let Some(features) = PairingFeatures::from_bytes(smp_data) {
                    let state = self.sessions.entry(aa).or_insert_with(|| SmpState::new(aa));
                    state.resp_features = Some(features.clone());
                    self.check_features(aa, &mut events);
                }
            }
            SMP_PAIRING_CONFIRM => {
                if smp_data.len() >= 16 {
                    let mut confirm = [0u8; 16];
                    confirm.copy_from_slice(&smp_data[..16]);

                    let state = self.sessions.entry(aa).or_insert_with(|| SmpState::new(aa));
                    let is_initiator = state.initiator_is_central == Some(from_central);

                    if is_initiator {
                        state.init_confirm = Some(confirm);
                    } else {
                        state.resp_confirm = Some(confirm);
                    }

                    events.push(SmpEvent::PairingConfirm {
                        aa,
                        from_initiator: is_initiator,
                        confirm,
                    });
                }
            }
            SMP_PAIRING_RANDOM => {
                if smp_data.len() >= 16 {
                    let mut random = [0u8; 16];
                    random.copy_from_slice(&smp_data[..16]);

                    let state = self.sessions.entry(aa).or_insert_with(|| SmpState::new(aa));
                    let is_initiator = state.initiator_is_central == Some(from_central);

                    if is_initiator {
                        state.init_random = Some(random);
                    } else {
                        state.resp_random = Some(random);
                    }

                    events.push(SmpEvent::PairingRandom {
                        aa,
                        from_initiator: is_initiator,
                        random,
                    });
                }
            }
            SMP_PAIRING_FAILED => {
                let reason = if smp_data.is_empty() { 0 } else { smp_data[0] };
                events.push(SmpEvent::PairingFailed { aa, reason });
                self.sessions.remove(&aa);
            }
            SMP_ENCRYPTION_INFORMATION => {
                if smp_data.len() >= 16 {
                    let mut ltk = [0u8; 16];
                    ltk.copy_from_slice(&smp_data[..16]);
                    events.push(SmpEvent::LtkDistributed { aa, ltk });
                }
            }
            SMP_CENTRAL_IDENTIFICATION => {
                // EDIV(2) + Rand(8) -- used with Legacy LTK, logged but not stored separately
            }
            SMP_IDENTITY_INFORMATION => {
                if smp_data.len() >= 16 {
                    let mut irk = [0u8; 16];
                    irk.copy_from_slice(&smp_data[..16]);
                    events.push(SmpEvent::IrkDistributed { aa, irk });
                }
            }
            SMP_IDENTITY_ADDRESS_INFORMATION => {
                if smp_data.len() >= 7 {
                    let addr_type = smp_data[0];
                    let mut addr = [0u8; 6];
                    addr.copy_from_slice(&smp_data[1..7]);
                    events.push(SmpEvent::IdentityAddress { aa, addr_type, addr });
                }
            }
            SMP_SIGNING_INFORMATION => {
                if smp_data.len() >= 16 {
                    let mut csrk = [0u8; 16];
                    csrk.copy_from_slice(&smp_data[..16]);
                    events.push(SmpEvent::CsrkDistributed { aa, csrk });
                }
            }
            SMP_PAIRING_PUBLIC_KEY => {
                let state = self.sessions.entry(aa).or_insert_with(|| SmpState::new(aa));
                let is_initiator = state.initiator_is_central == Some(from_central);
                events.push(SmpEvent::PublicKey { aa, from_initiator: is_initiator });
            }
            SMP_PAIRING_DHKEY_CHECK => {
                // DHKey check -- pairing is completing (SC mode)
            }
            SMP_SECURITY_REQUEST => {
                events.push(SmpEvent::PairingStarted { aa });
            }
            _ => {}
        }

        events
    }

    /// After both Pairing Request and Response are seen, determine the method and security level.
    fn check_features(&mut self, aa: u32, events: &mut Vec<SmpEvent>) {
        let state = match self.sessions.get_mut(&aa) {
            Some(s) => s,
            None => return,
        };

        let (init, resp) = match (&state.init_features, &state.resp_features) {
            (Some(i), Some(r)) => (i.clone(), r.clone()),
            _ => return,
        };

        // Determine security level
        let sc = init.has_secure_connections() && resp.has_secure_connections();
        state.security = if sc {
            SecurityLevel::SecureConnections
        } else {
            SecurityLevel::Legacy
        };

        // Determine pairing method (Vol 3 Part H, Section 2.3.5.1)
        let mitm = init.has_mitm() || resp.has_mitm();
        let oob = init.oob_data_flag != 0 || resp.oob_data_flag != 0;

        state.method = if oob {
            PairingMethod::OutOfBand
        } else if !mitm {
            PairingMethod::JustWorks
        } else if sc {
            // Secure Connections IO mapping (Table 2.8)
            sc_pairing_method(init.io_capability, resp.io_capability)
        } else {
            // Legacy IO mapping (Table 2.7)
            legacy_pairing_method(init.io_capability, resp.io_capability)
        };

        events.push(SmpEvent::FeaturesExchanged {
            aa,
            initiator: init.clone(),
            responder: resp.clone(),
            method: state.method,
            security: state.security,
        });

        // Flag weak pairing
        if state.method == PairingMethod::JustWorks {
            events.push(SmpEvent::WeakPairing {
                aa,
                reason: "JustWorks pairing -- no MITM protection, passively eavesdroppable"
                    .to_string(),
            });
        }

        if state.security == SecurityLevel::Legacy && state.method == PairingMethod::JustWorks {
            events.push(SmpEvent::WeakPairing {
                aa,
                reason: "Legacy JustWorks -- TK=0, LTK derivable from captured Confirm/Random"
                    .to_string(),
            });
        }

        if state.security == SecurityLevel::Legacy {
            events.push(SmpEvent::WeakPairing {
                aa,
                reason: format!(
                    "Legacy pairing ({:?}) -- vulnerable to passive eavesdropping of TK",
                    state.method
                ),
            });
        }
    }

    /// Get active session for a connection (for external inspection).
    pub fn get_session(&self, aa: u32) -> Option<(PairingMethod, SecurityLevel)> {
        self.sessions.get(&aa).map(|s| (s.method, s.security))
    }

    /// Clean up completed/stale sessions.
    pub fn remove_session(&mut self, aa: u32) {
        self.sessions.remove(&aa);
    }
}

/// Legacy pairing method from IO capabilities (BT Core Spec Table 2.7).
fn legacy_pairing_method(init_io: u8, resp_io: u8) -> PairingMethod {
    match (init_io, resp_io) {
        (IO_NO_INPUT_NO_OUTPUT, _) | (_, IO_NO_INPUT_NO_OUTPUT) => PairingMethod::JustWorks,
        (IO_DISPLAY_ONLY, IO_DISPLAY_ONLY) => PairingMethod::JustWorks,
        (IO_DISPLAY_ONLY, IO_DISPLAY_YESNO) => PairingMethod::JustWorks,
        (IO_DISPLAY_YESNO, IO_DISPLAY_ONLY) => PairingMethod::JustWorks,
        (IO_DISPLAY_YESNO, IO_DISPLAY_YESNO) => PairingMethod::JustWorks,
        (IO_DISPLAY_ONLY, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_DISPLAY_ONLY, IO_KEYBOARD_DISPLAY) => PairingMethod::PasskeyEntry,
        (IO_DISPLAY_YESNO, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_DISPLAY_YESNO, IO_KEYBOARD_DISPLAY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_ONLY, IO_DISPLAY_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_ONLY, IO_DISPLAY_YESNO) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_ONLY, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_ONLY, IO_KEYBOARD_DISPLAY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_DISPLAY, IO_DISPLAY_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_DISPLAY, IO_DISPLAY_YESNO) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_DISPLAY, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_DISPLAY, IO_KEYBOARD_DISPLAY) => PairingMethod::PasskeyEntry,
        _ => PairingMethod::JustWorks,
    }
}

/// Secure Connections pairing method from IO capabilities (BT Core Spec Table 2.8).
fn sc_pairing_method(init_io: u8, resp_io: u8) -> PairingMethod {
    match (init_io, resp_io) {
        (IO_NO_INPUT_NO_OUTPUT, _) | (_, IO_NO_INPUT_NO_OUTPUT) => PairingMethod::JustWorks,
        (IO_DISPLAY_ONLY, IO_DISPLAY_ONLY) => PairingMethod::JustWorks,
        (IO_DISPLAY_ONLY, IO_DISPLAY_YESNO) => PairingMethod::JustWorks,
        (IO_DISPLAY_YESNO, IO_DISPLAY_ONLY) => PairingMethod::JustWorks,
        (IO_DISPLAY_YESNO, IO_DISPLAY_YESNO) => PairingMethod::NumericComparison,
        (IO_DISPLAY_ONLY, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_DISPLAY_ONLY, IO_KEYBOARD_DISPLAY) => PairingMethod::PasskeyEntry,
        (IO_DISPLAY_YESNO, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_DISPLAY_YESNO, IO_KEYBOARD_DISPLAY) => PairingMethod::NumericComparison,
        (IO_KEYBOARD_ONLY, IO_DISPLAY_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_ONLY, IO_DISPLAY_YESNO) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_ONLY, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_ONLY, IO_KEYBOARD_DISPLAY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_DISPLAY, IO_DISPLAY_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_DISPLAY, IO_DISPLAY_YESNO) => PairingMethod::NumericComparison,
        (IO_KEYBOARD_DISPLAY, IO_KEYBOARD_ONLY) => PairingMethod::PasskeyEntry,
        (IO_KEYBOARD_DISPLAY, IO_KEYBOARD_DISPLAY) => PairingMethod::NumericComparison,
        _ => PairingMethod::JustWorks,
    }
}

/// Extract L2CAP CID and payload from a BLE data channel PDU.
/// Data channel PDU: LLID(2 bits in header) + Length + L2CAP(Length+CID) + ...
/// Returns (l2cap_cid, l2cap_payload) if this is an L2CAP start fragment (LLID=2).
pub fn extract_l2cap(pdu_data: &[u8]) -> Option<(u16, &[u8])> {
    // BLE data PDU: AA(4) + Header(2) + Payload(0-251) + CRC(3)
    // Header byte 0: LLID(bits 1:0), NESN(bit 2), SN(bit 3), MD(bit 4)
    // LLID=1: continuation, LLID=2: start of L2CAP, LLID=3: LL control
    if pdu_data.len() < 4 + 2 + 4 {
        return None;
    }

    let header = pdu_data[4]; // first header byte after AA
    let llid = header & 0x03;

    if llid != 2 {
        return None; // not a start-of-L2CAP fragment
    }

    let pdu_len = pdu_data[5] as usize;
    if pdu_len < 4 {
        return None; // L2CAP header is 4 bytes minimum (length + CID)
    }

    let l2cap_start = 6; // after AA(4) + Header(2)
    if l2cap_start + 4 > pdu_data.len() {
        return None;
    }

    let l2cap_len = u16::from_le_bytes([pdu_data[l2cap_start], pdu_data[l2cap_start + 1]]) as usize;
    let l2cap_cid = u16::from_le_bytes([pdu_data[l2cap_start + 2], pdu_data[l2cap_start + 3]]);

    let payload_start = l2cap_start + 4;
    let payload_end = (payload_start + l2cap_len).min(pdu_data.len() - 3); // exclude CRC

    if payload_start >= payload_end {
        return None;
    }

    Some((l2cap_cid, &pdu_data[payload_start..payload_end]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairing_features_parse() {
        // IO=NoInputNoOutput, OOB=0, AuthReq=0x01 (bonding), MaxKey=16, InitDist=0, RespDist=0
        let data = [IO_NO_INPUT_NO_OUTPUT, 0, AUTH_BONDING, 16, 0, 0];
        let f = PairingFeatures::from_bytes(&data).unwrap();
        assert_eq!(f.io_capability, IO_NO_INPUT_NO_OUTPUT);
        assert!(f.has_bonding());
        assert!(!f.has_mitm());
        assert!(!f.has_secure_connections());
    }

    #[test]
    fn test_legacy_justworks() {
        assert_eq!(
            legacy_pairing_method(IO_NO_INPUT_NO_OUTPUT, IO_NO_INPUT_NO_OUTPUT),
            PairingMethod::JustWorks
        );
        assert_eq!(
            legacy_pairing_method(IO_DISPLAY_ONLY, IO_DISPLAY_ONLY),
            PairingMethod::JustWorks
        );
    }

    #[test]
    fn test_legacy_passkey() {
        assert_eq!(
            legacy_pairing_method(IO_DISPLAY_ONLY, IO_KEYBOARD_ONLY),
            PairingMethod::PasskeyEntry
        );
        assert_eq!(
            legacy_pairing_method(IO_KEYBOARD_DISPLAY, IO_KEYBOARD_DISPLAY),
            PairingMethod::PasskeyEntry
        );
    }

    #[test]
    fn test_sc_numeric_comparison() {
        assert_eq!(
            sc_pairing_method(IO_DISPLAY_YESNO, IO_DISPLAY_YESNO),
            PairingMethod::NumericComparison
        );
        assert_eq!(
            sc_pairing_method(IO_KEYBOARD_DISPLAY, IO_DISPLAY_YESNO),
            PairingMethod::NumericComparison
        );
    }

    #[test]
    fn test_smp_parser_pairing_exchange() {
        let mut parser = SmpParser::new();
        let aa = 0x12345678u32;

        // Pairing Request: IO=DisplayYesNo, OOB=0, AuthReq=SC+MITM+Bonding, MaxKey=16
        let req_data = [
            SMP_PAIRING_REQUEST,
            IO_DISPLAY_YESNO, 0, AUTH_SC | AUTH_MITM | AUTH_BONDING, 16, 0x07, 0x07,
        ];
        let events = parser.parse_l2cap(aa, L2CAP_CID_SMP, &req_data, true);
        assert_eq!(events.len(), 1); // PairingStarted
        assert!(matches!(events[0], SmpEvent::PairingStarted { .. }));

        // Pairing Response: IO=DisplayYesNo, OOB=0, AuthReq=SC+MITM+Bonding
        let resp_data = [
            SMP_PAIRING_RESPONSE,
            IO_DISPLAY_YESNO, 0, AUTH_SC | AUTH_MITM | AUTH_BONDING, 16, 0x07, 0x07,
        ];
        let events = parser.parse_l2cap(aa, L2CAP_CID_SMP, &resp_data, false);

        // Should get FeaturesExchanged with NumericComparison + SecureConnections
        let feat_event = events.iter().find(|e| matches!(e, SmpEvent::FeaturesExchanged { .. }));
        assert!(feat_event.is_some());
        if let SmpEvent::FeaturesExchanged { method, security, .. } = feat_event.unwrap() {
            assert_eq!(*method, PairingMethod::NumericComparison);
            assert_eq!(*security, SecurityLevel::SecureConnections);
        }
    }

    #[test]
    fn test_smp_parser_weak_legacy_justworks() {
        let mut parser = SmpParser::new();
        let aa = 0xAABBCCDD;

        // Legacy JustWorks: NoInputNoOutput on both sides, no SC, no MITM
        let req = [SMP_PAIRING_REQUEST, IO_NO_INPUT_NO_OUTPUT, 0, AUTH_BONDING, 16, 0, 0];
        parser.parse_l2cap(aa, L2CAP_CID_SMP, &req, true);

        let resp = [SMP_PAIRING_RESPONSE, IO_NO_INPUT_NO_OUTPUT, 0, AUTH_BONDING, 16, 0, 0];
        let events = parser.parse_l2cap(aa, L2CAP_CID_SMP, &resp, false);

        // Should flag as weak pairing (Legacy + JustWorks)
        let weak_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, SmpEvent::WeakPairing { .. }))
            .collect();
        assert!(weak_events.len() >= 2); // JustWorks warning + Legacy warning
    }

    #[test]
    fn test_smp_ltk_distribution() {
        let mut parser = SmpParser::new();
        let aa = 0x11223344;

        let mut ltk_pdu = vec![SMP_ENCRYPTION_INFORMATION];
        ltk_pdu.extend_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                     0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10]);

        let events = parser.parse_l2cap(aa, L2CAP_CID_SMP, &ltk_pdu, false);
        assert_eq!(events.len(), 1);
        if let SmpEvent::LtkDistributed { ltk, .. } = &events[0] {
            assert_eq!(ltk[0], 0x01);
            assert_eq!(ltk[15], 0x10);
        } else {
            panic!("expected LtkDistributed");
        }
    }

    #[test]
    fn test_smp_irk_distribution() {
        let mut parser = SmpParser::new();
        let aa = 0x55667788;

        let mut irk_pdu = vec![SMP_IDENTITY_INFORMATION];
        irk_pdu.extend_from_slice(&[0xAA; 16]);

        let events = parser.parse_l2cap(aa, L2CAP_CID_SMP, &irk_pdu, false);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], SmpEvent::IrkDistributed { .. }));
    }

    #[test]
    fn test_non_smp_cid_ignored() {
        let mut parser = SmpParser::new();
        let events = parser.parse_l2cap(0x12345678, 0x0004, &[0x01, 0x00], true);
        assert!(events.is_empty());
    }
}
