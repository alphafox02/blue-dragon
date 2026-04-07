// Copyright 2026 CEMAXECUTER LLC
//
// BLE vulnerability scanner: passively detects security weaknesses from
// observed advertising and data channel traffic.
//
// Checks:
// - Weak pairing modes (via SMP analysis -- see smp.rs)
// - Unencrypted data channel traffic
// - Legacy pairing (brutable TK)
// - Known vulnerable device fingerprints (name, manufacturer, service UUID)
// - Missing security features (no MITM protection, no bonding)
// - Exposed sensitive GATT characteristics

/// Vulnerability severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Informational: non-security observation
    Info,
    /// Low: minor security concern
    Low,
    /// Medium: exploitable under certain conditions
    Medium,
    /// High: directly exploitable vulnerability
    High,
    /// Critical: no authentication/encryption, data fully exposed
    Critical,
}

/// A detected vulnerability or security observation.
#[derive(Debug, Clone)]
pub struct VulnAlert {
    pub severity: Severity,
    pub category: &'static str,
    pub description: String,
    pub mac: Option<String>,
    pub device_name: Option<String>,
    pub cve: Option<&'static str>,
}

impl VulnAlert {
    pub fn format_oneline(&self) -> String {
        let sev = match self.severity {
            Severity::Info => "INFO",
            Severity::Low => "LOW",
            Severity::Medium => "MED",
            Severity::High => "HIGH",
            Severity::Critical => "CRIT",
        };
        let mac_str = self.mac.as_deref().unwrap_or("?");
        let name_str = self.device_name.as_deref().unwrap_or("");
        if name_str.is_empty() {
            format!("[{}] {} {}: {}", sev, mac_str, self.category, self.description)
        } else {
            format!("[{}] {} ({}) {}: {}", sev, mac_str, name_str, self.category, self.description)
        }
    }
}

/// Known vulnerable device patterns.
struct DevicePattern {
    name_contains: Option<&'static str>,
    /// Manufacturer company ID (from BLE AD type 0xFF)
    manufacturer_id: Option<u16>,
    /// Service UUID substring
    service_uuid_contains: Option<&'static str>,
    severity: Severity,
    category: &'static str,
    description: &'static str,
    cve: Option<&'static str>,
}

/// Database of known vulnerable BLE device patterns.
/// Based on public CVE disclosures and security research.
const KNOWN_VULNS: &[DevicePattern] = &[
    // Smart locks with known bypass vulnerabilities
    DevicePattern {
        name_contains: Some("Aug"),
        manufacturer_id: None,
        service_uuid_contains: None,
        severity: Severity::Medium,
        category: "smart-lock",
        description: "August smart lock: historically vulnerable to replay attacks",
        cve: Some("CVE-2019-17098"),
    },
    DevicePattern {
        name_contains: Some("Tapplock"),
        manufacturer_id: None,
        service_uuid_contains: None,
        severity: Severity::High,
        category: "smart-lock",
        description: "Tapplock: BLE MAC used as unlock key, trivially cloneable",
        cve: None,
    },
    // Fitness trackers with weak/no authentication
    DevicePattern {
        name_contains: Some("MI Band"),
        manufacturer_id: None,
        service_uuid_contains: None,
        severity: Severity::Low,
        category: "fitness",
        description: "Xiaomi Mi Band: unencrypted BLE, activity data exposed",
        cve: None,
    },
    // Medical devices
    DevicePattern {
        name_contains: None,
        manufacturer_id: None,
        service_uuid_contains: Some("1808"), // Glucose
        severity: Severity::High,
        category: "medical",
        description: "BLE glucose monitor: health data potentially unencrypted",
        cve: None,
    },
    DevicePattern {
        name_contains: None,
        manufacturer_id: None,
        service_uuid_contains: Some("180d"), // Heart Rate
        severity: Severity::Medium,
        category: "medical",
        description: "BLE heart rate monitor: biometric data over BLE",
        cve: None,
    },
    DevicePattern {
        name_contains: None,
        manufacturer_id: None,
        service_uuid_contains: Some("1810"), // Blood Pressure
        severity: Severity::High,
        category: "medical",
        description: "BLE blood pressure monitor: health data potentially unencrypted",
        cve: None,
    },
    // Tile/AirTag trackers (privacy concern)
    DevicePattern {
        name_contains: Some("Tile"),
        manufacturer_id: None,
        service_uuid_contains: None,
        severity: Severity::Info,
        category: "tracker",
        description: "Tile tracker: BLE beacon, potential stalking tool",
        cve: None,
    },
    // SweynTooth family
    DevicePattern {
        name_contains: None,
        manufacturer_id: Some(0x000D), // Texas Instruments
        service_uuid_contains: None,
        severity: Severity::Medium,
        category: "sweyntooth",
        description: "TI BLE chip: check firmware for SweynTooth vulnerabilities",
        cve: Some("CVE-2019-19195"),
    },
    DevicePattern {
        name_contains: None,
        manufacturer_id: Some(0x0059), // Nordic Semiconductor
        service_uuid_contains: None,
        severity: Severity::Low,
        category: "sweyntooth",
        description: "Nordic nRF chip: verify firmware patched for SweynTooth",
        cve: Some("CVE-2019-17519"),
    },
];

/// Sensitive GATT service UUIDs that indicate security-relevant functionality.
const SENSITIVE_SERVICES: &[(&str, &str, Severity)] = &[
    ("1803", "Link Loss", Severity::Info),
    ("1802", "Immediate Alert", Severity::Info),
    ("180f", "Battery Service", Severity::Low),
    ("1812", "HID over GATT", Severity::High),
    ("fef5", "Dialog Semiconductor OTA", Severity::Medium),
    ("fe59", "Nordic DFU", Severity::Medium),
    ("1530", "Nordic Legacy DFU", Severity::Medium),
];

/// Check a device against the known vulnerability database.
/// `name`: device name from advertisement or GATT
/// `manufacturer_id`: company ID from manufacturer-specific AD data
/// `service_uuids`: list of service UUID strings (short or full form)
pub fn check_device(
    mac: &str,
    name: Option<&str>,
    manufacturer_id: Option<u16>,
    service_uuids: &[String],
) -> Vec<VulnAlert> {
    let mut alerts = Vec::new();

    for pattern in KNOWN_VULNS {
        let mut matched = false;

        if let Some(name_pat) = pattern.name_contains {
            if let Some(dev_name) = name {
                if dev_name.to_lowercase().contains(&name_pat.to_lowercase()) {
                    matched = true;
                }
            }
        }

        if let Some(mfr_id) = pattern.manufacturer_id {
            if manufacturer_id == Some(mfr_id) {
                matched = true;
            }
        }

        if let Some(svc_pat) = pattern.service_uuid_contains {
            for uuid in service_uuids {
                if uuid.to_lowercase().contains(svc_pat) {
                    matched = true;
                    break;
                }
            }
        }

        if matched {
            alerts.push(VulnAlert {
                severity: pattern.severity,
                category: pattern.category,
                description: pattern.description.to_string(),
                mac: Some(mac.to_string()),
                device_name: name.map(|s| s.to_string()),
                cve: pattern.cve,
            });
        }
    }

    // Check for sensitive GATT services
    for &(short_uuid, svc_name, severity) in SENSITIVE_SERVICES {
        for uuid in service_uuids {
            if uuid.to_lowercase().contains(short_uuid) {
                alerts.push(VulnAlert {
                    severity,
                    category: "sensitive-service",
                    description: format!("{} service exposed over BLE", svc_name),
                    mac: Some(mac.to_string()),
                    device_name: name.map(|s| s.to_string()),
                    cve: None,
                });
                break;
            }
        }
    }

    alerts
}

/// Check advertising PDU characteristics for security issues.
pub fn check_advertising(
    mac: &str,
    pdu_type: u8,
    name: Option<&str>,
) -> Vec<VulnAlert> {
    let mut alerts = Vec::new();

    // ADV_IND (0) = connectable undirected: any device can connect
    if pdu_type == 0 {
        alerts.push(VulnAlert {
            severity: Severity::Info,
            category: "connectable",
            description: "ADV_IND: device accepts connections from any central".to_string(),
            mac: Some(mac.to_string()),
            device_name: name.map(|s| s.to_string()),
            cve: None,
        });
    }

    alerts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_device_name_match() {
        let alerts = check_device(
            "aa:bb:cc:dd:ee:ff",
            Some("Tapplock Pro"),
            None,
            &[],
        );
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].severity, Severity::High);
        assert_eq!(alerts[0].category, "smart-lock");
    }

    #[test]
    fn test_check_device_service_uuid() {
        let uuids = vec!["00001808-0000-1000-8000-00805f9b34fb".to_string()];
        let alerts = check_device(
            "11:22:33:44:55:66",
            None,
            None,
            &uuids,
        );
        assert!(!alerts.is_empty());
        let glucose = alerts.iter().find(|a| a.category == "medical");
        assert!(glucose.is_some());
        assert_eq!(glucose.unwrap().severity, Severity::High);
    }

    #[test]
    fn test_check_device_manufacturer_id() {
        let alerts = check_device(
            "aa:bb:cc:dd:ee:ff",
            None,
            Some(0x0059), // Nordic
            &[],
        );
        assert!(!alerts.is_empty());
        let nordic = alerts.iter().find(|a| a.category == "sweyntooth");
        assert!(nordic.is_some());
    }

    #[test]
    fn test_check_device_sensitive_service() {
        let uuids = vec!["00001812-0000-1000-8000-00805f9b34fb".to_string()]; // HID
        let alerts = check_device(
            "aa:bb:cc:dd:ee:ff",
            Some("BLE Keyboard"),
            None,
            &uuids,
        );
        let hid = alerts.iter().find(|a| a.category == "sensitive-service");
        assert!(hid.is_some());
        assert_eq!(hid.unwrap().severity, Severity::High);
    }

    #[test]
    fn test_check_device_no_match() {
        let alerts = check_device(
            "aa:bb:cc:dd:ee:ff",
            Some("Generic Widget"),
            None,
            &[],
        );
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_format_oneline() {
        let alert = VulnAlert {
            severity: Severity::High,
            category: "smart-lock",
            description: "test vulnerability".to_string(),
            mac: Some("aa:bb:cc:dd:ee:ff".to_string()),
            device_name: Some("TestLock".to_string()),
            cve: Some("CVE-2024-1234"),
        };
        let line = alert.format_oneline();
        assert!(line.contains("[HIGH]"));
        assert!(line.contains("TestLock"));
        assert!(line.contains("test vulnerability"));
    }
}
