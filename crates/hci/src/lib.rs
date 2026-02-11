// Copyright 2025-2026 CEMAXECUTER LLC

use serde::{Deserialize, Serialize};
use std::time::Duration;

use bluer::{Address, Session};
use tokio::runtime::Runtime;
use tokio::time::timeout;

/// Timeout for a single GATT query (connect + discover + read + disconnect).
const GATT_TIMEOUT: Duration = Duration::from_secs(15);

/// Timeout for connecting to a device.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// A single GATT characteristic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GattCharacteristic {
    pub uuid: String,
    pub flags: Vec<String>,
    /// Value read from the characteristic (if readable), hex-encoded.
    pub value: Option<String>,
    /// UTF-8 interpretation of value (if valid UTF-8 and printable).
    pub value_str: Option<String>,
}

/// A single GATT service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GattService {
    pub uuid: String,
    pub primary: bool,
    pub characteristics: Vec<GattCharacteristic>,
}

/// Result of a GATT query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GattResult {
    pub mac: String,
    pub device_name: Option<String>,
    pub services: Vec<GattService>,
    pub error: Option<String>,
    pub timestamp: f64,
}

/// Check if a BLE PDU type indicates a connectable device.
/// ADV_IND (0) and ADV_DIRECT_IND (1) are connectable.
pub fn is_connectable(pdu_type: u8) -> bool {
    pdu_type == 0 || pdu_type == 1
}

/// HCI GATT prober. Owns a single-threaded tokio runtime for async BlueZ operations.
pub struct HciProber {
    rt: Runtime,
}

impl HciProber {
    /// Create a new HciProber with its own tokio runtime.
    pub fn new() -> Result<Self, String> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("failed to create tokio runtime: {}", e))?;
        Ok(Self { rt })
    }

    /// Check if an HCI adapter is available via BlueZ D-Bus.
    pub fn is_available(&self) -> bool {
        self.rt.block_on(async {
            let session = match Session::new().await {
                Ok(s) => s,
                Err(_) => return false,
            };
            let adapter = match session.default_adapter().await {
                Ok(a) => a,
                Err(_) => return false,
            };
            adapter.is_powered().await.unwrap_or(false)
        })
    }

    /// Query GATT services and characteristics for the given MAC address.
    /// This is a blocking call that runs async discovery on the internal runtime.
    pub fn query(&self, mac: &str) -> GattResult {
        let mac_owned = mac.to_string();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        self.rt.block_on(async {
            match timeout(GATT_TIMEOUT, discover_gatt(&mac_owned)).await {
                Ok(Ok(mut result)) => {
                    result.timestamp = now;
                    result
                }
                Ok(Err(e)) => GattResult {
                    mac: mac_owned,
                    device_name: None,
                    services: Vec::new(),
                    error: Some(e),
                    timestamp: now,
                },
                Err(_) => GattResult {
                    mac: mac_owned,
                    device_name: None,
                    services: Vec::new(),
                    error: Some("GATT query timed out".to_string()),
                    timestamp: now,
                },
            }
        })
    }
}

/// Convert CharacteristicFlags to a list of human-readable strings.
fn flags_to_strings(f: &bluer::gatt::CharacteristicFlags) -> Vec<String> {
    let mut v = Vec::new();
    if f.broadcast { v.push("broadcast".into()); }
    if f.read { v.push("read".into()); }
    if f.write_without_response { v.push("write-without-response".into()); }
    if f.write { v.push("write".into()); }
    if f.notify { v.push("notify".into()); }
    if f.indicate { v.push("indicate".into()); }
    if f.authenticated_signed_writes { v.push("authenticated-signed-writes".into()); }
    v
}

/// Core async GATT discovery logic.
async fn discover_gatt(mac: &str) -> Result<GattResult, String> {
    let addr: Address = mac.parse().map_err(|e| format!("invalid MAC '{}': {}", mac, e))?;

    let session = Session::new()
        .await
        .map_err(|e| format!("BlueZ session error: {}", e))?;
    let adapter = session
        .default_adapter()
        .await
        .map_err(|e| format!("no HCI adapter: {}", e))?;

    // Start discovery to ensure BlueZ knows about the device.
    // discover_devices() returns impl Stream; dropping it stops scanning.
    let disco = adapter
        .discover_devices()
        .await
        .map_err(|e| format!("discovery error: {}", e))?;

    // Wait up to 3 seconds for the target device to appear in BlueZ
    use futures::StreamExt;
    let device = {
        let mut found = adapter.device(addr).ok();
        if found.is_none() {
            let mut stream = Box::pin(disco);
            let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
            loop {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match timeout(remaining, stream.next()).await {
                    Ok(Some(bluer::AdapterEvent::DeviceAdded(a))) if a == addr => {
                        found = adapter.device(addr).ok();
                        break;
                    }
                    Ok(Some(_)) => continue,
                    _ => break,
                }
            }
        }
        found.ok_or_else(|| format!("device {} not found", mac))?
    };

    // Connect if not already connected
    let was_connected = device.is_connected().await.unwrap_or(false);
    if !was_connected {
        timeout(CONNECT_TIMEOUT, device.connect())
            .await
            .map_err(|_| format!("connect to {} timed out", mac))?
            .map_err(|e| format!("connect to {} failed: {}", mac, e))?;
    }

    // Wait for service resolution (BlueZ auto-discovers GATT after connect)
    let mut resolved = device.is_services_resolved().await.unwrap_or(false);
    if !resolved {
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            resolved = device.is_services_resolved().await.unwrap_or(false);
            if resolved {
                break;
            }
        }
    }

    let device_name = device.name().await.ok().flatten();

    // Enumerate services and characteristics
    let mut services = Vec::new();
    if resolved {
        if let Ok(svc_list) = device.services().await {
            for svc in svc_list {
                let svc_uuid = match svc.uuid().await {
                    Ok(u) => u.to_string(),
                    Err(_) => continue,
                };
                let primary = svc.primary().await.unwrap_or(true);

                let mut characteristics = Vec::new();
                if let Ok(char_list) = svc.characteristics().await {
                    for ch in char_list {
                        let ch_uuid = match ch.uuid().await {
                            Ok(u) => u.to_string(),
                            Err(_) => continue,
                        };
                        let ch_flags = ch.flags().await.unwrap_or_default();
                        let can_read = ch_flags.read;
                        let flags = flags_to_strings(&ch_flags);

                        let (value, value_str) = if can_read {
                            match timeout(Duration::from_secs(2), ch.read()).await {
                                Ok(Ok(v)) => {
                                    let hex = v.iter()
                                        .map(|b| format!("{:02x}", b))
                                        .collect::<String>();
                                    let utf8 = if v.iter().all(|&b| (0x20..0x7f).contains(&b) || b == 0x0a) {
                                        String::from_utf8(v).ok()
                                    } else {
                                        None
                                    };
                                    (Some(hex), utf8)
                                }
                                _ => (None, None),
                            }
                        } else {
                            (None, None)
                        };

                        characteristics.push(GattCharacteristic {
                            uuid: ch_uuid,
                            flags,
                            value,
                            value_str,
                        });
                    }
                }

                services.push(GattService {
                    uuid: svc_uuid,
                    primary,
                    characteristics,
                });
            }
        }
    }

    // Disconnect if we initiated the connection
    if !was_connected {
        let _ = device.disconnect().await;
    }

    Ok(GattResult {
        mac: mac.to_string(),
        device_name,
        services,
        error: None,
        timestamp: 0.0,
    })
}
