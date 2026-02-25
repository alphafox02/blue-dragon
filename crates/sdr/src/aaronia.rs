// Copyright 2025-2026 CEMAXECUTER LLC

//! Aaronia Spectran V6 SDR backend.
//!
//! Uses the RTSA API (libAaroniaRTSAAPI.so) in RAW mode to receive
//! wideband IQ samples. At 122 MHz receiver clock with no decimation,
//! covers the entire BLE band (80 MHz) plus guard.

use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam::channel::Sender;

use crate::{SampleBuf, SdrSource};

// ---------------------------------------------------------------------------
// FFI types and constants
// ---------------------------------------------------------------------------

type AarResult = u32;
const AAROK: AarResult = 0x0000_0000;
const AAR_EMPTY: AarResult = 0x0000_0001;

#[repr(C)]
struct AarHandle {
    d: *mut c_void,
}

#[repr(C)]
struct AarDevice {
    d: *mut c_void,
}

#[repr(C)]
struct AarConfig {
    d: *mut c_void,
}

// On Linux, wchar_t is 4 bytes (i32). The C header declares wchar_t serialNumber[120],
// so on Linux that's 120 * 4 = 480 bytes. bool is 1 byte in C.
#[repr(C)]
struct AarDeviceInfoReal {
    cbsize: i64,
    serial_number: [i32; 120], // wchar_t = 4 bytes on Linux
    ready: u8,
    boost: u8,
    superspeed: u8,
    active: u8,
}

#[repr(C)]
struct AarPacket {
    cbsize: i64,
    stream_id: u64,
    flags: u64,
    start_time: f64,
    end_time: f64,
    start_frequency: f64,
    step_frequency: f64,
    span_frequency: f64,
    rbw_frequency: f64,
    num: i64,
    total: i64,
    size: i64,
    stride: i64,
    fp32: *mut f32,
    interleave: i64,
}

// Packet flags
const PACKET_WARN_OVERFLOW: u64 = 0x0000_0000_0000_0100;
const PACKET_WARN_DROPPED: u64 = 0x0000_0000_0000_0200;

/// Default path to Aaronia RTSA Suite installation (contains paths.xml).
const AARONIA_INSTALL_DIR: &str = "/opt/aaronia-rtsa-suite/Aaronia-RTSA-Suite-PRO";

#[allow(dead_code)]
extern "C" {
    fn AARTSAAPI_Init(memory: u32) -> AarResult;
    fn AARTSAAPI_Init_With_Path(memory: u32, path: *const i32) -> AarResult;
    fn AARTSAAPI_Shutdown() -> AarResult;
    fn AARTSAAPI_Open(handle: *mut AarHandle) -> AarResult;
    fn AARTSAAPI_Close(handle: *mut AarHandle) -> AarResult;
    fn AARTSAAPI_RescanDevices(handle: *mut AarHandle, timeout: i32) -> AarResult;
    fn AARTSAAPI_EnumDevice(
        handle: *mut AarHandle,
        device_type: *const i32, // wchar_t*
        index: i32,
        dinfo: *mut AarDeviceInfoReal,
    ) -> AarResult;
    fn AARTSAAPI_OpenDevice(
        handle: *mut AarHandle,
        dhandle: *mut AarDevice,
        device_type: *const i32, // wchar_t*
        serial: *const i32,      // wchar_t*
    ) -> AarResult;
    fn AARTSAAPI_CloseDevice(handle: *mut AarHandle, dhandle: *mut AarDevice) -> AarResult;
    fn AARTSAAPI_ConnectDevice(dhandle: *mut AarDevice) -> AarResult;
    fn AARTSAAPI_DisconnectDevice(dhandle: *mut AarDevice) -> AarResult;
    fn AARTSAAPI_StartDevice(dhandle: *mut AarDevice) -> AarResult;
    fn AARTSAAPI_StopDevice(dhandle: *mut AarDevice) -> AarResult;
    fn AARTSAAPI_ConfigRoot(dhandle: *mut AarDevice, config: *mut AarConfig) -> AarResult;
    fn AARTSAAPI_ConfigFind(
        dhandle: *mut AarDevice,
        group: *mut AarConfig,
        config: *mut AarConfig,
        name: *const i32, // wchar_t*
    ) -> AarResult;
    fn AARTSAAPI_ConfigSetFloat(
        dhandle: *mut AarDevice,
        config: *mut AarConfig,
        value: f64,
    ) -> AarResult;
    fn AARTSAAPI_ConfigSetString(
        dhandle: *mut AarDevice,
        config: *mut AarConfig,
        value: *const i32, // wchar_t*
    ) -> AarResult;
    fn AARTSAAPI_GetPacket(
        dhandle: *mut AarDevice,
        channel: i32,
        index: i32,
        packet: *mut AarPacket,
    ) -> AarResult;
    fn AARTSAAPI_ConsumePackets(dhandle: *mut AarDevice, channel: i32, num: i32) -> AarResult;
    fn AARTSAAPI_ConfigGetFloat(
        dhandle: *mut AarDevice,
        config: *mut AarConfig,
        value: *mut f64,
    ) -> AarResult;
    fn AARTSAAPI_ConfigGetString(
        dhandle: *mut AarDevice,
        config: *mut AarConfig,
        value: *mut i32,
        size: *mut i64,
    ) -> AarResult;
    fn AARTSAAPI_ConfigGetInfo(
        dhandle: *mut AarDevice,
        config: *mut AarConfig,
        cinfo: *mut AarConfigInfo,
    ) -> AarResult;
}

#[repr(C)]
struct AarConfigInfo {
    cbsize: i64,
    name: [i32; 80],       // wchar_t[80]
    title: [i32; 120],     // wchar_t[120]
    config_type: i32,      // enum
    min_value: f64,
    max_value: f64,
    step_value: f64,
    unit: [i32; 10],       // wchar_t[10]
    options: [i32; 1000],  // wchar_t[1000]
    disabled_options: u64,
}

// ---------------------------------------------------------------------------
// Wide-char helpers
// ---------------------------------------------------------------------------

/// Convert a Rust &str to a null-terminated wchar_t array (i32 on Linux).
fn to_wchar(s: &str) -> Vec<i32> {
    let mut v: Vec<i32> = s.chars().map(|c| c as i32).collect();
    v.push(0);
    v
}

/// Convert wchar_t array to Rust String.
fn from_wchar(buf: &[i32]) -> String {
    buf.iter()
        .take_while(|&&c| c != 0)
        .filter_map(|&c| char::from_u32(c as u32))
        .collect()
}

// ---------------------------------------------------------------------------
// Device listing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AaroniaInfo {
    pub serial: String,
    pub ready: bool,
    pub superspeed: bool,
}

/// List connected Spectran V6 devices.
pub fn list_devices() -> Result<Vec<AaroniaInfo>, String> {
    unsafe {
        let install_path = to_wchar(AARONIA_INSTALL_DIR);
        let res = AARTSAAPI_Init_With_Path(0, install_path.as_ptr()); // SMALL
        if res != AAROK {
            return Err(format!("AARTSAAPI_Init failed: 0x{:08x}", res));
        }

        let mut h = AarHandle { d: ptr::null_mut() };
        let res = AARTSAAPI_Open(&mut h);
        if res != AAROK {
            AARTSAAPI_Shutdown();
            return Err(format!("AARTSAAPI_Open failed: 0x{:08x}", res));
        }

        let res = AARTSAAPI_RescanDevices(&mut h, 5000);
        if res != AAROK {
            AARTSAAPI_Close(&mut h);
            AARTSAAPI_Shutdown();
            return Err(format!("AARTSAAPI_RescanDevices failed: 0x{:08x}", res));
        }

        let device_type = to_wchar("spectranv6");
        let mut devices = Vec::new();
        let mut idx = 0;
        loop {
            let mut dinfo = std::mem::zeroed::<AarDeviceInfoReal>();
            dinfo.cbsize = std::mem::size_of::<AarDeviceInfoReal>() as i64;
            let res = AARTSAAPI_EnumDevice(
                &mut h,
                device_type.as_ptr(),
                idx,
                &mut dinfo,
            );
            if res != AAROK {
                break;
            }
            devices.push(AaroniaInfo {
                serial: from_wchar(&dinfo.serial_number),
                ready: dinfo.ready != 0,
                superspeed: dinfo.superspeed != 0,
            });
            idx += 1;
        }

        // Note: AARTSAAPI_Close() and AARTSAAPI_Shutdown() are intentionally skipped here.
        // The library spawns internal threads that can crash during teardown
        // when called in a short-lived context like --list. The OS will
        // clean up when the process exits.
        Ok(devices)
    }
}

// ---------------------------------------------------------------------------
// Receiver clock mapping
// ---------------------------------------------------------------------------

/// Map -C channel count to the best receiver clock and decimation.
/// We want the sample rate to equal num_channels * 1e6.
/// Available clocks: 46, 61, 77, 92, 122, 184, 245, 492 MHz.
fn select_clock(sample_rate_hz: u32) -> (&'static str, &'static str, u32) {
    // Receiver clock options in MHz, ascending
    let clocks: &[(u32, &str)] = &[
        (46_080_000, "46MHz"),
        (61_440_000, "61MHz"),
        (77_000_000, "77MHz"),
        (92_160_000, "92MHz"),
        (122_880_000, "122MHz"),
        (184_320_000, "184MHz"),
        (245_760_000, "245MHz"),
    ];

    // Try exact match first (clock / decimation = target rate)
    let decimations: &[(u32, &str)] = &[
        (1, "Full"),
        (2, "1 / 2"),
        (4, "1 / 4"),
        (8, "1 / 8"),
        (16, "1 / 16"),
        (32, "1 / 32"),
        (64, "1 / 64"),
        (128, "1 / 128"),
        (256, "1 / 256"),
        (512, "1 / 512"),
    ];

    for &(clock_hz, clock_name) in clocks {
        for &(div, div_name) in decimations {
            if clock_hz / div == sample_rate_hz {
                return (clock_name, div_name, clock_hz / div);
            }
        }
    }

    // No exact match -- pick smallest clock >= target rate
    for &(clock_hz, clock_name) in clocks {
        if clock_hz >= sample_rate_hz {
            return (clock_name, "Full", clock_hz);
        }
    }

    // Fallback to max
    ("245MHz", "Full", 245_760_000)
}

// ---------------------------------------------------------------------------
// SdrSource implementation
// ---------------------------------------------------------------------------

pub struct AaroniaSource {
    serial: String,
    sample_rate: u32,
    center_freq: u64,
    gain: f64,
    running: Arc<AtomicBool>,
    actual_sample_rate: u32,
}

impl AaroniaSource {
    pub fn new(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: f64,
    ) -> Result<Self, String> {
        // Interface format: "aaronia-SERIAL" or "aaronia" for first device
        let serial = if iface == "aaronia" {
            String::new()
        } else if let Some(rest) = iface.strip_prefix("aaronia-") {
            rest.to_string()
        } else {
            return Err(format!(
                "invalid Aaronia interface: '{}' (expected aaronia or aaronia-SERIAL)",
                iface
            ));
        };

        let (_, _, actual_rate) = select_clock(sample_rate);

        Ok(Self {
            serial,
            sample_rate,
            center_freq,
            gain,
            running: Arc::new(AtomicBool::new(false)),
            actual_sample_rate: actual_rate,
        })
    }

    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }
}

impl SdrSource for AaroniaSource {
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String> {
        self.running.store(true, Ordering::SeqCst);

        let (clock_name, decim_name, actual_rate) = select_clock(self.sample_rate);
        self.actual_sample_rate = actual_rate;

        unsafe {
            // Initialize library
            let install_path = to_wchar(AARONIA_INSTALL_DIR);
            let res = AARTSAAPI_Init_With_Path(1, install_path.as_ptr()); // MEDIUM
            if res != AAROK {
                return Err(format!("AARTSAAPI_Init failed: 0x{:08x}", res));
            }

            let mut h = AarHandle { d: ptr::null_mut() };
            let res = AARTSAAPI_Open(&mut h);
            if res != AAROK {
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_Open failed: 0x{:08x}", res));
            }

            let res = AARTSAAPI_RescanDevices(&mut h, 5000);
            if res != AAROK {
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_RescanDevices failed: 0x{:08x}", res));
            }

            // Find device serial
            let device_type_enum = to_wchar("spectranv6");
            let serial_wc = if self.serial.is_empty() {
                // Use first device
                let mut dinfo = std::mem::zeroed::<AarDeviceInfoReal>();
                dinfo.cbsize = std::mem::size_of::<AarDeviceInfoReal>() as i64;
                let res = AARTSAAPI_EnumDevice(
                    &mut h,
                    device_type_enum.as_ptr(),
                    0,
                    &mut dinfo,
                );
                if res != AAROK {
                    AARTSAAPI_Close(&mut h);
                    AARTSAAPI_Shutdown();
                    return Err("no Spectran V6 devices found".into());
                }
                let serial_str = from_wchar(&dinfo.serial_number);
                eprintln!("Aaronia: using device {}", serial_str);
                to_wchar(&serial_str)
            } else {
                to_wchar(&self.serial)
            };

            // Open device in raw mode
            let device_type_raw = to_wchar("spectranv6/raw");
            let mut d = AarDevice { d: ptr::null_mut() };
            let res = AARTSAAPI_OpenDevice(
                &mut h,
                &mut d,
                device_type_raw.as_ptr(),
                serial_wc.as_ptr(),
            );
            if res != AAROK {
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_OpenDevice failed: 0x{:08x}", res));
            }

            // Configure
            let mut root = AarConfig { d: ptr::null_mut() };
            let mut config = AarConfig { d: ptr::null_mut() };
            AARTSAAPI_ConfigRoot(&mut d, &mut root);

            // String configs
            let str_configs: &[(&str, &str)] = &[
                ("device/receiverchannel", "Rx1"),
                ("device/outputformat", "iq"),
                ("device/receiverclock", clock_name),
                ("main/decimation", decim_name),
                ("calibration/preamp", "Amp"),
                ("calibration/rffilter", "Auto Extended"),
                ("device/gaincontrol", "manual"),
            ];
            for &(key, val) in str_configs {
                let k = to_wchar(key);
                let v = to_wchar(val);
                let res = AARTSAAPI_ConfigFind(&mut d, &mut root, &mut config, k.as_ptr());
                if res == AAROK {
                    AARTSAAPI_ConfigSetString(&mut d, &mut config, v.as_ptr());
                }
            }

            // Center frequency
            let key = to_wchar("main/centerfreq");
            AARTSAAPI_ConfigFind(&mut d, &mut root, &mut config, key.as_ptr());
            AARTSAAPI_ConfigSetFloat(&mut d, &mut config, self.center_freq as f64);

            // Reference level -- [-20, 10] dBm with preamp "Auto"
            let reflevel = if (-20.0..=10.0).contains(&self.gain) {
                self.gain
            } else {
                -20.0
            };
            let key = to_wchar("main/reflevel");
            AARTSAAPI_ConfigFind(&mut d, &mut root, &mut config, key.as_ptr());
            AARTSAAPI_ConfigSetFloat(&mut d, &mut config, reflevel);

            // Connect
            let res = AARTSAAPI_ConnectDevice(&mut d);
            if res != AAROK {
                AARTSAAPI_CloseDevice(&mut h, &mut d);
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_ConnectDevice failed: 0x{:08x}", res));
            }

            // Start
            let res = AARTSAAPI_StartDevice(&mut d);
            if res != AAROK {
                AARTSAAPI_DisconnectDevice(&mut d);
                AARTSAAPI_CloseDevice(&mut h, &mut d);
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_StartDevice failed: 0x{:08x}", res));
            }

            log::info!(
                "Aaronia Spectran V6 streaming (clock={}, decim={}, rate={} MS/s, center={} MHz, reflevel={} dBm)",
                clock_name,
                decim_name,
                actual_rate / 1_000_000,
                self.center_freq / 1_000_000,
                reflevel,
            );

            // Auto-scale: measure RMS from first packet for f32→i16 conversion
            let scale = {
                let mut first_pkt = std::mem::zeroed::<AarPacket>();
                first_pkt.cbsize = std::mem::size_of::<AarPacket>() as i64;
                let mut retries = 0;
                loop {
                    let res = AARTSAAPI_GetPacket(&mut d, 0, 0, &mut first_pkt);
                    if res == AAROK {
                        break;
                    }
                    if res != AAR_EMPTY {
                        break;
                    }
                    retries += 1;
                    if retries > 2000 {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }

                let n = first_pkt.num as usize;
                let stride = first_pkt.stride as usize;
                let scale = if n > 0 && !first_pkt.fp32.is_null() && stride >= 2 {
                    let mut sum_sq: f64 = 0.0;
                    for i in 0..n {
                        let fi = *first_pkt.fp32.add(i * stride) as f64;
                        let fq = *first_pkt.fp32.add(i * stride + 1) as f64;
                        sum_sq += fi * fi + fq * fq;
                    }
                    let rms = (sum_sq / (n as f64 * 2.0)).sqrt();
                    // Scale so RMS noise ≈ 2048 in i16 (leaving headroom)
                    if rms > 0.0 {
                        2048.0 / rms as f32
                    } else {
                        1e6
                    }
                } else {
                    1e6
                };
                AARTSAAPI_ConsumePackets(&mut d, 0, 1);
                scale
            };

            log::info!("Aaronia f32→i16 scale factor: {:.0}", scale);

            // Receive loop -- convert f32 IQ to i16 IQ
            while self.running.load(Ordering::SeqCst) {
                let mut pkt = std::mem::zeroed::<AarPacket>();
                pkt.cbsize = std::mem::size_of::<AarPacket>() as i64;

                let res = AARTSAAPI_GetPacket(&mut d, 0, 0, &mut pkt);
                if res == AAR_EMPTY {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                if res != AAROK {
                    log::error!("AARTSAAPI_GetPacket error: 0x{:08x}", res);
                    break;
                }

                if pkt.flags & PACKET_WARN_OVERFLOW != 0 {
                    eprint!("O");
                }
                if pkt.flags & PACKET_WARN_DROPPED != 0 {
                    eprint!("D");
                }

                let num_samples = pkt.num as usize;
                if num_samples > 0 && !pkt.fp32.is_null() {
                    let stride = pkt.stride as usize;
                    let mut data = Vec::with_capacity(num_samples * 2);

                    for i in 0..num_samples {
                        let fi = *pkt.fp32.add(i * stride);
                        let fq = *pkt.fp32.add(i * stride + 1);
                        data.push((fi * scale).clamp(-32768.0, 32767.0) as i16);
                        data.push((fq * scale).clamp(-32768.0, 32767.0) as i16);
                    }

                    AARTSAAPI_ConsumePackets(&mut d, 0, 1);

                    if tx.send(SampleBuf { data, num_samples }).is_err() {
                        break;
                    }
                } else {
                    AARTSAAPI_ConsumePackets(&mut d, 0, 1);
                }
            }

            // Cleanup
            AARTSAAPI_StopDevice(&mut d);
            AARTSAAPI_DisconnectDevice(&mut d);
            AARTSAAPI_CloseDevice(&mut h, &mut d);
            AARTSAAPI_Close(&mut h);
            AARTSAAPI_Shutdown();

            log::info!("Aaronia streaming stopped");
        }

        Ok(())
    }

    fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }

    fn sample_rate(&self) -> u32 {
        self.actual_sample_rate
    }

    fn center_frequency(&self) -> u64 {
        self.center_freq
    }
}

// ---------------------------------------------------------------------------
// Handle for zero-copy recv_into path
// ---------------------------------------------------------------------------

/// Direct Aaronia handle for the recv_into() path used by the pipeline.
pub struct AaroniaHandle {
    h: AarHandle,
    d: AarDevice,
    max_samps: usize,
    pub running: Arc<AtomicBool>,
    overflow_count: u64,
    initialized: bool,
    /// f32-to-i8 scale factor, auto-computed from noise floor RMS.
    scale: f32,
}

unsafe impl Send for AaroniaHandle {}

impl AaroniaHandle {
    /// Open a Spectran V6 in raw mode and start streaming.
    /// `gain` is interpreted as reflevel in dBm (range [-20, 10]).
    /// If out of range (e.g. the default 60 from USRP), defaults to -20 dBm
    /// (maximum sensitivity for BLE).
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: f64,
        _antenna: Option<&str>,
    ) -> Result<Self, String> {
        let serial = if iface == "aaronia" {
            String::new()
        } else if let Some(rest) = iface.strip_prefix("aaronia-") {
            rest.to_string()
        } else {
            return Err(format!(
                "invalid Aaronia interface: '{}' (expected aaronia or aaronia-SERIAL)",
                iface
            ));
        };

        let (clock_name, decim_name, actual_rate) = select_clock(sample_rate);

        unsafe {
            let install_path = to_wchar(AARONIA_INSTALL_DIR);
            let res = AARTSAAPI_Init_With_Path(1, install_path.as_ptr());
            if res != AAROK {
                return Err(format!("AARTSAAPI_Init failed: 0x{:08x}", res));
            }

            let mut h = AarHandle { d: ptr::null_mut() };
            let res = AARTSAAPI_Open(&mut h);
            if res != AAROK {
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_Open failed: 0x{:08x}", res));
            }

            let res = AARTSAAPI_RescanDevices(&mut h, 5000);
            if res != AAROK {
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_RescanDevices failed: 0x{:08x}", res));
            }

            let device_type_enum = to_wchar("spectranv6");
            let serial_wc = if serial.is_empty() {
                let mut dinfo = std::mem::zeroed::<AarDeviceInfoReal>();
                dinfo.cbsize = std::mem::size_of::<AarDeviceInfoReal>() as i64;
                let res = AARTSAAPI_EnumDevice(
                    &mut h,
                    device_type_enum.as_ptr(),
                    0,
                    &mut dinfo,
                );
                if res != AAROK {
                    AARTSAAPI_Close(&mut h);
                    AARTSAAPI_Shutdown();
                    return Err("no Spectran V6 devices found".into());
                }
                let serial_str = from_wchar(&dinfo.serial_number);
                eprintln!("Aaronia: using device {}", serial_str);
                to_wchar(&serial_str)
            } else {
                to_wchar(&serial)
            };

            let device_type_raw = to_wchar("spectranv6/raw");
            let mut d = AarDevice { d: ptr::null_mut() };
            let res = AARTSAAPI_OpenDevice(
                &mut h,
                &mut d,
                device_type_raw.as_ptr(),
                serial_wc.as_ptr(),
            );
            if res != AAROK {
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_OpenDevice failed: 0x{:08x}", res));
            }

            // Configure
            let mut root = AarConfig { d: ptr::null_mut() };
            let mut config = AarConfig { d: ptr::null_mut() };
            AARTSAAPI_ConfigRoot(&mut d, &mut root);

            let configs: &[(&str, &str)] = &[
                ("device/receiverchannel", "Rx1"),
                ("device/outputformat", "iq"),
                ("device/receiverclock", clock_name),
                ("main/decimation", decim_name),
                ("calibration/preamp", "Amp"),
                ("calibration/rffilter", "Auto Extended"),
                ("device/gaincontrol", "manual"),
            ];
            for &(key, val) in configs {
                let k = to_wchar(key);
                let v = to_wchar(val);
                let res = AARTSAAPI_ConfigFind(&mut d, &mut root, &mut config, k.as_ptr());
                if res != AAROK {
                    eprintln!("Aaronia: ConfigFind({}) failed: 0x{:08x}", key, res);
                    continue;
                }
                let res = AARTSAAPI_ConfigSetString(&mut d, &mut config, v.as_ptr());
                if res != AAROK {
                    eprintln!("Aaronia: ConfigSet({} = {}) failed: 0x{:08x}", key, val, res);
                }
            }

            let key = to_wchar("main/centerfreq");
            AARTSAAPI_ConfigFind(&mut d, &mut root, &mut config, key.as_ptr());
            AARTSAAPI_ConfigSetFloat(&mut d, &mut config, center_freq as f64);

            // With preamp "Auto", reflevel range is [-20, 10] dBm
            let reflevel = if (-20.0..=10.0).contains(&gain) {
                gain
            } else {
                -20.0
            };
            let key = to_wchar("main/reflevel");
            AARTSAAPI_ConfigFind(&mut d, &mut root, &mut config, key.as_ptr());
            AARTSAAPI_ConfigSetFloat(&mut d, &mut config, reflevel);

            // Diagnostic: query preamp config info to see valid options
            {
                let key = to_wchar("calibration/preamp");
                let mut cfg = AarConfig { d: ptr::null_mut() };
                if AARTSAAPI_ConfigFind(&mut d, &mut root, &mut cfg, key.as_ptr()) == AAROK {
                    let mut cinfo = std::mem::zeroed::<AarConfigInfo>();
                    cinfo.cbsize = std::mem::size_of::<AarConfigInfo>() as i64;
                    if AARTSAAPI_ConfigGetInfo(&mut d, &mut cfg, &mut cinfo) == AAROK {
                        let opts = from_wchar(&cinfo.options);
                        let title = from_wchar(&cinfo.title);
                        // Read current value
                        let mut val_buf = [0i32; 256];
                        let mut val_size = 256i64;
                        AARTSAAPI_ConfigGetString(&mut d, &mut cfg, val_buf.as_mut_ptr(), &mut val_size);
                        let cur_val = from_wchar(&val_buf);
                        eprintln!("Aaronia preamp: title='{}', current='{}', options='{}'", title, cur_val, opts);
                    } else {
                        eprintln!("Aaronia: ConfigGetInfo(calibration/preamp) failed");
                    }
                } else {
                    eprintln!("Aaronia: ConfigFind(calibration/preamp) failed");
                }
            }

            // Connect and start
            let res = AARTSAAPI_ConnectDevice(&mut d);
            if res != AAROK {
                AARTSAAPI_CloseDevice(&mut h, &mut d);
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_ConnectDevice failed: 0x{:08x}", res));
            }

            let res = AARTSAAPI_StartDevice(&mut d);
            if res != AAROK {
                AARTSAAPI_DisconnectDevice(&mut d);
                AARTSAAPI_CloseDevice(&mut h, &mut d);
                AARTSAAPI_Close(&mut h);
                AARTSAAPI_Shutdown();
                return Err(format!("AARTSAAPI_StartDevice failed: 0x{:08x}", res));
            }

            // Wait for first packet to determine max_samps, then let the
            // receiver settle before computing auto-scale.
            let mut first_pkt = std::mem::zeroed::<AarPacket>();
            first_pkt.cbsize = std::mem::size_of::<AarPacket>() as i64;
            let mut retries = 0u32;
            loop {
                let res = AARTSAAPI_GetPacket(&mut d, 0, 0, &mut first_pkt);
                if res == AAROK {
                    break;
                }
                if res != AAR_EMPTY {
                    AARTSAAPI_StopDevice(&mut d);
                    AARTSAAPI_DisconnectDevice(&mut d);
                    AARTSAAPI_CloseDevice(&mut h, &mut d);
                    AARTSAAPI_Close(&mut h);
                    AARTSAAPI_Shutdown();
                    return Err(format!("failed to get first packet: 0x{:08x}", res));
                }
                retries += 1;
                if retries > 10000 {
                    AARTSAAPI_StopDevice(&mut d);
                    AARTSAAPI_DisconnectDevice(&mut d);
                    AARTSAAPI_CloseDevice(&mut h, &mut d);
                    AARTSAAPI_Close(&mut h);
                    AARTSAAPI_Shutdown();
                    return Err("timeout waiting for first Aaronia packet".into());
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }

            let max_samps = first_pkt.num as usize;
            AARTSAAPI_ConsumePackets(&mut d, 0, 1);

            // Wait ~500ms for receiver to settle past startup transient,
            // then drain any queued packets
            std::thread::sleep(std::time::Duration::from_millis(500));
            loop {
                let mut drain = std::mem::zeroed::<AarPacket>();
                drain.cbsize = std::mem::size_of::<AarPacket>() as i64;
                if AARTSAAPI_GetPacket(&mut d, 0, 0, &mut drain) == AAROK {
                    AARTSAAPI_ConsumePackets(&mut d, 0, 1);
                } else {
                    break;
                }
            }

            // Measure RMS over multiple packets for robust auto-scaling.
            // WiFi bursts in the 2.4 GHz band can inflate a single-packet
            // RMS measurement, producing a scale factor that's too low for
            // the actual noise floor.  We collect 20 per-packet RMS values
            // and use the 25th percentile (5th lowest) as the reference.
            let mut scale: f32 = 1e6;
            let mut rms_samples: Vec<f64> = Vec::with_capacity(20);
            retries = 0;
            while rms_samples.len() < 20 {
                let mut rms_pkt = std::mem::zeroed::<AarPacket>();
                rms_pkt.cbsize = std::mem::size_of::<AarPacket>() as i64;
                let res = AARTSAAPI_GetPacket(&mut d, 0, 0, &mut rms_pkt);
                if res == AAROK {
                    let n = rms_pkt.num as usize;
                    let stride = rms_pkt.stride as usize;
                    if n > 0 && !rms_pkt.fp32.is_null() && stride >= 2 {
                        let mut sum_sq: f64 = 0.0;
                        for i in 0..n {
                            let fi = *rms_pkt.fp32.add(i * stride) as f64;
                            let fq = *rms_pkt.fp32.add(i * stride + 1) as f64;
                            sum_sq += fi * fi + fq * fq;
                        }
                        let rms = (sum_sq / (n as f64 * 2.0)).sqrt();
                        if rms > 0.0 {
                            rms_samples.push(rms);
                        }
                    }
                    AARTSAAPI_ConsumePackets(&mut d, 0, 1);
                } else if res == AAR_EMPTY {
                    retries += 1;
                    if retries > 10000 { break; }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                } else {
                    break;
                }
            }
            if !rms_samples.is_empty() {
                rms_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                // 25th percentile: filters out WiFi burst outliers
                let idx = rms_samples.len() / 4;
                let rms = rms_samples[idx];
                // Target RMS noise at 2 LSBs in i8 (512 in i16 after *256).
                // Leaves ~36 dB headroom for BLE signals before clipping.
                scale = 2.0 / rms as f32;
                eprintln!(
                    "Aaronia auto-scale: rms={:.3e} (p25 of {} pkts), scale={:.0}",
                    rms, rms_samples.len(), scale
                );
            }

            log::info!(
                "Aaronia Spectran V6 open (clock={}, decim={}, rate={} MS/s, center={} MHz, reflevel={} dBm, pkt_size={}, scale={:.0})",
                clock_name,
                decim_name,
                actual_rate / 1_000_000,
                center_freq / 1_000_000,
                reflevel,
                max_samps,
                scale,
            );

            Ok(Self {
                h,
                d,
                max_samps,
                running: Arc::new(AtomicBool::new(true)),
                overflow_count: 0,
                initialized: true,
                scale,
            })
        }
    }

    /// Receive samples into an i8 buffer (for the GPU pipeline path).
    /// Returns number of complex samples received.
    pub fn recv_into(&mut self, buf: &mut [i8]) -> usize {
        let max_complex = buf.len() / 2;

        unsafe {
            let mut pkt = std::mem::zeroed::<AarPacket>();
            pkt.cbsize = std::mem::size_of::<AarPacket>() as i64;

            loop {
                let res = AARTSAAPI_GetPacket(&mut self.d, 0, 0, &mut pkt);
                if res == AAR_EMPTY {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                    continue;
                }
                if res != AAROK {
                    return 0;
                }
                break;
            }

            if pkt.flags & PACKET_WARN_OVERFLOW != 0 {
                self.overflow_count += 1;
            }

            let num_samples = (pkt.num as usize).min(max_complex);
            if num_samples > 0 && !pkt.fp32.is_null() {
                let stride = pkt.stride as usize;
                let scale = self.scale;
                for i in 0..num_samples {
                    let fi = *pkt.fp32.add(i * stride);
                    let fq = *pkt.fp32.add(i * stride + 1);
                    buf[i * 2] = (fi * scale).clamp(-128.0, 127.0) as i8;
                    buf[i * 2 + 1] = (fq * scale).clamp(-128.0, 127.0) as i8;
                }
            }

            AARTSAAPI_ConsumePackets(&mut self.d, 0, 1);

            num_samples
        }
    }

    /// Receive samples into an i16 buffer (for the CPU pipeline path).
    /// Converts f32 directly to i16 for full 16-bit precision, avoiding
    /// the 8-bit quantization bottleneck of the i8 path.
    /// Returns number of complex samples received.
    pub fn recv_into_i16(&mut self, buf: &mut [i16]) -> usize {
        let max_complex = buf.len() / 2;

        unsafe {
            let mut pkt = std::mem::zeroed::<AarPacket>();
            pkt.cbsize = std::mem::size_of::<AarPacket>() as i64;

            loop {
                let res = AARTSAAPI_GetPacket(&mut self.d, 0, 0, &mut pkt);
                if res == AAR_EMPTY {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                    continue;
                }
                if res != AAROK {
                    return 0;
                }
                break;
            }

            if pkt.flags & PACKET_WARN_OVERFLOW != 0 {
                self.overflow_count += 1;
            }

            let num_samples = (pkt.num as usize).min(max_complex);
            if num_samples > 0 && !pkt.fp32.is_null() {
                let stride = pkt.stride as usize;
                // scale * 256: same headroom as the i8 path but preserving
                // 8 extra bits of precision in the i16 representation.
                let scale16 = self.scale * 256.0;
                for i in 0..num_samples {
                    let fi = *pkt.fp32.add(i * stride);
                    let fq = *pkt.fp32.add(i * stride + 1);
                    buf[i * 2] = (fi * scale16).clamp(-32768.0, 32767.0) as i16;
                    buf[i * 2 + 1] = (fq * scale16).clamp(-32768.0, 32767.0) as i16;
                }
            }

            AARTSAAPI_ConsumePackets(&mut self.d, 0, 1);

            num_samples
        }
    }

    /// Set reference level at runtime.
    pub fn set_gain(&self, gain: f64) {
        unsafe {
            let mut root = AarConfig { d: ptr::null_mut() };
            let mut config = AarConfig { d: ptr::null_mut() };
            // Cast away const -- the API takes *mut but logically this is safe
            let d_ptr = &self.d as *const AarDevice as *mut AarDevice;
            AARTSAAPI_ConfigRoot(d_ptr, &mut root);
            let key = to_wchar("main/reflevel");
            AARTSAAPI_ConfigFind(d_ptr, &mut root, &mut config, key.as_ptr());
            AARTSAAPI_ConfigSetFloat(d_ptr, &mut config, gain);
        }
    }

    pub fn max_samps(&self) -> usize {
        self.max_samps
    }

    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }
}

impl Drop for AaroniaHandle {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                AARTSAAPI_StopDevice(&mut self.d);
                AARTSAAPI_DisconnectDevice(&mut self.d);
                AARTSAAPI_CloseDevice(&mut self.h, &mut self.d);
                AARTSAAPI_Close(&mut self.h);
                AARTSAAPI_Shutdown();
            }
            self.initialized = false;
        }
    }
}
