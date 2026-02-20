// Copyright 2025 CEMAXECUTER LLC

use crossbeam::channel::Sender;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{SampleBuf, SdrSource};

type BladerfDevice = c_void;
type BladerfStream = c_void;

// bladeRF constants
const BLADERF_MODULE_RX: c_int = 0;
const BLADERF_CHANNEL_RX_0: c_int = 0;
const BLADERF_GAIN_MGC: c_int = 1;
const BLADERF_FORMAT_SC8_Q7: c_int = 4;
const BLADERF_FORMAT_SC16_Q11: c_int = 1;

// bladerf_devinfo
#[repr(C)]
struct BladerfDevinfo {
    backend: c_int,
    serial: [c_char; 33],
    usb_bus: u8,
    usb_addr: u8,
    instance: c_uint,
    manufacturer: [c_char; 33],
    product: [c_char; 33],
}

extern "C" {
    fn bladerf_open(device: *mut *mut BladerfDevice, identifier: *const c_char) -> c_int;
    fn bladerf_close(device: *mut BladerfDevice);
    fn bladerf_set_frequency(
        dev: *mut BladerfDevice,
        ch: c_int,
        frequency: u64,
    ) -> c_int;
    fn bladerf_set_bandwidth(
        dev: *mut BladerfDevice,
        ch: c_int,
        bandwidth: c_uint,
        actual: *mut c_uint,
    ) -> c_int;
    fn bladerf_set_gain(dev: *mut BladerfDevice, ch: c_int, gain: c_int) -> c_int;
    fn bladerf_set_gain_mode(dev: *mut BladerfDevice, ch: c_int, mode: c_int) -> c_int;
    fn bladerf_set_sample_rate(
        dev: *mut BladerfDevice,
        ch: c_int,
        rate: c_uint,
        actual: *mut c_uint,
    ) -> c_int;
    fn bladerf_sync_config(
        dev: *mut BladerfDevice,
        layout: c_int,
        format: c_int,
        num_buffers: c_uint,
        buffer_size: c_uint,
        num_transfers: c_uint,
        stream_timeout: c_uint,
    ) -> c_int;
    fn bladerf_sync_rx(
        dev: *mut BladerfDevice,
        samples: *mut c_void,
        num_samples: c_uint,
        metadata: *mut c_void,
        timeout_ms: c_uint,
    ) -> c_int;
    fn bladerf_enable_module(dev: *mut BladerfDevice, ch: c_int, enable: bool) -> c_int;
    fn bladerf_get_device_list(devices: *mut *mut BladerfDevinfo) -> c_int;
    fn bladerf_free_device_list(devices: *mut BladerfDevinfo);
    fn bladerf_enable_feature(
        dev: *mut BladerfDevice,
        feature: c_uint,
        enable: bool,
    ) -> c_int;
}

// BLADERF_FEATURE_OVERSAMPLE = 1
const BLADERF_FEATURE_OVERSAMPLE: c_uint = 1;

#[derive(Debug, Clone)]
pub struct BladerfInfo {
    pub instance: u32,
    pub serial: String,
}

pub fn list_devices() -> Result<Vec<BladerfInfo>, String> {
    let mut devs: *mut BladerfDevinfo = ptr::null_mut();
    let count = unsafe { bladerf_get_device_list(&mut devs) };

    if count <= 0 || devs.is_null() {
        return Ok(Vec::new());
    }

    let mut devices = Vec::new();
    for i in 0..count as usize {
        let dev = unsafe { &*devs.add(i) };
        let serial = unsafe {
            std::ffi::CStr::from_ptr(dev.serial.as_ptr())
                .to_string_lossy()
                .to_string()
        };
        devices.push(BladerfInfo {
            instance: dev.instance,
            serial,
        });
    }

    unsafe { bladerf_free_device_list(devs) };
    Ok(devices)
}

fn parse_instance(iface: &str) -> Option<u32> {
    // "bladerf0" -> 0
    if iface.starts_with("bladerf") {
        iface[7..].parse().ok()
    } else {
        None
    }
}

pub struct BladerfSource {
    instance: u32,
    sample_rate: u32,
    center_freq: u64,
    gain: i32,
    running: Arc<AtomicBool>,
}

impl BladerfSource {
    pub fn new(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: i32,
    ) -> Result<Self, String> {
        let instance = parse_instance(iface)
            .ok_or_else(|| format!("invalid bladeRF interface: '{}' (expected bladerf0)", iface))?;

        Ok(Self {
            instance,
            sample_rate,
            center_freq,
            gain,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }
}

impl SdrSource for BladerfSource {
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String> {
        self.running.store(true, Ordering::SeqCst);

        let identifier = CString::new(format!("*:instance={}", self.instance))
            .map_err(|e| format!("CString error: {}", e))?;

        unsafe {
            let mut dev: *mut BladerfDevice = ptr::null_mut();
            let r = bladerf_open(&mut dev, identifier.as_ptr());
            if r != 0 {
                return Err(format!("bladerf_open failed: {}", r));
            }

            // Try oversample mode (SC8_Q7)
            let use_sc8 = bladerf_enable_feature(dev, BLADERF_FEATURE_OVERSAMPLE, true) == 0;
            let format = if use_sc8 {
                BLADERF_FORMAT_SC8_Q7
            } else {
                BLADERF_FORMAT_SC16_Q11
            };

            bladerf_set_bandwidth(dev, BLADERF_CHANNEL_RX_0, 56_000_000, ptr::null_mut());
            bladerf_set_frequency(dev, BLADERF_CHANNEL_RX_0, self.center_freq);
            bladerf_set_gain_mode(dev, BLADERF_CHANNEL_RX_0, BLADERF_GAIN_MGC);
            bladerf_set_gain(dev, BLADERF_CHANNEL_RX_0, self.gain);
            bladerf_set_sample_rate(
                dev,
                BLADERF_CHANNEL_RX_0,
                self.sample_rate,
                ptr::null_mut(),
            );

            let buf_size = 8192u32;
            let r = bladerf_sync_config(dev, BLADERF_MODULE_RX, format, 16, buf_size, 8, 3500);
            if r != 0 {
                bladerf_close(dev);
                return Err(format!("bladerf_sync_config failed: {}", r));
            }

            let r = bladerf_enable_module(dev, BLADERF_MODULE_RX, true);
            if r != 0 {
                bladerf_close(dev);
                return Err(format!("bladerf_enable_module failed: {}", r));
            }

            log::info!(
                "bladeRF streaming (instance={}, {} MHz, {} MS/s, gain={} dB, {})",
                self.instance,
                self.center_freq / 1_000_000,
                self.sample_rate / 1_000_000,
                self.gain,
                if use_sc8 { "SC8" } else { "SC16" },
            );

            if use_sc8 {
                let mut sc8_buf = vec![0i8; buf_size as usize * 2];
                while self.running.load(Ordering::SeqCst) {
                    let r = bladerf_sync_rx(
                        dev,
                        sc8_buf.as_mut_ptr() as *mut c_void,
                        buf_size,
                        ptr::null_mut(),
                        3500,
                    );
                    if r != 0 {
                        log::error!("bladerf_sync_rx error: {}", r);
                        break;
                    }

                    let num_samples = buf_size as usize;
                    let mut data = Vec::with_capacity(num_samples * 2);
                    for i in 0..num_samples * 2 {
                        data.push((sc8_buf[i] as i16) << 8);
                    }

                    if tx.send(SampleBuf { data, num_samples }).is_err() {
                        break;
                    }
                }
            } else {
                let mut sc16_buf = vec![0i16; buf_size as usize * 2];
                while self.running.load(Ordering::SeqCst) {
                    let r = bladerf_sync_rx(
                        dev,
                        sc16_buf.as_mut_ptr() as *mut c_void,
                        buf_size,
                        ptr::null_mut(),
                        3500,
                    );
                    if r != 0 {
                        log::error!("bladerf_sync_rx error: {}", r);
                        break;
                    }

                    let num_samples = buf_size as usize;
                    let mut data = Vec::with_capacity(num_samples * 2);
                    for i in 0..num_samples * 2 {
                        data.push(sc16_buf[i] >> 4); // SC16_Q11 -> roughly 8-bit range
                    }

                    if tx.send(SampleBuf { data, num_samples }).is_err() {
                        break;
                    }
                }
            }

            bladerf_enable_module(dev, BLADERF_MODULE_RX, false);
            bladerf_close(dev);
            log::info!("bladeRF streaming stopped");
        }

        Ok(())
    }

    fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn center_frequency(&self) -> u64 {
        self.center_freq
    }
}

/// Direct bladeRF handle for zero-copy recv_into path.
pub struct BladerfHandle {
    dev: *mut BladerfDevice,
    use_sc8: bool,
    max_samps: usize,
    sc16_buf: Vec<i16>,
    pub running: Arc<AtomicBool>,
    overflow_count: u64,
}

unsafe impl Send for BladerfHandle {}

impl BladerfHandle {
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: i32,
    ) -> Result<Self, String> {
        let instance = parse_instance(iface)
            .ok_or_else(|| format!("invalid bladeRF interface: '{}'", iface))?;

        let identifier = CString::new(format!("*:instance={}", instance))
            .map_err(|e| format!("CString error: {}", e))?;

        unsafe {
            let mut dev: *mut BladerfDevice = ptr::null_mut();
            let r = bladerf_open(&mut dev, identifier.as_ptr());
            if r != 0 {
                return Err(format!("bladerf_open failed: {}", r));
            }

            let use_sc8 = bladerf_enable_feature(dev, BLADERF_FEATURE_OVERSAMPLE, true) == 0;
            let format = if use_sc8 {
                BLADERF_FORMAT_SC8_Q7
            } else {
                BLADERF_FORMAT_SC16_Q11
            };

            bladerf_set_bandwidth(dev, BLADERF_CHANNEL_RX_0, 56_000_000, ptr::null_mut());
            bladerf_set_frequency(dev, BLADERF_CHANNEL_RX_0, center_freq);
            bladerf_set_gain_mode(dev, BLADERF_CHANNEL_RX_0, BLADERF_GAIN_MGC);
            bladerf_set_gain(dev, BLADERF_CHANNEL_RX_0, gain);
            bladerf_set_sample_rate(dev, BLADERF_CHANNEL_RX_0, sample_rate, ptr::null_mut());

            let buf_size = 8192u32;
            let r = bladerf_sync_config(dev, BLADERF_MODULE_RX, format, 16, buf_size, 8, 3500);
            if r != 0 {
                bladerf_close(dev);
                return Err(format!("bladerf_sync_config failed: {}", r));
            }

            let r = bladerf_enable_module(dev, BLADERF_MODULE_RX, true);
            if r != 0 {
                bladerf_close(dev);
                return Err(format!("bladerf_enable_module failed: {}", r));
            }

            let max_samps = buf_size as usize;
            let sc16_buf = if !use_sc8 {
                vec![0i16; max_samps * 2]
            } else {
                Vec::new()
            };

            Ok(Self {
                dev,
                use_sc8,
                max_samps,
                sc16_buf,
                running: Arc::new(AtomicBool::new(true)),
                overflow_count: 0,
            })
        }
    }

    pub fn recv_into(&mut self, buf: &mut [i8]) -> usize {
        let max_samps = (buf.len() / 2).min(self.max_samps);

        unsafe {
            if self.use_sc8 {
                let r = bladerf_sync_rx(
                    self.dev,
                    buf.as_mut_ptr() as *mut c_void,
                    max_samps as c_uint,
                    ptr::null_mut(),
                    3500,
                );
                if r != 0 {
                    return 0;
                }
                max_samps
            } else {
                let r = bladerf_sync_rx(
                    self.dev,
                    self.sc16_buf.as_mut_ptr() as *mut c_void,
                    max_samps as c_uint,
                    ptr::null_mut(),
                    3500,
                );
                if r != 0 {
                    return 0;
                }
                for i in 0..max_samps * 2 {
                    buf[i] = (self.sc16_buf[i] >> 8) as i8;
                }
                max_samps
            }
        }
    }

    pub fn max_samps(&self) -> usize {
        self.max_samps
    }

    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }
}

impl Drop for BladerfHandle {
    fn drop(&mut self) {
        unsafe {
            bladerf_enable_module(self.dev, BLADERF_MODULE_RX, false);
            bladerf_close(self.dev);
        }
    }
}
