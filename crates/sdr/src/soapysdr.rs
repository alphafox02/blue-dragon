// Copyright 2025 CEMAXECUTER LLC

use crossbeam::channel::Sender;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{SampleBuf, SdrSource};

type SoapyDevice = c_void;
type SoapyStream = c_void;

// SoapySDR constants
const SOAPY_SDR_RX: c_int = 0;
const SOAPY_SDR_TIMEOUT: c_int = -1;
const SOAPY_SDR_OVERFLOW: c_int = -4;

// SoapySDRKwargs
#[repr(C)]
struct SoapyKwargs {
    size: usize,
    keys: *mut *mut c_char,
    vals: *mut *mut c_char,
}

extern "C" {
    fn SoapySDRDevice_enumerate(args: *const SoapyKwargs, length: *mut usize) -> *mut SoapyKwargs;
    fn SoapySDRDevice_make(args: *const SoapyKwargs) -> *mut SoapyDevice;
    fn SoapySDRDevice_unmake(device: *mut SoapyDevice) -> c_int;
    fn SoapySDRDevice_setSampleRate(
        dev: *mut SoapyDevice,
        direction: c_int,
        channel: usize,
        rate: c_double,
    ) -> c_int;
    fn SoapySDRDevice_setFrequency(
        dev: *mut SoapyDevice,
        direction: c_int,
        channel: usize,
        frequency: c_double,
        args: *const SoapyKwargs,
    ) -> c_int;
    fn SoapySDRDevice_setGain(
        dev: *mut SoapyDevice,
        direction: c_int,
        channel: usize,
        value: c_double,
    ) -> c_int;
    fn SoapySDRDevice_setBandwidth(
        dev: *mut SoapyDevice,
        direction: c_int,
        channel: usize,
        bw: c_double,
    ) -> c_int;
    fn SoapySDRDevice_setupStream(
        dev: *mut SoapyDevice,
        direction: c_int,
        format: *const c_char,
        channels: *const usize,
        num_chans: usize,
        args: *const SoapyKwargs,
    ) -> *mut SoapyStream;
    fn SoapySDRDevice_activateStream(
        dev: *mut SoapyDevice,
        stream: *mut SoapyStream,
        flags: c_int,
        time_ns: i64,
        num_elems: usize,
    ) -> c_int;
    fn SoapySDRDevice_deactivateStream(
        dev: *mut SoapyDevice,
        stream: *mut SoapyStream,
        flags: c_int,
        time_ns: i64,
    ) -> c_int;
    fn SoapySDRDevice_closeStream(
        dev: *mut SoapyDevice,
        stream: *mut SoapyStream,
    ) -> c_int;
    fn SoapySDRDevice_readStream(
        dev: *mut SoapyDevice,
        stream: *mut SoapyStream,
        buffs: *const *mut c_void,
        num_elems: usize,
        flags: *mut c_int,
        time_ns: *mut i64,
        timeout_us: i64,
    ) -> c_int;
    fn SoapySDRDevice_getStreamMTU(
        dev: *mut SoapyDevice,
        stream: *mut SoapyStream,
    ) -> usize;
    fn SoapySDRDevice_getStreamFormats(
        dev: *mut SoapyDevice,
        direction: c_int,
        channel: usize,
        length: *mut usize,
    ) -> *mut *mut c_char;
    fn SoapySDRDevice_lastError() -> *const c_char;
    fn SoapySDRKwargsList_clear(info: *mut SoapyKwargs, length: usize);
    fn SoapySDRStrings_clear(strings: *mut *mut c_char, length: usize);
}

fn last_error() -> String {
    unsafe {
        let p = SoapySDRDevice_lastError();
        if p.is_null() {
            "unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(p)
                .to_string_lossy()
                .to_string()
        }
    }
}

#[derive(Debug, Clone)]
pub struct SoapyInfo {
    pub index: usize,
    pub driver: String,
    pub label: String,
}

pub fn list_devices() -> Result<Vec<SoapyInfo>, String> {
    let mut length: usize = 0;
    let results = unsafe { SoapySDRDevice_enumerate(ptr::null(), &mut length) };

    if results.is_null() || length == 0 {
        return Ok(Vec::new());
    }

    let mut devices = Vec::new();
    for i in 0..length {
        let kw = unsafe { &*results.add(i) };
        let mut driver = String::new();
        let mut label = String::new();

        for j in 0..kw.size {
            let key = unsafe {
                std::ffi::CStr::from_ptr(*kw.keys.add(j))
                    .to_string_lossy()
                    .to_string()
            };
            let val = unsafe {
                std::ffi::CStr::from_ptr(*kw.vals.add(j))
                    .to_string_lossy()
                    .to_string()
            };
            match key.as_str() {
                "driver" => driver = val,
                "label" => label = val,
                _ => {}
            }
        }

        devices.push(SoapyInfo {
            index: i,
            driver,
            label,
        });
    }

    unsafe { SoapySDRKwargsList_clear(results, length) };
    Ok(devices)
}

fn parse_index(iface: &str) -> Option<usize> {
    // "soapy-0" -> 0
    if iface.starts_with("soapy-") {
        iface[6..].parse().ok()
    } else {
        None
    }
}

/// Check if device supports CS8 format
fn supports_cs8(dev: *mut SoapyDevice) -> bool {
    let mut length: usize = 0;
    let formats =
        unsafe { SoapySDRDevice_getStreamFormats(dev, SOAPY_SDR_RX, 0, &mut length) };

    if formats.is_null() || length == 0 {
        return false;
    }

    let mut found = false;
    for i in 0..length {
        let fmt = unsafe {
            std::ffi::CStr::from_ptr(*formats.add(i))
                .to_string_lossy()
                .to_string()
        };
        if fmt == "CS8" {
            found = true;
            break;
        }
    }

    unsafe { SoapySDRStrings_clear(formats, length) };
    found
}

pub struct SoapySource {
    index: usize,
    sample_rate: u32,
    center_freq: u64,
    gain: f64,
    running: Arc<AtomicBool>,
}

impl SoapySource {
    pub fn new(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: f64,
    ) -> Result<Self, String> {
        let index = parse_index(iface).ok_or_else(|| {
            format!(
                "invalid SoapySDR interface: '{}' (expected soapy-N)",
                iface
            )
        })?;

        Ok(Self {
            index,
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

impl SdrSource for SoapySource {
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String> {
        self.running.store(true, Ordering::SeqCst);

        let mut length: usize = 0;
        let results = unsafe { SoapySDRDevice_enumerate(ptr::null(), &mut length) };
        if results.is_null() || self.index >= length {
            if !results.is_null() {
                unsafe { SoapySDRKwargsList_clear(results, length) };
            }
            return Err(format!(
                "SoapySDR device index {} not found ({} devices)",
                self.index, length
            ));
        }

        let dev = unsafe { SoapySDRDevice_make(results.add(self.index)) };
        unsafe { SoapySDRKwargsList_clear(results, length) };

        if dev.is_null() {
            return Err(format!("SoapySDR make failed: {}", last_error()));
        }

        unsafe {
            SoapySDRDevice_setSampleRate(dev, SOAPY_SDR_RX, 0, self.sample_rate as f64);
            SoapySDRDevice_setFrequency(
                dev,
                SOAPY_SDR_RX,
                0,
                self.center_freq as f64,
                ptr::null(),
            );
            let _ = SoapySDRDevice_setGain(dev, SOAPY_SDR_RX, 0, self.gain);
            let _ = SoapySDRDevice_setBandwidth(dev, SOAPY_SDR_RX, 0, self.sample_rate as f64);
        }

        let use_cs8 = supports_cs8(dev);
        let fmt_str = if use_cs8 {
            CString::new("CS8").unwrap()
        } else {
            CString::new("CS16").unwrap()
        };

        let channel: usize = 0;
        let stream = unsafe {
            SoapySDRDevice_setupStream(
                dev,
                SOAPY_SDR_RX,
                fmt_str.as_ptr(),
                &channel,
                1,
                ptr::null(),
            )
        };
        if stream.is_null() {
            unsafe { SoapySDRDevice_unmake(dev) };
            return Err(format!("SoapySDR setupStream failed: {}", last_error()));
        }

        let mtu = unsafe { SoapySDRDevice_getStreamMTU(dev, stream) };
        let mtu = if mtu == 0 { 65536 } else { mtu };

        let r = unsafe { SoapySDRDevice_activateStream(dev, stream, 0, 0, 0) };
        if r != 0 {
            unsafe {
                SoapySDRDevice_closeStream(dev, stream);
                SoapySDRDevice_unmake(dev);
            }
            return Err(format!("SoapySDR activateStream failed: {}", last_error()));
        }

        log::info!(
            "SoapySDR streaming (index={}, {} MHz, {} MS/s, gain={}, {})",
            self.index,
            self.center_freq / 1_000_000,
            self.sample_rate / 1_000_000,
            self.gain,
            if use_cs8 { "CS8" } else { "CS16" },
        );

        if use_cs8 {
            let mut cs8_buf = vec![0i8; mtu * 2];
            while self.running.load(Ordering::SeqCst) {
                let mut flags: c_int = 0;
                let mut time_ns: i64 = 0;
                let mut buf_ptr = cs8_buf.as_mut_ptr() as *mut c_void;

                let ret = unsafe {
                    SoapySDRDevice_readStream(
                        dev,
                        stream,
                        &mut buf_ptr,
                        mtu,
                        &mut flags,
                        &mut time_ns,
                        100_000,
                    )
                };

                if ret == SOAPY_SDR_TIMEOUT || ret == SOAPY_SDR_OVERFLOW {
                    continue;
                }
                if ret < 0 {
                    log::error!("SoapySDR read error: {}", ret);
                    break;
                }

                let num_samples = ret as usize;
                let mut data = Vec::with_capacity(num_samples * 2);
                for i in 0..num_samples * 2 {
                    data.push((cs8_buf[i] as i16) << 8);
                }

                if tx.send(SampleBuf { data, num_samples }).is_err() {
                    break;
                }
            }
        } else {
            let mut cs16_buf = vec![0i16; mtu * 2];
            while self.running.load(Ordering::SeqCst) {
                let mut flags: c_int = 0;
                let mut time_ns: i64 = 0;
                let mut buf_ptr = cs16_buf.as_mut_ptr() as *mut c_void;

                let ret = unsafe {
                    SoapySDRDevice_readStream(
                        dev,
                        stream,
                        &mut buf_ptr,
                        mtu,
                        &mut flags,
                        &mut time_ns,
                        100_000,
                    )
                };

                if ret == SOAPY_SDR_TIMEOUT || ret == SOAPY_SDR_OVERFLOW {
                    continue;
                }
                if ret < 0 {
                    log::error!("SoapySDR read error: {}", ret);
                    break;
                }

                let num_samples = ret as usize;
                let mut data = Vec::with_capacity(num_samples * 2);
                for i in 0..num_samples * 2 {
                    data.push(cs16_buf[i]);
                }

                if tx.send(SampleBuf { data, num_samples }).is_err() {
                    break;
                }
            }
        }

        unsafe {
            SoapySDRDevice_deactivateStream(dev, stream, 0, 0);
            SoapySDRDevice_closeStream(dev, stream);
            SoapySDRDevice_unmake(dev);
        }

        log::info!("SoapySDR streaming stopped");
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

/// Direct SoapySDR handle for zero-copy recv_into path.
pub struct SoapyHandle {
    dev: *mut SoapyDevice,
    stream: *mut SoapyStream,
    use_cs8: bool,
    mtu: usize,
    cs16_buf: Vec<i16>,
    pub running: Arc<AtomicBool>,
    overflow_count: u64,
}

unsafe impl Send for SoapyHandle {}

impl SoapyHandle {
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        gain: f64,
    ) -> Result<Self, String> {
        let index = parse_index(iface)
            .ok_or_else(|| format!("invalid SoapySDR interface: '{}'", iface))?;

        let mut length: usize = 0;
        let results = unsafe { SoapySDRDevice_enumerate(ptr::null(), &mut length) };
        if results.is_null() || index >= length {
            if !results.is_null() {
                unsafe { SoapySDRKwargsList_clear(results, length) };
            }
            return Err(format!("SoapySDR device index {} not found", index));
        }

        let dev = unsafe { SoapySDRDevice_make(results.add(index)) };
        unsafe { SoapySDRKwargsList_clear(results, length) };

        if dev.is_null() {
            return Err(format!("SoapySDR make failed: {}", last_error()));
        }

        unsafe {
            SoapySDRDevice_setSampleRate(dev, SOAPY_SDR_RX, 0, sample_rate as f64);
            SoapySDRDevice_setFrequency(dev, SOAPY_SDR_RX, 0, center_freq as f64, ptr::null());
            let _ = SoapySDRDevice_setGain(dev, SOAPY_SDR_RX, 0, gain);
            let _ = SoapySDRDevice_setBandwidth(dev, SOAPY_SDR_RX, 0, sample_rate as f64);
        }

        let use_cs8 = supports_cs8(dev);
        let fmt_str = if use_cs8 {
            CString::new("CS8").unwrap()
        } else {
            CString::new("CS16").unwrap()
        };

        let channel: usize = 0;
        let stream = unsafe {
            SoapySDRDevice_setupStream(
                dev,
                SOAPY_SDR_RX,
                fmt_str.as_ptr(),
                &channel,
                1,
                ptr::null(),
            )
        };
        if stream.is_null() {
            unsafe { SoapySDRDevice_unmake(dev) };
            return Err(format!("SoapySDR setupStream failed: {}", last_error()));
        }

        let mtu = unsafe { SoapySDRDevice_getStreamMTU(dev, stream) };
        let mtu = if mtu == 0 { 65536 } else { mtu };

        let r = unsafe { SoapySDRDevice_activateStream(dev, stream, 0, 0, 0) };
        if r != 0 {
            unsafe {
                SoapySDRDevice_closeStream(dev, stream);
                SoapySDRDevice_unmake(dev);
            }
            return Err(format!("SoapySDR activateStream failed: {}", last_error()));
        }

        let cs16_buf = if !use_cs8 {
            vec![0i16; mtu * 2]
        } else {
            Vec::new()
        };

        Ok(Self {
            dev,
            stream,
            use_cs8,
            mtu,
            cs16_buf,
            running: Arc::new(AtomicBool::new(true)),
            overflow_count: 0,
        })
    }

    pub fn recv_into(&mut self, buf: &mut [i8]) -> usize {
        let max_samps = (buf.len() / 2).min(self.mtu);
        let mut flags: c_int = 0;
        let mut time_ns: i64 = 0;

        if self.use_cs8 {
            let mut buf_ptr = buf.as_mut_ptr() as *mut c_void;
            let ret = unsafe {
                SoapySDRDevice_readStream(
                    self.dev,
                    self.stream,
                    &mut buf_ptr,
                    max_samps,
                    &mut flags,
                    &mut time_ns,
                    100_000,
                )
            };

            if ret == SOAPY_SDR_OVERFLOW {
                self.overflow_count += 1;
                return 0;
            }
            if ret < 0 {
                return 0;
            }
            ret as usize
        } else {
            let mut buf_ptr = self.cs16_buf.as_mut_ptr() as *mut c_void;
            let ret = unsafe {
                SoapySDRDevice_readStream(
                    self.dev,
                    self.stream,
                    &mut buf_ptr,
                    max_samps,
                    &mut flags,
                    &mut time_ns,
                    100_000,
                )
            };

            if ret == SOAPY_SDR_OVERFLOW {
                self.overflow_count += 1;
                return 0;
            }
            if ret < 0 {
                return 0;
            }

            let n = ret as usize;
            for i in 0..n * 2 {
                buf[i] = (self.cs16_buf[i] >> 8) as i8;
            }
            n
        }
    }

    pub fn max_samps(&self) -> usize {
        self.mtu
    }

    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }
}

impl Drop for SoapyHandle {
    fn drop(&mut self) {
        unsafe {
            SoapySDRDevice_deactivateStream(self.dev, self.stream, 0, 0);
            SoapySDRDevice_closeStream(self.dev, self.stream);
            SoapySDRDevice_unmake(self.dev);
        }
    }
}
