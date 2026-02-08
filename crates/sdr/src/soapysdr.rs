// Copyright 2025 CEMAXECUTER LLC

use crossbeam::channel::Sender;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{SampleBuf, SdrSource};

// All SoapySDR calls go through the C shim (csrc/soapy_shim.c) to avoid
// an ABI incompatibility between Rust FFI and the SoapyUHD module that
// causes segfaults in readStream when setupStream is called via Rust FFI.
extern "C" {
    fn soapy_shim_make(args: *const c_char) -> *mut c_void;
    fn soapy_shim_unmake(dev: *mut c_void);
    fn soapy_shim_set_sample_rate(dev: *mut c_void, rate: c_double) -> c_int;
    fn soapy_shim_set_frequency(dev: *mut c_void, freq: c_double) -> c_int;
    fn soapy_shim_set_gain(dev: *mut c_void, gain: c_double) -> c_int;
    fn soapy_shim_set_bandwidth(dev: *mut c_void, bw: c_double) -> c_int;
    fn soapy_shim_setup_stream(dev: *mut c_void, format: *const c_char) -> *mut c_void;
    fn soapy_shim_activate(dev: *mut c_void, stream: *mut c_void) -> c_int;
    fn soapy_shim_deactivate(dev: *mut c_void, stream: *mut c_void);
    fn soapy_shim_close(dev: *mut c_void, stream: *mut c_void);
    fn soapy_shim_get_mtu(dev: *mut c_void, stream: *mut c_void) -> usize;
    fn soapy_shim_read(
        dev: *mut c_void,
        stream: *mut c_void,
        buf: *mut c_void,
        num_samples: usize,
    ) -> c_int;
    fn soapy_shim_get_full_scale(dev: *mut c_void) -> c_double;
    fn soapy_shim_enumerate(
        labels: *mut [c_char; 64],
        drivers: *mut [c_char; 32],
        max_devices: usize,
    ) -> usize;
}

const SOAPY_SDR_TIMEOUT: c_int = -1;
const SOAPY_SDR_OVERFLOW: c_int = -4;

#[derive(Debug, Clone)]
pub struct SoapyInfo {
    pub index: usize,
    pub driver: String,
    pub label: String,
}

pub fn list_devices() -> Result<Vec<SoapyInfo>, String> {
    const MAX_DEVS: usize = 16;
    let mut labels = [[0 as c_char; 64]; MAX_DEVS];
    let mut drivers = [[0 as c_char; 32]; MAX_DEVS];

    let count =
        unsafe { soapy_shim_enumerate(labels.as_mut_ptr(), drivers.as_mut_ptr(), MAX_DEVS) };

    let mut devices = Vec::new();
    for i in 0..count {
        let label = unsafe {
            std::ffi::CStr::from_ptr(labels[i].as_ptr())
                .to_string_lossy()
                .to_string()
        };
        let driver = unsafe {
            std::ffi::CStr::from_ptr(drivers[i].as_ptr())
                .to_string_lossy()
                .to_string()
        };
        devices.push(SoapyInfo {
            index: i,
            driver,
            label,
        });
    }

    Ok(devices)
}

fn parse_index(iface: &str) -> Option<usize> {
    if iface.starts_with("soapy-") {
        iface[6..].parse().ok()
    } else {
        None
    }
}

/// Open a SoapySDR device via the C shim. Returns (dev, stream, use_cs8, mtu, cs16_shift).
/// cs16_shift is the right-shift amount to convert CS16 samples to i8 range.
fn open_device(
    index: usize,
    sample_rate: u32,
    center_freq: u64,
    gain: f64,
) -> Result<(*mut c_void, *mut c_void, bool, usize, u32), String> {
    // Enumerate to get the driver name for this index
    const MAX_DEVS: usize = 16;
    let mut labels = [[0 as c_char; 64]; MAX_DEVS];
    let mut drivers = [[0 as c_char; 32]; MAX_DEVS];
    let count =
        unsafe { soapy_shim_enumerate(labels.as_mut_ptr(), drivers.as_mut_ptr(), MAX_DEVS) };

    if index >= count {
        return Err(format!(
            "SoapySDR device index {} not found ({} devices)",
            index, count
        ));
    }

    let driver = unsafe {
        std::ffi::CStr::from_ptr(drivers[index].as_ptr())
            .to_string_lossy()
            .to_string()
    };

    let args = CString::new(format!("driver={}", driver))
        .map_err(|e| format!("CString error: {}", e))?;
    let dev = unsafe { soapy_shim_make(args.as_ptr()) };
    if dev.is_null() {
        return Err("SoapySDR make failed".to_string());
    }

    unsafe {
        soapy_shim_set_sample_rate(dev, sample_rate as f64);
        soapy_shim_set_frequency(dev, center_freq as f64);
        soapy_shim_set_gain(dev, gain);
        soapy_shim_set_bandwidth(dev, sample_rate as f64);
    }

    // Query native full-scale to compute CS16->i8 shift amount.
    // bladeRF SC16_Q11: fullScale=2048, shift=4. USRP int16: fullScale=32768, shift=8.
    let full_scale = unsafe { soapy_shim_get_full_scale(dev) };
    let cs16_shift = (full_scale.log2() as u32).saturating_sub(7);

    // Try CS16 first (native for most devices), fall back to CS8
    let cs16 = CString::new("CS16").unwrap();
    let cs8 = CString::new("CS8").unwrap();

    let (stream, use_cs8) = unsafe {
        let s = soapy_shim_setup_stream(dev, cs16.as_ptr());
        if !s.is_null() {
            (s, false)
        } else {
            let s = soapy_shim_setup_stream(dev, cs8.as_ptr());
            if !s.is_null() {
                (s, true)
            } else {
                soapy_shim_unmake(dev);
                return Err("SoapySDR setupStream failed (tried CS16, CS8)".to_string());
            }
        }
    };

    let mtu = unsafe { soapy_shim_get_mtu(dev, stream) };
    let mtu = if mtu == 0 || mtu > 262144 { 65536 } else { mtu };

    let r = unsafe { soapy_shim_activate(dev, stream) };
    if r != 0 {
        unsafe {
            soapy_shim_close(dev, stream);
            soapy_shim_unmake(dev);
        }
        return Err("SoapySDR activateStream failed".to_string());
    }

    log::info!(
        "SoapySDR open (index={}, driver={}, {} MHz, {} MS/s, gain={}, {}, mtu={}, fullScale={})",
        index,
        driver,
        center_freq / 1_000_000,
        sample_rate / 1_000_000,
        gain,
        if use_cs8 { "CS8" } else { "CS16" },
        mtu,
        full_scale,
    );

    Ok((dev, stream, use_cs8, mtu, cs16_shift))
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

        let (dev, stream, use_cs8, mtu, _cs16_shift) =
            open_device(self.index, self.sample_rate, self.center_freq, self.gain)?;

        if use_cs8 {
            let mut cs8_buf = vec![0i8; mtu * 2];
            while self.running.load(Ordering::SeqCst) {
                let ret = unsafe {
                    soapy_shim_read(dev, stream, cs8_buf.as_mut_ptr() as *mut c_void, mtu)
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
                let ret = unsafe {
                    soapy_shim_read(dev, stream, cs16_buf.as_mut_ptr() as *mut c_void, mtu)
                };

                if ret == SOAPY_SDR_TIMEOUT || ret == SOAPY_SDR_OVERFLOW {
                    continue;
                }
                if ret < 0 {
                    log::error!("SoapySDR read error: {}", ret);
                    break;
                }

                let num_samples = ret as usize;
                let data = cs16_buf[..num_samples * 2].to_vec();

                if tx.send(SampleBuf { data, num_samples }).is_err() {
                    break;
                }
            }
        }

        unsafe {
            soapy_shim_deactivate(dev, stream);
            soapy_shim_close(dev, stream);
            soapy_shim_unmake(dev);
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
    dev: *mut c_void,
    stream: *mut c_void,
    use_cs8: bool,
    mtu: usize,
    cs16_buf: Vec<i16>,
    cs16_shift: u32,
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

        let (dev, stream, use_cs8, mtu, cs16_shift) =
            open_device(index, sample_rate, center_freq, gain)?;

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
            cs16_shift,
            running: Arc::new(AtomicBool::new(true)),
            overflow_count: 0,
        })
    }

    pub fn recv_into(&mut self, buf: &mut [i8]) -> usize {
        let max_samps = (buf.len() / 2).min(self.mtu);

        if self.use_cs8 {
            let ret = unsafe {
                soapy_shim_read(
                    self.dev,
                    self.stream,
                    buf.as_mut_ptr() as *mut c_void,
                    max_samps,
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
            let ret = unsafe {
                soapy_shim_read(
                    self.dev,
                    self.stream,
                    self.cs16_buf.as_mut_ptr() as *mut c_void,
                    max_samps,
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
            let shift = self.cs16_shift;
            for i in 0..n * 2 {
                buf[i] = (self.cs16_buf[i] >> shift) as i8;
            }
            n
        }
    }

    pub fn set_gain(&self, gain: f64) {
        unsafe {
            soapy_shim_set_gain(self.dev, gain);
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
            soapy_shim_deactivate(self.dev, self.stream);
            soapy_shim_close(self.dev, self.stream);
            soapy_shim_unmake(self.dev);
        }
    }
}
