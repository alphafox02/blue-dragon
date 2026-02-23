// Copyright 2025-2026 CEMAXECUTER LLC

use crossbeam::channel::{bounded, Receiver, Sender};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{SampleBuf, SdrSource};

const HACKRF_SUCCESS: c_int = 0;

#[repr(C)]
struct HackrfDeviceList {
    serial_numbers: *mut *mut c_char,
    usb_board_ids: *mut c_int,
    usb_device_index: *mut c_int,
    devicecount: c_int,
    usb_devices: *mut *mut c_void,
    usb_devicecount: c_int,
}

#[repr(C)]
pub struct HackrfTransfer {
    pub device: *mut c_void,
    pub buffer: *mut u8,
    pub buffer_length: i32,
    pub valid_length: i32,
    pub rx_ctx: *mut c_void,
    pub tx_ctx: *mut c_void,
}

type HackrfDevice = c_void;

extern "C" {
    fn hackrf_init() -> c_int;
    fn hackrf_exit() -> c_int;
    fn hackrf_open(device: *mut *mut HackrfDevice) -> c_int;
    fn hackrf_open_by_serial(
        desired_serial_number: *const c_char,
        device: *mut *mut HackrfDevice,
    ) -> c_int;
    fn hackrf_close(device: *mut HackrfDevice) -> c_int;
    fn hackrf_set_sample_rate(device: *mut HackrfDevice, freq_hz: f64) -> c_int;
    fn hackrf_set_freq(device: *mut HackrfDevice, freq_hz: u64) -> c_int;
    fn hackrf_set_vga_gain(device: *mut HackrfDevice, value: u32) -> c_int;
    fn hackrf_set_lna_gain(device: *mut HackrfDevice, value: u32) -> c_int;
    fn hackrf_start_rx(
        device: *mut HackrfDevice,
        callback: unsafe extern "C" fn(*mut HackrfTransfer) -> c_int,
        rx_ctx: *mut c_void,
    ) -> c_int;
    fn hackrf_stop_rx(device: *mut HackrfDevice) -> c_int;
    fn hackrf_device_list() -> *mut HackrfDeviceList;
    fn hackrf_device_list_free(list: *mut HackrfDeviceList);
}

/// Information about a detected HackRF device
#[derive(Debug, Clone)]
pub struct HackrfInfo {
    pub serial: String,
}

/// List all available HackRF devices
pub fn list_devices() -> Result<Vec<HackrfInfo>, String> {
    unsafe {
        let r = hackrf_init();
        if r != HACKRF_SUCCESS {
            return Err(format!("hackrf_init failed: {}", r));
        }

        let list = hackrf_device_list();
        if list.is_null() {
            hackrf_exit();
            return Err("hackrf_device_list returned null".to_string());
        }

        let count = (*list).devicecount as usize;
        let mut devices = Vec::with_capacity(count);

        for i in 0..count {
            let serial_ptr = *(*list).serial_numbers.add(i);
            if serial_ptr.is_null() {
                continue;
            }
            let serial_full = std::ffi::CStr::from_ptr(serial_ptr)
                .to_string_lossy()
                .to_string();
            // Strip leading zeros from serial
            let serial = serial_full.trim_start_matches('0').to_string();
            devices.push(HackrfInfo { serial });
        }

        hackrf_device_list_free(list);
        hackrf_exit();
        Ok(devices)
    }
}

/// Extract serial from interface string like "hackrf-SERIAL"
fn parse_serial(iface: &str) -> Option<String> {
    let parts: Vec<&str> = iface.splitn(2, '-').collect();
    if parts.len() == 2 && parts[0] == "hackrf" {
        Some(parts[1].to_string())
    } else {
        None
    }
}

/// Context passed to the HackRF RX callback
struct RxContext {
    tx: Sender<SampleBuf>,
}

unsafe extern "C" fn rx_callback(transfer: *mut HackrfTransfer) -> c_int {
    let ctx = &*((*transfer).rx_ctx as *const RxContext);
    let valid = (*transfer).valid_length as usize;
    let buffer = (*transfer).buffer;

    // HackRF delivers unsigned bytes; reinterpret as i8 IQ pairs
    let num_samples = valid / 2;
    let mut data = Vec::with_capacity(num_samples * 2);
    for i in 0..num_samples * 2 {
        let sample = *(buffer.add(i) as *const i8);
        data.push((sample as i16) << 8);
    }

    let _ = ctx.tx.try_send(SampleBuf {
        data,
        num_samples,
    });

    0
}

/// Context for the zero-copy HackRF handle callback
struct HandleRxContext {
    tx: Sender<Vec<i8>>,
}

unsafe extern "C" fn handle_rx_callback(transfer: *mut HackrfTransfer) -> c_int {
    let ctx = &*((*transfer).rx_ctx as *const HandleRxContext);
    let valid = (*transfer).valid_length as usize;
    let buffer = (*transfer).buffer;

    let mut chunk = Vec::with_capacity(valid);
    for i in 0..valid {
        chunk.push(*(buffer.add(i) as *const i8));
    }

    let _ = ctx.tx.try_send(chunk);

    0
}

/// HackRF SDR source using the libhackrf C API
pub struct HackrfSource {
    serial: Option<String>,
    sample_rate: u32,
    center_freq: u64,
    lna_gain: u32,
    vga_gain: u32,
    running: Arc<AtomicBool>,
}

impl HackrfSource {
    pub fn new(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        lna_gain: u32,
        vga_gain: u32,
    ) -> Result<Self, String> {
        if sample_rate > 20_000_000 {
            return Err("HackRF sample rate must be 20 MHz or less".to_string());
        }

        let serial = if iface.is_empty() || iface == "hackrf" {
            None
        } else {
            Some(
                parse_serial(iface)
                    .ok_or_else(|| {
                        format!(
                            "invalid HackRF interface: '{}' (expected hackrf-SERIAL)",
                            iface
                        )
                    })?,
            )
        };

        Ok(Self {
            serial,
            sample_rate,
            center_freq,
            lna_gain,
            vga_gain,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }
}

impl SdrSource for HackrfSource {
    fn start(&mut self, tx: Sender<SampleBuf>) -> Result<(), String> {
        self.running.store(true, Ordering::SeqCst);

        unsafe {
            let r = hackrf_init();
            if r != HACKRF_SUCCESS {
                return Err(format!("hackrf_init failed: {}", r));
            }

            let mut dev: *mut HackrfDevice = ptr::null_mut();

            let r = if let Some(ref serial) = self.serial {
                let cs = CString::new(serial.as_str())
                    .map_err(|e| format!("CString error: {}", e))?;
                hackrf_open_by_serial(cs.as_ptr(), &mut dev)
            } else {
                hackrf_open(&mut dev)
            };

            if r != HACKRF_SUCCESS {
                hackrf_exit();
                return Err(format!("hackrf_open failed: {}", r));
            }

            log::info!("HackRF opened (serial={:?})", self.serial);

            let r = hackrf_set_sample_rate(dev, self.sample_rate as f64);
            if r != HACKRF_SUCCESS {
                hackrf_close(dev);
                hackrf_exit();
                return Err(format!("hackrf_set_sample_rate failed: {}", r));
            }

            let r = hackrf_set_freq(dev, self.center_freq);
            if r != HACKRF_SUCCESS {
                hackrf_close(dev);
                hackrf_exit();
                return Err(format!("hackrf_set_freq failed: {}", r));
            }

            let r = hackrf_set_lna_gain(dev, self.lna_gain);
            if r != HACKRF_SUCCESS {
                hackrf_close(dev);
                hackrf_exit();
                return Err(format!("hackrf_set_lna_gain failed: {}", r));
            }

            let r = hackrf_set_vga_gain(dev, self.vga_gain);
            if r != HACKRF_SUCCESS {
                hackrf_close(dev);
                hackrf_exit();
                return Err(format!("hackrf_set_vga_gain failed: {}", r));
            }

            // Allocate the RX context on the heap -- leaked intentionally (freed after stop)
            let ctx = Box::into_raw(Box::new(RxContext { tx }));

            let r = hackrf_start_rx(dev, rx_callback, ctx as *mut c_void);
            if r != HACKRF_SUCCESS {
                let _ = Box::from_raw(ctx);
                hackrf_close(dev);
                hackrf_exit();
                return Err(format!("hackrf_start_rx failed: {}", r));
            }

            log::info!(
                "HackRF streaming started ({} MHz, {} MS/s, LNA={} dB, VGA={} dB)",
                self.center_freq / 1_000_000,
                self.sample_rate / 1_000_000,
                self.lna_gain,
                self.vga_gain,
            );

            // Block until stopped
            while self.running.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            hackrf_stop_rx(dev);
            let _ = Box::from_raw(ctx);
            hackrf_close(dev);
            hackrf_exit();

            log::info!("HackRF streaming stopped");
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

/// Opened HackRF device handle for direct recv_into() calls (GPU path).
/// Since HackRF is callback-based, internally uses a crossbeam bounded channel:
/// the callback pushes raw i8 chunks, recv_into copies from the channel.
pub struct HackrfHandle {
    dev: *mut HackrfDevice,
    ctx: *mut HandleRxContext,
    rx: Receiver<Vec<i8>>,
    pending: Vec<i8>,
    pending_offset: usize,
    max_samps: usize,
    pub running: Arc<AtomicBool>,
    overflow_count: u64,
}

// HackRF device pointer is thread-safe (single owner)
unsafe impl Send for HackrfHandle {}

impl HackrfHandle {
    /// Open HackRF and start streaming. Returns a handle for recv_into() calls.
    pub fn open(
        iface: &str,
        sample_rate: u32,
        center_freq: u64,
        lna_gain: u32,
        vga_gain: u32,
    ) -> Result<Self, String> {
        if sample_rate > 20_000_000 {
            return Err("HackRF sample rate must be 20 MHz or less".to_string());
        }

        let serial = if iface.is_empty() || iface == "hackrf" {
            None
        } else {
            Some(parse_serial(iface).ok_or_else(|| {
                format!("invalid HackRF interface: '{}'", iface)
            })?)
        };

        // Bounded channel: 64 chunks should be enough backpressure
        let (tx, rx) = bounded::<Vec<i8>>(64);

        unsafe {
            let r = hackrf_init();
            if r != HACKRF_SUCCESS {
                return Err(format!("hackrf_init failed: {}", r));
            }

            let mut dev: *mut HackrfDevice = ptr::null_mut();

            let r = if let Some(ref s) = serial {
                let cs = CString::new(s.as_str())
                    .map_err(|e| format!("CString error: {}", e))?;
                hackrf_open_by_serial(cs.as_ptr(), &mut dev)
            } else {
                hackrf_open(&mut dev)
            };

            if r != HACKRF_SUCCESS {
                hackrf_exit();
                return Err(format!("hackrf_open failed: {}", r));
            }

            hackrf_set_sample_rate(dev, sample_rate as f64);
            hackrf_set_freq(dev, center_freq);
            hackrf_set_lna_gain(dev, lna_gain);
            hackrf_set_vga_gain(dev, vga_gain);

            let ctx = Box::into_raw(Box::new(HandleRxContext { tx }));

            let r = hackrf_start_rx(dev, handle_rx_callback, ctx as *mut c_void);
            if r != HACKRF_SUCCESS {
                let _ = Box::from_raw(ctx);
                hackrf_close(dev);
                hackrf_exit();
                return Err(format!("hackrf_start_rx failed: {}", r));
            }

            // HackRF delivers 262144 bytes per transfer by default (131072 complex samples)
            let max_samps = 131072;

            Ok(Self {
                dev,
                ctx,
                rx,
                pending: Vec::new(),
                pending_offset: 0,
                max_samps,
                running: Arc::new(AtomicBool::new(true)),
                overflow_count: 0,
            })
        }
    }

    /// Receive SC8 samples directly into a raw i8 buffer.
    /// `buf`: destination buffer (i8 IQ pairs).
    /// Returns: number of complex samples received, or 0 on timeout/error.
    pub fn recv_into(&mut self, buf: &mut [i8]) -> usize {
        let max_bytes = buf.len();
        let mut written = 0;

        // Drain any pending data from previous chunk
        while written < max_bytes && self.pending_offset < self.pending.len() {
            buf[written] = self.pending[self.pending_offset];
            written += 1;
            self.pending_offset += 1;
        }

        if written >= max_bytes {
            return written / 2;
        }

        // Try to receive a new chunk
        match self.rx.recv_timeout(std::time::Duration::from_secs(3)) {
            Ok(chunk) => {
                let avail = chunk.len().min(max_bytes - written);
                buf[written..written + avail].copy_from_slice(&chunk[..avail]);
                written += avail;

                if avail < chunk.len() {
                    self.pending = chunk;
                    self.pending_offset = avail;
                } else {
                    self.pending.clear();
                    self.pending_offset = 0;
                }
            }
            Err(_) => {
                // Timeout or disconnected
            }
        }

        written / 2
    }

    /// Set LNA and VGA gains at runtime (thread-safe FFI calls).
    pub fn set_gain(&self, lna: u32, vga: u32) {
        unsafe {
            hackrf_set_lna_gain(self.dev, lna);
            hackrf_set_vga_gain(self.dev, vga);
        }
    }

    pub fn max_samps(&self) -> usize {
        self.max_samps
    }

    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }
}

impl Drop for HackrfHandle {
    fn drop(&mut self) {
        unsafe {
            hackrf_stop_rx(self.dev);
            let _ = Box::from_raw(self.ctx);
            hackrf_close(self.dev);
            hackrf_exit();
        }

        if self.overflow_count > 0 {
            eprintln!("HackRF: {} overflows during capture", self.overflow_count);
        }
    }
}
