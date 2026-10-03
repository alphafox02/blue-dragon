// Copyright 2025-2026 CEMAXECUTER LLC

//! Loads a firmware image into an ESP32-S3's RAM through its ROM loader,
//! over the chip's USB Serial/JTAG port: the same steps as `esptool
//! load-ram`, without needing esptool. Nothing is written to flash, so a
//! power cycle restores the board.
//!
//! The port's DTR and RTS lines reset the chip into download mode. The ROM
//! then takes SLIP-framed commands: SYNC, and for each image segment
//! MEM_BEGIN followed by MEM_DATA blocks, then MEM_END, which starts the
//! image at its entry point.

use std::io::{Read, Write};
use std::time::{Duration, Instant};

const SLIP_END: u8 = 0xC0;
const SLIP_ESC: u8 = 0xDB;
const SLIP_ESC_END: u8 = 0xDC;
const SLIP_ESC_ESC: u8 = 0xDD;

const CMD_MEM_BEGIN: u8 = 0x05;
const CMD_MEM_END: u8 = 0x06;
const CMD_MEM_DATA: u8 = 0x07;
const CMD_SYNC: u8 = 0x08;

/// RAM is written in blocks of this many bytes.
const RAM_BLOCK: usize = 0x1800;
const CHECKSUM_SEED: u8 = 0xEF;

const IMAGE_MAGIC: u8 = 0xE9;
/// The image header's chip id for the ESP32-S3.
const CHIP_ID_ESP32S3: u16 = 9;
/// The common header (8 bytes) and the extended header (16 bytes).
const HEADER_BYTES: usize = 24;

/// One piece of an image, loaded at `addr`.
#[derive(Debug, PartialEq)]
pub struct Segment {
    pub addr: u32,
    pub data: Vec<u8>,
}

/// A firmware image as written by `esptool elf2image`.
#[derive(Debug)]
pub struct Image {
    pub entry: u32,
    pub segments: Vec<Segment>,
}

impl Image {
    /// Parses an ESP32-S3 application image (`.bin`).
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES || bytes[0] != IMAGE_MAGIC {
            return Err("not an ESP firmware image (bad magic)".to_string());
        }
        let chip = u16::from_le_bytes([bytes[12], bytes[13]]);
        if chip != CHIP_ID_ESP32S3 {
            return Err(format!("image is for chip id {}, not the ESP32-S3", chip));
        }
        let count = bytes[1] as usize;
        let entry = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let mut segments = Vec::with_capacity(count);
        let mut at = HEADER_BYTES;
        for i in 0..count {
            let head = bytes
                .get(at..at + 8)
                .ok_or_else(|| format!("image ends inside segment {} header", i))?;
            let addr = u32::from_le_bytes(head[0..4].try_into().unwrap());
            let len = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
            let data = bytes
                .get(at + 8..at + 8 + len)
                .ok_or_else(|| format!("image ends inside segment {}", i))?;
            segments.push(Segment { addr, data: data.to_vec() });
            at += 8 + len;
        }
        Ok(Self { entry, segments })
    }
}

fn slip_encode(packet: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packet.len() + 2);
    out.push(SLIP_END);
    for &b in packet {
        match b {
            SLIP_END => out.extend_from_slice(&[SLIP_ESC, SLIP_ESC_END]),
            SLIP_ESC => out.extend_from_slice(&[SLIP_ESC, SLIP_ESC_ESC]),
            _ => out.push(b),
        }
    }
    out.push(SLIP_END);
    out
}

fn checksum(data: &[u8]) -> u32 {
    data.iter().fold(CHECKSUM_SEED, |s, &b| s ^ b) as u32
}

/// A command packet: direction 0, opcode, payload length, checksum, payload.
fn command_packet(op: u8, payload: &[u8], check: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity(8 + payload.len());
    p.push(0);
    p.push(op);
    p.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    p.extend_from_slice(&check.to_le_bytes());
    p.extend_from_slice(payload);
    p
}

struct Rom {
    port: Box<dyn serialport::SerialPort>,
    /// Bytes of a frame being received.
    frame: Vec<u8>,
    in_frame: bool,
    escaped: bool,
}

impl Rom {
    /// Reads one SLIP frame, or None if `timeout` passes first.
    fn read_frame(&mut self, timeout: Duration) -> Option<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        let mut byte = [0u8; 1];
        while Instant::now() < deadline {
            match self.port.read(&mut byte) {
                Ok(1) => {}
                _ => continue,
            }
            let b = byte[0];
            if !self.in_frame {
                if b == SLIP_END {
                    self.in_frame = true;
                    self.frame.clear();
                }
                continue;
            }
            if self.escaped {
                self.escaped = false;
                self.frame.push(match b {
                    SLIP_ESC_END => SLIP_END,
                    SLIP_ESC_ESC => SLIP_ESC,
                    other => other,
                });
            } else if b == SLIP_ESC {
                self.escaped = true;
            } else if b == SLIP_END {
                if self.frame.is_empty() {
                    continue; // back-to-back delimiters
                }
                self.in_frame = false;
                return Some(std::mem::take(&mut self.frame));
            } else {
                self.frame.push(b);
            }
        }
        None
    }

    /// Sends a command and returns the data of the matching response,
    /// whose first byte is the ROM's status (0 = success).
    fn command(&mut self, op: u8, payload: &[u8], check: u32, timeout: Duration) -> Result<Vec<u8>, String> {
        self.port
            .write_all(&slip_encode(&command_packet(op, payload, check)))
            .map_err(|e| format!("write failed: {}", e))?;
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            let Some(frame) = self.read_frame(deadline - Instant::now()) else { break };
            // Response: direction 1, opcode, length, value, data.
            if frame.len() >= 8 && frame[0] == 1 && frame[1] == op {
                return Ok(frame[8..].to_vec());
            }
        }
        Err(format!("no response to ROM command {:#04x}", op))
    }

    fn check(&mut self, what: &str, op: u8, payload: &[u8], check: u32) -> Result<(), String> {
        let data = self.command(op, payload, check, Duration::from_secs(3))?;
        match data.first() {
            Some(0) => Ok(()),
            Some(_) => Err(format!("{} failed (ROM status {:02x?})", what, &data[..data.len().min(2)])),
            None => Err(format!("{} failed (empty response)", what)),
        }
    }

    /// Resets the chip into its ROM loader with the USB Serial/JTAG port's
    /// DTR and RTS lines (esptool's sequence for this peripheral).
    fn reset_to_download(&mut self) -> Result<(), String> {
        let step = Duration::from_millis(100);
        let mut set = |dtr: bool, rts: bool| -> Result<(), String> {
            self.port.write_request_to_send(rts).map_err(|e| e.to_string())?;
            self.port.write_data_terminal_ready(dtr).map_err(|e| e.to_string())
        };
        set(false, false)?; // idle
        std::thread::sleep(step);
        set(true, false)?; // IO0 low
        std::thread::sleep(step);
        // Into reset through (1, 1), not (0, 0).
        self.port.write_request_to_send(true).map_err(|e| e.to_string())?;
        self.port.write_data_terminal_ready(false).map_err(|e| e.to_string())?;
        self.port.write_request_to_send(true).map_err(|e| e.to_string())?;
        std::thread::sleep(step);
        self.port.write_data_terminal_ready(false).map_err(|e| e.to_string())?;
        self.port.write_request_to_send(false).map_err(|e| e.to_string())?; // out of reset
        Ok(())
    }

    fn sync(&mut self) -> Result<(), String> {
        let mut payload = vec![0x07, 0x07, 0x12, 0x20];
        payload.extend_from_slice(&[0x55; 32]);
        for _ in 0..5 {
            let _ = self.port.clear(serialport::ClearBuffer::Input);
            self.in_frame = false;
            if self.command(CMD_SYNC, &payload, 0, Duration::from_millis(100)).is_ok() {
                // The ROM answers SYNC several times; let the rest arrive and
                // drop it.
                std::thread::sleep(Duration::from_millis(100));
                let _ = self.port.clear(serialport::ClearBuffer::Input);
                self.in_frame = false;
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Err("no answer from the ROM loader".to_string())
    }
}

/// Resets the ESP32-S3 at `path` into its ROM loader, writes `image` to RAM
/// and starts it. The firmware needs about a second after this to bring up
/// its radio and answer.
pub fn load_ram(path: &str, image: &Image) -> Result<(), String> {
    let port = serialport::new(path, 115_200)
        .timeout(Duration::from_millis(20))
        .open()
        .map_err(|e| format!("cannot open {}: {}", path, e))?;
    let mut rom = Rom { port, frame: Vec::new(), in_frame: false, escaped: false };
    let mut synced = Err(String::new());
    for _ in 0..3 {
        rom.reset_to_download()?;
        synced = rom.sync();
        if synced.is_ok() {
            break;
        }
    }
    synced.map_err(|e| format!("{} on {}", e, path))?;

    for seg in &image.segments {
        let blocks = seg.data.len().div_ceil(RAM_BLOCK) as u32;
        let mut begin = Vec::with_capacity(16);
        for v in [seg.data.len() as u32, blocks, RAM_BLOCK as u32, seg.addr] {
            begin.extend_from_slice(&v.to_le_bytes());
        }
        rom.check("starting a RAM download", CMD_MEM_BEGIN, &begin, 0)?;
        for (seq, block) in seg.data.chunks(RAM_BLOCK).enumerate() {
            let mut payload = Vec::with_capacity(16 + block.len());
            for v in [block.len() as u32, seq as u32, 0, 0] {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            payload.extend_from_slice(block);
            rom.check("writing RAM", CMD_MEM_DATA, &payload, checksum(block))?;
        }
    }
    // Run from the entry point. The ROM may not finish answering before the
    // firmware takes over the port, so the answer is not required.
    let mut end = Vec::with_capacity(8);
    end.extend_from_slice(&u32::from(image.entry == 0).to_le_bytes());
    end.extend_from_slice(&image.entry.to_le_bytes());
    let _ = rom.command(CMD_MEM_END, &end, 0, Duration::from_millis(50));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slip_escapes_the_delimiter_and_the_escape() {
        assert_eq!(slip_encode(&[1, 0xC0, 2, 0xDB]), vec![0xC0, 1, 0xDB, 0xDC, 2, 0xDB, 0xDD, 0xC0]);
    }

    #[test]
    fn data_checksum_is_seeded_with_0xef() {
        assert_eq!(checksum(&[]), 0xEF);
        assert_eq!(checksum(&[0xEF]), 0);
        assert_eq!(checksum(&[0x01, 0x02]), 0xEC);
    }

    #[test]
    fn command_packet_layout() {
        assert_eq!(command_packet(CMD_SYNC, &[0xAA, 0xBB], 0x1234_5678), vec![0, 8, 2, 0, 0x78, 0x56, 0x34, 0x12, 0xAA, 0xBB]);
    }

    fn image(chip: u16, segments: &[(u32, &[u8])]) -> Vec<u8> {
        let mut b = vec![IMAGE_MAGIC, segments.len() as u8, 2, 0];
        b.extend_from_slice(&0x4038_7474u32.to_le_bytes());
        let mut ext = [0u8; 16];
        ext[4..6].copy_from_slice(&chip.to_le_bytes());
        b.extend_from_slice(&ext);
        for (addr, data) in segments {
            b.extend_from_slice(&addr.to_le_bytes());
            b.extend_from_slice(&(data.len() as u32).to_le_bytes());
            b.extend_from_slice(data);
        }
        b.extend_from_slice(&[0; 16]); // checksum padding, ignored
        b
    }

    #[test]
    fn parses_an_esp32s3_image() {
        let img = Image::parse(&image(9, &[(0x3FCA_0000, &[1, 2, 3]), (0x4037_4000, &[4; 5])])).unwrap();
        assert_eq!(img.entry, 0x4038_7474);
        assert_eq!(
            img.segments,
            vec![
                Segment { addr: 0x3FCA_0000, data: vec![1, 2, 3] },
                Segment { addr: 0x4037_4000, data: vec![4; 5] },
            ]
        );
    }

    #[test]
    fn rejects_other_chips_and_truncated_images() {
        assert!(Image::parse(&image(5, &[(0, &[1])])).unwrap_err().contains("chip id 5"));
        let mut cut = image(9, &[(0, &[1, 2, 3, 4])]);
        cut.truncate(HEADER_BYTES + 8 + 2);
        assert!(Image::parse(&cut).is_err());
        assert!(Image::parse(b"not an image at all, not an image").is_err());
    }
}
