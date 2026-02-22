# Blue Dragon

Wideband BLE and Classic Bluetooth (BR) passive sniffer written in Rust.
Captures packets across multiple channels simultaneously using a polyphase
channelizer. Wireshark-compatible PCAP output with optional ZMQ streaming,
GPS tagging, and remote sensor management.

## Supported Hardware

| SDR | Interface Flag | Bandwidth | Notes |
|-----|---------------|-----------|-------|
| USRP (B200/B210) | `-i usrp-MODEL-SERIAL` | 4-56 MHz | Best performance |
| HackRF One | `-i hackrf-SERIAL` | 4-20 MHz | 20 MHz max sample rate |
| bladeRF 2.0 | `-i bladerf0` | 4-96 MHz | Full-band capable |
| SoapySDR | `-i soapy-N` | Varies | Generic SDR support |

To list available SDR devices:

    blue-dragon --list

## Building

### DragonOS

DragonOS includes most SDR libraries. Install the Rust toolchain and
build dependencies:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env
    sudo apt install libzmq3-dev pkg-config

Build with USRP, ZMQ, and GPS support (recommended):

    cd blue-dragon
    cargo build --release --features "usrp,zmq,gps"

Build with additional SDR backends:

    cargo build --release --features "usrp,hackrf,bladerf,soapysdr,zmq,gps"

The binary is at `target/release/blue-dragon`.

### Debian / Ubuntu

    sudo apt install build-essential pkg-config
    sudo apt install libuhd-dev libhackrf-dev libbladerf-dev libsoapysdr-dev
    sudo apt install libzmq3-dev

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env

    cd blue-dragon
    cargo build --release --features "usrp,hackrf,bladerf,soapysdr,zmq,gps"

Optional GPU acceleration (OpenCL):

    sudo apt install ocl-icd-opencl-dev
    cargo build --release --features "usrp,zmq,gps,gpu"

### macOS (Homebrew)

    brew install uhd hackrf libbladerf soapysdr zeromq pkg-config
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env

    cd blue-dragon
    cargo build --release --features "usrp,hackrf,bladerf,soapysdr,zmq"

GPU acceleration on macOS requires Metal (future work -- OpenCL is
deprecated on macOS and Metal backend is not yet implemented).

### Feature Flags

Features are opt-in. Build only what you need:

| Feature | Description | System Dependency |
|---------|-------------|-------------------|
| `usrp` | USRP B200/B210 support (default) | libuhd-dev |
| `hackrf` | HackRF One support | libhackrf-dev |
| `bladerf` | bladeRF 2.0 support | libbladerf-dev |
| `soapysdr` | SoapySDR generic support | libsoapysdr-dev |
| `zmq` | ZMQ packet streaming + C2 | libzmq3-dev |
| `gps` | GPS tagging via gpsd | (no C lib -- uses TCP JSON) |
| `gpu` | OpenCL GPU acceleration | ocl-icd-opencl-dev |

## Usage

### Command-Line Options

```
Input (pick one):
    -f, --file FILE         Read input from IQ file
    -l, --live              Capture live from SDR

SDR settings:
    -i, --interface IFACE   SDR device (e.g. usrp-B210-FCO2P05)
    -c, --center-freq FREQ  Center frequency in MHz (default: 2441)
    -C, --channels N        Number of channels (default: 40)
    -g, --gain DB           SDR gain (default: 60)
    --hackrf-lna DB         HackRF LNA gain (default: 40)
    --hackrf-vga DB         HackRF VGA gain (default: 20)

Output:
    -w, --write FILE        Output PCAP to file or FIFO
    --check-crc             Enable BLE CRC-24 validation
    --stats                 Print performance statistics
    -v, --verbose           Verbose output

Network streaming:
    -Z, --zmq ENDPOINT     Stream to collector (e.g. tcp://collector:5555)
    --zmq-curve-key FILE   CurveZMQ encryption keyfile
    --sensor-id NAME       Sensor identity for multi-sensor deployments

GPS:
    --gpsd                  Tag packets with GPS from gpsd

IQ file options:
    --format FORMAT         Sample format: ci8, ci16, cf32 (default: ci16)

GPU:
    --no-gpu                Disable GPU acceleration (CPU-only)

Wireshark:
    --install               Install as Wireshark extcap plugin
    --list                  List available SDR interfaces
```

### Examples

Capture 40 channels centered on 2441 MHz using a USRP B210:

    blue-dragon -l -i usrp-B210-FCO2P05 -c 2441 -C 40 -w capture.pcap

Capture with CRC validation and stats:

    blue-dragon -l -i usrp-B210-FCO2P05 -c 2441 -C 40 --check-crc --stats

Capture using HackRF (20 MHz max):

    blue-dragon -l -i hackrf-0000000000000000 -c 2441 -C 20 --check-crc --stats

Stream packets over ZMQ to a remote dashboard:

    blue-dragon -l -c 2441 -C 40 --zmq tcp://collector:5555 --check-crc

Stream with CURVE encryption:

    blue-dragon -l -c 2441 -C 40 --zmq tcp://collector:5555 --zmq-curve-key server.key

Capture with GPS tagging:

    blue-dragon -l -c 2441 -C 40 --gpsd --zmq tcp://collector:5555

Read from a previously recorded IQ file:

    blue-dragon -f recording.ci16 -c 2441 -C 20 -w output.pcap --check-crc --stats

### Channel Count Guidelines

The `-C` flag controls how many 1 MHz channels the polyphase channelizer
creates. Each BLE/BT channel is 2 MHz wide. More channels = more spectrum
coverage = more packets, but requires more CPU.

| Channels | Bandwidth | BLE Coverage | Notes |
|----------|-----------|--------------|-------|
| 4 | 4 MHz | ~5% | Minimal, for testing |
| 20 | 20 MHz | ~25% | HackRF maximum |
| 40 | 40 MHz | ~50% | Good starting point |
| 48 | 48 MHz | ~60% | Better coverage |
| 56 | 56 MHz | ~70% | Near full coverage |
| 60 | 60 MHz | ~75% | Best CRC rates |
| 80 | 80 MHz | 100% | Full BLE band |
| 96 | 96 MHz | 100% | Full band with margin |

Best CRC validation rates are at channel counts that are multiples of 4
near 40, 48, and 60. This is a characteristic of the PFBCH2 filterbank,
not a bug. Use `--stats` to monitor real-time performance.

### Wireshark Integration

Install as a Wireshark extcap plugin:

    blue-dragon --install

This creates a symlink in `~/.config/wireshark/extcap/`. After
installation, plug in your SDR and launch Wireshark -- Blue Dragon
will appear in the interface list.

## ZMQ Streaming and Dashboard

Blue Dragon streams packets over ZMQ using the same wire format as the
C sniffer, so it works with the existing Python web dashboard.

    # Start dashboard (binds data on 5555, C2 on 5556):
    pip install pyzmq
    python3 tools/zmq_web_dashboard.py tcp://*:5555

    # Start sensor(s):
    blue-dragon -l -c 2441 -C 40 --zmq tcp://dashboard:5555 --sensor-id roof --check-crc
    blue-dragon -l -c 2441 -C 40 --zmq tcp://dashboard:5555 --sensor-id lobby --check-crc

    # Open http://localhost:8099

### Sensor C2 (Command and Control)

When connected via `--zmq`, a C2 control channel is automatically
established on data_port + 1 (e.g. 5556). Each sensor sends a JSON
heartbeat every 5 seconds. The dashboard Nodes tab shows live sensor
status, gain/squelch controls, and packet rate monitoring.

Runtime-tunable: SDR gain, squelch threshold.
Restart-required: center frequency, channel count (sensor restarts automatically).

### CURVE Encryption

    python3 tools/zmq_keygen.py server.key
    blue-dragon -l ... --zmq tcp://collector:5555 --zmq-curve-key server.key
    python3 tools/zmq_web_dashboard.py tcp://*:5555 --server-key server.key

## GPS Tagging

Requires a gpsd instance running with a USB GPS receiver:

    sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock
    blue-dragon -l -c 2441 -C 40 --gpsd --zmq tcp://collector:5555

GPS coordinates are embedded in the PCAP using PPI (Per-Packet Information)
headers, compatible with Wireshark and Kismet. The dashboard `--gps` flag
enables a live map display.

No `libgps-dev` is needed -- Blue Dragon connects directly to gpsd via
TCP JSON protocol on port 2947.

## Architecture

```
SDR (USRP / HackRF / BladeRF / SoapySDR)
    |
    | int8 IQ samples
    v
Polyphase Channelizer (PFB, AVX2/SSE2 SIMD)
    |
    | N x 1 MHz channels
    v
FFT (rustfft, pure Rust)           [or OpenCL GPU]
    |
    v
Burst Catcher (AGC + squelch, per-channel)
    |
    | detected bursts
    v
FSK Demodulator (atan2 discriminator, CFO correction)
    |
    | bit stream
    v
Protocol Decoder
    |-- BLE: preamble search, AA correlator, whitening, CRC-24
    |-- BLE connection following (CONNECT_IND tracking)
    |-- Classic BT: Barker code, FEC syndrome decode
    v
Output
    |-- PCAP file (DLT 255/256, PPI wrapping with GPS)
    |-- ZMQ PUB (multipart: sensor_id + GPS + PCAP record)
    |-- C2 heartbeat (JSON over ZMQ DEALER)
```

## Differences from C Version

| | Blue Dragon (Rust) | C version |
|---|---|---|
| FFT | rustfft (pure Rust, no FFTW) | FFTW3 |
| AGC | Custom (no liquid-dsp dependency) | liquid-dsp |
| GPS | TCP JSON to gpsd (no libgps) | libgps FFI |
| Build | `cargo build` | cmake + make |
| Threading | crossbeam channels | pthreads + custom queues |
| GPU | OpenCL + VkFFT (optional) | OpenCL + VkFFT / Metal |
| Wire format | Identical | Identical |
| Dashboard | Same Python dashboard | Same Python dashboard |

## Acknowledgments

Blue Dragon draws inspiration from
[ICE9 Bluetooth Sniffer](https://github.com/nicholasgasior/ice9-bluetooth-sniffer)
and [Blue Hydra](https://github.com/pwnieexpress/blue_hydra), with the
goal of building the best SDR-based Bluetooth monitoring tool possible.
The polyphase channelizer architecture, protocol decoding approach, and
ZMQ wire format originate from the ICE9 project.

## License

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

Copyright 2025-2026 CEMAXECUTER LLC
