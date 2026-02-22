# Blue Dragon

Wideband BLE and Classic Bluetooth (BR) passive sniffer written in Rust.
Captures packets across multiple channels simultaneously using a polyphase
channelizer. Decodes BLE 5 LE 1M, LE 2M, and LE Coded PHYs with Extended
Advertising support. Wireshark-compatible PCAP output with optional ZMQ
streaming, GPS tagging, and remote sensor management.

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

DragonOS includes most SDR libraries pre-installed (often from source in
`/usr/local/lib`). The build system uses pkg-config with fallback to
common install paths, so source-built libraries are detected automatically.

Install the Rust toolchain if not already present:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env

Install any missing build dependencies:

    sudo apt install build-essential pkg-config libzmq3-dev

Build with USRP, ZMQ, and GPS support (recommended):

    cd blue-dragon
    cargo build --release --features "usrp,zmq,gps"

Build with all SDR backends:

    cargo build --release --features "usrp,hackrf,bladerf,soapysdr,zmq,gps"

The binary is at `target/release/blue-dragon`.

**Note:** If a library is installed from source but the linker can't find
it, run `sudo ldconfig` to refresh the shared library cache.

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

### Raspberry Pi (4/5, 64-bit OS)

DragonOS Pi64 already includes SDR libraries (libhackrf, libbladerf,
libsoapysdr, libzmq, etc.) so only the Rust toolchain is needed:

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env

    cd blue-dragon
    cargo build --release --features "hackrf,bladerf,soapysdr,zmq,gps"

On a stock Raspberry Pi OS (64-bit), install the SDR libraries first:

    sudo apt install build-essential pkg-config
    sudo apt install libhackrf-dev libbladerf-dev libsoapysdr-dev libzmq3-dev

No FFTW dependency -- the FFT is pure Rust (rustfft), so it builds
cleanly on ARM without cross-compilation issues. The PFB channelizer
uses NEON SIMD on aarch64 automatically.

GPU acceleration is not recommended on Pi -- the VideoCore GPU cannot
keep up with the PFB+FFT workload and the submission overhead exceeds
any compute savings. The CPU NEON path is faster on Pi hardware.

USRP is possible but UHD on Pi is heavy. SoapySDR with an RTL-SDR,
Airspy, or HackRF is the better fit for Pi deployments.

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
| `hci` | Active GATT probing via HCI | libdbus-1-dev (for BlueZ D-Bus) |

## BLE 5 PHY Support

Blue Dragon decodes all three BLE PHY modes automatically. No flags
are needed -- all PHYs are tried on every burst.

| PHY | Data Rate | Range | Use Case |
|-----|-----------|-------|----------|
| LE 1M | 1 Mbps | Standard | Legacy advertising, most BLE traffic |
| LE 2M | 2 Mbps | Shorter | High-throughput data connections |
| LE Coded (S=8) | 125 kbps | 4x range | Long-range IoT, asset tracking |
| LE Coded (S=2) | 500 kbps | 2x range | Long-range with higher throughput |

The `--stats` output shows a per-PHY breakdown:

    BLE: 1523 (2M:47 coded:12)  BT: 8  CRC: 94.2%

### Extended Advertising

BLE 5 Extended Advertising (ADV_EXT_IND, PDU type 7) is parsed
automatically. The Common Extended Header is decoded to extract:

- AuxPtr: secondary advertising channel, offset, and PHY
- AdvA / TargetA: advertiser and target addresses
- ADI: advertising data identifier
- TxPower: transmit power level

Since Blue Dragon captures all channels simultaneously, both primary
and secondary advertisements are captured without needing to follow
AuxPtr chains.

### PCAP PHY Encoding

PCAP output uses LINKTYPE_BLUETOOTH_LE_LL_WITH_PHDR (DLT 256).
PHY type is encoded in the RF header flags (bits 14-15):

| Bits 14-15 | PHY | Wireshark Display |
|-----------|-----|-------------------|
| 0b00 | LE 1M | `LE 1M` |
| 0b01 | LE 2M | `LE 2M` |
| 0b10 | LE Coded | `LE Coded` |

LE Coded packets include a CI (Coding Indicator) byte between the
Access Address and PDU, per the PCAP specification. Wireshark 3.6+
recognizes all three PHY types natively.

## Usage

### Command-Line Options

```
Input (pick one):
    -f, --file FILE         Read input from IQ file
    -l, --live              Capture live from SDR

SDR settings:
    -i, --interface IFACE   SDR device (e.g. usrp-B210-SERIAL)
    -c, --center-freq FREQ  Center frequency in MHz (default: 2441)
    -C, --channels N        Number of channels (default: 40)
    -a, --all-channels      Full BLE band: sets -C 96 -c 2441
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

HCI:
    --hci                   Enable active GATT probing via system Bluetooth adapter

Wireshark:
    --install               Install as Wireshark extcap plugin
    --list                  List available SDR interfaces
```

### Examples

Capture 40 channels centered on 2441 MHz using a USRP B210:

    blue-dragon -l -i usrp-B210-SERIAL -c 2441 -C 40 -w capture.pcap

Capture with CRC validation and stats:

    blue-dragon -l -i usrp-B210-SERIAL -c 2441 -C 40 --check-crc --stats

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

The dashboard device table includes a PHY column showing which BLE PHY
was used by each device (1M, 2M, or Coded).

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

## HCI GATT Probing

With `--hci`, Blue Dragon can actively query GATT services and
characteristics on connectable BLE devices using the system's Bluetooth
adapter (hci0). This is opt-in -- without the flag, the sniffer is
100% passive.

    cargo build --release --features "usrp,zmq,hci"
    blue-dragon -l -c 2441 -C 40 --zmq tcp://dashboard:5555 --hci --check-crc

The dashboard marks connectable devices (ADV_IND, ADV_DIRECT_IND) with
a blue badge. Click a device row to open the detail panel, then click
"Query GATT" to enumerate services and characteristics via BlueZ.

GATT queries are routed only to the sensor(s) that have seen the target
device, not broadcast to all sensors.

**Range limitation:** The HCI adapter has a typical range of 10-30 meters,
much shorter than the SDR's passive capture range. GATT queries will only
succeed for devices within Bluetooth range of the sensor's hci0 adapter.
This makes the feature most useful when the sensor is physically close to
the target, or in deployments where sensors are distributed across a site.

Requires a powered Bluetooth adapter visible to BlueZ (`hciconfig hci0 up`).
The `bluer` crate communicates with BlueZ via D-Bus -- no raw HCI access
or special permissions beyond D-Bus policy are needed.

## Architecture

```
SDR (USRP / HackRF / BladeRF / SoapySDR)
    |
    | int8 IQ samples
    v
Polyphase Channelizer (PFB, AVX2/SSE2/NEON SIMD)
    |
    | N x 2 Msps channels
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
    | soft + hard bit streams
    v
Protocol Decoder
    |-- BLE LE 1M: preamble search, AA correlator, whitening, CRC-24
    |-- BLE LE 2M: 16-bit preamble, SPS=1 reslice, AA correlator
    |-- BLE LE Coded: 80-symbol preamble, FEC (Viterbi), pattern demap
    |-- BLE Extended Advertising: Common Extended Header, AuxPtr
    |-- BLE connection following (CONNECT_IND tracking)
    |-- Classic BT: Barker code, FEC syndrome decode
    v
Output
    |-- PCAP file (DLT 256 with PPI wrapping, PHY flags)
    |-- ZMQ PUB (multipart: sensor_id + GPS + PCAP record)
    |-- C2 heartbeat (JSON over ZMQ DEALER)
    |-- HCI GATT prober (opt-in, via system hci0 adapter)
```

## Performance

Tested on an Intel i7-12700H / NVIDIA RTX 3060 Laptop GPU with a
bladeRF 2.0 micro xA4 at 96 channels (full BLE band):

| Metric | Result |
|--------|--------|
| BLE CRC validation rate | 95-97% |
| Packet rate (quiet environment) | 50-70 pkt/s |
| Classic BT UAP recovery | Converges in ~12 packets |
| GPU PFB+FFT (OpenCL) | ~2x throughput vs CPU-only |
| CPU usage (GPU enabled) | ~30% of one core (recv + decode) |
| Memory usage | ~40 MB RSS |

The polyphase channelizer processes all 96 channels in real time on
both CPU (AVX2/SSE2/NEON) and GPU (OpenCL) paths. On hardware without
a GPU, the CPU path handles 96 channels comfortably on any modern
x86_64 or aarch64 processor.

LE 2M and LE Coded decoding adds negligible overhead -- the additional
PHY decoders only run when LE 1M decode fails on a burst, and the
preamble checks fail fast on non-matching bursts.

## Differences from C Version

| | Blue Dragon (Rust) | C version |
|---|---|---|
| FFT | rustfft (pure Rust, no FFTW) | FFTW3 |
| AGC | Custom (no liquid-dsp dependency) | liquid-dsp |
| GPS | TCP JSON to gpsd (no libgps) | libgps FFI |
| Build | `cargo build` | cmake + make |
| Threading | crossbeam channels | pthreads + custom queues |
| GPU | OpenCL + VkFFT (optional) | OpenCL + VkFFT / Metal |
| BLE 5 | LE 1M + LE 2M + LE Coded | LE 1M only |
| Wire format | Identical | Identical |
| Dashboard | Same Python dashboard | Same Python dashboard |

## Acknowledgments

The name Blue Dragon is inspired by
[Blue Hydra](https://github.com/pwnieexpress/blue_hydra).

## License

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

Copyright 2025-2026 CEMAXECUTER LLC
