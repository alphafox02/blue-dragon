# Blue Dragon

Wideband BLE and Classic Bluetooth passive sniffer written in Rust.

Most BLE sniffers capture one channel at a time. Blue Dragon uses a
polyphase filterbank channelizer to capture **up to 40 BLE channels
simultaneously** from a single SDR, decoding BLE 5 LE 1M, LE 2M, and
LE Coded PHYs plus Classic Bluetooth BR in the same passband. This
means no channel hopping and no missed packets -- every advertisement
and connection event across the captured bandwidth is seen.

Output is Wireshark-compatible PCAP with optional ZMQ streaming for
multi-sensor deployments, GPS tagging for drive surveys, and a web
dashboard for real-time monitoring.

## Status

| Feature | Status | Notes |
|---------|--------|-------|
| USRP B210 capture | Tested | USB 3, up to -C 40 validated |
| BLE LE 1M decoding | Tested | 95-96% CRC pass rate |
| BLE LE 2M decoding | Tested | |
| BLE LE Coded decoding | Tested | Low volume confirmed in drive tests |
| Classic BT BR detection | Tested | LAP extraction, UAP recovery |
| PCAP output (PPI) | Tested | Mixed BLE+BT per-packet DLT |
| GPS tagging (gpsd) | Tested | 100% packet tagging in drive tests |
| ZMQ streaming | Tested | PUB/SUB + C2 heartbeat |
| CurveZMQ encryption | Tested | Requires system libzmq with libsodium |
| Wireshark extcap | Tested | Live capture with GPS/CRC options |
| CPU SIMD (AVX2) | Tested | i7-12700H |
| GPU OpenCL | Tested | NVIDIA RTX 3060, Intel UHD iGPU |
| HackRF backend | Untested | Compiles, needs hardware validation |
| bladeRF backend | Tested | 88.7% CRC OTA at -g 30 |
| SoapySDR backend | Tested | |
| Aaronia Spectran V6 | Tested | 92 MHz BW, ~38% CRC OTA (tuning ongoing) |
| HCI GATT probing | Untested | Compiles, needs end-to-end test with --hci |
| HCI active scanning | Tested | --active-scan enriches device data |
| Channel counts -C 60+ | Untested | -C 40 validated, higher counts need testing |

## Supported Hardware

| SDR | Interface Flag | Bandwidth | ADC Bits | Notes |
|-----|---------------|-----------|----------|-------|
| USRP (B200/B210) | `-i usrp-MODEL-SERIAL` | 4-56 MHz | 12-bit | AD9361 (61.44 Msps, 56 MHz analog BW) |
| HackRF One | `-i hackrf-SERIAL` | 4-20 MHz | 8-bit | 20 MHz max sample rate |
| bladeRF 2.0 | `-i bladerf0` | 4-56 MHz (normal), up to 122 MHz (oversample) | 12-bit (normal) / 8-bit (oversample) | AD9361 (oversample overclocks beyond AD spec) |
| SoapySDR | `-i soapy-N` | Varies | Varies | Generic SDR support |
| Aaronia Spectran V6 | `-i aaronia` | 92 MHz | f32 | Real-time spectrum analyzer |

To list available SDR devices:

    blue-dragon --list

### SDR Gain Recommendations

The `-g` flag sets the SDR's receive gain in dB. The optimal value depends on
the SDR hardware and environment. **Too high clips the ADC (zero packets);
too low buries the signal in the noise floor (low CRC rate).**

| SDR | Default | OTA Recommended | Cabled (30 dB atten) | Notes |
|-----|---------|-----------------|----------------------|-------|
| USRP B210 | 60 | 40-50 | 60 | UHD auto-AGC not used |
| bladeRF 2.0 | 60 | **25-35** | 50-60 | Clips at 60 OTA -- use 30 |
| HackRF | 40 LNA / 20 VGA | TBD | TBD | Separate `--hackrf-lna` / `--hackrf-vga` |
| Aaronia Spectran V6 | N/A | N/A | N/A | Uses reflevel, not gain (auto-scaled) |
| SoapySDR | 60 | Device-dependent | Device-dependent | Depends on underlying hardware |

**Symptoms of gain too high:** BLE count = 0, all bursts fail decode (ADC saturation
clips the waveform so preamble/AA correlation fails). Fix: lower `-g`.

**Symptoms of gain too low:** Low CRC pass rate (< 50%), low BLE packet count.
Fix: raise `-g`.

Use `--stats` to monitor CRC rate in real time. Target: > 85% for a clean
environment, 70-90% typical for busy 2.4 GHz bands.

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

Optional GPU acceleration (OpenCL):

    sudo apt install ocl-icd-opencl-dev
    cargo build --release --features "usrp,zmq,gps,gpu"

DragonOS systems with an NVIDIA GPU typically have the OpenCL ICD loader
already installed. If `clinfo` shows your GPU, you only need the
`ocl-icd-opencl-dev` headers for the build. For Intel iGPU support,
install `intel-opencl-icd` as well.

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

### macOS (Homebrew) -- Untested

    brew install uhd hackrf libbladerf soapysdr zeromq pkg-config
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env

    cd blue-dragon
    cargo build --release --features "usrp,hackrf,bladerf,soapysdr,zmq"

GPU acceleration on macOS requires Metal (future work -- OpenCL is
deprecated on macOS and Metal backend is not yet implemented).

### Aaronia Spectran V6

Requires the Aaronia RTSA Suite Pro installed to `/opt/aaronia-rtsa-suite/`.
The SDK and runtime library (`libAaroniaRTSAAPI.so`) live in the
`Aaronia-RTSA-Suite-PRO/` subdirectory. The build system embeds an rpath
so the binary finds the library at runtime without `LD_LIBRARY_PATH`.

    cargo build --release --features "aaronia,zmq"

The Aaronia backend uses `spectranv6/raw` mode with `outputformat=iq` to
get wideband IQ samples. The 92.16 MHz clock gives 92 MHz of usable
bandwidth at `-C 92`:

    blue-dragon -l -i aaronia -C 92 --check-crc --stats

The `-g` flag is ignored for Aaronia. Instead, the backend sets the reference
level to -20 dBm and auto-scales the f32 samples to match the pipeline's
int8/int16 range. The auto-scale measures RMS over 20 packets at startup
and uses the 25th percentile to filter out WiFi burst outliers.

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
| `hci` | Active GATT probing + LE scanning via HCI | libdbus-1-dev (for BlueZ D-Bus) |
| `aaronia` | Aaronia Spectran V6 support | Aaronia RTSA Suite Pro |

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
    --antenna PORT          RX port (USRP: RX2/TX/RX, bladeRF: RX1/RX2)

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

BLE 5 Long Range:
    --coded-scan            Continuous LE Coded scan on advertising channels

GPU:
    --no-gpu                Disable GPU acceleration (CPU-only)

HCI:
    --hci                   Enable active GATT probing via system Bluetooth adapter
    --active-scan           Enable LE active scanning to enrich device data

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

Capture 92 MHz with Aaronia Spectran V6:

    blue-dragon -l -i aaronia -C 92 --check-crc --stats

Capture full BLE band with bladeRF at recommended OTA gain:

    blue-dragon -l -i bladerf0 -a -g 30 --check-crc --stats

Capture with active BLE scanning for device enrichment:

    blue-dragon -l -c 2441 -C 40 --hci --active-scan --zmq tcp://dashboard:5555 --check-crc

Capture BLE 5 Long Range (LE Coded PHY) on advertising channels:

    blue-dragon -l -i usrp-B210-SERIAL -c 2402 -C 4 --check-crc --coded-scan --stats

Without `--coded-scan`, coded decoding still runs on any squelch-triggered
burst that fails LE 1M and BT decode. The flag adds continuous sampling on
channels 37/38/39 to catch weak coded signals below the normal squelch
threshold.

Read from a previously recorded IQ file:

    blue-dragon -f recording.ci16 -c 2441 -C 20 -w output.pcap --check-crc --stats

### Channel Count Guidelines

The `-C` flag sets both the SDR sample rate and the number of 1 MHz FFT
bins in the polyphase channelizer: **`-C 40` = 40 MHz bandwidth at
40 Msps, split into 40 channels**.

BLE channels are spaced every **2 MHz** (ch 0 = 2402 MHz, ch 1 = 2404 MHz,
..., ch 39 = 2480 MHz), so only half the FFT bins land on BLE channel
centers. The other half sit between BLE channels (these still catch
Classic Bluetooth, which uses 1 MHz spacing). This means you need
roughly **2x the FFT bins to cover N BLE channels**:

| `-C` | Bandwidth | BLE Channels | Notes |
|------|-----------|-------------|-------|
| 4 | 4 MHz | ~2 of 40 | Minimal, for testing |
| 20 | 20 MHz | ~10 of 40 | HackRF maximum |
| 40 | 40 MHz | ~20 of 40 | Good starting point |
| 48 | 48 MHz | ~24 of 40 | Better coverage |
| 56 | 56 MHz | ~28 of 40 | Near full coverage |
| 60 | 60 MHz | ~30 of 40 | Best CRC rates |
| 80 | 80 MHz | 40 of 40 | Full BLE band (2402-2480 MHz) |
| 96 | 96 MHz | 40 of 40 | Full band + 8 MHz guard on each side |

**Why `-C 80` for full coverage?** The BLE band spans 2402-2480 MHz
(78 MHz). At 80 MHz centered on 2441 MHz, all 40 BLE channels fit
within the captured bandwidth.

**Why `-C 96` for bladeRF?** The extra 16 MHz (8 MHz per side) acts as
a guard band, preventing filter roll-off from degrading channels at the
band edges. The bladeRF 2.0 supports the wider sample rate natively.

**Tradeoff:** More channels = more CPU. At `-C 40` you capture half the
BLE band at half the compute cost. On constrained hardware (Raspberry Pi,
HackRF's 20 MHz limit), smaller values are necessary.

Best CRC validation rates are at channel counts that are multiples of 4
near 40, 48, and 60. This is a characteristic of the PFBCH2 filterbank,
not a bug. Use `--stats` to monitor real-time performance.

### Wireshark Integration

Install as a Wireshark extcap plugin:

    blue-dragon --install

This detects Wireshark's personal extcap path (via `tshark -G folders`)
and creates a symlink there. On Wireshark 4.2+ this is typically
`~/.local/lib/wireshark/extcap/`. After installation, plug in your SDR
and launch Wireshark -- Blue Dragon will appear in the interface list
with one entry per connected SDR.

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

CURVE encryption requires `libzmq3-dev` (system libzmq with libsodium).
The `.cargo/config.toml` overrides the Rust crate's vendored libzmq build
to link against the system library, which has full CURVE support.

    # Generate a keypair:
    python3 tools/zmq_keygen.py server.key

    # Start sensor with CURVE:
    blue-dragon -l ... --zmq tcp://collector:5555 --zmq-curve-key server.key

    # Start dashboard with CURVE:
    python3 tools/zmq_web_dashboard.py tcp://*:5555 --server-key server.key

The `server.key` contains both public and secret keys (keep it on the sensor
and dashboard hosts). The `server.key.pub` contains only the public key and
is safe to distribute.

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
SDR (USRP / HackRF / BladeRF / SoapySDR / Aaronia)
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
    |-- HCI LE active scanner (opt-in, enriches device data)
```

## Performance

Tested on Intel i7-12700H, USRP B210, -C 40 (20 BLE channels, 40 MHz),
WHAD ButteRFly advertiser through 30 dB attenuator:

| Metric | Result |
|--------|--------|
| BLE CRC validation rate | 92-95% |
| Packet rate (active environment) | 30-60 pkt/s |
| Classic BT UAP recovery | Converges in ~12 packets |
| Memory usage | ~40 MB RSS |

### GPU vs CPU Performance

The polyphase channelizer + FFT is the compute bottleneck. The CPU
path uses SIMD (AVX2/SSE2/NEON) and handles high channel counts well
on modern hardware. GPU acceleration (OpenCL) offloads this work and
may help on slower CPUs or at very high channel counts.

Comparison at -C 40 (20 BLE channels), USRP B210, i7-12700H:

| Compute backend | BLE/30s | CRC% | Overflow |
|-----------------|---------|------|----------|
| NVIDIA RTX 3060 (OpenCL) | 1,486 | 95.2% | 1 |
| Intel UHD iGPU (OpenCL) | 870 | 92.3% | 0 |
| CPU-only (AVX2) | 1,820 | 94.0% | 0 |

All three backends handle -C 40 without sample loss on this hardware.
The CPU AVX2 path is competitive with discrete GPU at moderate channel
counts. GPU offload becomes more beneficial at -C 60+ or on slower CPUs
without AVX2.

Build with GPU support:

    cargo build --release --features "usrp,zmq,gps,gpu"

Use `--no-gpu` to force CPU-only mode. On Raspberry Pi, the CPU NEON
path is faster than the VideoCore GPU -- do not use `gpu` on Pi.

LE 2M and LE Coded decoding adds negligible overhead -- the additional
PHY decoders only run when LE 1M decode fails on a burst, and the
preamble checks fail fast on non-matching bursts.

## Sample Precision

All SDR backends deliver samples as signed int8 (SC8) to the pipeline,
regardless of native ADC resolution. The PFB channelizer promotes i8 to
i16 internally (`(sample as i16) << 8`), but the lower 8 bits are always
zero -- effectively 8-bit precision in a 16-bit container.

| SDR | Native ADC | What enters pipeline | Precision lost |
|-----|-----------|---------------------|----------------|
| USRP B210 | 12-bit | i8 (UHD SC8 wire format) | 4 bits (24 dB) |
| bladeRF (normal) | 12-bit (SC16_Q11) | i8 (right-shift >> 4) | 4 bits (24 dB) |
| bladeRF (oversample) | SC8_Q7 | i8 (native) | 5 bits (30 dB) |
| HackRF | 8-bit | i8 (native) | None |
| SoapySDR (CS16) | Device-dependent | i8 (right-shift >> N) | Device-dependent |
| Aaronia Spectran V6 | 32-bit float | i8 (scaled from f32) | ~24 bits |

This means the USRP and bladeRF are only using 8 of their 12 available
ADC bits. A future i16 pipeline would preserve full ADC resolution for
all backends except HackRF (which is natively 8-bit). This primarily
affects dynamic range -- the ability to decode weak signals in the
presence of strong ones.

**USRP note:** The USRP B210 ADC is 12-bit, but we currently request
SC8 wire format from UHD. Switching to SC16 would preserve all 12 bits.
USB3 bandwidth is not a concern -- SC16 at 61.44 Msps is only 246 MB/s
vs USB3's ~400 MB/s sustained. The AD9361's analog filter passes 56 MHz
cleanly; at `-C 56` and below, all channels are within the analog
bandwidth. Above that, edge channels see filter roll-off.

**Aaronia note:** The Aaronia delivers 32-bit float samples. The current
i8 quantization loses significant precision. The backend has a
`recv_into_i16` method ready for a future i16 pipeline upgrade.

## Troubleshooting

### BLE count is 0 (zero packets)

1. **Gain too high (most common).** The ADC is clipping. Lower `-g`:
   - bladeRF OTA: try `-g 30` (default 60 clips)
   - USRP OTA: try `-g 40-50`
   - Cabled with 30 dB attenuator: `-g 60` is fine

2. **Wrong interface name.** Use `--list` to find your SDR, then pass
   the exact string to `-i`.

3. **Missing `-l` flag.** Live capture requires `-l` (or `--live`).

4. **Center frequency off-band.** Default 2441 MHz is good. Ensure
   the SDR is tuned to the 2.4 GHz ISM band.

### CRC rate is low (< 50%)

1. **Gain too low.** Raise `-g` until CRC improves.
2. **Gain too high.** Also causes poor CRC -- the waveform clips and
   correlation degrades. Try lowering gain.
3. **Heavy WiFi environment.** 2.4 GHz WiFi shares the band with BLE
   and can raise the noise floor. Normal to see 70-85% CRC in busy
   environments.
4. **Not using `--check-crc`.** Without this flag, CRC is never
   computed and shows `0/0`. This is not an error -- add `--check-crc`
   to enable validation.

### Aaronia: library not found

If you see `libAaroniaRTSAAPI.so: cannot open shared object file`:
the RTSA Suite is not installed to the expected path, or the rpath
is not embedded. Verify the library exists:

    ls /opt/aaronia-rtsa-suite/Aaronia-RTSA-Suite-PRO/libAaroniaRTSAAPI.so

If installed elsewhere, either symlink or set `LD_LIBRARY_PATH`.
Rebuilding with `--features aaronia` when the directory exists will
embed the correct rpath automatically.

### ZMQ: "Invalid argument" on bind

The ZMQ endpoint format is `tcp://host:port` for connecting to a
remote dashboard, or `tcp://*:port` for binding locally. Ensure the
port is not already in use.

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
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

Copyright 2025-2026 CEMAXECUTER LLC
