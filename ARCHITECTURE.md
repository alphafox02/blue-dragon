# Blue Dragon Architecture

## Crate Layout

```
blue-dragon/
├── crates/
│   ├── app/          bd-app       CLI entry point, pipeline orchestration
│   ├── dsp/          bd-dsp       PFB channelizer, FFT, burst detection, FSK demod
│   ├── sdr/          bd-sdr       SDR hardware abstraction (USRP, HackRF, BladeRF, SoapySDR, Spectran V6)
│   ├── protocol/     bd-protocol  BLE and Classic BT packet decoding
│   ├── output/       bd-output    PCAP writer, ZMQ streaming, GPS client, C2 control
│   └── gpu/          bd-gpu       Optional OpenCL GPU acceleration
└── tools/
    ├── zmq_web_dashboard.py       Web dashboard (device tracking, C2, maps)
    ├── zmq_subscriber.py          ZMQ subscriber test harness
    ├── zmq_keygen.py              CurveZMQ key generator
    └── zmq_gps_test_pub.py        GPS test publisher
```

**Dependency graph:**

```
bd-app ──> bd-dsp ──> bd-protocol
  │          │
  │          └──> rustfft, num-complex, crossbeam
  │
  ├──> bd-sdr ──> crossbeam (+ C FFI: libuhd, libhackrf, libbladeRF, libSoapySDR)
  │
  ├──> bd-output ──> bd-protocol (+ optional: zmq, serde_json)
  │
  └──> bd-gpu ──> num-complex (+ C FFI: OpenCL, VkFFT)
```

All SDR backends and optional features are compile-time gated:

| Feature | Crate | System Dependency |
|---------|-------|-------------------|
| `usrp` | bd-sdr | libuhd |
| `hackrf` | bd-sdr | libhackrf |
| `bladerf` | bd-sdr | libbladeRF |
| `soapysdr` | bd-sdr | libSoapySDR |
| `aaronia` | bd-sdr | RTSA Suite (libAaroniaRTSAAPI) |
| `zmq` | bd-output | libzmq |
| `gps` | bd-output | none (TCP JSON to gpsd) |
| `gpu` | bd-gpu | OpenCL runtime |

## Threading Model (Live Capture)

```
                           ┌─────────────┐
                           │  Ctrl-C     │
                           │  handler    │
                           └──────┬──────┘
                                  │ sets running=false
                                  v
┌──────────┐  bounded(32)  ┌──────────────┐  bounded(4) x N  ┌────────────────┐
│ sdr-recv │──────────────>│   pfb-fft    │──────────────────>│  burst-0..N-1  │
│          │  Vec<i16>     │  (or GPU)    │  BatchMsg         │  (8 workers)   │
└──────────┘               └──────────────┘                   └───────┬────────┘
     ^                                                                │
     │ polls gain_pending                                   bounded(512) Burst
     │ (AtomicI32)                                                    │
     │                                                                v
┌────┴───────┐             ┌──────────────┐               ┌───────────────────┐
│ c2-dispatch│<────────────│ c2-control   │               │      decode       │
│            │  bounded(16)│ (DEALER)     │               │                   │
└────────────┘ ControlCmd  └──────────────┘               │ FSK demod         │
     │                          ^                         │ BLE detect/CRC    │
     │ stores in atomics        │ heartbeat               │ BT detect/FEC     │
     │ gain_pending             │ every 5s                │ Connection track   │
     │ squelch_pending          │                         │ PCAP write        │
     v                          └─────────────────────────│ ZMQ publish       │
  burst workers poll                                      │ Stats print       │
  squelch_pending                                         └───────────────────┘
```

### Thread roles

| Thread | Name | What it does |
|--------|------|-------------|
| 1 | `sdr-recv` | Drains SDR hardware via `recv_into_i16()`, sends i16 buffers to PFB. Accumulates multiple recv calls to ~4080 samples before sending (batching). Polls `gain_pending` atomic for C2 gain changes. |
| 2 | main | PFB channelizer + IFFT (CPU path) or GPU submit loop. Feeds i16 samples directly to polyphase filter, broadcasts channel data to workers. |
| 3-10 | `burst-0` .. `burst-7` | Each assigned a disjoint subset of BLE channels. Per-channel AGC + squelch detection. Polls `squelch_pending` atomic. Sends detected bursts to decode. |
| 11 | `decode` | FSK demodulation, BLE/BT protocol decode, CRC validation, PCAP output, ZMQ publish. Updates heartbeat stats. |
| 12 | `c2-control` | ZMQ DEALER socket to dashboard ROUTER. Sends JSON heartbeat every 5s. Receives commands. (Only with `--zmq`.) |
| 13 | `c2-dispatch` | Reads ControlCommand channel, stores gain/squelch in atomics for other threads to pick up. (Only with `--zmq`.) |
| 14 | `hci-scan` | LE active scanning via BlueZ. Sends ScanResult to ZMQ. Rate-limited to 1 update per MAC per 10s. (Only with `--active-scan`.) |

### Inter-thread communication

| Channel | Type | Capacity | Payload |
|---------|------|----------|---------|
| SDR -> PFB | crossbeam bounded | 32 | `Vec<i16>` (native-precision samples) |
| PFB -> workers | crossbeam bounded | 4 per worker | `BatchMsg` (channel samples) |
| Workers -> decode | crossbeam bounded | 512 (shared) | `Burst` (IQ + metadata) |
| C2 control -> dispatch | crossbeam bounded | 16 | `ControlCommand` enum |

| Atomic | Type | Purpose |
|--------|------|---------|
| `running` | `AtomicBool` | Global stop flag |
| `gain_pending` | `AtomicI32` | Pending gain change (x10 encoding, sentinel=i32::MIN) |
| `squelch_pending` | `AtomicI32` | Pending squelch change (x10 encoding, sentinel=i32::MIN) |
| `overflow_count` | `AtomicU64` | SDR underrun counter |

Shared mutex: `Arc<Mutex<HeartbeatState>>` -- written by decode thread (pkt_rate, crc_pct), read by C2 control thread for heartbeat JSON.

## Data Flow

```
SDR (native precision: SC16, CS8, or f32 depending on backend)
  │
  │ sdr-recv thread: recv_into_i16(&mut [i16])
  │ Each backend converts to i16 at native precision
  v
PFB Channelizer (Type 2, M channels, semi_len=4)
  │ Kaiser window prototype, fc=0.75/M, 60 dB stopband
  │ SIMD: AVX2 (x86_64), SSE2 (x86_64 fallback), NEON (aarch64)
  │ Fixed-point i16 arithmetic
  v
Inverse FFT (rustfft, M-point)
  │ Output: M complex32 channels per step
  │ Normalization: 1/M
  v
Burst Catcher (per BLE channel)
  │ AGC envelope tracking (alpha=0.25)
  │ Squelch threshold (default -45 dB, C2 tunable)
  │ Accumulates samples until signal timeout
  v
FSK Demodulator
  │ Frequency discriminator: arg(y[n] * conj(y[n-1])) / pi
  │ CFO correction via 64-symbol median filter
  │ Hard bit decisions at SPS=2
  │ Soft demod values preserved for FEC
  v
Protocol Decode (tried in order, first success wins)
  │
  ├── 1. Classic BT: Barker code detection, FEC syndrome decode
  │
  ├── 2. BLE LE 1M preamble-first (SPS=2)
  │     Check 8-bit alternating preamble, try 3 AA positions
  │     Whitening (127-bit pre-computed table), CRC-24
  │
  ├── 3. BLE LE 1M AA correlator (SPS=2)
  │     Normalized template match (threshold 0.6)
  │     Hamming distance to known AAs <= 4
  │
  ├── 4. BLE LE 2M preamble-first (SPS=1)
  │     Reslice existing demod at SPS=1 (no re-run of discriminator)
  │     16-bit preamble, max PDU 251 bytes
  │
  ├── 5. BLE LE 2M AA correlator (SPS=1)
  │     Second correlator parameterized for SPS=1
  │
  └── 6. BLE LE Coded
        80-symbol coded preamble (00111100 x10)
        FEC Block 1 at S=8: AA(32 bits) + CI(2 bits), Viterbi decode
        FEC Block 2 at S=CI: PDU + CRC, soft pattern demap, Viterbi decode
        Convolutional code: rate 1/2, K=4, g0=17o, g1=15o (8 trellis states)
  │
  ├── BLE Extended Advertising: parse Common Extended Header (AuxPtr, ADI, TxPower)
  ├── BLE connection following: CONNECT_IND tracking table
  ├── BLE CRC-24: reflected polynomial 0xDA6000
  │     Init = reflect24(0x555555) for advertising
  │     Init from CONNECT_IND for data channels
  └── BLE whitening: 127-bit pre-computed lookup table
  v
Output
  ├── PCAP file: DLT 192 (PPI) wrapping per-packet DLT 255/256
  │     PPI-GPS field type 30002 for GPS coordinates
  │     PHY type in flags bits 14-15 (0=1M, 1=2M, 2=Coded)
  ├── ZMQ PUB: multipart [sensor_id?] [gps_frame?] [pcap_record]
  └── C2 heartbeat: JSON over DEALER socket every 5s
```

## SDR Backends

All backends implement a `Handle` type with:
- `recv_into_i16(&mut [i16]) -> usize` -- receive samples at native precision (CPU path)
- `recv_into(&mut [i8]) -> usize` -- receive SC8 samples (GPU path)
- `max_samps() -> usize` -- maximum batch size
- `overflow_count() -> u64` -- underrun counter
- `set_gain(gain)` -- runtime tunable via C2

| Backend | Native Format | CPU i16 Conversion | Effective Bits | Max Sample Rate |
|---------|--------------|-------------------|----------------|-----------------|
| USRP | SC16 (12-bit ADC, left-justified) | Direct from UHD | 12 | 61.44 Msps (56 MHz analog BW) |
| HackRF | CS8 (unsigned reinterpreted as signed) | `<< 8` (no extra precision) | 8 | 20 MHz |
| bladeRF (normal) | SC16_Q11 (12-bit) | `<< 4` to fill i16 range | 12 | 61.44 Msps (56 MHz analog BW) |
| bladeRF (oversample) | SC8_Q7 | `<< 8` (no extra precision) | 8 | 122.88 Msps |
| SoapySDR | CS8 preferred, CS16/CF32 fallback | Dynamic left-shift | Device-dependent | Device-dependent |
| Spectran V6 | f32 (32-bit float IQ) | f32 * scale -> i16 | 16 | 92-245 MHz (clock-dependent) |

The `SdrHandle` enum dispatches to the correct backend at runtime. HackRF
has separate LNA/VGA gain controls; all others use a single gain value.
The Spectran V6 backend maps `-g N` to reference level -N dBm with
auto-scaling from f32 to i16.

### Spectran V6

The Spectran V6 backend (`sdr/src/aaronia.rs`) uses the RTSA API in
`spectranv6/raw` mode to receive wideband IQ samples as 32-bit float
pairs. Key details:

- **Clock:** Device-dependent. At `-C 92`, the 92.16 MHz clock is used,
  giving 92 channels at 1.0017 MHz each (0.17% wider than 1 MHz).
  Not all clocks are supported on all devices; the backend probes the
  device and errors with guidance if the requested `-C` is incompatible.
  Typical supported clocks: 92, 122, 184, 245 MHz.
- **Decimation:** 1:1 (full rate, no decimation).
- **Reference level:** Maps `-g N` to reflevel -N dBm. Default `-g 20`
  gives -20 dBm (most sensitive). Lower `-g` values increase headroom
  for strong WiFi signals at the cost of sensitivity.
- **Auto-scale:** At startup, measures RMS over 20 packets. Uses the 25th
  percentile RMS to compute a scale factor that maps noise floor to ~2
  LSBs in i8 (or ~512 in i16). This filters out WiFi burst outliers
  that would inflate a single-packet RMS measurement.
- **Packet format:** Each RTSA API packet contains `num` complex samples
  with `stride` floats between samples (stride >= 2). The backend reads
  `fp32[i*stride]` (I) and `fp32[i*stride+1]` (Q) per sample.
- **recv_into_i16:** CPU pipeline path. Scales f32 by `scale * 256` to
  fill the i16 range, preserving ~16 bits of effective precision.
- **Overflow detection:** Checks `PACKET_WARN_OVERFLOW` flag per packet.
- **No sudo required:** The RTSA API handles USB communication internally.

### Precision chain (CPU i16 pipeline)

```
SDR ADC (8-32 bit native)
    |
    | recv_into_i16(&mut [i16]): each backend delivers native precision
    | USRP: SC16 wire format (12-bit, left-justified in i16)
    | bladeRF normal: SC16_Q11 << 4 (12-bit, fills i16 range)
    | bladeRF oversample: SC8 << 8 (8-bit, no extra precision)
    | HackRF: CS8 << 8 (8-bit, no extra precision)
    | Spectran V6: f32 * scale * 256 -> i16
    v
crossbeam channel: Vec<i16> (bounded, 32 slots)
    |
    | PFB thread: feeds i16 directly (no promotion needed)
    v
PFB channelizer: i16 fixed-point arithmetic
    |
    | IFFT: output Complex32 (f32 pairs)
    v
Burst catcher / demod / decode: all f32
```

The i16 pipeline preserves the full native ADC resolution for 12-bit
SDRs (USRP, bladeRF), giving ~24 dB more dynamic range than the
previous i8 path. The recv thread accumulates multiple recv calls
to ~4080 samples before sending, compensating for smaller per-recv
batch sizes with wider sample formats (e.g., USRP SC16 halves
max_samps compared to SC8).

The GPU path still uses `recv_into(&mut [i8])` and i8 buffers for
OpenCL kernel compatibility.

## GPU Path

When `--gpu` is enabled (and not `--no-gpu`), the PFB channelizer and FFT
run on the GPU via OpenCL:

- C implementation in `gpu/csrc/gpu_pfb_fft.c` using VkFFT for FFT
- Rust wrapper in `gpu/src/lib.rs` via cc-built FFI
- Double-buffered: GPU processes batch N while host fills batch N+1
- Batch size: 4096 PFB steps (configurable)
- Output: float pairs (complex32) returned to Rust
- Eliminates PFB as CPU bottleneck; ~2x throughput improvement

The GPU path replaces only PFB+FFT. Burst detection, demodulation, and
protocol decode remain on CPU (these are per-burst, not bulk DSP).

## BLE 5 Multi-PHY Decode

The PFB channelizer outputs 2 Msps per channel, which is sufficient for
all three BLE PHYs without any channelizer changes:

| PHY | Symbol Rate | SPS at 2 Msps | Deviation | Preamble |
|-----|------------|---------------|-----------|----------|
| LE 1M | 1 Msym/s | 2 | +/-250 kHz | 8 bits (10101010) |
| LE 2M | 2 Msym/s | 1 | +/-500 kHz | 16 bits (1010...1010) |
| LE Coded | 1 Msym/s | 2 | +/-250 kHz | 80 symbols (00111100 x10) |

### LE 2M

LE 2M is the same GFSK modulation at double the symbol rate. At SPS=1,
the `reslice()` function re-extracts hard bits from the existing
CFO-corrected analog demod signal -- no FM discriminator re-run needed.
The +/-500 kHz deviation is partially clipped by the PFB channel filter,
but sufficient for reliable detection.

A second `AaCorrelator` instance parameterized with `with_sps(1)` handles
LE 2M access address matching.

### LE Coded

LE Coded uses the same 1 Msym/s GFSK physical layer but adds forward
error correction on top. The coded preamble (80 symbols of `00111100`
pattern) is highly distinctive and has essentially zero false-positive
rate.

The FEC pipeline:
1. **Preamble detection**: correlate for 80-symbol pattern at SPS=2
2. **Block 1** (always S=8): soft pattern demap, Viterbi decode to get AA (32 bits) + CI (2 bits) + TERM1 (3 flush)
3. **CI decode**: CI=0 means S=8 (125 kbps), CI=1 means S=2 (500 kbps)
4. **Block 2** (at S per CI): soft pattern demap, Viterbi decode to get PDU + CRC

The convolutional code (rate 1/2, K=4, generators g0=17o and g1=15o)
has 8 trellis states, so the Viterbi decoder runs in microseconds. Soft
pattern demapping uses the analog demod values (not hard bits) for
maximum coding gain.

### Extended Advertising

ADV_EXT_IND (PDU type 7) appears on primary advertising channels using
any PHY. The Common Extended Header is parsed to extract AuxPtr (secondary
channel, timing offset, PHY), advertiser address, ADI, and transmit power.
Since Blue Dragon captures all channels simultaneously, secondary
advertisements are already being received without needing to follow
AuxPtr chains.

### Performance Impact

Steps 4-6 of the decode chain only execute when steps 1-3 fail to decode
a burst. The LE 2M reslice is a single-pass re-sample of the existing
demod buffer (~1 microsecond). The coded preamble check fails fast on
non-matching bursts (~0.5 microseconds). Net overhead on the LE 1M path
is zero.

## ZMQ Wire Format

Packets are sent as ZMQ multipart messages, compatible with the Python
dashboard:

```
Frame 1 (optional): sensor_id (UTF-8 string, SNDMORE)
Frame 2 (optional): GPS frame (24 bytes: 3x f64 LE = lat, lon, alt, SNDMORE)
Frame 3: packet data
```

Packet data format:
```
Byte 0:    type (0x01=BLE, 0x02=BT)
Bytes 1-16: pcaprec_hdr (ts_sec, ts_usec, incl_len, orig_len, all u32 LE)
Bytes 17+:  DLT-specific header + payload
```

BLE uses DLT 256 (BLUETOOTH_LE_LL_WITH_PHDR): 10-byte RF info header
followed by the raw BLE PDU (AA + header + payload + CRC).

Classic BT uses DLT 255 (BLUETOOTH_BREDR_BB): 22-byte BR/EDR header
(rf_channel, signal, noise, ac_errors, LAP, flags) followed by optional
7-byte raw FEC-encoded header (used for UAP recovery on the dashboard).

## PCAP Format

Global header: DLT 192 (PPI) to support mixed BLE + Classic BT packets
in the same capture file with per-packet GPS coordinates.

Per-packet structure:
```
pcaprec_hdr (16 bytes)
PPI header (8 bytes: version=0, flags=0, length=8, dlt=255|256)
DLT-specific header + payload
```

When GPS is active, a PPI-GPS field (type 30002) is inserted between
the PPI header and the DLT payload, carrying lat/lon/alt in fixed3_7
format. Wireshark displays these natively.

## C2 Control Protocol

Dashboard ROUTER binds on data_port + 1 (e.g., 5556). Sensors DEALER
connect with identity = sensor_id.

**Heartbeat** (sensor -> dashboard, every 5s):
```json
{
  "type": "heartbeat",
  "sensor_id": "roof",
  "sdr": "bladerf",
  "center_freq": 2441,
  "channels": 96,
  "gain": {"value": 30.0},
  "squelch": -45.0,
  "total_pkts": 12345,
  "pkt_rate": 52.3,
  "crc_pct": 96.1,
  "uptime": 3600,
  "gps": [42.3601, -71.0589, 10.5]
}
```

**Commands** (dashboard -> sensor):

| Command | Fields | Runtime? |
|---------|--------|----------|
| `set_gain` | `gain`, optional `lna`/`vga` | Yes |
| `set_squelch` | `threshold` (-100 to -5) | Yes |
| `get_status` | (none) | Yes (triggers heartbeat) |
| `query_gatt` | `mac` | Yes (requires `--hci`) |
| `restart` | `center_freq`, `channels` | Requires restart |

Runtime commands update atomics that the recv and worker threads poll.
Restart is acknowledged but not yet implemented (would require execv).
