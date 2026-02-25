# Blue Dragon Architecture

## Crate Layout

```
blue-dragon/
├── crates/
│   ├── app/          bd-app       CLI entry point, pipeline orchestration
│   ├── dsp/          bd-dsp       PFB channelizer, FFT, burst detection, FSK demod
│   ├── sdr/          bd-sdr       SDR hardware abstraction (USRP, HackRF, BladeRF, SoapySDR, Aaronia)
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
| `aaronia` | bd-sdr | Aaronia RTSA Suite (libAaroniaRTSAAPI) |
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
│          │  Vec<i8>      │  (or GPU)    │  BatchMsg         │  (8 workers)   │
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
| 1 | `sdr-recv` | Drains SDR hardware via `recv_into()`, sends SC8 buffers to PFB. Polls `gain_pending` atomic for C2 gain changes and calls `sdr.set_gain()`. |
| 2 | main | PFB channelizer + IFFT (CPU path) or GPU submit loop. Converts SC8 to i16, runs polyphase filter, broadcasts channel data to workers. |
| 3-10 | `burst-0` .. `burst-7` | Each assigned a disjoint subset of BLE channels. Per-channel AGC + squelch detection. Polls `squelch_pending` atomic. Sends detected bursts to decode. |
| 11 | `decode` | FSK demodulation, BLE/BT protocol decode, CRC validation, PCAP output, ZMQ publish. Updates heartbeat stats. |
| 12 | `c2-control` | ZMQ DEALER socket to dashboard ROUTER. Sends JSON heartbeat every 5s. Receives commands. (Only with `--zmq`.) |
| 13 | `c2-dispatch` | Reads ControlCommand channel, stores gain/squelch in atomics for other threads to pick up. (Only with `--zmq`.) |
| 14 | `hci-scan` | LE active scanning via BlueZ. Sends ScanResult to ZMQ. Rate-limited to 1 update per MAC per 10s. (Only with `--active-scan`.) |

### Inter-thread communication

| Channel | Type | Capacity | Payload |
|---------|------|----------|---------|
| SDR -> PFB | crossbeam bounded | 32 | `Vec<i8>` (SC8 samples) |
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
SDR (SC8: signed int8 IQ pairs, or f32 scaled to i8 for Aaronia)
  │
  │ sdr-recv thread: recv_into(&mut [i8])
  v
Convert to i16 (left-shift by 8)
  │
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
- `recv_into(&mut [i8]) -> usize` -- receive SC8 samples
- `max_samps() -> usize` -- maximum batch size
- `overflow_count() -> u64` -- underrun counter
- `set_gain(gain)` -- runtime tunable via C2

| Backend | Native Format | Conversion | Effective Bits | Max Sample Rate |
|---------|--------------|------------|----------------|-----------------|
| USRP | SC8 (UHD converts 12-bit internally) | None | 8 | 61.44 Msps (56 MHz analog BW) |
| HackRF | CS8 (unsigned reinterpreted as signed) | None | 8 | 20 MHz |
| bladeRF | SC16_Q11 (12-bit) / SC8_Q7 (oversample) | >>4 to i8 | 8 (12 native up to 56 MHz) | 61.44 Msps normal (56 MHz analog BW), 122.88 Msps oversample |
| SoapySDR | CS8 preferred, CF32/CS16 fallback | Format-dependent | 8 | Device-dependent |
| Aaronia | f32 (32-bit float IQ) | f32 * scale -> i8 | 8 (f32 native) | 92.16 MHz |

The `SdrHandle` enum dispatches to the correct backend at runtime. HackRF
has separate LNA/VGA gain controls; all others use a single gain value.
The Aaronia backend ignores the `-g` gain flag and instead uses an internal
reference level (-20 dBm) with auto-scaling.

### Aaronia Spectran V6

The Aaronia backend (`sdr/src/aaronia.rs`) uses the RTSA API in
`spectranv6/raw` mode to receive wideband IQ samples as 32-bit float
pairs. Key details:

- **Clock:** 92,160,000 Hz (92.16 MHz). Requested as 92 MHz but the API
  selects the nearest supported clock, which is 92.16 MHz. At `-C 92`,
  this gives 92 channels at 1.0017 MHz each (0.17% wider than 1 MHz).
- **Decimation:** 1:1 (full rate, no decimation).
- **Reference level:** -20 dBm. This sets the ADC full-scale. Lower values
  increase sensitivity but clip on strong signals (WiFi).
- **Auto-scale:** At startup, measures RMS over 20 packets. Uses the 25th
  percentile RMS to compute a scale factor that maps noise floor to ~2
  LSBs in i8. This filters out WiFi burst outliers that would inflate
  a single-packet RMS measurement. Target: ~36 dB headroom for BLE
  signals before clipping.
- **Packet format:** Each RTSA API packet contains `num` complex samples
  with `stride` floats between samples (stride >= 2). The backend reads
  `fp32[i*stride]` (I) and `fp32[i*stride+1]` (Q) per sample.
- **recv_into_i16:** An alternate method exists for future i16 pipeline
  support. Scales f32 by `scale * 256` to fill the i16 range.
- **Overflow detection:** Checks `PACKET_WARN_OVERFLOW` flag per packet.
- **No sudo required:** The RTSA API handles USB communication internally.

### Precision chain (current i8 pipeline)

```
SDR ADC (8-32 bit native)
    |
    | recv_into(&mut [i8]): all backends truncate/scale to signed 8-bit
    v
crossbeam channel: Vec<i8> (bounded, 32 slots)
    |
    | PFB thread: (sample as i16) << 8 -- sign-extend to i16
    v
PFB channelizer: i16 fixed-point arithmetic
    |
    | IFFT: output Complex32 (f32 pairs)
    v
Burst catcher / demod / decode: all f32
```

The i8-to-i16 promotion via left-shift preserves the sign but fills
the lower 8 bits with zeros. All backends effectively deliver 8-bit
precision regardless of native ADC resolution. A future i16 pipeline
would change `Vec<i8>` to `Vec<i16>` and have each backend deliver
native precision, improving dynamic range for USRP (+4 bits/24 dB),
bladeRF (+4 bits/24 dB), and Aaronia (+8 bits/48 dB).

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
