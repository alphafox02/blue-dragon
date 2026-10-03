# Blue Dragon

Wideband BLE and Classic Bluetooth passive sniffer written in Rust.

Most BLE sniffers capture one channel at a time. Blue Dragon uses a
polyphase filterbank channelizer to capture **up to 40 BLE channels
simultaneously** from a single SDR, decoding BLE 5 LE 1M, LE 2M, and
LE Coded PHYs plus Classic Bluetooth BR/EDR detection in the same
passband. Wideband capture avoids receiver-side channel hopping, subject to
the SDR bandwidth, squelch, signal quality, and decoder limits.

Output is Wireshark-compatible PCAP with optional ZMQ streaming for
multi-sensor deployments, GPS tagging for drive surveys, and a web
dashboard for real-time monitoring.

## Status

| Feature | Status | Notes |
|---------|--------|-------|
| USRP B210 capture | Tested | USB 3, validated at -C 40 and -C 60 |
| BLE LE 1M decoding | Tested | 95-96% CRC pass rate |
| BLE LE 2M decoding | Tested | |
| BLE LE Coded decoding | Tested | Controlled Coded-PHY advertising and drive captures confirmed |
| Classic BT BR detection | Tested | LAP extraction and CRC-valid BR payloads confirmed OTA |
| Classic BT UAP recovery | Tested | Autonomous OTA recovery confirmed from two clock-consistent, CRC-valid payloads |
| Classic BT EDR decoding | Experimental | Sync, DQPSK/8DPSK, matched filter, and CRC paths are unit-tested; CRC-valid OTA EDR not yet confirmed |
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
| Spectran V6 | Tested | 245 MHz BW, ~91% CRC OTA (`-C 245 --aaronia-decim 2`) |
| RFNM (Lime) | Tested | Full BLE band (122.88 Msps), 71-78% CRC OTA |
| HCI GATT probing | Untested | Compiles, needs end-to-end test with --hci |
| HCI active scanning | Tested | --active-scan enriches device data |
| Channel counts -C 60+ | Tested | -C 40 and -C 60 validated on USRP and bladeRF |
| ESP32-S3 (eSpDR, USB) | Experimental | One board covers 16 MHz; five tiled boards cover the whole band (88-95% CRC OTA) |

## Supported Hardware

| SDR | Interface Flag | Bandwidth | ADC Bits | Notes |
|-----|---------------|-----------|----------|-------|
| USRP (B200/B210) | `-i usrp-MODEL-SERIAL` | 4-56 MHz | 12-bit | AD9361 (61.44 Msps, 56 MHz analog BW) |
| HackRF One | `-i hackrf-SERIAL` | 4-20 MHz | 8-bit | 20 MHz max sample rate |
| bladeRF 2.0 | `-i bladerf0` | 4-56 MHz (normal), up to 122 MHz (oversample) | 12-bit (normal) / 8-bit (oversample) | AD9361 (oversample overclocks beyond AD spec) |
| SoapySDR | `-i soapy-N` | Varies | Varies | Generic SDR support |
| Spectran V6 | `-i aaronia` | 46-245 MHz | f32 | Supported `-C` values: 46, 61, 77, 92, 122, 184, 245 (device-dependent). Other values snap up to the nearest supported clock automatically. |
| RFNM (Lime) | `-i rfnm` or `-i rfnm-SERIAL` | 122 MHz | 12-bit | 122.88 Msps base clock, all 40 BLE channels |
| Epiq Sidekiq family | `-i sidekiq-SERIAL` | per-device | 12 or 16 (per-device) | Bit depth, LO range, sample-rate range and gain index range are queried from the device at open; the recv path scales samples to i16 per the reported ADC resolution. Family includes Stretch / m.2-2280 / m.2 (3042) / mPCIe (AD9361/4, 12-bit); X2 / X4 / X40 / Nv100 / Nvm2 (16-bit). Opt-in `--features sidekiq`; requires libsidekiq SDK (`$Sidekiq_DIR` or `~/sidekiq_sdk_current`). |
| ESP32-S3 (eSpDR, USB) | `-i espdr0` or `-i espdr:/dev/ttyACM0` | 16 MHz | 10-bit | Experimental. The ESP's own radio as a receiver, over its USB port with no extra hardware; the ESP sends bursts cut to their channel. Five ESPs tile the whole band (`-i espdr -C 80`); one to four share it through the ESP's 80 MHz fold. `--espdr-load` loads the firmware. Opt-in `--features espdr`; see [ESP32-S3](#esp32-s3-espdr). |

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
| Spectran V6 | 60 | 30-60 | 20-30 | `-g N` → reflevel -N dBm, clamped [-36, 10]; preamp=Auto, auto-scaled f32→i16 |
| RFNM (Lime) | 30 | 20-30 | 30 | Lime gain range -24 to 30 dB |
| Epiq Sidekiq | varies | **30-35** | TBD | `-g` is the device's RX gain *index* (range varies per model, read from the SDK at open time). On AD9361-based cards (Stretch / m.2-2280, m.2-3042, mPCIe) each step ≈ 1 dB; indices 30-35 measured highest CRC on a populated office BLE band in our testing. `--sidekiq-agc` switches to the SDK's auto-gain, `--sidekiq-no-dc` disables FPGA DC offset correction (on by default), `--sidekiq-gpsdo` enables FPGA GPSDO on cards with an integrated GPS receiver. **Recommended production config: build with `--features sidekiq,zmq,gps,gpu` and run `-C 60 -g 30`** for 85%+ CRC at 60 MHz of in-band BLE capture with the Intel/AMD integrated GPU PFB. Avoid prime-number `-C` values (e.g. 53, 59, 61) with the GPU PFB; CPU path handles them fine. |
| SoapySDR | 60 | Device-dependent | Device-dependent | Depends on underlying hardware |
| ESP32-S3 | 60 (too high) | 20-30 with an antenna, 44-56 on a PCB antenna | TBD | `-g` is a gain-table selector (0-127, not dB) and not linear; tune per board with `BD_ESPDR_GAINS` |

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

ESP32-S3 boards as receivers (see [ESP32-S3](#esp32-s3-espdr)) need the
`espdr` feature and libudev's headers:

    sudo apt install libudev-dev
    cargo build --release --features "usrp,hackrf,bladerf,soapysdr,zmq,gps,espdr"

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

ESP32-S3 support adds the `espdr` feature and `libudev-dev`, as above.

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

### Spectran V6

Requires the RTSA Suite Pro installed to `/opt/aaronia-rtsa-suite/`.
The SDK and runtime library (`libAaroniaRTSAAPI.so`) live in the
`Aaronia-RTSA-Suite-PRO/` subdirectory. The build system embeds an rpath
so the binary finds the library at runtime without `LD_LIBRARY_PATH`.

    cargo build --release --features "aaronia,zmq"

The Spectran V6 backend uses `spectranv6/raw` mode with `outputformat=iq`
to get wideband IQ samples. Quick start (recommended for full BLE band):

    blue-dragon -l -i aaronia -C 92 --aaronia-decim 2 --check-crc --stats

Measured CRC pass rate at this setting: **~90% OTA** in a typical office
RF environment. The `--aaronia-decim 2` halfband path uses the device's
internal DC notch and consistently outperforms the Full-decimation mode.

Other shortcuts:

    blue-dragon -l -i aaronia -C 92 --check-crc --stats             # Full decim, ~85-87% OTA
    blue-dragon -l -i aaronia -a --check-crc --stats                # all 40 BLE channels

#### `-C` (channels / sample rate)

Aaronia receiver clocks are at integer-MHz granularity: **46, 61, 77, 92,
122, 184, 245**. Pass `-C N` where `N` is one of these values, or any
other value -- non-matching values snap up to the next supported clock
with a warning, so `-C 40` (the CLI default) becomes `-C 46`. Older
firmware or non-ECO models may not support 46/61/77 MHz; the backend
falls back to the next-higher clock automatically and tells you on stderr.

`-a` / `--all-channels` resolves to `-C 92` on Aaronia (the 92.16 MHz
clock comfortably covers the full 78 MHz BLE band, 2402-2480 MHz). Other
backends keep the historical `-C 96`.

The device's actual sample rate is read from `packet.stepFrequency`
(per the vendor docs) -- the 92 MHz clock is really 92.16 MHz, and the
per-channel resampler corrects the 0.17% timing offset transparently.

#### `--aaronia-decim` (halfband / quarterband / ...)

`--aaronia-decim D` sets the device decimation factor (`1` = Full default,
`2` = halfband, `4`, `8`, ... up to `512`). Decimation > 1 runs the ADC
at `D ×` the effective rate and applies a digital halfband filter,
giving sharper antialiasing **and a hardware DC notch** that suppresses
LO leakage / IQ-imbalance DC artifacts at the cost of more USB bandwidth
before decimation. The DC notch in particular tends to clean up demod
quality near band-center.

`-C` and `--aaronia-decim` combine: `-C N` is the *effective* bandwidth
in MHz; the device clock is auto-chosen as the smallest supported clock
≥ `N × D`, and `-C` snaps to whatever effective bandwidth that clock
actually produces after decimation.

Examples:

    -C 46  --aaronia-decim 2    # clock 92 MHz / 2 = 46.08 MS/s + DC notch
    -C 92  --aaronia-decim 2    # clock 184 MHz / 2 = 92.16 MS/s, full BLE band
    -C 122 --aaronia-decim 2    # clock 245 MHz / 2 = 122.88 MS/s
    -C 184 --aaronia-decim 2    # 245 / 2 = 122 effective; -C snaps down to 122
    -C 92  --aaronia-decim 4    # clock 245 / 4 = 61 effective; -C snaps to 61

#### `-g` (gain → reference level)

`-g N` maps to reference level `-N dBm`, clamped to `[-36, 10]`. Default
`-g 60` → **reflevel -36 dBm** (most sensitive). Examples: `-g 20` →
-20 dBm, `-g 10` → -10 dBm, `-g 0` → 0 dBm (most headroom for strong
nearby transmitters).

The preamp is set to `"Auto"` by default, letting the device pick the
right amplifier path (Disabled / Amp / Preamp / Both) for the configured
reflevel. The RX filter is `"Auto Extended"` for full BLE-band coverage.

The backend auto-scales the f32 samples to i16 using RMS measured over
20 packets at startup (25th percentile to filter WiFi burst outliers).
The chosen scale is logged on stderr at open time:
`Aaronia auto-scale: rms=... scale=...`.

#### Live diagnostics

Every 5 s during streaming, the SDR thread logs warn-flag deltas if any
of the device's data-quality bits fire:

    Aaronia warn flags (5s deltas): overflow=N dropped=N inaccurate=N resampled=N time_disc=N

Silent zeros means clean capture. Non-zero `dropped` indicates USB
backpressure; non-zero `inaccurate` is a clock or calibration issue;
non-zero `resampled` means the API is internally resampling and the
device rate may differ from the requested clock.

### RFNM

Requires librfnm and spdlog installed (typically from source to
`/usr/local`). The RFNM with Lime daughtercard runs at 122.88 Msps,
covering the entire BLE band (all 40 channels) simultaneously. GPU
acceleration is required at this sample rate.

    cargo build --release --features "rfnm,gpu,zmq"

    blue-dragon -l -i rfnm -C 122 -c 2441 -g 30 --check-crc --stats

The 122.88 MHz base clock doesn't divide evenly into 1 MHz channels
(122.88/61 = 2.0144 Msps per channel). The FSK demodulator automatically
resamples to correct the timing drift. This is transparent and does not
affect other SDR backends.

### ESP32-S3 (eSpDR)

Experimental. An ESP32-S3 can act as a 2.4 GHz IQ receiver using the
[eSpDR](https://github.com/h0m3us3r/eSpDR) radio firmware, which exposes
the chip's undocumented sample-dump engine. The original eSpDR streams
80 Msps through an external FPGA; the
[alphafox02/eSpDR](https://github.com/alphafox02/eSpDR) fork adds a mode
that uses only the ESP's own USB port, so a bare ESP32-S3 dev board works.
One ESP covers 16 MHz; five on a powered USB hub cover the whole band.

#### Quick start

Build with the `espdr` feature (it needs `libudev-dev`), and get the
firmware image `iq-source.bin` from the fork's
[releases](https://github.com/alphafox02/eSpDR/releases):

    cargo build --release --features espdr
    # one ESP, centred on advertising channel 38
    blue-dragon -l -i espdr0 -C 16 -c 2426 -g 28 --check-crc --stats --espdr-load --espdr-image iq-source.bin
    # five ESPs, the whole band (see "Whole band with several ESPs")
    BD_ESPDR_GAINS=44,28,44,44,56 blue-dragon -l -i espdr -C 80 -c 2441 --check-crc --stats --espdr-load --espdr-image iq-source.bin

The ESPs run the fork's firmware from RAM: nothing is written to flash,
and a power cycle restores a board. `--espdr-load` makes blue-dragon load
it into any ESP that is not running it, or runs an older revision, before
starting; ESPs already running it are used as they are. The loader talks to
the ESP32-S3's ROM directly, so esptool is not needed. Without
`--espdr-image`, it looks for `/usr/share/espdr/iq-source.bin`, then
`/usr/local/share/espdr/iq-source.bin` (`BD_ESPDR_IMAGE` also names an
image), so a distribution can install the image once.

`--espdr-load` is opt-in because `-i espdr` takes every attached ESP32-S3,
and one running other firmware would be reset into this one. Without it,
an ESP that does not answer stops blue-dragon with a message saying so, and
one running an older firmware revision is used with a warning. The fork's
own loader does the same from Python (`python3 usb/load.py --image
iq-source.bin`, which uses esptool), and building the image yourself takes
`. $IDF_PATH/export.sh` (ESP-IDF v5.5.3 or later) and `make -C esp32s3`.

#### Settings

Use `-C 16` with a single ESP (`-i espdr0`; `-C 80` with a single named ESP
takes 0.2 ms snapshots at 80 Msps). `-c` sets the ESP's LO (2210-2790 MHz);
2426 centres the window on advertising channel 38. `-g` is the ESP's
gain-table selector (0-127, not dB), and the table is not linear: with an
antenna attached, about 20-30 avoids clipping, while boards on their PCB
antenna did best at 44-56, and above about 60 decoding collapsed. With
several ESPs attached, `espdr1`, `espdr2`, ... select them in the order of
their USB serial port names, and `BD_ESPDR_GAINS` sets each board's gain.

#### How it works

USB Full Speed (about 1 MB/s) cannot carry the full stream, so the ESP
watches a 16 MHz window continuously and sends only the bursts above the
noise floor, each timestamped; blue-dragon places them on a true timeline
and fills the gaps with noise at the reported floor. The ESP also cuts each
burst down to its own channel at 4 Msps with the S3's vector unit, a
quarter of the data, and blue-dragon restores it to the window; this lets
about twice as many packets through and bursts up to 3 ms (a whole 3-DH5).
Wi-Fi bursts are dropped on the ESP to save USB bandwidth.

On a busy band with an antenna at 2426 MHz this decoded 514 BLE packets
with a valid CRC in 45 s (257 with whole-window bursts); with a Classic
link flooded by `l2ping` at 2441 MHz, about 11,000 Classic framings and a
few EDR packets in 38 s. The USB link is still the limit on a busy band, and
bursts the ESP could not send are reported as overflows. Older firmware
without channelization sends whole-window bursts, and firmware without
streaming falls back to 1 ms snapshots automatically.

The ESP's DC offset is often 20 dB above the noise in a 1 MHz channel. When
five boards tile the band, each LO sits between two channels, and
blue-dragon removes the offset from each tiled board's channelized bursts.
Decoding one board's `l2ping` recording both ways, the two Classic channels
beside its LO went from 10 packets each to about 100, with every other
channel unchanged (they received 240-420 each, so those two remain weaker).
A single ESP or folded ESPs can have a channel on the LO, where removing the
offset would also take part of the packet, so their bursts are left as
received.

#### Options for testing

| Variable | Effect |
|---|---|
| `BD_ESPDR_WIDE=1` | whole-window bursts instead of channelized ones (keeps two simultaneous signals on different channels) |
| `BD_ESPDR_KEEP_WIDEBAND=1` | keep Wi-Fi bursts instead of dropping them on the ESP |
| `BD_ESPDR_SNAPSHOT=1` | 1 ms snapshots instead of streaming |
| `BD_ESPDR_TELEMETRY=1` | once-per-second receiver diagnostics; the stream format is unchanged |
| `BD_ESPDR_TRIGGER_RATIO=N` | burst-to-noise power ratio, 3 to 31 (default 4, 6 dB) |
| `BD_ESPDR_FILTER=N` | baseband filter code, 0-63 (see below) |
| `BD_ESPDR_LAYOUT=...` | `tile`, `tile-ble` or `fold` (see below) |

#### Whole band with several ESPs

    BD_ESPDR_GAINS=44,28,44,44,56 blue-dragon -l -i espdr -C 80 -c 2441 --check-crc --stats --espdr-load

`-C 80` with `-i espdr` (every attached ESP) or a comma list (`-i
espdr0,espdr2`) makes the ESPs one 80 MHz receiver. blue-dragon picks the
layout from how many there are:

| ESPs | Layout | What each ESP hears | Channels reported |
|---|---|---|---|
| 5 | tiled | its own 16 MHz, tuned 16 MHz apart | the true channel |
| 1-4 | folded | the whole band, folded into 16 MHz | true for BLE, may be a fold for Classic |

`BD_ESPDR_LAYOUT=tile`, `tile-ble`, or `fold` overrides the choice. `tile`
is the five-board default and retains every Classic channel. `tile-ble` is
an opt-in five-board layout for `-c 2441` that favours BLE advertising
reception as described below. `BD_ESPDR_GAINS` sets each board's gain in
board order. It is worth tuning per board, since boards of the same kind
differ: here one decoded nothing at 44 but 150 packets on channel 39 in
15 s at 48 or 56, and the board with an external antenna did best at
20-28. With
`BD_ESPDR_GAINS=44,28,44,44,56` the five boards received 742 valid BLE
advertising packets in 30 s, against 294 at a uniform 52 (40 for the
external antenna). A single antenna feeding every board through a powered
splitter would even them out.

**The baseband filter.** The ESP's 16 Msps samples are its 80 Msps capture
decimated without a digital filter, so its analog baseband filter is all
that keeps signals outside the window from folding in. The eSpDR firmware
leaves it wide open (about 69 MHz), and then a signal 23 MHz away comes
through as strongly as one in the window. blue-dragon narrows it to code 54
(set `BD_ESPDR_FILTER`, 0-63, larger is narrower) for a single ESP and when
tiled; measured on channel 39, a signal 9 or 23 MHz outside the window then
no longer came through while one 7 MHz inside still did, and the noise
floor dropped. The register mapping and bandwidth calibration came from the
[ESP-SDR](https://github.com/ESPARGOS/esp-sdr) project.

**Tiled.** Each burst is placed once at the frequency it came from. The five
LOs are staggered by half a MHz so the integer-MHz Bluetooth channels fall
inside a tile rather than on its boundary. Five ESPs on one USB hub, four on
their PCB antennas, received 894 BLE
advertising packets with a valid CRC in 25 s across channels 37, 38 and 39,
and Classic packets at their true channels (15,000 in 38 s during an
`l2ping` flood, with the link's UAP recovered). A 29 s run with the staggered
LOs received 5,968 Classic packets and all four former boundary channels
(2417, 2433, 2449 and 2465 MHz).

A board's outermost channels, 7.5 MHz from its LO, lose several dB to the
baseband filter. In the complete layout that edge falls on 2480 MHz
(advertising channel 39), where one test received 8-18 valid packets in 15
s. `BD_ESPDR_LAYOUT=tile-ble` tunes the top board 1 MHz higher and brought
that to 56-91 packets, increasing the total by about a third. It deliberately
does not receive 2465 MHz (Classic channel 63), so use it for BLE-focused
captures rather than as the general default. The program prints that
trade-off at startup when the profile is selected.

Current eSpDR firmware also latches each board's sample count on common USB
start-of-frame boundaries (every device below one USB host sees the same
1 ms frame numbers). blue-dragon fits those observations to the reference
board, removing USB arrival latency from tiled timing and tracking the
crystals' drift (here -3 to +2 ppm). On the five-board hub the boards'
sample clocks agreed to 1.2-1.8 us RMS (under about 7 us peak). During an
`l2ping` flood, the share of Classic packets within 7.5 us of their link's
slot timing rose from about a fifth with arrival timing alone to all of
them, and the link's UAP was recovered. Older firmware remains usable with
arrival timing.

**Folded.** With the filter open each ESP hears about 80 MHz around its LO
folded into its window (flat to about 25 MHz from the LO, 5 dB down at 39,
gone by 55), so a burst is known only to within a multiple of 16 MHz. Every
ESP is tuned to the same LO and sends only its share of the channel
positions, and each burst is placed at every frequency it could have come
from; a BLE packet passes its CRC only at the channel it was sent on, so the
decoders sort them out. Two ESPs received 514 valid advertising packets in
25 s across all three channels. Classic packets decode at every fold, since
their whitening does not depend on the channel; a piconet sends on one
channel at a time, so the repeats are dropped and each packet is reported
once, though its reported channel may be any of the folds. A BLE packet
that fails its CRC is likewise dropped when a copy at another fold passes
(failing packets are held about 30 ms for this). Every board also sends the
bursts where channel 38 folds, so each board's sample clock is measured
against a reference board's from the same packets, to a fraction of a
microsecond; the output timeline is the reference board's own sample count,
so packets keep the slot timing a single receiver would give them.

Classic packets reported at a position that can contain a 16 MHz image carry
the standard `RF Channel Aliasing` flag in PCAP and ZMQ output
(`btbredr_rf.flags.rf_channel_aliasing` in Wireshark). This covers every
reported channel in folded mode, the outer positions of each tile, and the
two outer positions of a single 16 Msps receiver. The flag preserves the
packet and its observed channel while making the uncertainty explicit; it
does not guess which of the two RF channels transmitted it.

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
| `aaronia` | Spectran V6 support | RTSA Suite Pro |
| `rfnm` | RFNM (Lime daughtercard) support | librfnm, spdlog |
| `sidekiq` | Epiq Sidekiq family support | libsidekiq SDK (`$Sidekiq_DIR` or `~/sidekiq_sdk_current`) |
| `espdr` | ESP32-S3 receiver over USB (eSpDR firmware) | libudev-dev |

#### Sidekiq DMA buffer tuning (high-rate capture only)

PCIe-attached Sidekiq cards stream IQ samples through a fixed-size DMA
ring buffer in kernel memory. The default `RingBufferPacketCount=2048`
gives 8 MB of buffer (~33 ms at 61 Msps), which is enough at low rates
on a quiet host but can overflow at the AD9361's 61 Msps ceiling when
the host has any latency jitter (VMs, browsers, builds, etc.). Symptom
is a non-zero **drops** counter rising in the periodic Sidekiq log line.

To raise the buffer:

```
sudo rmmod dmadriver
sudo insmod /home/$USER/sidekiq_image_current/driver/$(uname -r)/dmadriver.ko \
    RingBufferPacketCount=8192   # 32 MB, ~130 ms at 61 Msps
cat /sys/module/dmadriver/parameters/RingBufferPacketCount   # verify
```

To persist across reboots, drop a config file into the SDK's driver
config directory (the load script picks it up via `modprobe --config`):

```
echo 'options dmadriver RingBufferPacketCount=8192' \
    | sudo tee $HOME/sidekiq_image_current/driver/driver_config/dmadriver.conf
```

Recommendation by card type:
- High-rate AD9361 cards (Stretch / m.2-2280, m.2-3042, mPCIe): 8192
- 16-bit / wider cards (Nv100, Nvm2, X4, X40): 8192 or higher
- Low-rate or embedded targets (Z2/Z3u): default is fine, do not raise

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
    --burst-file FILE       Replay a compact channelized burst capture
    -l, --live              Capture live from SDR

SDR settings:
    -i, --interface IFACE   SDR device (e.g. usrp-B210-SERIAL)
    -c, --center-freq FREQ  Center frequency in MHz (default: 2441)
    -C, --channels N        Number of channels (default: 40)
    -a, --all-channels      Full BLE band: sets -C 96 -c 2441
    -g, --gain DB           SDR gain (default: 60)
    -s, --squelch DB        Squelch threshold (default: -45)
    --antenna PORT          RX port (USRP: RX2/TX/RX, bladeRF: RX1/RX2)
    --hackrf-lna DB         HackRF LNA gain (default: 40)
    --hackrf-vga DB         HackRF VGA gain (default: 20)
    --aaronia-decim D       Spectran V6 decimation (1, 2, 4, ... 512)
    --sidekiq-agc           Sidekiq: AD9361 AGC instead of the -g gain index
    --sidekiq-no-dc         Sidekiq: disable FPGA DC offset correction
    --sidekiq-gpsdo         Sidekiq: enable the card's GPSDO
    --espdr-load            ESP32-S3: load the firmware into boards that need it
    --espdr-image PATH      ESP32-S3: firmware image for --espdr-load

Output:
    -w, --write FILE        Output PCAP to file or FIFO
    --write-bursts FILE     Record channelized IQ bursts for replay
    --burst-limit-mb N      Stop burst recording at N MiB (default: 512; 0=unlimited)
    --check-crc             Enable BLE CRC-24 validation (Classic payload CRC is always required)
    --classic-address ADDR  Trust a known Classic BD_ADDR (repeatable)
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
    --sample-rate HZ        Sample rate of the file

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

Capture 92 MHz with Spectran V6:

    blue-dragon -l -i aaronia -C 92 --check-crc --stats

Capture full BLE band with bladeRF at recommended OTA gain:

    blue-dragon -l -i bladerf0 -a -g 30 --check-crc --stats

Capture full BLE band (all 40 channels) with RFNM:

    blue-dragon -l -i rfnm -C 122 -c 2441 -g 30 --check-crc --stats

Capture the whole band with five ESP32-S3 boards, loading their firmware:

    BD_ESPDR_GAINS=44,28,44,44,56 blue-dragon -l -i espdr -C 80 -c 2441 --check-crc --stats --espdr-load

Capture with active BLE scanning for device enrichment:

    blue-dragon -l -c 2441 -C 40 --hci --active-scan --zmq tcp://dashboard:5555 --check-crc

Capture BLE 5 Long Range (LE Coded PHY) on advertising channels:

    blue-dragon -l -i usrp-B210-SERIAL -c 2402 -C 4 --check-crc --coded-scan --stats

Without `--coded-scan`, coded decoding still runs on any squelch-triggered
burst that fails LE 1M and BT decode. The flag adds continuous sampling on
channels 37/38/39 to catch weak coded signals below the normal squelch
threshold. Overlapping scan windows and the normal squelch path are
de-duplicated, so one RF transmission is reported once.

Read from a previously recorded IQ file:

    blue-dragon -f recording.ci16 -c 2441 -C 20 -w output.pcap --check-crc --stats

Record a bounded regression capture with a bladeRF, then replay it later:

    blue-dragon -l -i bladerf0 -a -g 30 --write-bursts lab.bdb --burst-limit-mb 512 --check-crc
    blue-dragon --burst-file lab.bdb -w replay.pcap --check-crc --stats

Supply a Classic address known from an independent source when validating
header and payload decoding:

    blue-dragon --burst-file lab.bdb --classic-address 10:20:30:40:50:60 -w replay.pcap

`--classic-address` supplies trusted LAP/UAP ground truth. It does not claim
that the address was recovered from RF, and it does not bypass payload CRC
validation.

Compact burst files contain only the 2 Msps channelized windows selected by
the squelch or coded scanner, with timestamps, frequency, RSSI, and noise
metadata. IQ is scaled per record and stored as interleaved signed 16-bit
samples. This makes them practical regression artifacts while retaining both
successful decodes and rejected bursts needed to check false positives.

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
SDR (USRP / HackRF / bladeRF / SoapySDR / Spectran V6 / RFNM / Sidekiq,
     or ESP32-S3 bursts placed on a continuous timeline)
    |
    | int16 IQ samples (native precision)
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
    |-- Classic BT BR/EDR: Barker code, FEC/HEC, DPSK sync, payload CRC
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
| Classic BT UAP recovery | Autonomous CRC/clock recovery confirmed OTA |
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

The CPU pipeline receives int16 (i16) IQ samples from all backends,
preserving native ADC resolution where possible. The GPU pipeline
currently uses int8 (i8) for OpenCL kernel compatibility.

| SDR | Native ADC | CPU Pipeline (i16) | GPU Pipeline (i8) |
|-----|-----------|-------------------|-------------------|
| USRP B210 | 12-bit | SC16 wire format, full 12 bits | SC8, 8 bits |
| bladeRF (normal) | 12-bit (SC16_Q11) | `<< 4` to fill i16 range | `>> 4` to i8 |
| bladeRF (oversample) | SC8_Q7 | `<< 8` (no extra precision) | Native i8 |
| HackRF | 8-bit | `<< 8` (no extra precision) | Native i8 |
| SoapySDR (CS16) | Device-dependent | Dynamic left-shift | Right-shift to i8 |
| Spectran V6 | 32-bit float | Scaled f32 → i16 | Scaled f32 → i8 |
| RFNM (Lime) | 12-bit | CS16 native (12-bit left-shifted to i16) | CS16 → i16 GPU path |

The ESP32-S3's 10-bit samples are scaled by 64 into the i16 range. The i16
pipeline gives 12-bit SDRs (USRP, bladeRF) their full dynamic
range -- about 24 dB more than the i8 path. HackRF is natively 8-bit,
so both paths are equivalent. The Spectran V6 benefits from 16-bit
quantization of its float samples instead of 8-bit.

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

### ESP32-S3: "not running the eSpDR firmware"

The ESP did not answer the eSpDR handshake: it was power-cycled (the
firmware lives in RAM), runs other firmware, or was left busy. Run with
`--espdr-load` (and `--espdr-image` if the image is not installed in
`/usr/share/espdr/`), or replug the board and load it again. A warning that
a board runs an older firmware revision is fixed the same way. Rising
`overflow` counts in `--stats` mean the ESP had more bursts than USB could
carry; that is normal on a busy band.

### Spectran V6: library not found

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

ESP32-S3 reception builds on h0m3us3r's
[eSpDR](https://github.com/h0m3us3r/eSpDR) radio firmware, and the
baseband filter calibration came from the
[ESP-SDR](https://github.com/ESPARGOS/esp-sdr) project.

## License

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

Copyright 2025-2026 CEMAXECUTER LLC
