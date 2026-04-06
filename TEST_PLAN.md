# Blue Dragon Test Plan

## Hardware Required
- USRP B210 or bladeRF 2.0 micro (SDR RX/TX)
- WHAD ButteRFly dongle (BLE injection/sniffing)
- Intel BT 5.3 adapter (hci0, for HCI-based tests)
- Any BLE device (phone, fitness tracker, smart bulb) as target

## Test Status Legend
- [ ] Not tested
- [x] Passed
- [~] Partial / needs investigation
- [-] Not applicable (hardware missing)

---

## Phase 1-2: Core Pipeline + VITA 49

### SDR Backends
- [x] bladeRF RX: `blue-dragon -l -i bladerf0 -C 4 -c 2402 -g 40 --stats --check-crc`
  - 215 BLE/5s, 99.1% CRC with --hci --active-scan
- [ ] USRP RX: `blue-dragon -l -i usrp-B210-FCO2P05 -C 40 -c 2441 -g 60 --stats --check-crc`
- [ ] HackRF RX: `blue-dragon -l -i hackrf -C 20 -c 2441 --stats --check-crc`
- [ ] Aaronia Spectran V6 RX
- [ ] RFNM RX
- [ ] SoapySDR RX
- [ ] VITA 49 UDP input: `blue-dragon -l -i vita49 -C 40 -c 2441 --stats`
- [x] File input: `blue-dragon -f recording.ci16 --sample-rate 40M -c 2441 --format ci16 --stats --check-crc`
  - BLE + BT packets decoded, 86% CRC from ci16 file

### PCAP Output
- [x] Write PCAP: `-w output.pcap`, open in Wireshark
  - Packets from file and live capture. tshark decodes BLE + BT correctly
- [ ] PCAP with GPS: `--gpsd`, verify GPS coordinates in PPI headers
- [ ] PCAP error logging: fill disk, verify error message on stderr

### Wireshark Extcap
- [ ] `blue-dragon --extcap-interfaces` lists available SDRs
- [ ] Live capture in Wireshark via extcap

### ZMQ Streaming + Dashboard
- [x] ZMQ PUB + C2: `blue-dragon -l -i bladerf0 -Z tcp://127.0.0.1:5555 --hci --active-scan`
  - C2 connected, ZMQ PUB streaming
- [x] Dashboard receives packets: `python3 tools/zmq_web_dashboard.py tcp://*:5555 --no-db`
  - Devices found with names decoded, sensor connected via C2
- [ ] CurveZMQ encryption: `--zmq-curve-key keyfile`

---

## Phase 3: Passive Intelligence

### 3.1 Connection Hop Following
- [ ] Capture CONNECT_IND packet (pair two BLE devices near SDR)
- [ ] Verify connection tracked in stats: `conns: 1`
- [ ] Verify data channel packets decoded with connection CRC init
- [ ] (Code-only) Unit tests pass: `cargo test -p bd-protocol -- hop`

### 3.2 SMP Pairing Analysis
- [ ] Pair phone to BLE device near SDR, verify stderr shows:
  - `SMP: connection 0xXXXXXXXX pairing JustWorks (Legacy)`
  - `SMP WARNING: ...` for weak pairing
- [ ] Capture LTK distribution: `SMP: connection 0xXXXX LTK captured`
- [ ] Dashboard receives SMP events via `smp:` ZMQ topic
- [ ] (Code-only) Unit tests: `cargo test -p bd-protocol -- smp`

### 3.3 AES-CCM Decryption
- [ ] Load known LTK via `--keys` file (not yet implemented in CLI)
- [ ] Decrypt captured encrypted connection
- [ ] (Code-only) Unit tests: `cargo test -p bd-protocol -- crypto`
  - [x] AES-128 NIST vector, nonce construction, session key derivation

### 3.4 IRK Identity Resolution
- [ ] Dashboard `--irk-file` resolves RPAs: device shows as "resolved" mac_type
- [ ] SMP-captured IRK auto-added to dashboard resolver
- [ ] (Code-only) Unit tests: `cargo test -p bd-protocol -- rpa`
  - [x] RPA detection, ah() function, resolve match/no-match

### 3.5 Vulnerability Scanner
- [ ] Detect DJI drone (manufacturer data match)
- [ ] Detect HID-over-GATT keyboard (service UUID 0x1812)
- [ ] Detect medical device (glucose/heart rate service)
- [ ] (Code-only) Unit tests: `cargo test -p bd-protocol -- vuln`
  - [x] Name match, UUID match, manufacturer match, sensitive service

---

## Phase 4: Active HCI Capabilities

### 4.1 GATT Read (existing)
- [ ] `query_gatt` via dashboard: click device -> "query GATT" button
- [ ] Verify services/characteristics displayed in device detail panel
- [ ] C2 command: `{"cmd":"query_gatt","mac":"AA:BB:CC:DD:EE:FF"}`

### 4.1b GATT Write (new)
- [ ] Dashboard: click "write" button next to writable characteristic
- [ ] Enter hex data, verify write succeeds
- [ ] C2 command: `{"cmd":"write_gatt","mac":"...","char_uuid":"...","data":"0102ff"}`
- [ ] Verify error on non-writable characteristic

### 4.2 Advertisement Spoofing
- [ ] C2 command: `{"cmd":"spoof_adv","name":"FakeDevice","connectable":false,"duration":30}`
- [ ] Verify "FakeDevice" visible on phone BLE scanner
- [ ] Verify spoofed advertisement visible on bladeRF/USRP capture
- [ ] Test connectable mode (peripheral)

### 4.3 L2CAP Connection Flood
- [ ] C2 command: `{"cmd":"l2cap_flood","mac":"...","count":10,"hold_secs":10}`
- [ ] Verify target device becomes unresponsive to new connections
- [ ] Verify connections release after hold period
- [ ] Verify error handling for unreachable device

### 4.5 MITM Relay
- [ ] Framework only -- requires integration wiring
- [ ] Test with HCI adapter (peripheral) + WHAD (central)
- [ ] Verify relay state machine transitions

---

## Phase 5: TX / WHAD / Fuzzing

### 5.1 SDR TX (bladeRF)
- [ ] `BladerfTxHandle::open()` on bladeRF 2.0
- [ ] Transmit GFSK-modulated BLE packet
- [ ] Verify received by WHAD dongle or second SDR

### 5.2 GFSK Modulator
- [x] Unit tests pass: Gaussian filter, output length, unit magnitude, i16 range
- [ ] Generate BLE packet IQ, play via bladeRF TX
- [ ] Verify decoded by WHAD or blue-dragon on another SDR

### 5.3 WHAD Native Rust
- [x] Device discovery: FW 1.0.1, type=1, capabilities parsed correctly
- [x] BLE sniff: 123 packets/3s on ch37, all CRC valid
- [x] Peripheral mode TX: test marker received by bladeRF (84 packets)
- [x] Injection test: bladeRF decoded 135 WHAD-transmitted packets
- [x] Peripheral mode TX: "RAW-INJECT-99" -> bladeRF decode: 135 packets (2026-04-05)
- [ ] Connection sniffing: `sniff_conn_req()`
- [ ] Raw PDU injection via `send_raw_pdu()` (vs peripheral mode)
- [ ] Reactive jamming: test with controlled BLE connection

### 5.5 BLE Fuzzer
- [x] Unit tests: case generation, exhaustion, reproducibility
- [ ] Generate fuzz cases from default ADV_IND
- [ ] Transmit fuzz cases via WHAD `send_raw_pdu()`
- [ ] Monitor target device for crashes/anomalies
- [ ] Log results with case index for reproducibility

---

## Dashboard Integration

### Core Display
- [ ] Device table updates in real-time
- [ ] RSSI, name, manufacturer data displayed
- [ ] PHY type shown (1M, 2M, Coded)
- [ ] Connection tracking displayed
- [ ] GPS map (if --gpsd enabled)

### New Features
- [ ] SMP events displayed (pairing method, weak pairing warnings)
- [ ] GATT write button appears for writable characteristics
- [ ] GATT write prompt and hex validation works
- [ ] Sensor heartbeat shows correct SDR type
- [ ] HackRF gain shows LNA/VGA sliders (if HackRF connected)

### C2 Control
- [ ] Set gain via dashboard slider
- [ ] Set squelch via dashboard slider
- [ ] Query GATT from device detail panel
- [ ] Write GATT from device detail panel
- [ ] Spoof advertisement via API: `POST /api/c2/spoof_adv`
- [ ] L2CAP flood via API: `POST /api/c2/l2cap_flood`
- [ ] Multi-sensor management (multiple blue-dragon instances)

---

## Cross-Device Loopback Tests (Verified)

| TX Source | RX Source | Test | Result |
|-----------|-----------|------|--------|
| WHAD ButteRFly | bladeRF 2.0 | BD-TEST-42 adv | 84 pkts decoded, 97% CRC |
| WHAD ButteRFly | bladeRF 2.0 | INJECT-OK adv | 84 pkts decoded, 97% CRC |
| WHAD ButteRFly | bladeRF 2.0 | RAW-INJECT-99 | 135 pkts decoded |
| bladeRF + HCI | Dashboard | Full stack ZMQ | Devices + sensor via C2 |
| Ambient BLE | bladeRF 2.0 | Passive capture | 99% CRC, multiple devices |
| Ambient BLE | WHAD ButteRFly | Passive sniff (Rust) | 123 pkts/3s, CRC valid |
| Ambient BLE | WHAD (Rust) | Device discovery | FW version + capabilities OK |
