#!/usr/bin/env python3
# Copyright 2025-2026 CEMAXECUTER LLC
"""
Live web dashboard for blue-dragon ZMQ streams.

Binds SUB and C2 ROUTER sockets; sensors connect to us. Presents a
Kismet-style device list with BLE device fingerprinting, CRC-gated
tracking, aggregate summaries, and per-sensor C2 controls. Features a
privacy toggle to mask MAC addresses (useful for video/streaming).

Usage:
    # Basic - listen for sensors on port 5555 (C2 auto-binds on 5556):
    python3 zmq_web_dashboard.py tcp://*:5555

    # Custom HTTP port:
    python3 zmq_web_dashboard.py tcp://*:5555 --port 8080

    # With GPS map:
    python3 zmq_web_dashboard.py tcp://*:5555 --gps

    # Encrypted connection:
    python3 zmq_web_dashboard.py tcp://*:5555 --server-key server.key

    # Also write PCAP while viewing dashboard:
    python3 zmq_web_dashboard.py tcp://*:5555 -w capture.pcap --gps

    # Update Bluetooth device database (Nordic Semiconductor, MIT licensed):
    python3 zmq_web_dashboard.py tcp://*:5555 --update-bt-db

    # RPA resolution with Identity Resolving Keys:
    python3 zmq_web_dashboard.py tcp://*:5555 --irk-file irks.txt

Requirements:
    pip install pyzmq
    pip install cryptography  (only needed for --irk-file)
"""

import argparse
import json
import math
import os
import queue
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError

try:
    import zmq
except ImportError:
    print("pyzmq is required: pip install pyzmq", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# SQLite device persistence
# ---------------------------------------------------------------------------
class DeviceDB:
    """SQLite-backed device persistence for cross-session tracking."""

    DEFAULT_PATH = Path.home() / ".cache" / "blue-dragon" / "devices.db"

    def __init__(self, db_path=None):
        self.path = Path(db_path) if db_path else self.DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.Lock()
        self._create_tables()
        self.session_id = self._start_session()
        self._known_keys = self._load_known_keys()
        self._changes = 0

    def _create_tables(self):
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS devices (
                    dev_key TEXT PRIMARY KEY,
                    protocol TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    name TEXT DEFAULT '',
                    mfr TEXT DEFAULT '',
                    identity TEXT,
                    mac_type TEXT DEFAULT '',
                    total_pkts INTEGER DEFAULT 0,
                    best_rssi INTEGER DEFAULT -127,
                    services TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time REAL NOT NULL,
                    end_time REAL
                );
                CREATE INDEX IF NOT EXISTS idx_devices_last_seen
                    ON devices(last_seen);
            """)

    def _start_session(self):
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO sessions (start_time) VALUES (?)",
                (time.time(),))
            return cur.lastrowid

    def _load_known_keys(self):
        cur = self.conn.execute("SELECT dev_key FROM devices")
        return set(row[0] for row in cur.fetchall())

    def is_new(self, dev_key):
        return dev_key not in self._known_keys

    def upsert(self, dev_key, protocol, now, name="", mfr="", identity=None,
               mac_type="", rssi=-127, services=""):
        with self.lock:
            self.conn.execute("""
                INSERT INTO devices
                    (dev_key, protocol, first_seen, last_seen, name, mfr,
                     identity, mac_type, total_pkts, best_rssi, services)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(dev_key) DO UPDATE SET
                    last_seen = MAX(excluded.last_seen, last_seen),
                    total_pkts = total_pkts + 1,
                    best_rssi = MAX(excluded.best_rssi, best_rssi),
                    name = CASE WHEN excluded.name != '' THEN excluded.name
                                ELSE name END,
                    mfr = CASE WHEN excluded.mfr != '' THEN excluded.mfr
                               ELSE mfr END,
                    identity = COALESCE(excluded.identity, identity),
                    services = CASE WHEN length(excluded.services) > length(services)
                                    THEN excluded.services ELSE services END
            """, (dev_key, protocol, now, now, name, mfr, identity, mac_type,
                  rssi, services))
            self._known_keys.add(dev_key)
            self._changes += 1
            if self._changes % 100 == 0:
                self.conn.commit()

    def get_first_seen(self, dev_key):
        cur = self.conn.execute(
            "SELECT first_seen FROM devices WHERE dev_key = ?", (dev_key,))
        row = cur.fetchone()
        return row[0] if row else None

    def close(self):
        with self.lock:
            self.conn.execute(
                "UPDATE sessions SET end_time = ? WHERE session_id = ?",
                (time.time(), self.session_id))
            self.conn.commit()
            self.conn.close()


# ---------------------------------------------------------------------------
# Device alerting
# ---------------------------------------------------------------------------
class AlertManager:
    """Device alerting via shell commands, webhooks, and browser notifications."""

    def __init__(self, watch_file=None, alert_cmd=None, alert_new=False,
                 webhook_url=None, cooldown=300, db=None):
        self.watch_set = set()
        self.alert_cmd = alert_cmd
        self.alert_new = alert_new
        self.webhook_url = webhook_url
        self.cooldown = cooldown
        self.db = db
        self.last_alert = {}
        self.lock = threading.Lock()
        self.pending_browser_alerts = []

        if watch_file:
            self._load_watch_file(watch_file)

    def _load_watch_file(self, path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                self.watch_set.add(line.lower())
        print(f"  Alert file: loaded {len(self.watch_set)} watch entries",
              file=sys.stderr)

    def check(self, dev_key, device_info):
        now = time.time()

        with self.lock:
            if dev_key in self.last_alert:
                if now - self.last_alert[dev_key] < self.cooldown:
                    return

            should_alert = False
            alert_reason = ""

            if dev_key.lower() in self.watch_set:
                should_alert = True
                alert_reason = "watchlist"
            elif device_info.get("identity"):
                label = f"[{device_info['identity']}]"
                if label.lower() in self.watch_set:
                    should_alert = True
                    alert_reason = "watchlist-identity"

            if not should_alert and self.alert_new and self.db:
                if self.db.is_new(dev_key):
                    should_alert = True
                    alert_reason = "new-device"

            if not should_alert:
                return

            self.last_alert[dev_key] = now

        self._fire_alert(dev_key, device_info, alert_reason)

    def _fire_alert(self, dev_key, device_info, reason):
        alert_data = {
            "mac": dev_key,
            "name": device_info.get("name", ""),
            "mfr": device_info.get("mfr", ""),
            "rssi": device_info.get("rssi", -127),
            "protocol": device_info.get("protocol", "BLE"),
            "reason": reason,
            "timestamp": time.time(),
            "identity": device_info.get("identity"),
        }

        if self.alert_cmd:
            self._exec_cmd(alert_data)

        if self.webhook_url:
            self._send_webhook(alert_data)

        with self.lock:
            self.pending_browser_alerts.append(alert_data)
            if len(self.pending_browser_alerts) > 100:
                self.pending_browser_alerts = self.pending_browser_alerts[-100:]

    def _exec_cmd(self, alert_data):
        env = os.environ.copy()
        env["ALERT_MAC"] = str(alert_data["mac"])
        env["ALERT_NAME"] = str(alert_data.get("name", ""))
        env["ALERT_MFR"] = str(alert_data.get("mfr", ""))
        env["ALERT_RSSI"] = str(alert_data.get("rssi", ""))
        env["ALERT_PROTOCOL"] = str(alert_data.get("protocol", "BLE"))
        env["ALERT_REASON"] = str(alert_data["reason"])
        env["ALERT_IDENTITY"] = str(alert_data.get("identity") or "")
        try:
            subprocess.Popen(self.alert_cmd, shell=True, env=env,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"  Alert cmd failed: {e}", file=sys.stderr)

    def _send_webhook(self, alert_data):
        import urllib.request
        try:
            data = json.dumps(alert_data).encode()
            req = urllib.request.Request(
                self.webhook_url, data=data,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"  Webhook failed: {e}", file=sys.stderr)

    def get_pending_alerts(self):
        with self.lock:
            alerts = list(self.pending_browser_alerts)
            self.pending_browser_alerts.clear()
            return alerts


# ---------------------------------------------------------------------------
# IRK / RPA resolution (BT Core Spec Vol 3, Part H, Section 2.2.2)
# ---------------------------------------------------------------------------
_irk_list = []  # list of (label, 16-byte key); populated by --irk-file

def _load_irk_file(path):
    """Load IRKs from file. Returns list of (label, bytes)."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # noqa: F811
    except ImportError:
        print("Error: --irk-file requires the 'cryptography' package\n"
              "       pip install cryptography", file=sys.stderr)
        sys.exit(1)
    irks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            label, hex_key = line.split(":", 1)
            try:
                key_bytes = bytes.fromhex(hex_key.strip())
            except ValueError:
                print(f"  Warning: skipping IRK '{label}' (bad hex)", file=sys.stderr)
                continue
            if len(key_bytes) != 16:
                print(f"  Warning: skipping IRK '{label}' (not 16 bytes)", file=sys.stderr)
                continue
            irks.append((label.strip(), key_bytes))
    return irks


def _bt_ah(irk, prand):
    """BT ah() function: AES-128-ECB(IRK, 0x00*13 || prand) -> last 3 bytes."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    cipher = Cipher(algorithms.AES(irk), modes.ECB())
    enc = cipher.encryptor()
    ct = enc.update(b'\x00' * 13 + prand) + enc.finalize()
    return ct[-3:]


def _resolve_rpa(mac_str):
    """Check MAC against all loaded IRKs. Returns label or None."""
    if not _irk_list:
        return None
    parts = mac_str.replace("-", ":").split(":")
    if len(parts) != 6:
        return None
    try:
        addr = bytes(int(b, 16) for b in parts)
    except ValueError:
        return None
    if len(addr) != 6 or (addr[0] >> 6) != 0b01:
        return None  # not an RPA
    prand = addr[:3]
    expected = addr[3:]
    for label, irk in _irk_list:
        if _bt_ah(irk, prand) == expected:
            return label
    return None


# ---------------------------------------------------------------------------
# BLE / PCAP structures (matching pcap.c / zmq_subscriber.py)
# ---------------------------------------------------------------------------
PCAP_GLOBAL_HDR = struct.Struct("<IHHiIII")
PCAP_REC_HDR = struct.Struct("<IIII")
BLE_RF_HDR = struct.Struct("<bbbBIH")
ZMQ_GPS_FRAME = struct.Struct("<ddd")

LE_DEWHITENED         = 0x0001
LE_SIGNAL_POWER_VALID = 0x0002
LE_NOISE_POWER_VALID  = 0x0004
LE_CRC_CHECKED        = 0x0400
LE_CRC_VALID          = 0x0800
LE_PHY_MASK           = 0xC000
LE_PHY_SHIFT          = 14

BLE_ADV_AA = 0x8E89BED6

# PPI structures for GPS PCAP writing
DLT_BLUETOOTH_LE_LL_WITH_PHDR = 256
DLT_PPI = 192
PPI_HDR = struct.Struct("<BBHI")
PPI_FIELD_HDR = struct.Struct("<HH")
PPI_GPS = struct.Struct("<BBHIIIII")
PPI_FIELD_GPS = 30002
PPI_GPS_FLAG_GPSFLAGS = 0x00000001
PPI_GPS_FLAG_LAT      = 0x00000002
PPI_GPS_FLAG_LON      = 0x00000004
PPI_GPS_FLAG_ALT      = 0x00000008
PPI_HDR_SIZE = PPI_HDR.size
PPI_GPS_SIZE = PPI_HDR.size + PPI_FIELD_HDR.size + PPI_GPS.size


def ppi_fixed3_7(val):
    return int((val + 180.0) * 1e7)

def ppi_fixed6_4(val):
    return int((val + 180000.0) * 1e4)

def build_ppi_gps_header(lat, lon, alt, dlt=DLT_BLUETOOTH_LE_LL_WITH_PHDR):
    ppi_hdr = PPI_HDR.pack(0, 0, PPI_GPS_SIZE, dlt)
    fld_hdr = PPI_FIELD_HDR.pack(PPI_FIELD_GPS, PPI_GPS.size)
    gps_data = PPI_GPS.pack(
        2, 0, PPI_GPS.size,
        PPI_GPS_FLAG_GPSFLAGS | PPI_GPS_FLAG_LAT | PPI_GPS_FLAG_LON | PPI_GPS_FLAG_ALT,
        0, ppi_fixed3_7(lat), ppi_fixed3_7(lon), ppi_fixed6_4(alt),
    )
    return ppi_hdr + fld_hdr + gps_data

def build_ppi_passthrough_header(dlt=DLT_BLUETOOTH_LE_LL_WITH_PHDR):
    return PPI_HDR.pack(0, 0, PPI_HDR_SIZE, dlt)


# ---------------------------------------------------------------------------
# Classic BT (BR/EDR) support
# ---------------------------------------------------------------------------
DLT_BLUETOOTH_BREDR_BB = 255
BREDR_BB_HDR = struct.Struct("<BbbBBBhIIIH")  # 22 bytes

# ZMQ packet type prefix bytes (prepended by sniffer C code)
ZMQ_PKT_TYPE_BLE = 0x00
ZMQ_PKT_TYPE_BT  = 0x01

BREDR_SIGNAL_POWER_VALID = 0x0004
BREDR_NOISE_POWER_VALID  = 0x0008


def parse_bt_packet(data):
    """Parse a Classic BT PCAP record (pcaprec_hdr + BREDR_BB header + optional raw header)."""
    if len(data) < PCAP_REC_HDR.size + BREDR_BB_HDR.size:
        return None

    ts_sec, ts_usec, incl_len, orig_len = PCAP_REC_HDR.unpack_from(data, 0)
    offset = PCAP_REC_HDR.size

    (rf_channel, signal_power, noise_power, ac_offenses,
     ptt, corr_hdr, corr_payload, lap, ref_lap_uap,
     bt_header, flags) = BREDR_BB_HDR.unpack_from(data, offset)
    offset += BREDR_BB_HDR.size

    # Raw FEC-encoded header bytes (7 bytes = 54 bits packed LSB-first)
    raw_header = None
    if len(data) >= offset + 7:
        raw_header = data[offset:offset + 7]

    mac = f"bt:{(lap >> 16) & 0xff:02x}:{(lap >> 8) & 0xff:02x}:{lap & 0xff:02x}"

    return {
        "timestamp": ts_sec + ts_usec / 1e6,
        "rf_channel": rf_channel,
        "freq_mhz": 2402 + rf_channel,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "lap": lap,
        "ac_errors": ac_offenses,
        "mac": mac,
        "mac_type": "bt-lap",
        "crc_checked": False,
        "crc_valid": None,
        "is_adv": False,
        "pdu_type": "BT",
        "fingerprint": {},
        "protocol": "BT",
        "raw_header": raw_header,
        "data_len": incl_len,
    }


class UAPEstimator:
    """Estimate Classic BT UAP from accumulated packet headers.

    Uses the same HEC-reversal approach as libbtbb: for each packet's
    54-bit FEC-encoded header, try all 64 CLK1-6 whitening sequences,
    reverse the HEC LFSR to find which UAP would produce the observed
    HEC, and accumulate votes. The correct UAP consistently appears
    while wrong CLK1-6 values produce random UAPs.
    """

    def __init__(self):
        self.uap_votes = [0] * 256
        self.packet_count = 0
        self.converged_uap = None
        self.confidence = 0.0

    def add_header(self, raw_header):
        """Process a 54-bit FEC-encoded header (7 bytes, LSB-first packed)."""
        if raw_header is None or len(raw_header) < 7:
            return

        self.packet_count += 1

        # Unpack 54 bits from 7 bytes
        bits_54 = []
        for byte_idx in range(7):
            for bit_idx in range(8):
                if byte_idx * 8 + bit_idx >= 54:
                    break
                bits_54.append((raw_header[byte_idx] >> bit_idx) & 1)

        # Try all 64 CLK1-6 whitening sequences
        # Whitening is applied AFTER FEC encoding, so we must de-whiten
        # the 54 bits first, then FEC decode to 18 bits
        for clk in range(64):
            wh = self._whitening_bits(clk, 54)
            dw54 = [bits_54[j] ^ wh[j] for j in range(54)]

            # FEC decode: 1/3 majority voting -> 18 bits
            bits_18 = []
            for i in range(18):
                b0 = dw54[i * 3]
                b1 = dw54[i * 3 + 1]
                b2 = dw54[i * 3 + 2]
                bits_18.append(1 if (b0 + b1 + b2) >= 2 else 0)

            # Pack 10-bit header payload and 8-bit HEC
            header_10 = 0
            for j in range(10):
                header_10 |= bits_18[j] << j
            hec_8 = 0
            for j in range(8):
                hec_8 |= bits_18[10 + j] << j

            uap = self._uap_from_hec(header_10, hec_8)
            self.uap_votes[uap] += 1

        # Check convergence at thresholds
        if self.packet_count in (5, 10, 20, 50) or self.packet_count % 50 == 0:
            self._check_convergence()

    @staticmethod
    def _whitening_bits(clk1_6, n):
        """Generate n bits of BT data whitening (x^7 + x^4 + 1 LFSR)."""
        reg = 0x40 | (clk1_6 & 0x3f)  # bit 6 forced to 1
        bits = []
        for _ in range(n):
            bits.append(reg & 1)
            fb = ((reg >> 6) ^ (reg >> 3)) & 1
            reg = (reg >> 1) | (fb << 6)
        return bits

    @staticmethod
    def _reverse8(byte):
        """Bit-reverse an 8-bit value."""
        byte = ((byte & 0xF0) >> 4) | ((byte & 0x0F) << 4)
        byte = ((byte & 0xCC) >> 2) | ((byte & 0x33) << 2)
        byte = ((byte & 0xAA) >> 1) | ((byte & 0x55) << 1)
        return byte & 0xFF

    @staticmethod
    def _uap_from_hec(header_10, hec):
        """Reverse HEC LFSR to find UAP (same algorithm as libbtbb).

        The BT spec initializes the HEC LFSR with reverse(UAP), so after
        unwinding the LFSR we must bit-reverse the result to get the UAP.
        """
        reg = hec
        for i in range(9, -1, -1):
            if reg & 0x80:
                reg ^= 0x65
            reg = ((reg << 1) & 0xff) | (((reg >> 7) ^ ((header_10 >> i) & 1)) & 1)
        return UAPEstimator._reverse8(reg)

    def _check_convergence(self):
        if self.packet_count < 5:
            return
        best_uap = max(range(256), key=lambda i: self.uap_votes[i])
        best_count = self.uap_votes[best_uap]
        # Expected noise per UAP: each packet adds 63 random votes
        # spread across 256 candidates (E = packet_count * 63/256)
        expected_noise = self.packet_count * 63.0 / 256.0
        # The correct UAP gets packet_count deterministic votes on top of noise
        signal = best_count - expected_noise
        if signal > 0 and self.packet_count > 0:
            conf = signal / self.packet_count
            if conf >= 0.5:
                self.converged_uap = best_uap
                self.confidence = min(conf, 1.0)

    def get_result(self):
        """Return (uap, confidence) or (None, 0.0) if not converged."""
        if self.converged_uap is not None:
            return (self.converged_uap, self.confidence)
        return (None, 0.0)


# ---------------------------------------------------------------------------
# Packet parsing
# ---------------------------------------------------------------------------
def channel_to_freq(ch):
    return 2402 + ch * 2


def extract_mac(ble_data, aa, phy="1M"):
    """Extract advertiser MAC from BLE advertising PDU.

    For ADV_EXT_IND (PDU type 7), the payload is a Common Extended Header
    which may or may not contain AdvA. Only extract MAC if the AdvA flag
    (bit 0 of extended header flags) is set.
    """
    if aa != BLE_ADV_AA:
        return None
    # LE Coded packets have CI byte between AA and PDU:
    #   AA(4) + CI(1) + Header(2) + AdvA(6) = 13 bytes minimum
    # LE 1M/2M:
    #   AA(4) + Header(2) + AdvA(6) = 12 bytes minimum
    ci_offset = 1 if phy == "Coded" else 0
    hdr_pos = 4 + ci_offset  # position of PDU header byte
    if len(ble_data) < hdr_pos + 2:
        return None
    pdu_type = ble_data[hdr_pos] & 0x0F

    # PDU type 7 (ADV_EXT_IND / AUX_ADV_IND / AUX_CHAIN_IND):
    # Payload is Common Extended Header, not AdvA at fixed offset.
    # Parse ext header flags to find AdvA if present.
    if pdu_type == 7:
        # ext_hdr starts after PDU header (2 bytes)
        ext_start = hdr_pos + 2
        if len(ble_data) < ext_start + 2:
            return None
        ext_hdr_len = ble_data[ext_start] & 0x3F
        if ext_hdr_len < 1:
            return None
        flags = ble_data[ext_start + 1]
        if not (flags & 0x01):
            return None  # no AdvA present
        # AdvA is first field after flags byte (6 bytes)
        adva_pos = ext_start + 2
        if len(ble_data) < adva_pos + 6:
            return None
        mac_bytes = ble_data[adva_pos:adva_pos + 6]
        return ":".join(f"{b:02x}" for b in reversed(mac_bytes))

    min_len = 12 + ci_offset
    if len(ble_data) < min_len:
        return None
    mac_start = 6 + ci_offset
    mac_bytes = ble_data[mac_start:mac_start + 6]
    return ":".join(f"{b:02x}" for b in reversed(mac_bytes))


# ---------------------------------------------------------------------------
# BLE device fingerprinting
# ---------------------------------------------------------------------------
BLE_COMPANY_IDS = {
    0x0001: "Nokia",
    0x0002: "Intel",
    0x0006: "Microsoft",
    0x000A: "Qualcomm",
    0x000D: "Texas Instruments",
    0x000F: "Broadcom",
    0x004C: "Apple",
    0x004F: "Meta (Oculus)",
    0x0059: "Nordic",
    0x0075: "Samsung",
    0x0087: "Garmin",
    0x009E: "Bose",
    0x00D2: "Bose",
    0x00E0: "Google",
    0x012D: "Sony",
    0x0131: "Huawei",
    0x0157: "Anhui Huami (Amazfit)",
    0x0171: "Amazon",
    0x01DA: "Fitbit",
    0x0310: "Tile",
    0x038F: "Xiaomi",
    0x0822: "Shenzhen Goodix",
}

APPLE_CONTINUITY_TYPES = {
    0x01: "Setup",
    0x02: "iBeacon",
    0x05: "AirDrop",
    0x07: "AirPods",
    0x09: "AirPlay",
    0x0C: "Handoff",
    0x0D: "Hotspot",
    0x0E: "Hotspot Src",
    0x0F: "Nearby Action",
    0x10: "Nearby Info",
    0x12: "Find My",
}

AD_TYPE_FLAGS            = 0x01
AD_TYPE_UUID16_INCOMPLETE = 0x02
AD_TYPE_UUID16_COMPLETE  = 0x03
AD_TYPE_NAME_SHORT       = 0x08
AD_TYPE_NAME_COMPLETE    = 0x09
AD_TYPE_TX_POWER         = 0x0A
AD_TYPE_SVC_DATA_16      = 0x16  # Service Data - 16-bit UUID (very common)
AD_TYPE_APPEARANCE       = 0x19
AD_TYPE_MANUFACTURER     = 0xFF

BLE_APPEARANCE = {
    0x0000: "Unknown",
    0x0040: "Phone",
    0x0080: "Computer",
    0x00C0: "Watch",
    0x00C1: "Sports Watch",
    0x0100: "Clock",
    0x0140: "Display",
    0x0180: "Remote",
    0x01C0: "Eyeglasses",
    0x0200: "Tag",
    0x0240: "Keyring",
    0x0300: "Pulse Oximeter",
    0x03C0: "Heart Rate Sensor",
    0x0440: "Blood Pressure",
    0x04C0: "HID",
    0x04C1: "Keyboard",
    0x04C2: "Mouse",
    0x04C3: "Joystick",
    0x04C4: "Gamepad",
    0x0540: "Barcode Scanner",
    0x0580: "Thermometer",
    0x0940: "Hearing Aid",
    0x0CC0: "Sensor",
    0x0CC1: "Motion Sensor",
}

BLE_UUID16_SERVICES = {
    # Standard GATT services
    0x1800: "Generic Access",
    0x1801: "Generic Attribute",
    0x1802: "Immediate Alert",
    0x1803: "Link Loss",
    0x1804: "Tx Power",
    0x1805: "Current Time",
    0x1806: "Ref Time Update",
    0x1807: "Next DST Change",
    0x1808: "Glucose",
    0x1809: "Health Thermo",
    0x180A: "Device Info",
    0x180D: "Heart Rate",
    0x180E: "Phone Alert",
    0x180F: "Battery",
    0x1810: "Blood Pressure",
    0x1811: "Alert Notify",
    0x1812: "HID",
    0x1813: "Scan Params",
    0x1814: "Running Speed",
    0x1816: "Cycling Speed",
    0x1818: "Cycling Power",
    0x1819: "Location Nav",
    0x181A: "Environmental",
    0x181B: "Body Composition",
    0x181C: "User Data",
    0x181D: "Weight Scale",
    0x181E: "Bond Mgmt",
    0x1820: "IP Support",
    0x1821: "Indoor Positioning",
    0x1822: "Pulse Oximeter",
    0x1824: "Transport Discovery",
    0x1825: "Object Transfer",
    0x1826: "Fitness Machine",
    0x1827: "Mesh Provisioning",
    0x1828: "Mesh Proxy",
    0x1829: "Reconnect Config",
    0x183A: "Audio Input",
    0x183B: "Volume Control",
    0x183C: "Volume Offset",
    0x183E: "Coord Set ID",
    0x184D: "Microphone",
    0x184E: "Audio Stream",
    0x1853: "Common Audio",
    0x1854: "Hearing Access",
    0x1855: "Telephony",
    0x1856: "Media Control",
    0x1857: "Generic Media",
    0x1858: "Constant Tone",
    0x1859: "Object ID",
    # Member company 16-bit UUIDs (0xFCxx, 0xFDxx, 0xFExx range)
    0xFCB2: "Apple",
    0xFD62: "Fitbit",
    0xFD6F: "Exposure Notify",
    0xFD69: "Samsung",
    0xFD82: "Loop (Disney)",
    0xFDA6: "OPPO",
    0xFDB5: "OnePlus",
    0xFDCF: "Nreal",
    0xFDDF: "Harman (JBL)",
    0xFDF0: "Google Nearby",
    0xFE03: "Amazon",
    0xFE07: "Sonos",
    0xFE0D: "Xiaomi",
    0xFE0F: "Philips",
    0xFE13: "Apple ANCS",
    0xFE26: "Google",
    0xFE2C: "Google Fast Pair",
    0xFE43: "Andreas Stihl",
    0xFE50: "Google",
    0xFE59: "Nordic DFU",
    0xFE6E: "JBL",
    0xFE78: "Garmin",
    0xFE8A: "Apple MFi",
    0xFE95: "Xiaomi Mi",
    0xFE9F: "Google",
    0xFEA0: "Google",
    0xFEAA: "Eddystone",
    0xFEAD: "Tile",
    0xFEB2: "Microsoft",
    0xFEB8: "Facebook",
    0xFEBB: "Adafruit",
    0xFEBE: "Bose",
    0xFEC7: "Apple Notification",
    0xFEC8: "Apple MIDI",
    0xFEC9: "Apple ANCS",
    0xFED4: "Apple",
    0xFED8: "Google Thread",
    0xFEDF: "Design SHIFT",
    0xFEE7: "Tencent",
    0xFEED: "Tile",
    0xFEF3: "Google",
    0xFEF5: "Dialog Semi",
}

AD_TYPE_UUID16_LIST = {0x02, 0x03}  # 16-bit UUID lists only (0x04-05 = 32-bit, 0x06-07 = 128-bit)

# ---------------------------------------------------------------------------
# Auto-updatable Bluetooth numbers database
# ---------------------------------------------------------------------------
NORDIC_DB_BASE = "https://raw.githubusercontent.com/NordicSemiconductor/bluetooth-numbers-database/master/v1"
NORDIC_COMPANY_IDS_URL = f"{NORDIC_DB_BASE}/company_ids.json"
NORDIC_SERVICE_UUIDS_URL = f"{NORDIC_DB_BASE}/service_uuids.json"

# Cache directory: ~/.cache/blue-dragon/
BT_DB_CACHE_DIR = Path.home() / ".cache" / "blue-dragon"


def bt_db_update(quiet=False):
    """Download latest Bluetooth numbers from Nordic Semiconductor database."""
    BT_DB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        (NORDIC_COMPANY_IDS_URL, BT_DB_CACHE_DIR / "company_ids.json"),
        (NORDIC_SERVICE_UUIDS_URL, BT_DB_CACHE_DIR / "service_uuids.json"),
    ]

    for url, path in files:
        try:
            if not quiet:
                print(f"  Downloading {url} ...", file=sys.stderr)
            with urlopen(url, timeout=15) as resp:
                data = resp.read()
            # Validate JSON before writing
            json.loads(data)
            path.write_bytes(data)
            if not quiet:
                print(f"  Saved to {path}", file=sys.stderr)
        except (URLError, OSError, json.JSONDecodeError) as e:
            print(f"  WARNING: Failed to download {url}: {e}", file=sys.stderr)
            return False

    if not quiet:
        print(f"  Bluetooth numbers database updated.", file=sys.stderr)
    return True


def bt_db_load():
    """Load cached Nordic database and merge over hardcoded dicts.

    Hardcoded entries take priority (they include curated short names),
    but any codes NOT in the hardcoded dicts get filled in from the
    Nordic database.
    """
    # Company IDs
    cid_path = BT_DB_CACHE_DIR / "company_ids.json"
    if cid_path.exists():
        try:
            entries = json.loads(cid_path.read_text())
            added = 0
            for entry in entries:
                code = entry.get("code")
                name = entry.get("name", "")
                if code is not None and code not in BLE_COMPANY_IDS and name:
                    BLE_COMPANY_IDS[code] = name
                    added += 1
            print(f"  BT DB: loaded {added} additional company IDs from cache", file=sys.stderr)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Failed to load {cid_path}: {e}", file=sys.stderr)

    # Service UUIDs
    svc_path = BT_DB_CACHE_DIR / "service_uuids.json"
    if svc_path.exists():
        try:
            entries = json.loads(svc_path.read_text())
            added = 0
            for entry in entries:
                uuid_str = entry.get("uuid", "")
                name = entry.get("name", "")
                try:
                    uuid_int = int(uuid_str, 16)
                except (ValueError, TypeError):
                    continue
                if uuid_int not in BLE_UUID16_SERVICES and name:
                    BLE_UUID16_SERVICES[uuid_int] = name
                    added += 1
            print(f"  BT DB: loaded {added} additional service UUIDs from cache", file=sys.stderr)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Failed to load {svc_path}: {e}", file=sys.stderr)


def classify_mac_type(ble_data, ci_offset=0):
    """Classify MAC address type from PDU header TxAdd bit."""
    hdr_pos = 4 + ci_offset
    if len(ble_data) < hdr_pos + 2:
        return "unknown"
    pdu_type = ble_data[hdr_pos] & 0x0F
    # TxAdd is bit 6 of the first header byte
    tx_add = (ble_data[hdr_pos] >> 6) & 1
    if tx_add == 0:
        return "public"

    # For PDU type 7 (extended advertising), AdvA is in the ext header,
    # not at the fixed offset. Find it via the flags byte.
    if pdu_type == 7:
        ext_start = hdr_pos + 2
        if len(ble_data) < ext_start + 2:
            return "random"
        flags = ble_data[ext_start + 1]
        if not (flags & 0x01):
            return "unknown"  # no AdvA present
        adva_end = ext_start + 2 + 6
        if len(ble_data) < adva_end:
            return "random"
        msb = ble_data[adva_end - 1]
    else:
        # Standard: AdvA at Header(2) + AdvA(6) from hdr_pos
        mac_end = hdr_pos + 2 + 6
        if len(ble_data) < mac_end:
            return "random"
        msb = ble_data[mac_end - 1]

    top2 = (msb >> 6) & 0x03
    if top2 == 0x03:
        return "static"       # static random -- persists per boot
    elif top2 == 0x01:
        return "resolvable"   # resolvable private -- rotates, can be resolved with IRK
    elif top2 == 0x00:
        return "non-resolv"   # non-resolvable private -- fully anonymous
    return "random"


def parse_ad_structures(adv_data):
    """Parse AD structures from BLE advertising data payload."""
    result = {
        "name": None,
        "manufacturer": None,
        "company_id": None,
        "tx_power": None,
        "appearance": None,
        "apple_type": None,
        "mfr_data": None,
        "services": [],
    }
    i = 0
    while i < len(adv_data):
        if i + 1 >= len(adv_data):
            break
        length = adv_data[i]
        if length == 0 or i + length >= len(adv_data):
            break
        ad_type = adv_data[i + 1]
        ad_data = adv_data[i + 2 : i + 1 + length]

        if ad_type in (AD_TYPE_NAME_COMPLETE, AD_TYPE_NAME_SHORT):
            try:
                result["name"] = ad_data.decode("utf-8", errors="replace")
            except Exception:
                pass

        elif ad_type == AD_TYPE_TX_POWER and len(ad_data) >= 1:
            result["tx_power"] = struct.unpack("b", ad_data[:1])[0]

        elif ad_type == AD_TYPE_APPEARANCE and len(ad_data) >= 2:
            code = struct.unpack("<H", ad_data[:2])[0]
            # Match by category (top bits) or exact
            result["appearance"] = BLE_APPEARANCE.get(
                code, BLE_APPEARANCE.get(code & 0xFFC0, f"0x{code:04x}"))

        elif ad_type in AD_TYPE_UUID16_LIST and len(ad_data) >= 2:
            # Parse 16-bit service UUIDs (complete or incomplete lists)
            for j in range(0, len(ad_data) - 1, 2):
                uuid16 = struct.unpack("<H", ad_data[j:j+2])[0]
                svc = BLE_UUID16_SERVICES.get(uuid16, f"0x{uuid16:04x}")
                if svc not in result["services"]:
                    result["services"].append(svc)

        elif ad_type == AD_TYPE_SVC_DATA_16 and len(ad_data) >= 2:
            # Service Data: first 2 bytes are the 16-bit UUID, rest is data
            uuid16 = struct.unpack("<H", ad_data[:2])[0]
            svc = BLE_UUID16_SERVICES.get(uuid16, f"0x{uuid16:04x}")
            if svc not in result["services"]:
                result["services"].append(svc)

        elif ad_type == AD_TYPE_MANUFACTURER and len(ad_data) >= 2:
            cid = struct.unpack("<H", ad_data[:2])[0]
            result["company_id"] = cid
            result["manufacturer"] = BLE_COMPANY_IDS.get(cid, f"0x{cid:04x}")
            result["mfr_data"] = ad_data[2:]

            # Apple Continuity protocol parsing
            if cid == 0x004C and len(ad_data) >= 3:
                apple_msg_type = ad_data[2]
                result["apple_type"] = APPLE_CONTINUITY_TYPES.get(
                    apple_msg_type, f"0x{apple_msg_type:02x}")

        i += 1 + length

    return result


def parse_ble_packet(data):
    """Parse a ZMQ message containing a PCAP record."""
    if len(data) < PCAP_REC_HDR.size + BLE_RF_HDR.size:
        return None

    ts_sec, ts_usec, incl_len, orig_len = PCAP_REC_HDR.unpack_from(data, 0)
    offset = PCAP_REC_HDR.size

    rf_channel, signal_power, noise_power, aa_offenses, ref_aa, flags = \
        BLE_RF_HDR.unpack_from(data, offset)
    offset += BLE_RF_HDR.size

    ble_data = data[offset:]
    aa = struct.unpack_from("<I", ble_data, 0)[0] if len(ble_data) >= 4 else 0

    crc_checked = bool(flags & LE_CRC_CHECKED)
    crc_valid = bool(flags & LE_CRC_VALID) if crc_checked else None
    phy_code = (flags & LE_PHY_MASK) >> LE_PHY_SHIFT
    phy = ("1M", "2M", "Coded", "?")[min(phy_code, 3)]
    is_adv = (aa == BLE_ADV_AA)
    # LE Coded packets have CI byte between AA and PDU
    ci_offset = 1 if phy == "Coded" else 0
    mac = extract_mac(ble_data, aa, phy)

    # Extract PDU type for advertising packets
    pdu_type = None
    pdu_type_name = None
    mac_type = None
    fingerprint = {}
    if is_adv and len(ble_data) >= 6 + ci_offset:
        pdu_type = ble_data[4 + ci_offset] & 0x0F
        pdu_names = {
            0: "ADV_IND", 1: "ADV_DIRECT", 2: "ADV_NONCONN",
            3: "SCAN_REQ", 4: "SCAN_RSP", 5: "CONNECT_IND",
            6: "ADV_SCAN_IND", 7: "ADV_EXT",
        }
        pdu_type_name = pdu_names.get(pdu_type, f"ADV_{pdu_type}")
        # Distinguish ADV_EXT_IND (no AdvA) from AUX_ADV_IND (has AdvA)
        if pdu_type == 7 and mac:
            pdu_type_name = "AUX_ADV"
        mac_type = classify_mac_type(ble_data, ci_offset)

        # AD structure parsing:
        # Types 0 (ADV_IND), 2 (ADV_NONCONN), 4 (SCAN_RSP), 6 (ADV_SCAN_IND)
        # carry AD structures at fixed offset (Header + AdvA).
        # Type 7 (ADV_EXT / AUX_ADV_IND) carries AD data after the variable-
        # length Common Extended Header.
        # Types 1 (ADV_DIRECT), 3 (SCAN_REQ), 5 (CONNECT_IND) do NOT have AD.
        crc_ok = (crc_valid is True) or (not crc_checked)
        hdr_pos = 4 + ci_offset
        if crc_ok and pdu_type in {0, 2, 4, 6} and len(ble_data) > 12 + ci_offset:
            pdu_len = ble_data[5 + ci_offset]
            adv_data_start = 12 + ci_offset  # AA(4) [+CI(1)] + Header(2) + AdvA(6)
            adv_data_end = min(6 + ci_offset + pdu_len, len(ble_data))
            if adv_data_end > adv_data_start:
                fingerprint = parse_ad_structures(
                    ble_data[adv_data_start:adv_data_end])
        elif crc_ok and pdu_type == 7:
            # Extended advertising: AD data follows the Common Extended Header
            ext_start = hdr_pos + 2
            if len(ble_data) >= ext_start + 2:
                ext_hdr_len = ble_data[ext_start] & 0x3F
                pdu_len = ble_data[hdr_pos + 1]
                ad_start = ext_start + 1 + ext_hdr_len  # after ext_hdr_len byte + header
                ad_end = min(hdr_pos + 2 + pdu_len, len(ble_data))
                if ad_end > ad_start:
                    fingerprint = parse_ad_structures(ble_data[ad_start:ad_end])

    # Parse CONNECT_IND payload for connection tracking
    conn_info = None
    if is_adv and pdu_type == 5 and len(ble_data) >= 40 + ci_offset:
        pdu_len = ble_data[5 + ci_offset]
        if pdu_len == 34:
            b = ci_offset
            init_addr = ":".join(f"{b2:02x}" for b2 in reversed(ble_data[6+b:12+b]))
            adv_addr  = ":".join(f"{b2:02x}" for b2 in reversed(ble_data[12+b:18+b]))
            conn_aa   = struct.unpack_from("<I", ble_data, 18+b)[0]
            crc_init  = ble_data[22+b] | (ble_data[23+b] << 8) | (ble_data[24+b] << 16)
            interval  = struct.unpack_from("<H", ble_data, 28+b)[0]
            latency   = struct.unpack_from("<H", ble_data, 30+b)[0]
            sup_timeout = struct.unpack_from("<H", ble_data, 32+b)[0]
            ch_map    = ble_data[34+b:39+b]
            hop       = ble_data[39+b] & 0x1F
            n_ch      = sum(bin(x).count('1') for x in ch_map[:4]) + bin(ch_map[4] & 0x1F).count('1')
            init_type = (ble_data[4+b] >> 6) & 1
            adv_type  = (ble_data[4+b] >> 7) & 1
            conn_info = {
                "init_addr": init_addr,
                "adv_addr": adv_addr,
                "init_type": "random" if init_type else "public",
                "adv_type": "random" if adv_type else "public",
                "conn_aa": conn_aa,
                "crc_init": crc_init,
                "interval_ms": round(interval * 1.25, 2),
                "latency": latency,
                "timeout_ms": sup_timeout * 10,
                "hop": hop,
                "used_channels": n_ch,
            }

    # Connectable: ADV_IND (0) and ADV_DIRECT_IND (1) indicate the device
    # accepts connections.
    connectable = pdu_type in (0, 1) if pdu_type is not None else False

    return {
        "timestamp": ts_sec + ts_usec / 1e6,
        "rf_channel": rf_channel,
        "freq_mhz": channel_to_freq(rf_channel),
        "signal_power": signal_power,
        "noise_power": noise_power,
        "aa": aa,
        "flags": flags,
        "crc_checked": crc_checked,
        "crc_valid": crc_valid,
        "is_adv": is_adv,
        "pdu_type": pdu_type_name or ("ADV" if is_adv else "DATA"),
        "mac": mac,
        "mac_type": mac_type,
        "data_len": len(ble_data),
        "fingerprint": fingerprint,
        "protocol": "BLE",
        "phy": phy,
        "conn_info": conn_info,
        "connectable": connectable,
    }


# ---------------------------------------------------------------------------
# Multilateration (position estimation from multiple sensors)
# ---------------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two lat/lon points."""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _estimate_device_position(sensor_obs, sensors, tx_power, path_loss_exp):
    """Estimate device position from multiple sensor distance estimates.

    Returns (lat, lon, uncertainty_m, num_sensors) or None.
    """
    points = []
    for sid, obs in sensor_obs.items():
        s = sensors.get(sid)
        if not s or s["lat"] is None:
            continue
        if obs["rssi_cnt"] == 0:
            continue
        rssi_avg = obs["rssi_sum"] / obs["rssi_cnt"]
        measured_1m = tx_power - 41
        dist = 10 ** ((measured_1m - rssi_avg) / (10.0 * path_loss_exp))
        weight = min(obs["rssi_cnt"], 100) / 100.0
        points.append((s["lat"], s["lon"], dist, weight))

    if len(points) < 2:
        return None

    # Initial guess: centroid
    lat0 = sum(p[0] for p in points) / len(points)
    lon0 = sum(p[1] for p in points) / len(points)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))

    sensors_m = [(
        (lon - lon0) * m_per_deg_lon,
        (lat - lat0) * m_per_deg_lat,
        dist, w
    ) for lat, lon, dist, w in points]

    # Gauss-Newton weighted least squares
    x_est, y_est = 0.0, 0.0
    for _ in range(20):
        A = []
        b = []
        W = []
        for sx, sy, sd, sw in sensors_m:
            dx = x_est - sx
            dy = y_est - sy
            d_calc = math.sqrt(dx * dx + dy * dy)
            if d_calc < 0.1:
                d_calc = 0.1
            A.append([dx / d_calc, dy / d_calc])
            b.append(sd - d_calc)
            W.append(sw)

        n = len(A)
        ATA = [[0, 0], [0, 0]]
        ATb = [0, 0]
        for i in range(n):
            for j in range(2):
                for k in range(2):
                    ATA[j][k] += W[i] * A[i][j] * A[i][k]
                ATb[j] += W[i] * A[i][j] * b[i]

        det = ATA[0][0] * ATA[1][1] - ATA[0][1] * ATA[1][0]
        if abs(det) < 1e-10:
            break
        ddx = (ATA[1][1] * ATb[0] - ATA[0][1] * ATb[1]) / det
        ddy = (-ATA[1][0] * ATb[0] + ATA[0][0] * ATb[1]) / det
        x_est += ddx
        y_est += ddy
        if abs(ddx) < 0.01 and abs(ddy) < 0.01:
            break

    est_lat = lat0 + y_est / m_per_deg_lat
    est_lon = lon0 + x_est / m_per_deg_lon

    residuals = []
    for sx, sy, sd, sw in sensors_m:
        dx = x_est - sx
        dy = y_est - sy
        d_calc = math.sqrt(dx * dx + dy * dy)
        residuals.append((sd - d_calc) ** 2 * sw)
    uncertainty = math.sqrt(sum(residuals) / max(len(residuals), 1))

    return (est_lat, est_lon, uncertainty, len(points))


# ---------------------------------------------------------------------------
# Shared dashboard state (device-centric, Kismet-style)
# ---------------------------------------------------------------------------
class DashboardState:
    def __init__(self, static_positions=None, path_loss_exp=2.0):
        self.lock = threading.Lock()
        self.total_packets = 0
        self.crc_valid = 0
        self.crc_invalid = 0
        self.data_packets = 0
        self.gps_count = 0
        self.last_gps = None
        self.start_time = time.time()
        self.sse_queues = []
        # Device table: mac -> device info dict
        self.devices = {}
        # Channel activity: rf_channel -> packet count (per protocol)
        self.ble_channel_counts = {}
        self.bt_channel_counts = {}
        self._dirty = True
        # Multi-sensor tracking
        self.sensors = {}  # sensor_id -> {lat, lon, last_seen, pkts}
        self.device_sensor_rssi = {}  # dev_key -> {sensor_id -> {rssi_sum, rssi_cnt, last}}
        self.static_positions = static_positions or {}
        self.path_loss_exp = path_loss_exp
        # Persistence and alerting (set externally after construction)
        self.db = None          # DeviceDB instance
        self.alert_mgr = None   # AlertManager instance
        # BLE connection tracking (from CONNECT_IND packets)
        self.ble_connections = {}  # conn_aa (int) -> connection info dict
        # C2 control state
        self.c2_sensor_status = {}   # sensor_id -> heartbeat data + last_heartbeat
        self.c2_outbox = queue.Queue()  # (sensor_id_bytes, json_bytes) for ROUTER

    def add_packet(self, pkt, gps_info=None, sensor_id=None):
        with self.lock:
            self.total_packets += 1
            if pkt["crc_checked"]:
                if pkt["crc_valid"]:
                    self.crc_valid += 1
                else:
                    self.crc_invalid += 1
            if gps_info:
                self.gps_count += 1
                self.last_gps = gps_info

            # Update sensor state
            if sensor_id:
                if sensor_id not in self.sensors:
                    self.sensors[sensor_id] = {"lat": None, "lon": None,
                                               "last_seen": 0, "pkts": 0}
                s = self.sensors[sensor_id]
                s["last_seen"] = time.time()
                s["pkts"] += 1
                if gps_info:
                    s["lat"] = round(gps_info[0], 6)
                    s["lon"] = round(gps_info[1], 6)
                elif sensor_id in self.static_positions:
                    pos = self.static_positions[sensor_id]
                    s["lat"] = pos[0]
                    s["lon"] = pos[1]

            # Track channel activity per protocol
            rf_ch = pkt.get("rf_channel")
            protocol = pkt.get("protocol", "BLE")
            if rf_ch is not None:
                if protocol == "BT":
                    self.bt_channel_counts[rf_ch] = self.bt_channel_counts.get(rf_ch, 0) + 1
                else:
                    self.ble_channel_counts[rf_ch] = self.ble_channel_counts.get(rf_ch, 0) + 1

            # Track CONNECT_IND -> connection table
            conn_info = pkt.get("conn_info")
            if conn_info:
                conn_aa = conn_info["conn_aa"]
                self.ble_connections[conn_aa] = {
                    **conn_info,
                    "created": pkt["timestamp"],
                    "last_seen": pkt["timestamp"],
                    "data_pkts": 0,
                }

            mac = pkt["mac"]

            # Data channel packets: resolve MAC via connection table
            if not mac and not pkt["is_adv"]:
                aa = pkt.get("aa", 0)
                conn = self.ble_connections.get(aa)
                if conn:
                    conn["last_seen"] = pkt["timestamp"]
                    conn["data_pkts"] = conn.get("data_pkts", 0) + 1
                    mac = conn["adv_addr"]
                    pkt["pdu_type"] = "DATA"

            if not mac:
                self.data_packets += 1
                return

            # Only create/update device entries for CRC-valid packets.
            # CRC-failed packets have bit errors that corrupt MAC addresses,
            # PDU types, and AD data -- creating phantom device entries.
            # When CRC checking is off, allow all packets (backward compat).
            crc_ok = (pkt["crc_valid"] is True) or (not pkt["crc_checked"])
            if not crc_ok:
                return

            # IRK resolution: if this RPA matches a known IRK, use the
            # identity label as the device key (merging all rotating MACs)
            identity = None
            dev_key = mac
            if _irk_list and pkt.get("mac_type") == "resolvable":
                label = _resolve_rpa(mac)
                if label:
                    identity = label
                    dev_key = f"[{label}]"

            now = round(pkt["timestamp"], 6)
            protocol = pkt.get("protocol", "BLE")
            fp = pkt.get("fingerprint", {})
            rssi = pkt["signal_power"]

            if dev_key in self.devices:
                d = self.devices[dev_key]
                d["pkts"] += 1
                d["last"] = now
                d["freq"] = pkt["freq_mhz"]
                d["type"] = pkt["pdu_type"]
                d["phy"] = pkt.get("phy", "1M")
                # Track connectable status (sticky -- once seen connectable, stays)
                if pkt.get("connectable"):
                    d["connectable"] = True
                # RSSI tracking: best, min, sum, count
                if rssi > d["rssi"]:
                    d["rssi"] = rssi
                if rssi < d["rssi_min"]:
                    d["rssi_min"] = rssi
                d["rssi_sum"] += rssi
                d["rssi_cnt"] += 1
                if pkt["crc_valid"]:
                    d["crc_ok"] += 1
                elif pkt["crc_valid"] is False:
                    d["crc_bad"] += 1
                if gps_info:
                    d["lat"] = round(gps_info[0], 6)
                    d["lon"] = round(gps_info[1], 6)
                # Update fingerprint fields (keep best info seen)
                if fp.get("name") and not d.get("name"):
                    d["name"] = fp["name"]
                if fp.get("manufacturer") and not d.get("mfr"):
                    d["mfr"] = fp["manufacturer"]
                if fp.get("apple_type") and not d.get("apple"):
                    d["apple"] = fp["apple_type"]
                if fp.get("appearance") and not d.get("appear"):
                    d["appear"] = fp["appearance"]
                if fp.get("tx_power") is not None and d.get("tx_pwr") is None:
                    d["tx_pwr"] = fp["tx_power"]
                for svc in fp.get("services") or []:
                    if svc not in d["services"]:
                        d["services"].append(svc)
                # Track resolved RPA addresses
                if identity and "rpa_addrs" in d:
                    d["rpa_addrs"].add(mac)
                # UAP estimation for Classic BT
                if protocol == "BT" and "_uap_est" in d:
                    raw_hdr = pkt.get("raw_header")
                    if raw_hdr:
                        d["_uap_est"].add_header(raw_hdr)
                        uap, conf = d["_uap_est"].get_result()
                        if uap is not None:
                            d["uap"] = uap
                            d["uap_conf"] = conf
            else:
                d = {
                    "mac": dev_key,
                    "protocol": protocol,
                    "first": now,
                    "last": now,
                    "freq": pkt["freq_mhz"],
                    "rssi": rssi,
                    "rssi_min": rssi,
                    "rssi_sum": rssi,
                    "rssi_cnt": 1,
                    "type": pkt["pdu_type"],
                    "phy": pkt.get("phy", "1M"),
                    "pkts": 1,
                    "crc_ok": 1 if pkt["crc_valid"] else 0,
                    "crc_bad": 1 if pkt["crc_valid"] is False else 0,
                    "mac_type": pkt.get("mac_type", ""),
                    "name": fp.get("name") or "",
                    "mfr": fp.get("manufacturer") or "",
                    "apple": fp.get("apple_type") or "",
                    "appear": fp.get("appearance") or "",
                    "tx_pwr": fp.get("tx_power"),
                    "services": list(fp.get("services") or []),
                    "identity": identity,
                    "connectable": bool(pkt.get("connectable")),
                }
                if identity:
                    d["rpa_addrs"] = {mac}
                    d["mac_type"] = "resolved"
                if gps_info:
                    d["lat"] = round(gps_info[0], 6)
                    d["lon"] = round(gps_info[1], 6)
                # Classic BT: set up UAP estimator
                if protocol == "BT":
                    d["_uap_est"] = UAPEstimator()
                    d["uap"] = None
                    d["uap_conf"] = 0.0
                    raw_hdr = pkt.get("raw_header")
                    if raw_hdr:
                        d["_uap_est"].add_header(raw_hdr)
                self.devices[dev_key] = d

            # Per-sensor RSSI tracking (for multilateration)
            if sensor_id:
                if dev_key not in self.device_sensor_rssi:
                    self.device_sensor_rssi[dev_key] = {}
                dsr = self.device_sensor_rssi[dev_key]
                if sensor_id not in dsr:
                    dsr[sensor_id] = {"rssi_sum": 0, "rssi_cnt": 0, "last": 0}
                dsr[sensor_id]["rssi_sum"] += rssi
                dsr[sensor_id]["rssi_cnt"] += 1
                dsr[sensor_id]["last"] = now

            # Persist to SQLite
            if self.db:
                svc_str = "|".join(d.get("services", []))
                self.db.upsert(
                    dev_key=dev_key,
                    protocol=d.get("protocol", "BLE"),
                    now=now,
                    name=d.get("name", ""),
                    mfr=d.get("mfr", ""),
                    identity=d.get("identity"),
                    mac_type=d.get("mac_type", ""),
                    rssi=rssi,
                    services=svc_str,
                )

            # Alerting
            if self.alert_mgr:
                self.alert_mgr.check(dev_key, d)

            self._dirty = True

    def get_stats(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            crc_total = self.crc_valid + self.crc_invalid
            return {
                "total": self.total_packets,
                "rate": round(self.total_packets / max(elapsed, 0.001), 1),
                "crc_pct": round(100.0 * self.crc_valid / crc_total, 1) if crc_total > 0 else None,
                "crc_valid": self.crc_valid,
                "crc_invalid": self.crc_invalid,
                "macs": len(self.devices),
                "data_pkts": self.data_packets,
                "connections": len(self.ble_connections),
                "gps_count": self.gps_count,
                "last_gps": list(self.last_gps[:2]) if self.last_gps else None,
                "uptime": round(elapsed, 1),
            }

    def get_devices(self):
        """Return device list sorted by last-seen (most recent first)."""
        with self.lock:
            self._dirty = False
            devs = []
            for d in self.devices.values():
                # Add computed avg RSSI for the JSON output
                dd = dict(d)
                dd["rssi_avg"] = round(d["rssi_sum"] / d["rssi_cnt"]) if d["rssi_cnt"] else d["rssi"]
                # Distance estimation from TX power + avg RSSI
                tx = d.get("tx_pwr")
                if tx is not None and dd["rssi_avg"] < 0:
                    measured = tx - 41  # RSSI at 1m for BLE 2.4 GHz
                    dd["est_dist"] = round(10 ** ((measured - dd["rssi_avg"]) / (10.0 * self.path_loss_exp)), 1)
                else:
                    dd["est_dist"] = None
                # Multilateration
                dev_key = d["mac"]
                dd["est_lat"] = None
                dd["est_lon"] = None
                dd["est_unc"] = None
                dd["num_sensors"] = 0
                dd["sensor_ids"] = []
                if dev_key in self.device_sensor_rssi:
                    dd["sensor_ids"] = sorted(self.device_sensor_rssi[dev_key].keys())
                    dd["num_sensors"] = len(dd["sensor_ids"])
                if tx is not None and dev_key in self.device_sensor_rssi:
                    obs = self.device_sensor_rssi[dev_key]
                    result = _estimate_device_position(
                        obs, self.sensors, tx, self.path_loss_exp)
                    if result:
                        dd["est_lat"] = round(result[0], 6)
                        dd["est_lon"] = round(result[1], 6)
                        dd["est_unc"] = round(result[2], 1)
                # Persistence fields
                dd["is_new"] = False
                dd["first_ever"] = None
                if self.db:
                    dd["is_new"] = self.db.is_new(dev_key)
                    fs = self.db.get_first_seen(dev_key)
                    if fs is not None:
                        dd["first_ever"] = fs
                # IRK resolution fields
                rpa_addrs = d.get("rpa_addrs")
                dd["identity"] = d.get("identity")
                dd["rpa_count"] = len(rpa_addrs) if rpa_addrs else 0
                if rpa_addrs:
                    del dd["rpa_addrs"]
                # Don't send internal accumulators or non-serializable objects
                del dd["rssi_sum"]
                del dd["rssi_cnt"]
                dd.pop("_uap_est", None)
                devs.append(dd)
            devs.sort(key=lambda x: x["last"], reverse=True)
            return devs

    def get_sensors(self):
        """Return list of known sensors with their positions."""
        with self.lock:
            result = []
            seen = set()
            for sid, s in self.sensors.items():
                seen.add(sid)
                result.append({
                    "id": sid, "lat": s["lat"], "lon": s["lon"],
                    "last_seen": s["last_seen"], "pkts": s["pkts"],
                })
            for label, (lat, lon) in self.static_positions.items():
                if label not in seen:
                    result.append({
                        "id": label, "lat": lat, "lon": lon,
                        "last_seen": 0, "pkts": 0,
                    })
            return result

    def get_summary(self):
        """Return aggregate breakdowns for the summary tab."""
        with self.lock:
            by_mfr = {}
            by_mac_type = {}
            by_pdu = {}
            by_svc = {}
            by_protocol = {}
            svc_device_count = 0
            top_talkers = []
            for d in self.devices.values():
                # Protocol
                proto = d.get("protocol", "BLE")
                by_protocol[proto] = by_protocol.get(proto, 0) + 1
                # Manufacturer
                m = d["mfr"] or "Unknown"
                by_mfr[m] = by_mfr.get(m, 0) + 1
                # MAC type
                mt = d["mac_type"] or "unknown"
                by_mac_type[mt] = by_mac_type.get(mt, 0) + 1
                # PDU type
                pt = d["type"]
                by_pdu[pt] = by_pdu.get(pt, 0) + 1
                # Services (count unique devices per service)
                svcs = d.get("services", [])
                if svcs:
                    svc_device_count += 1
                for svc in svcs:
                    by_svc[svc] = by_svc.get(svc, 0) + 1
                # Top talkers
                top_talkers.append((d["mac"], d["pkts"], d["mfr"], d["name"]))

            top_talkers.sort(key=lambda x: x[1], reverse=True)
            top10 = [{"mac": t[0], "pkts": t[1], "mfr": t[2], "name": t[3]}
                     for t in top_talkers[:15]]

            # Sort breakdowns by count descending
            def sorted_dict(d):
                return sorted(d.items(), key=lambda x: x[1], reverse=True)

            # BLE channel distribution: convert rf_channel to BLE channel number
            ble_ch_dist = {}
            for rf_ch, cnt in self.ble_channel_counts.items():
                freq = 2402 + rf_ch * 2
                if freq == 2402:
                    ble_ch = 37
                elif freq == 2426:
                    ble_ch = 38
                elif freq == 2480:
                    ble_ch = 39
                elif 2404 <= freq <= 2424:
                    ble_ch = (freq - 2404) // 2
                elif 2428 <= freq <= 2478:
                    ble_ch = (freq - 2428) // 2 + 11
                else:
                    ble_ch = rf_ch
                ble_ch_dist[ble_ch] = ble_ch_dist.get(ble_ch, 0) + cnt

            # BT channel distribution: rf_channel is already BT channel (freq = 2402 + ch)
            bt_ch_dist = {}
            for rf_ch, cnt in self.bt_channel_counts.items():
                bt_ch_dist[rf_ch] = bt_ch_dist.get(rf_ch, 0) + cnt

            return {
                "by_protocol": sorted_dict(by_protocol),
                "by_mfr": sorted_dict(by_mfr),
                "by_mac_type": sorted_dict(by_mac_type),
                "by_pdu": sorted_dict(by_pdu),
                "by_svc": sorted_dict(by_svc),
                "svc_devices": svc_device_count,
                "total_devices": len(self.devices),
                "top_talkers": top10,
                "ble_channels": sorted(ble_ch_dist.items()),
                "bt_channels": sorted(bt_ch_dist.items()),
            }

    def get_connections(self):
        """Return active BLE connections for display."""
        with self.lock:
            now = time.time()
            result = []
            stale = []
            for aa, c in self.ble_connections.items():
                age = now - c["last_seen"]
                # Expire connections with no data after supervision timeout
                # (or 60s if unknown)
                sup_timeout = c.get("timeout_ms", 60000) / 1000.0
                if age > max(sup_timeout, 60):
                    stale.append(aa)
                    continue
                result.append({
                    "aa": f"0x{aa:08X}",
                    "init_addr": c["init_addr"],
                    "adv_addr": c["adv_addr"],
                    "interval_ms": c.get("interval_ms", 0),
                    "latency": c.get("latency", 0),
                    "hop": c.get("hop", 0),
                    "used_channels": c.get("used_channels", 0),
                    "data_pkts": c.get("data_pkts", 0),
                    "age": round(age, 1),
                    "created": c.get("created", 0),
                })
            for aa in stale:
                del self.ble_connections[aa]
            return result

    def is_dirty(self):
        with self.lock:
            return self._dirty

    def register_sse(self):
        q = queue.Queue(maxsize=50)
        with self.lock:
            self.sse_queues.append(q)
        return q

    def unregister_sse(self, q):
        with self.lock:
            if q in self.sse_queues:
                self.sse_queues.remove(q)

    def update_sensor_c2(self, sensor_id, heartbeat):
        """Update C2 sensor status from a heartbeat message."""
        with self.lock:
            heartbeat["last_heartbeat"] = time.time()
            self.c2_sensor_status[sensor_id] = heartbeat

    def send_c2_command(self, sensor_id, cmd, params=None):
        """Queue a C2 command for delivery by the ROUTER thread."""
        msg = {"cmd": cmd, "req_id": f"r{int(time.time()*1000) % 1000000}"}
        if params:
            msg.update(params)
        self.c2_outbox.put((sensor_id.encode(), json.dumps(msg).encode()))

    def get_sensor_nodes(self):
        """Return merged sensor list with C2 heartbeat data."""
        with self.lock:
            now = time.time()
            nodes = {}
            # Start with data-path sensors
            for sid, s in self.sensors.items():
                nodes[sid] = {
                    "id": sid,
                    "status": "online" if (now - s["last_seen"]) < 15 else
                              "stale" if (now - s["last_seen"]) < 60 else "offline",
                    "lat": s["lat"], "lon": s["lon"],
                    "pkts": s["pkts"], "last_seen": s["last_seen"],
                    "sdr": None, "center_freq": None, "channels": None,
                    "gain": None, "squelch": None,
                    "pkt_rate": None, "crc_pct": None,
                    "uptime": None, "has_c2": False,
                }
            # Merge C2 heartbeat data
            for sid, hb in self.c2_sensor_status.items():
                age = now - hb.get("last_heartbeat", 0)
                status = "online" if age < 15 else "stale" if age < 60 else "offline"
                if sid in nodes:
                    n = nodes[sid]
                    n["has_c2"] = True
                    n["status"] = status
                    n["sdr"] = hb.get("sdr")
                    n["center_freq"] = hb.get("center_freq")
                    n["channels"] = hb.get("channels")
                    n["gain"] = hb.get("gain")
                    n["squelch"] = hb.get("squelch")
                    n["pkt_rate"] = hb.get("pkt_rate")
                    n["crc_pct"] = hb.get("crc_pct")
                    n["uptime"] = hb.get("uptime")
                    if hb.get("gps"):
                        n["lat"] = hb["gps"][0]
                        n["lon"] = hb["gps"][1]
                else:
                    nodes[sid] = {
                        "id": sid, "status": status, "has_c2": True,
                        "lat": hb.get("gps", [None])[0] if hb.get("gps") else None,
                        "lon": hb.get("gps", [None, None])[1] if hb.get("gps") else None,
                        "pkts": 0, "last_seen": hb.get("last_heartbeat", 0),
                        "sdr": hb.get("sdr"),
                        "center_freq": hb.get("center_freq"),
                        "channels": hb.get("channels"),
                        "gain": hb.get("gain"),
                        "squelch": hb.get("squelch"),
                        "pkt_rate": hb.get("pkt_rate"),
                        "crc_pct": hb.get("crc_pct"),
                        "uptime": hb.get("uptime"),
                    }
            # Add static positions that haven't been seen
            for label, (lat, lon) in self.static_positions.items():
                if label not in nodes:
                    nodes[label] = {
                        "id": label, "status": "offline", "has_c2": False,
                        "lat": lat, "lon": lon,
                        "pkts": 0, "last_seen": 0,
                        "sdr": None, "center_freq": None, "channels": None,
                        "gain": None, "squelch": None,
                        "pkt_rate": None, "crc_pct": None, "uptime": None,
                    }
            return list(nodes.values())


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
state = DashboardState()
gps_enabled = False


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/events":
            self._serve_sse()
        elif path == "/api/stats":
            self._serve_json(state.get_stats())
        elif path == "/api/devices":
            self._serve_json(state.get_devices())
        elif path == "/api/summary":
            self._serve_json(state.get_summary())
        elif path == "/api/sensors":
            self._serve_json(state.get_sensors())
        elif path == "/api/nodes":
            self._serve_json(state.get_sensor_nodes())
        elif path == "/api/export.csv":
            self._serve_csv()
        elif path == "/api/export.json":
            self._serve_export_json()
        else:
            self.send_error(404)

    def _read_post_body(self, max_bytes=65536):
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        if length > max_bytes:
            return None
        return json.loads(self.rfile.read(length))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/watch":
            try:
                body = self._read_post_body()
                if body is None:
                    self.send_error(413)
                    return
                mac = body.get("mac", "")
                watch = body.get("watch", False)
                if state.alert_mgr:
                    if watch:
                        state.alert_mgr.watch_set.add(mac.lower())
                    else:
                        state.alert_mgr.watch_set.discard(mac.lower())
                self._serve_json({"ok": True})
            except Exception:
                self.send_error(400)
        elif path == "/api/gatt/query":
            self._handle_gatt_query()
        elif path.startswith("/api/c2/"):
            self._handle_c2(path)
        else:
            self.send_error(404)

    def _handle_gatt_query(self):
        try:
            body = self._read_post_body()
            if body is None:
                self.send_error(413)
                return
        except Exception:
            self.send_error(400)
            return

        mac = body.get("mac", "").lower()
        if not mac:
            self._serve_json({"ok": False, "error": "mac required"})
            return

        # Only send to sensors that have actually seen this device
        with state.lock:
            c2_sensors = set(state.c2_sensor_status.keys())
            seen_by = set(state.device_sensor_rssi.get(mac, {}).keys())
            target_sensors = list(c2_sensors & seen_by)

        if not target_sensors:
            # No sensors with C2 have seen this device
            if not c2_sensors:
                self._serve_json({"ok": False, "error": "no sensors with C2 connected"})
            else:
                self._serve_json({"ok": False, "error": "no C2-connected sensor has seen this device"})
            return

        for sid in target_sensors:
            state.send_c2_command(sid, "query_gatt", {"mac": mac})

        self._serve_json({"status": "ok",
                          "message": f"GATT query sent to {len(target_sensors)} sensor(s)"})

    def _handle_c2(self, path):
        try:
            body = self._read_post_body()
            if body is None:
                self.send_error(413)
                return
        except Exception:
            self.send_error(400)
            return

        sensor_id = body.get("sensor_id", "")
        if not sensor_id:
            self._serve_json({"ok": False, "error": "sensor_id required"})
            return

        if path == "/api/c2/set_gain":
            params = {}
            if "gain" in body:
                g = float(body["gain"])
                if not (0 <= g <= 100):
                    self._serve_json({"ok": False, "error": "gain out of range (0-100)"})
                    return
                params["gain"] = g
            if "lna" in body:
                v = int(body["lna"])
                if not (0 <= v <= 40):
                    self._serve_json({"ok": False, "error": "lna out of range (0-40)"})
                    return
                params["lna"] = v
            if "vga" in body:
                v = int(body["vga"])
                if not (0 <= v <= 62):
                    self._serve_json({"ok": False, "error": "vga out of range (0-62)"})
                    return
                params["vga"] = v
            state.send_c2_command(sensor_id, "set_gain", params)
            self._serve_json({"ok": True})
        elif path == "/api/c2/set_squelch":
            threshold = float(body.get("threshold", -45))
            if not (-100 <= threshold <= -5):
                self._serve_json({"ok": False, "error": "threshold out of range (-100 to -5)"})
                return
            state.send_c2_command(sensor_id, "set_squelch",
                                  {"threshold": threshold})
            self._serve_json({"ok": True})
        elif path == "/api/c2/restart":
            params = {}
            if "center_freq" in body:
                f = int(body["center_freq"])
                if not (2400 <= f <= 2500):
                    self._serve_json({"ok": False, "error": "center_freq out of range (2400-2500 MHz)"})
                    return
                params["center_freq"] = f
            if "channels" in body:
                c = int(body["channels"])
                if not (1 <= c <= 96):
                    self._serve_json({"ok": False, "error": "channels out of range (1-96)"})
                    return
                params["channels"] = c
            state.send_c2_command(sensor_id, "restart", params)
            self._serve_json({"ok": True})
        elif path == "/api/c2/get_status":
            state.send_c2_command(sensor_id, "get_status")
            self._serve_json({"ok": True})
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = DASHBOARD_HTML.replace("__GPS_ENABLED__", "true" if gps_enabled else "false")
        self.wfile.write(html.encode())

    def _serve_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_csv(self):
        devs = state.get_devices()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", "attachment; filename=ble_devices.csv")
        self.end_headers()
        cols = ["mac", "protocol", "mac_type", "identity", "rpa_count",
                "uap", "uap_conf", "mfr", "apple",
                "name", "appear", "services", "type", "rssi", "rssi_min",
                "rssi_avg", "tx_pwr", "est_dist", "num_sensors",
                "est_lat", "est_lon", "est_unc", "pkts", "crc_ok",
                "crc_bad", "freq", "first", "last", "first_ever", "is_new"]
        self.wfile.write((",".join(cols) + "\n").encode())
        for d in devs:
            row = []
            for c in cols:
                v = d.get(c, "")
                if c == "services":
                    v = "|".join(d.get("services", []))
                elif v is None:
                    v = ""
                row.append(str(v).replace(",", ";"))
            self.wfile.write((",".join(row) + "\n").encode())

    def _serve_export_json(self):
        data = {
            "stats": state.get_stats(),
            "summary": state.get_summary(),
            "devices": state.get_devices(),
        }
        payload = json.dumps(data, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Disposition", "attachment; filename=ble_capture.json")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = state.register_sse()
        try:
            while running:
                time.sleep(1.0)
                # Push full device list + stats + summary once per second
                devs = state.get_devices()
                stats = state.get_stats()
                summary = state.get_summary()
                sensors = state.get_sensors()
                nodes = state.get_sensor_nodes()
                conns = state.get_connections()
                alerts = []
                if state.alert_mgr:
                    alerts = state.alert_mgr.get_pending_alerts()
                payload = json.dumps({"stats": stats, "devices": devs,
                                      "summary": summary, "sensors": sensors,
                                      "nodes": nodes, "connections": conns,
                                      "alerts": alerts})
                self.wfile.write(f"event: update\ndata: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            state.unregister_sse(q)

    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# ZMQ receiver thread
# ---------------------------------------------------------------------------
running = True


def parse_zmq_frames(frames):
    """Parse ZMQ multipart message, returning (sensor_id, gps_info, pcap_data).

    Handles all frame formats:
      1 frame:  [PCAP]                     -> (None, None, pcap)
      2 frames: [GPS_24b] [PCAP]           -> (None, gps, pcap)   legacy
      2 frames: [SENSOR_ID] [PCAP]         -> (id, None, pcap)    new, no GPS
      3 frames: [SENSOR_ID] [GPS_24b] [PCAP] -> (id, gps, pcap)  new, with GPS
    """
    sensor_id = None
    gps_info = None

    if len(frames) == 1:
        pcap_data = frames[0]
    elif len(frames) == 2:
        if len(frames[0]) == ZMQ_GPS_FRAME.size:
            lat, lon, alt = ZMQ_GPS_FRAME.unpack(frames[0])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                gps_info = (lat, lon, alt)
            else:
                sensor_id = frames[0].decode("utf-8", errors="replace")
        else:
            sensor_id = frames[0].decode("utf-8", errors="replace")
        pcap_data = frames[1]
    elif len(frames) >= 3:
        sensor_id = frames[0].decode("utf-8", errors="replace")
        if len(frames[1]) == ZMQ_GPS_FRAME.size:
            lat, lon, alt = ZMQ_GPS_FRAME.unpack(frames[1])
            gps_info = (lat, lon, alt)
        pcap_data = frames[-1]
    else:
        return None, None, None

    return sensor_id, gps_info, pcap_data


def _process_zmq_message(frames, pcap_file, use_gps, fallback_sensor_id=None):
    """Process a ZMQ message: parse frames, write PCAP, update state."""
    sensor_id, gps_info, pcap_data = parse_zmq_frames(frames)
    if pcap_data is None:
        return

    if sensor_id is None:
        sensor_id = fallback_sensor_id

    # Detect type prefix byte (new format: 0x00=BLE, 0x01=BT)
    pkt_type = ZMQ_PKT_TYPE_BLE
    raw_pcap = pcap_data
    if len(pcap_data) > PCAP_REC_HDR.size and pcap_data[0] <= ZMQ_PKT_TYPE_BT:
        pkt_type = pcap_data[0]
        raw_pcap = pcap_data[1:]

    # Write PCAP if requested (always PPI-wrapped for mixed BLE/BT support)
    if pcap_file:
        pkt_dlt = DLT_BLUETOOTH_LE_LL_WITH_PHDR if pkt_type == ZMQ_PKT_TYPE_BLE else DLT_BLUETOOTH_BREDR_BB
        ts_sec, ts_usec, incl_len, orig_len = PCAP_REC_HDR.unpack_from(raw_pcap, 0)
        payload = raw_pcap[PCAP_REC_HDR.size:]
        if gps_info:
            ppi_hdr = build_ppi_gps_header(*gps_info, dlt=pkt_dlt)
        else:
            ppi_hdr = build_ppi_passthrough_header(dlt=pkt_dlt)
        new_len = len(ppi_hdr) + len(payload)
        pcap_file.write(PCAP_REC_HDR.pack(ts_sec, ts_usec, new_len, new_len))
        pcap_file.write(ppi_hdr)
        pcap_file.write(payload)
        pcap_file.flush()

    # Parse based on packet type
    if pkt_type == ZMQ_PKT_TYPE_BT:
        pkt = parse_bt_packet(raw_pcap)
    else:
        pkt = parse_ble_packet(raw_pcap)

    if pkt:
        state.add_packet(pkt, gps_info, sensor_id)


def _process_gatt_message(frames):
    """Process a gatt: topic ZMQ message and store results in device state."""
    # Expected frames: [b"gatt:", sensor_id_bytes, json_payload]
    # or: [b"gatt:", json_payload] (no sensor_id)
    if len(frames) < 2:
        return
    try:
        payload = frames[-1]
        result = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    mac = result.get("mac", "").lower()
    if not mac:
        return

    with state.lock:
        if mac in state.devices:
            state.devices[mac]["gatt"] = result
            state._dirty = True


def zmq_receiver(endpoints, server_key_path, pcap_file, use_gps):
    """Bind SUB socket(s) and receive packets from connecting sensors."""
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 1000)
    if server_key_path:
        server_public_key = parse_server_pubkey(server_key_path)
        client_public, client_secret = zmq.curve_keypair()
        sub.setsockopt(zmq.CURVE_SERVERKEY, server_public_key)
        sub.setsockopt(zmq.CURVE_PUBLICKEY, client_public)
        sub.setsockopt(zmq.CURVE_SECRETKEY, client_secret)
    for ep in endpoints:
        sub.bind(ep)
        print(f"  Listening on {ep}", file=sys.stderr)

    while running:
        try:
            frames = sub.recv_multipart()
        except zmq.Again:
            continue
        except zmq.ZMQError:
            break

        # Handle gatt: topic messages (frame 0 = topic, frame 1 = sensor_id,
        # frame 2 = JSON payload)
        if frames and frames[0] == b"gatt:":
            _process_gatt_message(frames)
            continue

        _process_zmq_message(frames, pcap_file, use_gps)

    sub.close()
    ctx.term()


def parse_server_pubkey(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("server_public_key="):
                return line.split("=", 1)[1].encode()
            if line.startswith("public_key="):
                return line.split("=", 1)[1].encode()
    raise ValueError(f"No public key found in {path}")


def control_router_thread(data_port, server_key_path):
    """ROUTER thread for C2 control channel. Binds on data_port + 1."""
    control_port = data_port + 1
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)

    if server_key_path:
        server_public_key = parse_server_pubkey(server_key_path)
        router.setsockopt(zmq.CURVE_SERVER, 1)
        # ROUTER is CURVE server -- needs server secret key
        # For simplicity, read both public and secret from the keyfile
        with open(server_key_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("secret_key="):
                    router.setsockopt(zmq.CURVE_SECRETKEY,
                                      line.split("=", 1)[1].encode())
                elif line.startswith("public_key=") or line.startswith("server_public_key="):
                    router.setsockopt(zmq.CURVE_PUBLICKEY,
                                      line.split("=", 1)[1].encode())

    router.bind(f"tcp://*:{control_port}")
    print(f"  C2 control: tcp://*:{control_port} (ROUTER)", file=sys.stderr)

    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)

    while running:
        try:
            ready = dict(poller.poll(1000))
        except zmq.ZMQError:
            break

        # Receive heartbeats from sensors
        if router in ready:
            try:
                frames = router.recv_multipart(zmq.NOBLOCK)
                if len(frames) >= 2:
                    identity = frames[0]
                    payload = frames[-1]
                    try:
                        msg = json.loads(payload)
                        msg_type = msg.get("type", "")
                        if msg_type == "heartbeat":
                            sensor_id = msg.get("sensor_id",
                                                identity.decode("utf-8", "replace"))
                            state.update_sensor_c2(sensor_id, msg)
                        elif msg_type == "response":
                            pass  # could log C2 responses
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            except zmq.Again:
                pass

        # Drain command outbox and send via ROUTER
        while not state.c2_outbox.empty():
            try:
                sensor_id_bytes, cmd_bytes = state.c2_outbox.get_nowait()
                # Find the ROUTER identity for this sensor_id
                # The sensor's DEALER identity IS the sensor_id string
                router.send_multipart([sensor_id_bytes, cmd_bytes])
            except queue.Empty:
                break
            except zmq.ZMQError as e:
                print(f"  C2 send error: {e}", file=sys.stderr)
                break

    router.close()
    ctx.term()


# ---------------------------------------------------------------------------
# Dashboard HTML (self-contained, no external deps except Leaflet CDN for map)
# ---------------------------------------------------------------------------
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blue Dragon</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 12px/1.4 monospace; background: #1a1a1a; color: #ccc; }

.toolbar { padding: 4px 8px; background: #252525; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; font-size: 11px; }
.toolbar b { color: #fff; }
.status { color: #888; }
.status.ok { color: #7c4; }
.spacer { flex: 1; }
button { font: 11px monospace; background: #333; color: #ccc; border: 1px solid #555;
         padding: 2px 8px; cursor: pointer; }
button:hover { background: #444; }
button.on { background: #653; border-color: #a75; color: #fa8; }

.tabs { display: flex; gap: 0; }
.tab { padding: 2px 10px; border: 1px solid #555; border-bottom: none; cursor: pointer;
       background: #252525; color: #888; margin-right: -1px; }
.tab.active { background: #1a1a1a; color: #ccc; border-bottom: 1px solid #1a1a1a;
              position: relative; z-index: 1; }
.tab:hover:not(.active) { color: #aaa; }

.stats { padding: 3px 8px; background: #202020; border-bottom: 1px solid #333;
         font-size: 11px; color: #888; display: flex; flex-wrap: wrap; gap: 0; }
.stats span { margin-right: 16px; }
.stats .val { color: #ccc; }

.panel { display: none; }
.panel.active { display: flex; flex-direction: column; height: calc(100vh - 52px); }

.table-area { flex: 1; overflow: auto; }
.pager { padding: 4px 8px; background: #252525; border-top: 1px solid #333;
         display: flex; align-items: center; gap: 8px; font-size: 11px; }
.pager button:disabled { opacity: 0.3; cursor: default; }
.pager select { font: 11px monospace; background: #333; color: #ccc; border: 1px solid #555;
                padding: 1px 4px; }

table { width: 100%; border-collapse: collapse; }
thead { position: sticky; top: 0; z-index: 2; }
th { background: #252525; border-bottom: 1px solid #444; padding: 4px 8px;
     text-align: left; color: #888; font-weight: normal; cursor: pointer; }
th:hover { color: #ccc; }
th.sorted::after { content: ' \25bc'; }
th.sorted.asc::after { content: ' \25b2'; }
td { padding: 3px 8px; border-bottom: 1px solid #2a2a2a; white-space: nowrap; }
tr:hover td { background: #222; }
tr.fresh td { background: #1a2a1a; }

.dim { color: #666; }
.grn { color: #7c4; }
.red { color: #c44; }
.blu { color: #68f; }
.org { color: #ca6; }
.yel { color: #ec5; }
.masked { color: #666; }
.badge { display: inline-block; padding: 0 4px; border-radius: 2px; font-size: 9px;
         font-weight: bold; }
.badge-ble { background: #234; color: #68f; }
.badge-bt { background: #342; color: #ca6; }
select.filter { font: 11px monospace; background: #333; color: #ccc; border: 1px solid #555;
                padding: 1px 4px; }

#map { flex: 1; width: 100%; }
.empty { padding: 40px; text-align: center; color: #555; }
.count { text-align: right; }

/* Summary panel */
.summary-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;
                padding: 12px; overflow: auto; flex: 1; }
.summary-card { background: #202020; border: 1px solid #333; padding: 8px; }
.summary-card h3 { color: #888; font-size: 11px; font-weight: normal;
                   text-transform: uppercase; margin-bottom: 6px; border-bottom: 1px solid #333;
                   padding-bottom: 4px; }
.summary-card.wide { grid-column: span 2; }
.summary-card.full { grid-column: span 3; }
.bar-row { display: flex; align-items: center; margin: 2px 0; font-size: 11px; }
.bar-label { width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 12px; background: #2a2a2a; margin: 0 6px; position: relative; }
.bar-fill { height: 100%; position: absolute; left: 0; top: 0; }
.bar-val { width: 50px; text-align: right; color: #888; }
.talker-row { display: flex; font-size: 11px; margin: 1px 0; padding: 2px 0;
              border-bottom: 1px solid #2a2a2a; }
.talker-row .rank { width: 20px; color: #555; }
.talker-row .mac { width: 140px; }
.talker-row .info { flex: 1; color: #888; overflow: hidden; text-overflow: ellipsis; }
.talker-row .cnt { width: 60px; text-align: right; }
.ch-grid { display: flex; flex-wrap: wrap; gap: 2px; }
.ch-cell { width: 28px; height: 22px; display: flex; align-items: center; justify-content: center;
           font-size: 9px; border: 1px solid #333; }
.export-bar { padding: 4px 8px; background: #202020; border-top: 1px solid #333;
              display: flex; gap: 8px; align-items: center; font-size: 11px; }
/* Nodes panel */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
.dot-online { background: #4c4; }
.dot-stale { background: #cc4; }
.dot-offline { background: #c44; }
.node-ctrl { display: flex; align-items: center; gap: 6px; }
.node-ctrl input[type=range] { width: 80px; accent-color: #68f; }
.node-ctrl input[type=number] { width: 60px; font: 11px monospace; background: #333;
  color: #ccc; border: 1px solid #555; padding: 1px 3px; }
.node-ctrl button { font-size: 10px; padding: 1px 6px; }
.node-ctrl .val-label { font-size: 10px; color: #888; min-width: 30px; text-align: right; }
/* Device detail slide-out panel */
.detail-overlay { display:none; position:fixed; top:0; right:0; bottom:0;
  width:420px; max-width:90vw; background:#1e1e1e; border-left:1px solid #444;
  z-index:100; overflow-y:auto; padding:12px 16px; box-shadow:-4px 0 16px rgba(0,0,0,0.5); }
.detail-overlay.open { display:block; }
.detail-close { position:absolute; top:6px; right:10px; background:none; border:none;
  color:#888; font-size:18px; cursor:pointer; }
.detail-close:hover { color:#ccc; }
.detail-hdr { font-size:13px; color:#fff; margin-bottom:10px; border-bottom:1px solid #333;
  padding-bottom:8px; }
.detail-row { display:flex; margin:3px 0; font-size:11px; }
.detail-row .lbl { color:#666; width:90px; flex-shrink:0; }
.detail-row .val { color:#ccc; word-break:break-all; }
.detail-section { margin-top:12px; }
.detail-section h4 { color:#888; font-size:11px; text-transform:uppercase;
  margin-bottom:4px; border-bottom:1px solid #333; padding-bottom:3px; font-weight:normal; }
.gatt-svc { margin:6px 0; padding:4px 0 4px 8px; border-left:2px solid #345; }
.gatt-svc-uuid { color:#68f; font-size:11px; font-weight:bold; }
.gatt-svc-label { color:#888; font-size:10px; margin-left:6px; }
.gatt-char { margin:3px 0 3px 12px; padding:2px 0 2px 8px; border-left:1px solid #333; }
.gatt-char-uuid { color:#ca6; font-size:10px; }
.gatt-char-flags { color:#555; font-size:9px; margin-left:4px; }
.gatt-char-val { color:#7c4; font-size:10px; margin-top:1px; word-break:break-all; }
.gatt-error { color:#c44; font-size:11px; margin-top:4px; }
.gatt-btn { font:11px monospace; background:#234; color:#68f; border:1px solid #345;
  padding:3px 12px; cursor:pointer; margin-top:6px; }
.gatt-btn:hover { background:#345; }
.gatt-btn:disabled { opacity:0.4; cursor:default; }
.gatt-pending { color:#888; font-size:11px; margin-top:4px; }
.conn-badge { display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#345; margin-right:3px; vertical-align:middle; }
.conn-badge.yes { background:#68f; }
@media (max-width: 900px) {
  .summary-grid { grid-template-columns: 1fr; }
  .summary-card.wide, .summary-card.full { grid-column: span 1; }
  .detail-overlay { width: 100vw; max-width: 100vw; }
}
</style>
</head>
<body>

<div class="toolbar">
  <b>blue-dragon</b>
  <span class="status" id="conn">disconnected</span>
  <div class="tabs" id="tabBar">
    <div class="tab active" data-tab="devices" onclick="switchTab('devices')">devices</div>
    <div class="tab" data-tab="summary" onclick="switchTab('summary')">summary</div>
    <div class="tab" data-tab="nodes" onclick="switchTab('nodes')">nodes</div>
  </div>
  <select class="filter" id="protoFilter" onchange="curPage=1;renderDevices()" title="Protocol filter">
    <option value="all">all</option>
    <option value="BLE">BLE only</option>
    <option value="BT">BT only</option>
  </select>
  <select class="filter" id="sensorFilter" onchange="curPage=1;renderDevices()" title="Sensor filter">
    <option value="all">all sensors</option>
  </select>
  <span class="spacer"></span>
  <button onclick="location.href='/api/export.csv'">export CSV</button>
  <button onclick="location.href='/api/export.json'">export JSON</button>
  <button id="privBtn" class="on" onclick="togglePrivacy()">MAC hidden</button>
</div>

<div class="stats">
  <span>pkts: <span class="val" id="sTotal">0</span></span>
  <span>rate: <span class="val" id="sRate">0</span>/s</span>
  <span>crc: <span class="val" id="sCrc">--</span></span>
  <span>devices: <span class="val" id="sMacs">0</span></span>
  <span>data: <span class="val" id="sData">0</span></span>
  <span>conns: <span class="val" id="sConns">0</span></span>
  <span>up: <span class="val" id="sUp">0s</span></span>
</div>

<div class="panel active" id="panelDevices">
  <div class="table-area" id="tWrap">
    <table>
      <thead><tr>
        <th data-col="last" class="sorted">last seen</th>
        <th data-col="protocol">proto</th>
        <th data-col="mac">mac</th>
        <th data-col="mac_type">addr</th>
        <th data-col="mfr">manufacturer</th>
        <th data-col="name">name</th>
        <th data-col="services">services</th>
        <th data-col="type">type</th>
        <th data-col="phy">phy</th>
        <th data-col="rssi">rssi</th>
        <th data-col="est_dist">dist</th>
        <th data-col="num_sensors">sensors</th>
        <th data-col="pkts" class="count">pkts</th>
        <th data-col="crc">crc %</th>
        <th data-col="first">first seen</th>
        <th></th>
      </tr></thead>
      <tbody id="tb"></tbody>
    </table>
    <div class="empty" id="empty">waiting for devices...</div>
    <div class="pager" id="pager" style="display:none">
      <button id="pgPrev" onclick="curPage--;renderDevices()">&laquo; prev</button>
      <span id="pgInfo" class="dim"></span>
      <button id="pgNext" onclick="curPage++;renderDevices()">next &raquo;</button>
      <select id="pgSize" onchange="pageSize=+this.value;curPage=1;renderDevices()" title="Devices per page">
        <option value="25">25</option>
        <option value="50" selected>50</option>
        <option value="100">100</option>
      </select>
    </div>
  </div>
</div>

<div class="panel" id="panelSummary">
  <div class="summary-grid" id="summaryGrid">
    <div class="summary-card" id="cardProto"><h3>by protocol</h3></div>
    <div class="summary-card" id="cardMfr"><h3>by manufacturer</h3></div>
    <div class="summary-card" id="cardAddr"><h3>by address type</h3></div>
    <div class="summary-card" id="cardPdu"><h3>by PDU type</h3></div>
    <div class="summary-card" id="cardSvc"><h3>services seen (devices)</h3></div>
    <div class="summary-card wide" id="cardTop"><h3>top talkers (by packets)</h3></div>
    <div class="summary-card full" id="cardBleCh"><h3>BLE channel activity</h3></div>
    <div class="summary-card full" id="cardBtCh"><h3>BT channel activity</h3></div>
  </div>
</div>

<div class="panel" id="panelMap">
  <div id="map"></div>
</div>

<div class="panel" id="panelNodes">
  <div class="table-area">
    <table>
      <thead><tr>
        <th>status</th>
        <th>sensor</th>
        <th>sdr</th>
        <th>freq (MHz)</th>
        <th>channels</th>
        <th>gain</th>
        <th>squelch</th>
        <th>pkt rate</th>
        <th>crc %</th>
        <th>uptime</th>
        <th>gps</th>
        <th>controls</th>
      </tr></thead>
      <tbody id="nodesTb"></tbody>
    </table>
    <div class="empty" id="nodesEmpty">no sensor nodes connected</div>
    <h3 style="margin:12px 0 4px 0">BLE Connections</h3>
    <table class="dev-table" style="margin-bottom:0">
      <thead><tr>
        <th>initiator</th>
        <th>advertiser</th>
        <th>access addr</th>
        <th>interval</th>
        <th>hop</th>
        <th>channels</th>
        <th>data pkts</th>
        <th>age</th>
      </tr></thead>
      <tbody id="connTb"></tbody>
    </table>
    <div class="empty" id="connEmpty">no active connections</div>
  </div>
</div>

<div class="detail-overlay" id="detailPanel">
  <button class="detail-close" onclick="closeDetail()">&times;</button>
  <div id="detailContent"></div>
</div>

<script>
let priv = true, map = null, marker = null, trail = [], curTab = 'devices';
let sortCol = 'last', sortAsc = false;
const GPS = __GPS_ENABLED__;
const tb = document.getElementById('tb');
let devices = [], summary = null;
let curPage = 1, pageSize = 50;
let knownSensors = new Set();

function switchTab(name) {
  curTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab===name));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel'+name.charAt(0).toUpperCase()+name.slice(1)).classList.add('active');
  if (name === 'map' && map) map.invalidateSize();
  if (name === 'summary' && summary) renderSummary(summary);
}

function togglePrivacy() {
  priv = !priv;
  const b = document.getElementById('privBtn');
  b.textContent = priv ? 'MAC hidden' : 'MAC visible';
  b.classList.toggle('on', priv);
  renderDevices();
  if (summary) renderSummary(summary);
}

function mask(m) { return m ? 'xx:xx:xx:xx:xx:xx' : ''; }

function fmtT(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US',{hour12:false})+'.'+String(d.getMilliseconds()).padStart(3,'0');
}

function fmtUp(s) {
  if (s < 60) return Math.round(s)+'s';
  if (s < 3600) return Math.floor(s/60)+'m'+Math.round(s%60)+'s';
  return Math.floor(s/3600)+'h'+Math.floor((s%3600)/60)+'m';
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function ago(ts) {
  const s = (Date.now()/1000) - ts;
  if (s < 2) return 'now';
  if (s < 60) return Math.round(s)+'s ago';
  if (s < 3600) return Math.floor(s/60)+'m ago';
  return Math.floor(s/3600)+'h ago';
}

/* Column sorting */
document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) { sortAsc = !sortAsc; }
    else { sortCol = col; sortAsc = false; }
    curPage = 1;
    document.querySelectorAll('th').forEach(h => { h.classList.remove('sorted','asc'); });
    th.classList.add('sorted');
    if (sortAsc) th.classList.add('asc');
    renderDevices();
  });
});

function sortDevices(devs) {
  const dir = sortAsc ? 1 : -1;
  return devs.slice().sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (sortCol === 'services') { va = (a.services||[]).join(','); vb = (b.services||[]).join(','); }
    if (typeof va === 'string') return dir * va.localeCompare(vb);
    return dir * ((va||0) - (vb||0));
  });
}

function addrCls(t) {
  if (t==='public') return 'red';
  if (t==='static') return 'org';
  if (t==='resolvable') return 'yel';
  if (t==='resolved') return 'grn';
  return 'dim';
}

function mfrLabel(d) {
  let s = d.mfr || '';
  if (d.apple) s = s ? s+' '+d.apple : d.apple;
  if (d.appear && d.appear !== 'Unknown') s = s ? s+' ('+d.appear+')' : d.appear;
  return s;
}

function rssiLabel(d) {
  if (d.rssi === d.rssi_min) return d.rssi;
  return d.rssi_min + '/' + d.rssi_avg + '/' + d.rssi;
}

function svcLabel(d) {
  const s = d.services || [];
  if (!s.length) return '';
  if (s.length <= 2) return s.join(', ');
  return s.slice(0,2).join(', ') + ' +' + (s.length-2);
}

function protoBadge(proto) {
  if (proto === 'BT') return '<span class="badge badge-bt">BT</span>';
  return '<span class="badge badge-ble">BLE</span>';
}

function btMacLabel(d) {
  // Classic BT: show LAP, and UAP if converged
  if (d.uap != null) {
    const uap = ('0'+d.uap.toString(16)).slice(-2);
    return d.mac.replace('bt:', 'bt:'+uap+':');
  }
  if (d.uap_conf != null && d.uap_conf === 0 && d.pkts > 0) {
    return d.mac + ' <span class="dim" style="font-size:9px">(UAP?)</span>';
  }
  return d.mac;
}

function updateSensorFilter() {
  // Collect all sensor IDs from device data
  for (const d of devices) {
    if (d.sensor_ids) d.sensor_ids.forEach(s => knownSensors.add(s));
  }
  const sel = document.getElementById('sensorFilter');
  const cur = sel.value;
  const opts = ['all', ...Array.from(knownSensors).sort()];
  if (sel.options.length !== opts.length) {
    sel.innerHTML = '';
    for (const o of opts) {
      const opt = document.createElement('option');
      opt.value = o;
      opt.textContent = o === 'all' ? 'all sensors' : o;
      sel.appendChild(opt);
    }
    sel.value = cur && opts.includes(cur) ? cur : 'all';
  }
}

function renderDevices() {
  updateSensorFilter();
  const now = Date.now() / 1000;
  const pf = document.getElementById('protoFilter').value;
  const sf = document.getElementById('sensorFilter').value;
  let filtered = devices;
  if (pf !== 'all') filtered = filtered.filter(d => (d.protocol||'BLE') === pf);
  if (sf !== 'all') filtered = filtered.filter(d => d.sensor_ids && d.sensor_ids.includes(sf));
  const sorted = sortDevices(filtered);
  document.getElementById('empty').style.display = sorted.length ? 'none' : 'block';

  // Pagination
  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  if (curPage > totalPages) curPage = totalPages;
  if (curPage < 1) curPage = 1;
  const start = (curPage - 1) * pageSize;
  const page = sorted.slice(start, start + pageSize);

  const pager = document.getElementById('pager');
  if (sorted.length > pageSize) {
    pager.style.display = 'flex';
    document.getElementById('pgPrev').disabled = curPage <= 1;
    document.getElementById('pgNext').disabled = curPage >= totalPages;
    document.getElementById('pgInfo').textContent =
      (start+1)+'-'+Math.min(start+pageSize, sorted.length)+' of '+sorted.length+' devices (page '+curPage+'/'+totalPages+')';
  } else {
    pager.style.display = sorted.length ? 'flex' : 'none';
    document.getElementById('pgPrev').disabled = true;
    document.getElementById('pgNext').disabled = true;
    document.getElementById('pgInfo').textContent = sorted.length+' devices';
  }

  const frag = document.createDocumentFragment();
  for (const d of page) {
    const tr = document.createElement('tr');
    const fresh = (now - d.last) < 3;
    if (fresh) tr.className = 'fresh';
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', ((dev) => (e) => {
      if (e.target.tagName === 'BUTTON') return;
      openDetail(dev);
    })(d));
    const proto = d.protocol || 'BLE';
    const connBadge = d.connectable ? '<span class="conn-badge yes" title="connectable"></span>' : '';
    let mc, macCls;
    if (proto === 'BT') {
      mc = priv ? 'bt:xx:xx:xx' : btMacLabel(d);
      macCls = priv ? 'masked' : 'org';
    } else {
      mc = priv ? mask(d.mac) : d.mac;
      macCls = priv ? 'masked' : '';
      if (d.identity) {
        mc = priv ? '[hidden]' : d.mac;
        macCls = 'grn';
        if (!priv && d.rpa_count > 1) mc += ' ('+d.rpa_count+' addrs)';
      }
    }
    if (d.is_new) mc += ' <span style="color:#f55;font-size:9px;font-weight:bold">NEW</span>';
    const dist = d.est_dist != null ? '~'+d.est_dist+'m' : '';
    const total = d.crc_ok + d.crc_bad;
    const crcPct = total > 0 ? Math.round(100*d.crc_ok/total)+'%' : '-';
    const crcCls = total > 0 ? (d.crc_ok/total > 0.8 ? 'grn' : d.crc_ok/total > 0.4 ? 'yel' : 'red') : 'dim';
    const isAdv = d.type !== 'DATA' && d.type !== 'BT';
    const mt = d.mac_type || '';
    tr.innerHTML =
      `<td class="dim">${ago(d.last)}</td>`+
      `<td>${connBadge}${protoBadge(proto)}</td>`+
      `<td class="${macCls}">${mc}</td>`+
      `<td class="${addrCls(mt)}">${mt}</td>`+
      `<td>${esc(mfrLabel(d))}</td>`+
      `<td class="blu">${esc(d.name)}</td>`+
      `<td class="dim">${esc(svcLabel(d))}</td>`+
      `<td class="${isAdv?'blu':'org'}">${d.type}</td>`+
      `<td class="${d.phy!=='1M'?'org':'dim'}">${d.phy||'1M'}</td>`+
      `<td>${rssiLabel(d)}</td>`+
      `<td class="dim">${dist}</td>`+
      `<td class="dim">${d.num_sensors||''}</td>`+
      `<td class="count">${d.pkts.toLocaleString()}</td>`+
      `<td class="${crcCls}">${crcPct}</td>`+
      `<td class="dim">${fmtT(d.first)}</td>`+
      `<td><button onclick="toggleWatch('${d.mac.replace(/'/g,"\\'")}')" style="font-size:9px;padding:1px 5px;cursor:pointer;background:${watched.has(d.mac)?'#754':'#333'};color:#ccc;border:1px solid #555;border-radius:3px">${watched.has(d.mac)?'unwatch':'watch'}</button></td>`;
    frag.appendChild(tr);
  }
  tb.innerHTML = '';
  tb.appendChild(frag);
}

/* Summary tab rendering */
const barColors = {
  mfr: '#68f', addr: '#ca6', pdu: '#7c4', svc: '#c6f'
};

function renderBarChart(containerId, data, color, maxItems) {
  const el = document.getElementById(containerId);
  const h3 = el.querySelector('h3').outerHTML;
  if (!data || !data.length) { el.innerHTML = h3 + '<div class="dim" style="padding:8px">no data</div>'; return; }
  const max = data[0][1];
  const items = data.slice(0, maxItems || 12);
  let html = h3;
  for (const [label, count] of items) {
    const pct = max > 0 ? Math.round(100 * count / max) : 0;
    html += `<div class="bar-row"><span class="bar-label">${esc(label)}</span>`+
      `<span class="bar-track"><span class="bar-fill" style="width:${pct}%;background:${color};opacity:0.6"></span></span>`+
      `<span class="bar-val">${count.toLocaleString()}</span></div>`;
  }
  if (data.length > items.length) {
    html += `<div class="bar-row dim" style="justify-content:center">+${data.length-items.length} more</div>`;
  }
  el.innerHTML = html;
}

function renderTopTalkers(data) {
  const el = document.getElementById('cardTop');
  const h3 = el.querySelector('h3').outerHTML;
  if (!data || !data.length) { el.innerHTML = h3 + '<div class="dim" style="padding:8px">no data</div>'; return; }
  let html = h3;
  data.forEach((t, i) => {
    const mc = priv ? mask(t.mac) : t.mac;
    const info = [t.mfr, t.name].filter(Boolean).join(' - ') || '';
    html += `<div class="talker-row"><span class="rank">${i+1}</span>`+
      `<span class="mac ${priv?'masked':''}">${esc(mc)}</span>`+
      `<span class="info">${esc(info)}</span>`+
      `<span class="cnt">${t.pkts.toLocaleString()}</span></div>`;
  });
  el.innerHTML = html;
}

function renderChannels(cardId, data, color, advHighlight) {
  const el = document.getElementById(cardId);
  const h3 = el.querySelector('h3').outerHTML;
  if (!data || !data.length) { el.innerHTML = h3 + '<div class="dim" style="padding:8px">no data</div>'; return; }
  const max = Math.max(...data.map(d => d[1]));
  let html = h3 + '<div class="ch-grid">';
  for (const [ch, cnt] of data) {
    const intensity = max > 0 ? Math.max(0.15, Math.sqrt(cnt / max)) : 0;
    const r = Math.round(30 + intensity * color[0]);
    const g = Math.round(30 + intensity * color[1]);
    const b = Math.round(30 + intensity * color[2]);
    const isAdv = advHighlight && ch >= 37;
    const border = isAdv ? 'border-color:#ca6' : '';
    html += `<div class="ch-cell" style="background:rgb(${r},${g},${b});${border}" `+
      `title="Ch ${ch}: ${cnt.toLocaleString()} pkts">${ch}</div>`;
  }
  html += '</div>';
  el.innerHTML = html;
}

function renderSummary(s) {
  renderBarChart('cardProto', s.by_protocol, '#8c8', 4);
  renderBarChart('cardMfr', s.by_mfr, barColors.mfr, 12);
  renderBarChart('cardAddr', s.by_mac_type, barColors.addr, 8);
  renderBarChart('cardPdu', s.by_pdu, barColors.pdu, 8);
  renderBarChart('cardSvc', s.by_svc, barColors.svc, 12);
  /* Add context line showing how many devices have services at all */
  const svcEl = document.getElementById('cardSvc');
  if (s.total_devices > 0) {
    let ctx = svcEl.querySelector('.svc-ctx');
    if (!ctx) { ctx = document.createElement('div'); ctx.className = 'svc-ctx dim'; ctx.style.cssText = 'font-size:10px;padding:4px 0 0;border-top:1px solid #333;margin-top:4px'; svcEl.appendChild(ctx); }
    ctx.textContent = s.svc_devices + ' of ' + s.total_devices + ' devices advertise GATT services (most use mfr-specific data)';
  }
  renderTopTalkers(s.top_talkers);
  renderChannels('cardBleCh', s.ble_channels, [80,140,60], true);
  renderChannels('cardBtCh', s.bt_channels, [140,100,40], false);
}

function updStats(s) {
  document.getElementById('sTotal').textContent = s.total.toLocaleString();
  document.getElementById('sRate').textContent = s.rate;
  document.getElementById('sCrc').textContent = s.crc_pct!==null ? s.crc_pct+'%' : '--';
  document.getElementById('sMacs').textContent = s.macs;
  document.getElementById('sData').textContent = s.data_pkts.toLocaleString();
  document.getElementById('sConns').textContent = s.connections || 0;
  document.getElementById('sUp').textContent = fmtUp(s.uptime);
  if (GPS && s.last_gps && map) {
    const ll = s.last_gps;
    if (!marker) {
      marker = L.circleMarker(ll,{radius:6,color:'#68f',fillColor:'#68f',fillOpacity:0.8,weight:1}).addTo(map);
      map.setView(ll,15);
    } else {
      marker.setLatLng(ll);
    }
    trail.push(ll);
    if (trail.length>500) trail=trail.slice(-500);
    if (window._t) map.removeLayer(window._t);
    if (trail.length>1) window._t=L.polyline(trail,{color:'#68f',weight:2,opacity:0.5}).addTo(map);
  }
}

/* --- Multi-sensor map markers --- */
let sensorMarkers = {};
let deviceMarkers = {};

function updateSensors(sensors) {
  if (!map) return;
  for (const s of sensors) {
    if (s.lat == null || s.lon == null) continue;
    if (sensorMarkers[s.id]) {
      sensorMarkers[s.id].setLatLng([s.lat, s.lon]);
      sensorMarkers[s.id].unbindTooltip();
      sensorMarkers[s.id].bindTooltip(s.id+' ('+s.pkts.toLocaleString()+' pkts)');
    } else {
      const m = L.marker([s.lat, s.lon], {
        icon: L.divIcon({
          className: '',
          html: '<div style="background:#f80;color:#000;padding:2px 6px;border-radius:3px;font:bold 10px monospace;white-space:nowrap;border:1px solid #a50">'+s.id+'</div>',
          iconSize: null, iconAnchor: [0, 0]
        })
      }).addTo(map);
      m.bindTooltip(s.id+' ('+s.pkts.toLocaleString()+' pkts)');
      sensorMarkers[s.id] = m;
      map.setView([s.lat, s.lon], 15);
    }
  }
}

function updateDevicePositions(devs) {
  if (!map) return;
  const now = Date.now()/1000;
  const shown = new Set();
  for (const d of devs) {
    if (d.est_lat == null || d.est_lon == null) continue;
    const key = d.mac;
    shown.add(key);
    let color = d.num_sensors >= 3 ? '#4c4' : d.num_sensors === 2 ? '#cc4' : '#c44';
    const unc = d.est_unc || 50;
    const label = priv ? 'device' : (d.name || d.mac);
    const tip = label+' ~'+(d.est_dist||'?')+'m, '+d.num_sensors+' sensor'+(d.num_sensors!==1?'s':'');
    if (deviceMarkers[key]) {
      deviceMarkers[key].dot.setLatLng([d.est_lat, d.est_lon]);
      deviceMarkers[key].ring.setLatLng([d.est_lat, d.est_lon]);
      deviceMarkers[key].ring.setRadius(unc);
      deviceMarkers[key].dot.setStyle({color:color, fillColor:color});
      deviceMarkers[key].ring.setStyle({color:color, fillColor:color});
      deviceMarkers[key].dot.unbindTooltip();
      deviceMarkers[key].dot.bindTooltip(tip);
    } else {
      const dot = L.circleMarker([d.est_lat, d.est_lon], {
        radius:5, color:color, fillColor:color, fillOpacity:0.7, weight:1
      }).addTo(map);
      const ring = L.circle([d.est_lat, d.est_lon], {
        radius:unc, color:color, fillColor:color, fillOpacity:0.1,
        weight:1, dashArray:'4 4'
      }).addTo(map);
      dot.bindTooltip(tip);
      deviceMarkers[key] = {dot, ring};
    }
  }
  for (const key of Object.keys(deviceMarkers)) {
    if (!shown.has(key)) {
      map.removeLayer(deviceMarkers[key].dot);
      map.removeLayer(deviceMarkers[key].ring);
      delete deviceMarkers[key];
    }
  }
}

if (GPS) {
  const tab = document.createElement('div');
  tab.className = 'tab';
  tab.dataset.tab = 'map';
  tab.textContent = 'map';
  tab.onclick = function(){ switchTab('map'); };
  document.getElementById('tabBar').appendChild(tab);
  const lk=document.createElement('link'); lk.rel='stylesheet';
  lk.href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'; document.head.appendChild(lk);
  const sc=document.createElement('script');
  sc.src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  sc.onload=function(){
    map=L.map('map',{zoomControl:true}).setView([0,0],2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'OSM'}).addTo(map);
  }; document.body.appendChild(sc);
}

const es = new EventSource('/events');
const cn = document.getElementById('conn');
es.onopen = ()=>{ cn.textContent='connected'; cn.className='status ok'; };
es.onerror = ()=>{ cn.textContent='disconnected'; cn.className='status'; };
// Alerting: notification permission + watched devices
if ('Notification' in window && Notification.permission === 'default') {
  Notification.requestPermission();
}
const watched = new Set();
function toggleWatch(mac) {
  if (watched.has(mac)) watched.delete(mac);
  else watched.add(mac);
  renderDevices();
  fetch('/api/watch', {
    method: 'POST',
    body: JSON.stringify({mac: mac, watch: watched.has(mac)}),
    headers: {'Content-Type': 'application/json'}
  }).catch(()=>{});
}

/* --- Nodes tab --- */
let nodesData = [];
let nodesDragging = false;

function c2Post(endpoint, body) {
  fetch(endpoint, {
    method: 'POST',
    body: JSON.stringify(body),
    headers: {'Content-Type': 'application/json'}
  }).catch(()=>{});
}

function c2SetGain(sensorId, sdr, val, which) {
  const body = {sensor_id: sensorId};
  if (sdr === 'hackrf') {
    body[which] = parseInt(val);
  } else {
    body.gain = parseFloat(val);
  }
  c2Post('/api/c2/set_gain', body);
}

function c2SetSquelch(sensorId, val) {
  c2Post('/api/c2/set_squelch', {sensor_id: sensorId, threshold: parseFloat(val)});
}

function c2Restart(sensorId) {
  const freqEl = document.getElementById('restart-freq-'+sensorId);
  const chEl = document.getElementById('restart-ch-'+sensorId);
  const params = {sensor_id: sensorId};
  if (freqEl && freqEl.value) params.center_freq = parseInt(freqEl.value);
  if (chEl && chEl.value) params.channels = parseInt(chEl.value);
  if (!confirm('Restart sensor "'+sensorId+'"? This will briefly disconnect it.')) return;
  c2Post('/api/c2/restart', params);
}

function c2GetStatus(sensorId) {
  c2Post('/api/c2/get_status', {sensor_id: sensorId});
}

/* Show slider value live while dragging (label next to slider) */
function sliderPreview(el, labelId) {
  const lbl = labelId ? document.getElementById(labelId) : el.nextElementSibling;
  if (lbl) lbl.textContent = el.value;
}

function renderNodes() {
  if (nodesDragging) return;  /* don't clobber sliders mid-drag */
  const ntb = document.getElementById('nodesTb');
  const empty = document.getElementById('nodesEmpty');
  if (!nodesData.length) { ntb.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  const frag = document.createDocumentFragment();
  for (const n of nodesData) {
    const tr = document.createElement('tr');
    const dotCls = n.status === 'online' ? 'dot-online' : n.status === 'stale' ? 'dot-stale' : 'dot-offline';
    const gps = (n.lat != null && n.lon != null) ? n.lat.toFixed(4)+', '+n.lon.toFixed(4) : '-';
    const rate = n.pkt_rate != null ? n.pkt_rate.toFixed(1) : '-';
    const crc = n.crc_pct != null ? n.crc_pct.toFixed(1)+'%' : '-';
    const up = n.uptime != null ? fmtUp(n.uptime) : '-';
    let gainHtml = '-';
    if (n.has_c2 && n.gain != null) {
      const sid = n.id.replace(/[^a-zA-Z0-9_-]/g,'_');
      if (n.sdr === 'hackrf' && n.gain.lna != null) {
        gainHtml = '<div class="node-ctrl">' +
          'LNA <input type="range" min="0" max="40" step="8" value="'+n.gain.lna+'" ' +
          'oninput="sliderPreview(this,\'gl-'+sid+'\')" ' +
          'onmousedown="nodesDragging=true" ontouchstart="nodesDragging=true" ' +
          'onchange="nodesDragging=false;c2SetGain(\''+n.id+'\',\'hackrf\',this.value,\'lna\')">' +
          '<span class="val-label" id="gl-'+sid+'">'+n.gain.lna+'</span>' +
          'VGA <input type="range" min="0" max="62" step="2" value="'+n.gain.vga+'" ' +
          'oninput="sliderPreview(this,\'gv-'+sid+'\')" ' +
          'onmousedown="nodesDragging=true" ontouchstart="nodesDragging=true" ' +
          'onchange="nodesDragging=false;c2SetGain(\''+n.id+'\',\'hackrf\',this.value,\'vga\')">' +
          '<span class="val-label" id="gv-'+sid+'">'+n.gain.vga+'</span></div>';
      } else {
        const gv = n.gain.value != null ? n.gain.value : n.gain;
        gainHtml = '<div class="node-ctrl">' +
          '<input type="range" min="0" max="76" step="1" value="'+gv+'" ' +
          'oninput="sliderPreview(this)" ' +
          'onmousedown="nodesDragging=true" ontouchstart="nodesDragging=true" ' +
          'onchange="nodesDragging=false;c2SetGain(\''+n.id+'\',\''+n.sdr+'\',this.value,\'gain\')">' +
          '<span class="val-label">'+gv+'</span></div>';
      }
    }
    let sqlHtml = '-';
    if (n.has_c2 && n.squelch != null) {
      sqlHtml = '<div class="node-ctrl">' +
        '<input type="range" min="-80" max="-10" step="1" value="'+n.squelch+'" ' +
        'oninput="sliderPreview(this)" ' +
        'onmousedown="nodesDragging=true" ontouchstart="nodesDragging=true" ' +
        'onchange="nodesDragging=false;c2SetSquelch(\''+n.id+'\',this.value)">' +
        '<span class="val-label">'+n.squelch+'</span></div>';
    }
    let ctrlHtml = '';
    if (n.has_c2) {
      const sid = n.id.replace(/[^a-zA-Z0-9_-]/g,'_');
      ctrlHtml = '<div class="node-ctrl">' +
        '<input type="number" id="restart-freq-'+sid+'" placeholder="freq" value="'+(n.center_freq||'')+'" style="width:55px" onfocus="nodesDragging=true" onblur="nodesDragging=false">' +
        '<input type="number" id="restart-ch-'+sid+'" placeholder="ch" value="'+(n.channels||'')+'" style="width:40px" onfocus="nodesDragging=true" onblur="nodesDragging=false">' +
        '<button onclick="c2Restart(\''+n.id+'\')" style="color:#f88">restart</button>' +
        '<button onclick="c2GetStatus(\''+n.id+'\')">refresh</button></div>';
    }
    tr.innerHTML =
      '<td><span class="dot '+dotCls+'"></span>'+esc(n.status)+'</td>' +
      '<td>'+esc(n.id)+'</td>' +
      '<td>'+esc(n.sdr||'-')+'</td>' +
      '<td>'+(n.center_freq||'-')+'</td>' +
      '<td>'+(n.channels||'-')+'</td>' +
      '<td>'+gainHtml+'</td>' +
      '<td>'+sqlHtml+'</td>' +
      '<td>'+rate+'</td>' +
      '<td>'+crc+'</td>' +
      '<td>'+up+'</td>' +
      '<td class="dim">'+gps+'</td>' +
      '<td>'+ctrlHtml+'</td>';
    frag.appendChild(tr);
  }
  ntb.innerHTML = '';
  ntb.appendChild(frag);
}

let connectionsData = [];

function renderConnections() {
  const ctb = document.getElementById('connTb');
  const empty = document.getElementById('connEmpty');
  if (!connectionsData.length) { ctb.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  const frag = document.createDocumentFragment();
  for (const c of connectionsData) {
    const tr = document.createElement('tr');
    const ageStr = c.age < 60 ? Math.round(c.age)+'s' : Math.floor(c.age/60)+'m';
    tr.innerHTML =
      '<td>'+esc(c.init_addr)+'</td>' +
      '<td>'+esc(c.adv_addr)+'</td>' +
      '<td class="dim">'+esc(c.aa)+'</td>' +
      '<td>'+c.interval_ms+'ms</td>' +
      '<td>'+c.hop+'</td>' +
      '<td>'+c.used_channels+'/37</td>' +
      '<td class="count">'+c.data_pkts.toLocaleString()+'</td>' +
      '<td class="dim">'+ageStr+'</td>';
    frag.appendChild(tr);
  }
  ctb.innerHTML = '';
  ctb.appendChild(frag);
}

/* --- Device detail panel --- */
let detailMac = null;
let gattPending = {};

function openDetail(d) {
  detailMac = d.mac;
  renderDetail(d);
  document.getElementById('detailPanel').classList.add('open');
}

function closeDetail() {
  detailMac = null;
  document.getElementById('detailPanel').classList.remove('open');
}

function renderDetail(d) {
  const el = document.getElementById('detailContent');
  const mc = priv ? mask(d.mac) : d.mac;
  const proto = d.protocol || 'BLE';
  const total = d.crc_ok + d.crc_bad;
  const crcPct = total > 0 ? Math.round(100*d.crc_ok/total)+'%' : '-';
  const connLabel = d.connectable ? '<span class="grn">yes</span>' : '<span class="dim">no</span>';

  let html = '<div class="detail-hdr">' + esc(mc) + ' ' + protoBadge(proto) + '</div>';
  html += '<div class="detail-row"><span class="lbl">Name</span><span class="val blu">' + esc(d.name||'-') + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">Manufacturer</span><span class="val">' + esc(mfrLabel(d)||'-') + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">Addr type</span><span class="val ' + addrCls(d.mac_type||'') + '">' + esc(d.mac_type||'-') + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">Connectable</span><span class="val">' + connLabel + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">RSSI</span><span class="val">' + rssiLabel(d) + ' dBm</span></div>';
  if (d.tx_pwr != null) html += '<div class="detail-row"><span class="lbl">TX power</span><span class="val">' + d.tx_pwr + ' dBm</span></div>';
  if (d.est_dist != null) html += '<div class="detail-row"><span class="lbl">Est. dist</span><span class="val">~' + d.est_dist + ' m</span></div>';
  html += '<div class="detail-row"><span class="lbl">Packets</span><span class="val">' + d.pkts.toLocaleString() + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">CRC</span><span class="val">' + crcPct + ' (' + d.crc_ok + ' ok, ' + d.crc_bad + ' bad)</span></div>';
  html += '<div class="detail-row"><span class="lbl">PDU type</span><span class="val">' + esc(d.type) + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">PHY</span><span class="val">' + esc(d.phy||'1M') + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">First seen</span><span class="val">' + fmtT(d.first) + '</span></div>';
  html += '<div class="detail-row"><span class="lbl">Last seen</span><span class="val">' + ago(d.last) + '</span></div>';
  if (d.num_sensors > 0) html += '<div class="detail-row"><span class="lbl">Sensors</span><span class="val">' + d.num_sensors + (d.sensor_ids ? ' (' + d.sensor_ids.join(', ') + ')' : '') + '</span></div>';

  // AD services
  if (d.services && d.services.length) {
    html += '<div class="detail-section"><h4>Advertised Services</h4>';
    for (const s of d.services) html += '<div style="font-size:10px;color:#c6f;margin:1px 0">' + esc(s) + '</div>';
    html += '</div>';
  }

  // GATT section
  html += '<div class="detail-section"><h4>GATT Profile</h4>';
  if (d.gatt) {
    html += renderGattResult(d.gatt);
  } else if (gattPending[d.mac]) {
    html += '<div class="gatt-pending">Querying GATT services...</div>';
  } else {
    html += '<div class="dim" style="font-size:10px">No GATT data available.</div>';
  }
  if (d.connectable) {
    const btnDisabled = gattPending[d.mac] ? ' disabled' : '';
    const btnLabel = gattPending[d.mac] ? 'querying...' : (d.gatt ? 'refresh GATT' : 'query GATT');
    html += '<button class="gatt-btn"' + btnDisabled + ' onclick="queryGatt(\'' + d.mac.replace(/'/g,"\\'") + '\')">' + btnLabel + '</button>';
  }
  html += '</div>';

  el.innerHTML = html;
}

function renderGattResult(g) {
  if (g.error) {
    return '<div class="gatt-error">Error: ' + esc(g.error) + '</div>';
  }
  let html = '';
  if (g.device_name) {
    html += '<div class="detail-row"><span class="lbl">Device name</span><span class="val blu">' + esc(g.device_name) + '</span></div>';
  }
  const svcs = g.services || [];
  if (!svcs.length) {
    return html + '<div class="dim" style="font-size:10px">No services found.</div>';
  }
  for (const svc of svcs) {
    html += '<div class="gatt-svc">';
    html += '<span class="gatt-svc-uuid">' + esc(svc.uuid) + '</span>';
    if (svc.primary === false) html += '<span class="gatt-svc-label">(secondary)</span>';
    const chars = svc.characteristics || [];
    for (const ch of chars) {
      html += '<div class="gatt-char">';
      html += '<span class="gatt-char-uuid">' + esc(ch.uuid) + '</span>';
      if (ch.flags && ch.flags.length) html += '<span class="gatt-char-flags">[' + ch.flags.join(', ') + ']</span>';
      if (ch.value_str) html += '<div class="gatt-char-val">"' + esc(ch.value_str) + '"</div>';
      else if (ch.value) html += '<div class="gatt-char-val">' + esc(ch.value) + '</div>';
      html += '</div>';
    }
    html += '</div>';
  }
  return html;
}

function queryGatt(mac) {
  gattPending[mac] = true;
  fetch('/api/gatt/query', {
    method: 'POST',
    body: JSON.stringify({mac: mac}),
    headers: {'Content-Type': 'application/json'}
  }).then(r => r.json()).then(r => {
    if (r.error || (!r.status && r.ok === false)) {
      gattPending[mac] = false;
      if (r.error) alert('GATT query failed: ' + r.error);
    }
    // Re-render detail if still showing this device
    if (detailMac === mac) {
      const dev = devices.find(d => d.mac === mac);
      if (dev) renderDetail(dev);
    }
  }).catch(() => { gattPending[mac] = false; });
  // Re-render immediately to show pending state
  const dev = devices.find(d => d.mac === mac);
  if (dev) renderDetail(dev);
}

es.addEventListener('update', e => {
  const d = JSON.parse(e.data);
  devices = d.devices;
  summary = d.summary;
  if (d.nodes) nodesData = d.nodes;
  if (d.connections) connectionsData = d.connections;
  updStats(d.stats);
  renderDevices();
  if (curTab === 'summary') renderSummary(summary);
  if (curTab === 'nodes') { renderNodes(); renderConnections(); }
  if (d.sensors && map) updateSensors(d.sensors);
  if (devices && map) updateDevicePositions(devices);
  // Update detail panel if open
  if (detailMac) {
    const dev = devices.find(dd => dd.mac === detailMac);
    if (dev) {
      // Clear pending state if GATT data arrived
      if (dev.gatt && gattPending[detailMac]) {
        delete gattPending[detailMac];
      }
      renderDetail(dev);
    }
  }
  // Browser alerts
  if (d.alerts && d.alerts.length > 0) {
    for (const a of d.alerts) {
      const title = a.reason === 'new-device' ? 'New Device' : 'Watched Device';
      const body = a.mac + ' ' + (a.name||'') + ' (' + a.protocol + ', RSSI: ' + a.rssi + ')';
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, { body: body, tag: a.mac });
      }
      const tb2 = document.querySelector('.toolbar');
      if (tb2) { tb2.style.background='#532'; setTimeout(()=>{tb2.style.background='#252525';},2000); }
    }
  }
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global running, gps_enabled

    parser = argparse.ArgumentParser(
        description="Live web dashboard for blue-dragon")
    parser.add_argument("endpoints", nargs="+",
                        help="ZMQ endpoints (e.g. tcp://localhost:5555)")
    parser.add_argument("-p", "--port", type=int, default=8099,
                        help="HTTP port for dashboard (default: 8099)")
    parser.add_argument("-w", "--write", metavar="FILE",
                        help="Also write received packets to PCAP file")
    parser.add_argument("--gps", action="store_true",
                        help="Enable GPS column and map display")
    parser.add_argument("--server-key", metavar="FILE",
                        help="Server public key file for CURVE encryption")
    parser.add_argument("--update-bt-db", action="store_true",
                        help="Download/update Bluetooth numbers database from Nordic Semiconductor, then run")
    parser.add_argument("--irk-file", metavar="FILE",
                        help="File of IRKs for RPA resolution (format: label:hex per line)")
    parser.add_argument("--sensor-pos", action="append", metavar="LABEL:LAT,LON",
                        help="Static position for sensor without GPS (repeatable)")
    parser.add_argument("--path-loss-exp", type=float, default=2.0,
                        help="Path loss exponent for distance estimation (default: 2.0)")
    # SQLite persistence
    parser.add_argument("--db", metavar="FILE",
                        help="SQLite database path (default: ~/.cache/blue-dragon/devices.db)")
    parser.add_argument("--no-db", action="store_true",
                        help="Disable SQLite device persistence")
    # Device alerting
    parser.add_argument("--alert-file", metavar="FILE",
                        help="File of MAC/LAP/identity labels to watch (one per line)")
    parser.add_argument("--alert-cmd", metavar="CMD",
                        help="Shell command on alert (env: ALERT_MAC, ALERT_NAME, ALERT_RSSI, ...)")
    parser.add_argument("--alert-new", action="store_true",
                        help="Alert on devices never seen before (requires DB)")
    parser.add_argument("--alert-webhook", metavar="URL",
                        help="HTTP POST URL for alert notifications")
    parser.add_argument("--alert-cooldown", type=int, default=300,
                        help="Seconds between re-alerts for same device (default: 300)")
    args = parser.parse_args()

    # Update Bluetooth numbers database if requested
    if args.update_bt_db:
        print("\n  Updating Bluetooth numbers database...", file=sys.stderr)
        bt_db_update()

    # Load cached Bluetooth numbers (merges over hardcoded fallbacks)
    bt_db_load()

    # Load IRKs for RPA resolution
    global _irk_list
    if args.irk_file:
        _irk_list = _load_irk_file(args.irk_file)
        if _irk_list:
            print(f"  Loaded {len(_irk_list)} IRK(s) for RPA resolution", file=sys.stderr)
        else:
            print("  Warning: no valid IRKs found in file", file=sys.stderr)

    gps_enabled = args.gps

    # Parse static sensor positions
    static_positions = {}
    if args.sensor_pos:
        for sp in args.sensor_pos:
            try:
                label, coords = sp.split(":", 1)
                lat, lon = coords.split(",")
                static_positions[label.strip()] = (float(lat), float(lon))
            except (ValueError, IndexError):
                print(f"  Warning: invalid --sensor-pos '{sp}' (expected LABEL:LAT,LON)",
                      file=sys.stderr)

    # Re-initialize state with multilateration config
    global state
    state = DashboardState(static_positions=static_positions,
                           path_loss_exp=args.path_loss_exp)

    # SQLite persistence (on by default)
    if not args.no_db:
        db_path = args.db or DeviceDB.DEFAULT_PATH
        state.db = DeviceDB(db_path)

    # Device alerting
    if args.alert_file or args.alert_new or args.alert_cmd or args.alert_webhook:
        state.alert_mgr = AlertManager(
            watch_file=args.alert_file,
            alert_cmd=args.alert_cmd,
            alert_new=args.alert_new,
            webhook_url=args.alert_webhook,
            cooldown=args.alert_cooldown,
            db=state.db,
        )
        if args.alert_new and not state.db:
            print("  Warning: --alert-new requires database (remove --no-db)",
                  file=sys.stderr)

    def sig_handler(sig, frame):
        global running
        running = False
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    pcap_file = None
    if args.write:
        pcap_file = open(args.write, "wb")
        # Always PPI (DLT 192) to support mixed BLE + Classic BT per-packet DLT
        snaplen = PPI_GPS_SIZE + 4 + 2 + 255 + 3
        pcap_file.write(PCAP_GLOBAL_HDR.pack(0xA1B2C3D4, 2, 4, 0, 0, snaplen, DLT_PPI))
        pcap_file.flush()

    # Start ZMQ receiver thread (always binds, sensors connect to us)
    zmq_thread = threading.Thread(
        target=zmq_receiver,
        args=(args.endpoints, args.server_key, pcap_file, args.gps),
        daemon=True,
    )
    zmq_thread.start()

    # Start C2 ROUTER thread (control port = data port + 1)
    try:
        ep = args.endpoints[0]
        parsed = urlparse(ep.replace("tcp://", "http://"))
        c2_port = (parsed.port or 5555) + 1
    except Exception:
        c2_port = 5556
    c2_thread = threading.Thread(
        target=control_router_thread,
        args=(c2_port - 1, args.server_key),
        daemon=True,
    )
    c2_thread.start()

    # Start HTTP server
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    httpd.timeout = 1

    print(f"\n  Blue Dragon - Web Dashboard", file=sys.stderr)
    print(f"  {'='*40}", file=sys.stderr)
    print(f"  Dashboard:  http://localhost:{args.port}", file=sys.stderr)
    if args.write:
        print(f"  PCAP file:  {args.write} (DLT 192 PPI, BLE+BT)", file=sys.stderr)
    print(f"  Privacy:    MAC addresses hidden by default", file=sys.stderr)
    if _irk_list:
        print(f"  IRK file:   {args.irk_file} ({len(_irk_list)} key(s))", file=sys.stderr)
    n = args.path_loss_exp
    print(f"  Distance:   estimated from TX power + RSSI (n={n})", file=sys.stderr)
    if state.db:
        n_known = len(state.db._known_keys)
        print(f"  Database:   {state.db.path} ({n_known} known devices)", file=sys.stderr)
    else:
        print(f"  Database:   disabled (--no-db)", file=sys.stderr)
    if static_positions:
        for label, (lat, lon) in static_positions.items():
            print(f"  Sensor pos: {label} ({lat}, {lon})", file=sys.stderr)
    print(f"  C2 control: tcp://*:{c2_port} (ROUTER, sensors connect here)",
          file=sys.stderr)
    print(f"  {'='*40}\n", file=sys.stderr)

    try:
        while running:
            httpd.handle_request()
    except (KeyboardInterrupt, SystemExit):
        pass

    running = False
    print("\nShutting down...", file=sys.stderr)
    zmq_thread.join(timeout=3)
    c2_thread.join(timeout=3)
    httpd.server_close()

    if pcap_file:
        pcap_file.close()
        print(f"PCAP written to {args.write}", file=sys.stderr)

    if state.db:
        state.db.close()
        print(f"Database saved.", file=sys.stderr)

    print("Dashboard stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
