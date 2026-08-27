#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "zmq_web_dashboard.py"
SPEC = importlib.util.spec_from_file_location("zmq_web_dashboard", MODULE_PATH)
dashboard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)


# Produced by bd_output::pcap::zmq_build_bt. The Rust test pins the same bytes.
VERIFIED_EDR3_ZMQ_HEX = (
    "010100000000000000190000001900000027d8a6003200ffff"
    "6050400060504030e9960200b70f010203"
)


def classic_record(*, flags, payload=b"", uap=0x30, lap=0x405060,
                   packet_type=0x0D, transport_rate=0x32, timestamp=1.0):
    bt_header = (packet_type & 0x0F) << 3
    ref_lap_uap = (uap << 24) | lap
    body = dashboard.BREDR_BB_HDR.pack(
        39, -40, -90, 0, transport_rate, 0, -1,
        lap, ref_lap_uap, bt_header, flags,
    ) + payload
    sec = int(timestamp)
    usec = int((timestamp - sec) * 1_000_000)
    return dashboard.PCAP_REC_HDR.pack(sec, usec, len(body), len(body)) + body


class ClassicDashboardTests(unittest.TestCase):
    def test_parses_exact_rust_zmq_contract_fixture(self):
        message = bytes.fromhex(VERIFIED_EDR3_ZMQ_HEX)
        self.assertEqual(message[0], dashboard.ZMQ_PKT_TYPE_BT)
        pkt = dashboard.parse_bt_packet(message[1:])

        self.assertEqual(pkt["mac"], "bt:40:50:60")
        self.assertEqual(pkt["uap"], 0x30)
        self.assertEqual(pkt["pdu_type"], "3-DH5/EV5")
        self.assertEqual(pkt["phy"], "EDR3")
        self.assertEqual(pkt["payload"], b"\x01\x02\x03")

    def test_accepts_sensor_verified_uap_and_edr_metadata(self):
        flags = (
            dashboard.BREDR_DEWHITENED
            | dashboard.BREDR_SIGNAL_POWER_VALID
            | dashboard.BREDR_NOISE_POWER_VALID
            | dashboard.BREDR_REFLAP_VALID
            | dashboard.BREDR_REFUAP_VALID
            | dashboard.BREDR_HEC_CHECKED
            | dashboard.BREDR_HEC_VALID
            | dashboard.BREDR_PAYLOAD_PRESENT
            | dashboard.BREDR_CRC_CHECKED
            | dashboard.BREDR_CRC_VALID
        )
        pkt = dashboard.parse_bt_packet(
            classic_record(flags=flags, payload=b"\x01\x02\x03"))

        self.assertEqual(pkt["uap"], 0x30)
        self.assertTrue(pkt["uap_verified"])
        self.assertEqual(pkt["pdu_type"], "3-DH5/EV5")
        self.assertEqual(pkt["phy"], "EDR3")
        self.assertTrue(pkt["crc_valid"])
        self.assertEqual(pkt["payload"], b"\x01\x02\x03")
        self.assertNotIn("raw_header", pkt)

    def test_payload_bytes_cannot_be_mistaken_for_uap_evidence(self):
        flags = dashboard.BREDR_PAYLOAD_PRESENT | dashboard.BREDR_REFLAP_VALID
        pkt = dashboard.parse_bt_packet(
            classic_record(flags=flags, payload=b"\xff" * 7))

        self.assertIsNone(pkt["uap"])
        self.assertFalse(pkt["uap_verified"])
        self.assertEqual(pkt["pdu_type"], "BT")

    def test_lap_record_upgrades_after_sensor_verification(self):
        state = dashboard.DashboardState()
        lap_flags = dashboard.BREDR_REFLAP_VALID
        verified_flags = (
            lap_flags
            | dashboard.BREDR_REFUAP_VALID
            | dashboard.BREDR_HEC_CHECKED
            | dashboard.BREDR_HEC_VALID
        )
        state.add_packet(
            dashboard.parse_bt_packet(classic_record(flags=lap_flags)),
            sensor_id="sensor-a",
        )
        state.add_packet(
            dashboard.parse_bt_packet(
                classic_record(flags=verified_flags, timestamp=2.0)),
            sensor_id="sensor-a",
        )

        devices = state.get_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["mac"], "bt:40:50:60")
        self.assertEqual(devices[0]["uap"], 0x30)
        self.assertEqual(devices[0]["uap_conf"], 1.0)
        self.assertEqual(devices[0]["pkts"], 2)
        self.assertIn("bt:30:40:50:60", state.devices)
        self.assertNotIn("bt:40:50:60", state.devices)

    def test_unverified_lap_needs_three_nearby_sightings(self):
        state = dashboard.DashboardState()
        flags = dashboard.BREDR_REFLAP_VALID
        for timestamp in (1.0, 2.0):
            state.add_packet(dashboard.parse_bt_packet(
                classic_record(flags=flags, timestamp=timestamp)))
        self.assertEqual(state.get_devices(), [])

        state.add_packet(dashboard.parse_bt_packet(
            classic_record(flags=flags, timestamp=3.0)), sensor_id="sensor-a")
        devices = state.get_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["pkts"], 3)

    def test_shared_lap_is_not_merged_without_verified_uap(self):
        state = dashboard.DashboardState()
        verified = (
            dashboard.BREDR_REFLAP_VALID
            | dashboard.BREDR_REFUAP_VALID
            | dashboard.BREDR_HEC_CHECKED
            | dashboard.BREDR_HEC_VALID
        )
        state.add_packet(dashboard.parse_bt_packet(
            classic_record(flags=verified, uap=0x30)))
        state.add_packet(dashboard.parse_bt_packet(
            classic_record(flags=verified, uap=0x42, timestamp=2.0)))
        state.add_packet(dashboard.parse_bt_packet(
            classic_record(flags=dashboard.BREDR_REFLAP_VALID,
                           timestamp=3.0)))

        self.assertEqual(set(state.devices), {
            "bt:30:40:50:60", "bt:42:40:50:60",
        })
        self.assertEqual([d["pkts"] for d in state.devices.values()], [1, 1])

    def test_database_merges_provisional_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = dashboard.DeviceDB(Path(tmp) / "devices.db")
            db.upsert("bt:40:50:60", "BT", 1.0, rssi=-50)
            db.upsert("bt:40:50:60", "BT", 2.0, rssi=-40)
            db.upsert("bt:30:40:50:60", "BT", 3.0, rssi=-60)
            db.migrate_key("bt:40:50:60", "bt:30:40:50:60")
            rows = db.conn.execute(
                "SELECT dev_key, first_seen, last_seen, total_pkts, best_rssi "
                "FROM devices").fetchall()
            self.assertEqual(rows, [("bt:30:40:50:60", 1.0, 3.0, 3, -40)])
            db.close()


if __name__ == "__main__":
    unittest.main()
