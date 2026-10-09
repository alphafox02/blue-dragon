"""Synthetic capture contracts for both standalone ZMQ consumers.

Socket, signal and subscriber file I/O are mocked; no RF or device DB is used.
"""

import importlib.util
import struct
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from unittest import mock


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"zmq": types.ModuleType("zmq")}):
        spec.loader.exec_module(module)
    return module


subscriber = load_tool("zmq_subscriber")
dashboard = load_tool("zmq_web_dashboard")
GPS = (10.25, -20.5, 24.5)
GPS_FRAME = subscriber.ZMQ_GPS_FRAME.pack(*GPS)
SENSOR_24 = b"123456789012345678901234"
# The Rust zmq_build_bt contract also pinned by the existing dashboard tests.
CLASSIC_MESSAGE = bytes.fromhex(
    "010100000000000000190000001900000027d8a6003200ffff"
    "6050400060504030e9960200b70f010203")


def ble_message(seconds=1000, original_extra=0):
    packet = subscriber.BLE_RF_HDR.pack(
        12, -45, -95, 0, subscriber.BLE_ADV_AA,
        subscriber.LE_DEWHITENED | subscriber.LE_CRC_CHECKED | subscriber.LE_CRC_VALID,
    ) + struct.pack("<I", subscriber.BLE_ADV_AA) + bytes.fromhex(
        "0006010203040506000000")
    return b"\x00" + subscriber.PCAP_REC_HDR.pack(
        seconds, 1234, len(packet), len(packet) + original_extra) + packet


def records(data):
    offset = subscriber.PCAP_GLOBAL_HDR.size
    result = []
    while offset < len(data):
        header = subscriber.PCAP_REC_HDR.unpack_from(data, offset)
        offset += subscriber.PCAP_REC_HDR.size
        payload = data[offset:offset + header[2]]
        if len(payload) != header[2]:
            raise AssertionError("truncated output record")
        result.append((header, payload))
        offset += header[2]
    return result


def fake_socket(module, messages):
    pending = iter(messages)
    socket = mock.Mock()

    def receive_next():
        try:
            return next(pending)
        except StopIteration:
            module.running = False
            return []

    socket.recv_multipart.side_effect = receive_next
    context = mock.Mock()
    context.socket.return_value = socket
    fake_zmq = types.SimpleNamespace(Context=lambda: context, SUB=1,
                                     SUBSCRIBE=2, RCVTIMEO=3,
                                     Again=TimeoutError, ZMQError=OSError)
    return fake_zmq, socket, context


def capture_subscriber(messages, gps=False):
    fake_zmq, socket, context = fake_socket(subscriber, messages)

    class Capture(BytesIO):
        def close(self):
            pass

    capture, stderr = Capture(), StringIO()
    args = ["subscriber", "inproc://mock", "-w", "mock.pcap"]
    if gps:
        args.append("--gps")
    with mock.patch.object(subscriber, "zmq", fake_zmq), \
            mock.patch.object(subscriber, "running", True), \
            mock.patch.object(subscriber.signal, "signal"), \
            mock.patch.object(sys, "argv", args), \
            mock.patch("builtins.open", return_value=capture), \
            redirect_stdout(StringIO()), redirect_stderr(stderr):
        subscriber.main()
    socket.close.assert_called_once()
    context.term.assert_called_once()
    return capture.getvalue(), stderr.getvalue()


def dashboard_header():
    """Exercise the dashboard's real file-header path with all services stubbed."""
    class Capture(BytesIO):
        def close(self):
            pass

    capture = Capture()
    args = ["dashboard", "inproc://mock", "--no-db", "-w", "mock.pcap"]
    with mock.patch.object(dashboard, "running", False), \
            mock.patch.object(dashboard, "state"), \
            mock.patch.object(dashboard, "gps_enabled", False), \
            mock.patch.object(dashboard, "bt_db_load"), \
            mock.patch.object(dashboard, "ThreadingHTTPServer"), \
            mock.patch.object(dashboard.threading, "Thread"), \
            mock.patch.object(dashboard.signal, "signal"), \
            mock.patch.object(sys, "argv", args), \
            mock.patch("builtins.open", return_value=capture), \
            redirect_stderr(StringIO()):
        dashboard.main()
    return capture.getvalue()


def capture_dashboard(messages, gps=False):
    fake_zmq, socket, context = fake_socket(dashboard, messages)
    capture, state = BytesIO(), dashboard.DashboardState()
    capture.write(dashboard_header())
    with mock.patch.object(dashboard, "zmq", fake_zmq), \
            mock.patch.object(dashboard, "running", True), \
            mock.patch.object(dashboard, "state", state), \
            redirect_stderr(StringIO()):
        dashboard.zmq_receiver(["inproc://mock"], None, capture, gps)
    socket.close.assert_called_once()
    context.term.assert_called_once()
    return capture.getvalue(), state


class CaptureEnvelopeTests(unittest.TestCase):
    def test_standard_types_and_legacy_timestamp_bytes(self):
        for tool in (subscriber, dashboard):
            for seconds in (0, 1, 2, 37, 255, 1000):
                record = ble_message(seconds)[1:]
                with self.subTest(tool=tool.__name__, seconds=seconds):
                    self.assertEqual(tool.parse_packet_message(record), (0, record))
                    self.assertEqual(tool.parse_packet_message(b"\x00" + record), (0, record))
            self.assertEqual(tool.parse_packet_message(CLASSIC_MESSAGE),
                             (1, CLASSIC_MESSAGE[1:]))

    def test_invalid_prefix_lengths_and_short_headers(self):
        record = CLASSIC_MESSAGE[1:]
        for tool in (subscriber, dashboard):
            for message in (
                    b"", b"\x01", b"{\"long_json\": \"not a packet record\"}",
                    CLASSIC_MESSAGE[:-1], CLASSIC_MESSAGE + b"\0",
                    b"\x02" + record, b"\x03" + record,
                    b"\x01" + tool.PCAP_REC_HDR.pack(1, 1_000_000, 25, 25) + record[16:],
                    b"\x01" + tool.PCAP_REC_HDR.pack(1, 0, 25, 24) + record[16:],
                    b"\x01" + tool.PCAP_REC_HDR.pack(1, 0, 21, 21) + b"\0" * 21,
                    b"\x00" + tool.PCAP_REC_HDR.pack(1, 0, 9, 9) + b"\0" * 9,
                    b"\x00" + tool.PCAP_REC_HDR.pack(1, 0, 25, 0xFFFFFFFF) + record[16:]):
                with self.subTest(tool=tool.__name__, message=message[:20]):
                    self.assertIsNone(tool.parse_packet_message(message))

    def test_maximum_inner_record_leaves_room_for_gps_wrapping(self):
        for tool in (subscriber, dashboard):
            for size, accepted in ((65499, True), (65500, False)):
                payload = CLASSIC_MESSAGE[17:39] + bytes(size - 22)
                message = b"\x01" + tool.PCAP_REC_HDR.pack(1000, 1234, size, size) + payload
                with self.subTest(tool=tool.__name__, size=size):
                    parsed = tool.parse_packet_message(message)
                    self.assertEqual(parsed is not None, accepted)

    def test_all_layouts_include_24_byte_ids(self):
        record = ble_message()
        cases = [([record], (None, None, record)),
                 ([b"sensor-a", record], ("sensor-a", None, record)),
                 ([GPS_FRAME, record], (None, GPS, record)),
                 ([b"sensor-a", GPS_FRAME, record], ("sensor-a", GPS, record)),
                 ([SENSOR_24, GPS_FRAME, record], (SENSOR_24.decode(), GPS, record)),
                 ([SENSOR_24, record], (SENSOR_24.decode(), None, record))]
        for tool in (subscriber, dashboard):
            for frames, expected in cases:
                with self.subTest(tool=tool.__name__, frames=frames[:-1]):
                    self.assertEqual(tool.parse_zmq_frames(frames), expected)

    def test_binary_record_ending_with_colon_is_not_a_topic(self):
        record = ble_message()[:-1] + b":"
        for tool in (subscriber, dashboard):
            self.assertEqual(tool.parse_zmq_frames([record]), (None, None, record))

    def test_bad_gps_retains_packet_without_unsafe_metadata(self):
        record = ble_message()
        for tool in (subscriber, dashboard):
            for frame in (b"short", tool.ZMQ_GPS_FRAME.pack(float("nan"), 0, 0),
                          tool.ZMQ_GPS_FRAME.pack(91, 0, 0),
                          tool.ZMQ_GPS_FRAME.pack(0, 181, 0),
                          tool.ZMQ_GPS_FRAME.pack(0, 0, float("inf")),
                          tool.ZMQ_GPS_FRAME.pack(0, 0, 1e308),
                          tool.ZMQ_GPS_FRAME.pack(0, 0, -1e308)):
                with self.subTest(tool=tool.__name__, frame=frame):
                    self.assertEqual(tool.parse_zmq_frames([SENSOR_24, frame, record]),
                                     (SENSOR_24.decode(), None, record))
            for frames in ([], [b"a", GPS_FRAME, b"extra", record]):
                self.assertEqual(tool.parse_zmq_frames(frames), (None, None, None))

    def test_topics_cannot_be_sensor_ids_even_with_packet_shaped_bodies(self):
        for tool in (subscriber, dashboard):
            for topic in (b"gatt:", b"scan:", b"smp:", b"future_topic:",
                          b"future/format:", b"x" * 100 + b":"):
                for frames in ([topic, CLASSIC_MESSAGE],
                               [topic, b"sensor-a", CLASSIC_MESSAGE]):
                    self.assertEqual(tool.parse_zmq_frames(frames), (None, None, None))


class CaptureLoopTests(unittest.TestCase):
    def test_capture_headers_allow_full_sized_records(self):
        for gps in (False, True):
            data, _ = capture_subscriber([], gps=gps)
            self.assertEqual(subscriber.PCAP_GLOBAL_HDR.unpack_from(data)[5], 65535)
        self.assertEqual(subscriber.PCAP_GLOBAL_HDR.unpack_from(dashboard_header())[5], 65535)

    def test_large_classic_record_survives_capture_and_gps_wrapping(self):
        payload = CLASSIC_MESSAGE[17:39] + bytes(range(256)) + bytes(83)
        message = b"\x01" + subscriber.PCAP_REC_HDR.pack(1000, 1234, len(payload), len(payload)) + payload
        for gps in (False, True):
            frames = [[SENSOR_24, GPS_FRAME, message]]
            subscriber_data, _ = capture_subscriber(frames, gps=gps)
            dashboard_data, _ = capture_dashboard(frames, gps=gps)
            for data, expected_ppi in ((subscriber_data, 36 if gps else 0),
                                       (dashboard_data, 36)):
                header, wrapped = records(data)[0]
                self.assertEqual(header[:2], (1000, 1234))
                if expected_ppi:
                    self.assertEqual(subscriber.PPI_HDR.unpack_from(wrapped),
                                     (0, 0, expected_ppi, 255))
                self.assertEqual(wrapped[expected_ppi:], payload)
                self.assertEqual(header[3], header[2])
                self.assertGreater(header[2], 300)
                self.assertLessEqual(header[2], subscriber.PCAP_GLOBAL_HDR.unpack_from(data)[5])

    def test_mixed_capture_gps_dlt_timestamp_payload_and_counts(self):
        messages = [[ble_message()], [SENSOR_24, ble_message()[1:]],
                    [GPS_FRAME, ble_message()[1:]],
                    [b"classic-sensor", GPS_FRAME, CLASSIC_MESSAGE],
                    [SENSOR_24, GPS_FRAME, CLASSIC_MESSAGE]]
        data, summary = capture_subscriber(messages, gps=True)
        other, state = capture_dashboard(messages, gps=True)
        self.assertEqual(data, other)
        self.assertIn("5 packets in", summary)
        self.assertIn("GPS: 3 packets with coordinates", summary)
        self.assertIn(SENSOR_24.decode(), state.sensors)
        self.assertEqual(subscriber.PCAP_GLOBAL_HDR.unpack_from(data)[6], 192)
        for index, (header, payload) in enumerate(records(data)):
            dlt = 256 if index < 3 else 255
            ppi = subscriber.PPI_HDR.unpack_from(payload)
            self.assertEqual(ppi[3], dlt)
            self.assertEqual(ppi[2], subscriber.PPI_HDR_SIZE if index < 2 else subscriber.PPI_GPS_SIZE)
            message = ble_message() if dlt == 256 else CLASSIC_MESSAGE
            self.assertEqual(header[:2], subscriber.PCAP_REC_HDR.unpack_from(message, 1)[:2])
            self.assertEqual(payload[ppi[2]:], message[17:])
            self.assertEqual(header[2], header[3])
            if index >= 2:
                self.assertEqual(payload[:ppi[2]], subscriber.build_ppi_gps_header(*GPS, dlt=dlt))

    def test_default_subscriber_encapsulation_is_unchanged(self):
        messages = [[SENSOR_24, GPS_FRAME, ble_message()], [CLASSIC_MESSAGE]]
        data, summary = capture_subscriber(messages)
        self.assertIn("2 packets in", summary)
        self.assertEqual(subscriber.PCAP_GLOBAL_HDR.unpack_from(data)[6], 256)
        self.assertEqual(data[24:], ble_message()[1:] + CLASSIC_MESSAGE[1:])
        # The separate PPI-default change corrects default Classic labeling.

    def test_dashboard_preserves_received_gps_without_display_flag(self):
        captures = []
        for display_gps in (False, True):
            data, state = capture_dashboard(
                [[SENSOR_24, GPS_FRAME, ble_message()]], gps=display_gps)
            payload = records(data)[0][1]
            self.assertEqual(payload[:subscriber.PPI_GPS_SIZE],
                             subscriber.build_ppi_gps_header(*GPS, dlt=256))
            self.assertEqual(state.sensors[SENSOR_24.decode()]["lat"], GPS[0])
            captures.append(data)
        self.assertEqual(*captures)

    def test_invalid_records_and_topics_then_valid_packet(self):
        messages = [[], [b"short"], [CLASSIC_MESSAGE[:-1]],
                    [b"a", GPS_FRAME, b"extra", ble_message()],
                    [b"\x02" + CLASSIC_MESSAGE[1:]],
                    [b"gatt:", b"sensor-a", b"{}"],
                    [b"scan:", b"sensor-a", b"{}"],
                    [b"smp:", b"sensor-a", b"{}"],
                    [b"future:", b"sensor-a", b"{bad json"],
                    [b"future:", CLASSIC_MESSAGE],
                    [b"future:", b"sensor-a", CLASSIC_MESSAGE],
                    [SENSOR_24, b"bad GPS", ble_message()]]
        for gps in (False, True):
            data, summary = capture_subscriber(messages, gps=gps)
            other, state = capture_dashboard(messages, gps=gps)
            self.assertEqual(len(records(data)), 1)
            self.assertEqual(len(records(other)), 1)
            self.assertIn("1 packets in", summary)
            self.assertEqual(set(state.sensors), {SENSOR_24.decode()})
            self.assertEqual(len(state.devices), 1)

    def test_captured_truncation_preserves_original_length(self):
        message = ble_message(original_extra=17)
        for gps in (False, True):
            data, _ = capture_subscriber([[SENSOR_24, GPS_FRAME, message]], gps=gps)
            other, _ = capture_dashboard([[SENSOR_24, GPS_FRAME, message]], gps=gps)
            for capture in (data, other):
                header, _ = records(capture)[0]
                self.assertEqual(header[3] - header[2], 17)


if __name__ == "__main__":
    unittest.main()
