"""PCAP file-header checks with mocked sockets and no device database."""

import importlib.util
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
            mock.patch.object(subscriber, "running", False), \
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


class CaptureHeaderTests(unittest.TestCase):
    def test_capture_headers_allow_full_sized_records(self):
        for gps in (False, True):
            data, _ = capture_subscriber([], gps=gps)
            self.assertEqual(subscriber.PCAP_GLOBAL_HDR.unpack_from(data)[5], 65535)
        self.assertEqual(subscriber.PCAP_GLOBAL_HDR.unpack_from(dashboard_header())[5], 65535)


if __name__ == "__main__":
    unittest.main()
