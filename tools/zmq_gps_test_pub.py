#!/usr/bin/env python3
# Copyright 2025-2026 CEMAXECUTER LLC
"""
Test publisher that sends fake BLE packets with GPS coordinates over ZMQ.

Simulates a sniffer with --gpsd by sending multipart messages:
  frame 1: GPS struct (lat, lon, alt as 3 doubles)
  frame 2: PCAP record (same format as real sniffer)

Usage:
    # Start fake publisher:
    python3 zmq_gps_test_pub.py

    # In another terminal, subscribe with GPS:
    python3 zmq_subscriber.py tcp://localhost:5555 -w test.pcap --gps

    # Then verify in Wireshark:
    wireshark test.pcap

Requirements:
    pip install pyzmq
"""

import argparse
import struct
import sys
import time

try:
    import zmq
except ImportError:
    print("pyzmq is required: pip install pyzmq", file=sys.stderr)
    sys.exit(1)

# Structures matching pcap.c
PCAP_REC_HDR = struct.Struct("<IIII")
BLE_RF_HDR = struct.Struct("<bbbBIH")
ZMQ_GPS_FRAME = struct.Struct("<ddd")

LE_DEWHITENED = 0x0001
LE_SIGNAL_POWER_VALID = 0x0002
LE_NOISE_POWER_VALID = 0x0004
BLE_ADV_AA = 0x8E89BED6


def make_fake_ble_adv():
    """Create a minimal fake BLE advertising packet."""
    aa = struct.pack("<I", BLE_ADV_AA)
    # ADV_IND PDU: type=0, length=12
    header = bytes([0x00, 12])
    # Fake address + some payload
    addr = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66])
    ad_data = bytes([0x02, 0x01, 0x06, 0x03, 0x03, 0x0F])
    # Fake CRC (3 bytes)
    crc = bytes([0xAA, 0xBB, 0xCC])
    return aa + header + addr + ad_data + crc


def main():
    parser = argparse.ArgumentParser(
        description="Fake GPS BLE publisher for testing")
    parser.add_argument("-e", "--endpoint", default="tcp://*:5555",
                        help="ZMQ bind endpoint (default: tcp://*:5555)")
    parser.add_argument("-n", "--count", type=int, default=100,
                        help="Number of packets to send (default: 100)")
    parser.add_argument("-d", "--delay", type=float, default=0.01,
                        help="Delay between packets in seconds (default: 0.01)")
    parser.add_argument("--lat", type=float, default=37.7749,
                        help="Starting latitude (default: 37.7749 / San Francisco)")
    parser.add_argument("--lon", type=float, default=-122.4194,
                        help="Starting longitude (default: -122.4194)")
    parser.add_argument("--alt", type=float, default=10.0,
                        help="Altitude in meters (default: 10.0)")
    parser.add_argument("--no-gps", action="store_true",
                        help="Send single-frame messages (no GPS)")
    parser.add_argument("--mix", action="store_true",
                        help="Alternate between GPS and no-GPS packets")
    args = parser.parse_args()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.endpoint)
    print(f"Publishing on {args.endpoint}", file=sys.stderr)
    print(f"Sending {args.count} packets with {args.delay}s delay", file=sys.stderr)

    # Give subscribers time to connect
    time.sleep(1.0)

    ble_pkt = make_fake_ble_adv()
    sent = 0

    for i in range(args.count):
        now = time.time()
        ts_sec = int(now)
        ts_usec = int((now - ts_sec) * 1e6)

        # BLE RF Info header
        rf_channel = 38  # advertising channel
        le_hdr = BLE_RF_HDR.pack(
            rf_channel, -60, -90, 0, 0,
            LE_DEWHITENED | LE_SIGNAL_POWER_VALID | LE_NOISE_POWER_VALID
        )

        # PCAP record header
        payload_len = len(le_hdr) + len(ble_pkt)
        rec_hdr = PCAP_REC_HDR.pack(ts_sec, ts_usec, payload_len, payload_len)

        pcap_record = rec_hdr + le_hdr + ble_pkt

        # Simulate movement (drift lat slightly each packet)
        lat = args.lat + i * 0.00001
        lon = args.lon + i * 0.000005

        send_gps = not args.no_gps
        if args.mix:
            send_gps = (i % 2 == 0)

        if send_gps:
            gps_frame = ZMQ_GPS_FRAME.pack(lat, lon, args.alt)
            pub.send(gps_frame, zmq.SNDMORE)
            pub.send(pcap_record)
        else:
            pub.send(pcap_record)

        sent += 1
        if sent % 10 == 0:
            gps_str = f" GPS={lat:.5f},{lon:.5f}" if send_gps else " (no GPS)"
            print(f"Sent {sent}/{args.count}{gps_str}", file=sys.stderr)

        time.sleep(args.delay)

    print(f"\nDone: sent {sent} packets", file=sys.stderr)
    pub.close()
    ctx.term()


if __name__ == "__main__":
    main()
