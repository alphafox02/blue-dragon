#!/usr/bin/env python3
# Copyright 2025-2026 CEMAXECUTER LLC
# Copyright 2025-2026 CEMAXECUTER LLC
"""
ZMQ subscriber test harness for blue-dragon.

Binds a SUB socket and receives BLE/BT packets as PCAP records from
connecting sensors. Can display live packet info, write to a PCAP file,
or both.

Usage:
    # Display live packets (sensors connect to us on port 5555):
    python3 zmq_subscriber.py tcp://*:5555

    # Write to PCAP file:
    python3 zmq_subscriber.py tcp://*:5555 -w output.pcap

    # Quiet mode (only write PCAP, no terminal output):
    python3 zmq_subscriber.py tcp://*:5555 -w output.pcap -q

    # Encrypted connection (CURVE):
    python3 zmq_subscriber.py tcp://*:5555 --server-key server.key

    # Write PCAP with GPS tagging (PPI headers, DLT 192):
    python3 zmq_subscriber.py tcp://*:5555 -w output.pcap --gps

Requirements:
    pip install pyzmq
"""

import argparse
import signal
import struct
import sys
import time

try:
    import zmq
except ImportError:
    print("pyzmq is required: pip install pyzmq", file=sys.stderr)
    sys.exit(1)

# PCAP structures (matching pcap.c)
PCAP_GLOBAL_HDR = struct.Struct("<IHHiIII")
PCAP_REC_HDR = struct.Struct("<IIII")      # ts_sec, ts_usec, incl_len, orig_len
BLE_RF_HDR = struct.Struct("<bbbBIH")      # rf_channel, signal_power, noise_power,
                                            # aa_offenses, ref_aa, flags

# ZMQ GPS frame (matches zmq_gps_frame_t in pcap.h)
ZMQ_GPS_FRAME = struct.Struct("<ddd")      # latitude, longitude, altitude

# BLE RF Info flags
LE_DEWHITENED         = 0x0001
LE_SIGNAL_POWER_VALID = 0x0002
LE_NOISE_POWER_VALID  = 0x0004
LE_CRC_CHECKED        = 0x0400
LE_CRC_VALID          = 0x0800

# BLE advertising channel access address
BLE_ADV_AA = 0x8E89BED6

DLT_BLUETOOTH_LE_LL_WITH_PHDR = 256
DLT_PPI = 192

# PPI header structures
PPI_HDR = struct.Struct("<BBHI")           # version, flags, len, dlt
PPI_FIELD_HDR = struct.Struct("<HH")       # type, datalen
PPI_GPS = struct.Struct("<BBHIIIII")        # ver, pad, len, present, gps_flags,
                                            # lat, lon, alt (unsigned offset encoding)

PPI_FIELD_GPS = 30002
PPI_GPS_FLAG_GPSFLAGS = 0x00000001  # bit 0: GPSFlags field present
PPI_GPS_FLAG_LAT      = 0x00000002  # bit 1: latitude
PPI_GPS_FLAG_LON      = 0x00000004  # bit 2: longitude
PPI_GPS_FLAG_ALT      = 0x00000008  # bit 3: altitude

PPI_HDR_SIZE = PPI_HDR.size                                    # 8 bytes
PPI_GPS_SIZE = PPI_HDR.size + PPI_FIELD_HDR.size + PPI_GPS.size  # 8 + 4 + 24 = 36 bytes

running = True


def signal_handler(sig, frame):
    global running
    running = False


def channel_to_freq(ch):
    """Convert BLE RF channel number to frequency in MHz."""
    return 2402 + ch * 2


def ppi_fixed3_7(val):
    """Convert float to PPI fixed3_7 format (unsigned with +180 offset)."""
    return int((val + 180.0) * 1e7)


def ppi_fixed6_4(val):
    """Convert float to PPI fixed6_4 format (unsigned with +180000m offset)."""
    return int((val + 180000.0) * 1e4)


def build_ppi_gps_header(lat, lon, alt):
    """Build a PPI header with GPS field."""
    ppi_hdr = PPI_HDR.pack(0, 0, PPI_GPS_SIZE, DLT_BLUETOOTH_LE_LL_WITH_PHDR)
    fld_hdr = PPI_FIELD_HDR.pack(PPI_FIELD_GPS, PPI_GPS.size)
    gps_data = PPI_GPS.pack(
        2, 0, PPI_GPS.size,
        PPI_GPS_FLAG_GPSFLAGS | PPI_GPS_FLAG_LAT | PPI_GPS_FLAG_LON | PPI_GPS_FLAG_ALT,
        0,
        ppi_fixed3_7(lat),
        ppi_fixed3_7(lon),
        ppi_fixed6_4(alt),
    )
    return ppi_hdr + fld_hdr + gps_data


def build_ppi_passthrough_header():
    """Build a minimal PPI header with no fields (passthrough)."""
    return PPI_HDR.pack(0, 0, PPI_HDR_SIZE, DLT_BLUETOOTH_LE_LL_WITH_PHDR)


def parse_ble_packet(data):
    """Parse a ZMQ message containing a PCAP record."""
    if len(data) < PCAP_REC_HDR.size + BLE_RF_HDR.size:
        return None

    # Parse PCAP record header
    ts_sec, ts_usec, incl_len, orig_len = PCAP_REC_HDR.unpack_from(data, 0)
    offset = PCAP_REC_HDR.size

    # Parse BLE RF Info header
    rf_channel, signal_power, noise_power, aa_offenses, ref_aa, flags = \
        BLE_RF_HDR.unpack_from(data, offset)
    offset += BLE_RF_HDR.size

    # Remaining bytes are the BLE packet (AA + PDU + CRC)
    ble_data = data[offset:]

    # Extract access address (first 4 bytes, little-endian)
    aa = struct.unpack_from("<I", ble_data, 0)[0] if len(ble_data) >= 4 else 0

    crc_checked = bool(flags & LE_CRC_CHECKED)
    crc_valid = bool(flags & LE_CRC_VALID) if crc_checked else None

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
        "data_len": len(ble_data),
        "ble_data": ble_data,
    }


def write_pcap_header(f, use_ppi=False):
    """Write PCAP global header."""
    dlt = DLT_PPI if use_ppi else DLT_BLUETOOTH_LE_LL_WITH_PHDR
    snaplen = 4 + 2 + 255 + 3
    if use_ppi:
        snaplen += PPI_GPS_SIZE
    f.write(PCAP_GLOBAL_HDR.pack(
        0xA1B2C3D4,  # magic
        2, 4,         # version
        0,            # timezone
        0,            # sigfigs
        snaplen,
        dlt,
    ))
    f.flush()


def write_ppi_pcap_record(f, pcap_data, gps_info):
    """Write a PPI-wrapped PCAP record to file."""
    # Parse the original PCAP record header to get timestamps
    ts_sec, ts_usec, incl_len, orig_len = PCAP_REC_HDR.unpack_from(pcap_data, 0)
    payload = pcap_data[PCAP_REC_HDR.size:]

    if gps_info:
        ppi_hdr = build_ppi_gps_header(gps_info[0], gps_info[1], gps_info[2])
    else:
        ppi_hdr = build_ppi_passthrough_header()

    new_len = len(ppi_hdr) + len(payload)
    new_rec_hdr = PCAP_REC_HDR.pack(ts_sec, ts_usec, new_len, new_len)
    f.write(new_rec_hdr)
    f.write(ppi_hdr)
    f.write(payload)
    f.flush()


def parse_server_pubkey(path):
    """Read server public key from a .pub key file."""
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


def main():
    parser = argparse.ArgumentParser(
        description="ZMQ subscriber for blue-dragon")
    parser.add_argument("endpoints", nargs="+",
                        help="ZMQ endpoints (e.g. tcp://localhost:5555)")
    parser.add_argument("-w", "--write", metavar="FILE",
                        help="Write received packets to PCAP file")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress terminal output")
    parser.add_argument("-t", "--timeout", type=int, default=0,
                        help="Exit after N seconds (0 = run forever)")
    parser.add_argument("--server-key", metavar="FILE",
                        help="Server public key file for CURVE encryption")
    parser.add_argument("--gps", action="store_true",
                        help="Write PCAP with PPI GPS headers (DLT 192)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all messages
    sub.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout for clean shutdown

    if args.server_key:
        server_public_key = parse_server_pubkey(args.server_key)
        client_public, client_secret = zmq.curve_keypair()
        sub.setsockopt(zmq.CURVE_SERVERKEY, server_public_key)
        sub.setsockopt(zmq.CURVE_PUBLICKEY, client_public)
        sub.setsockopt(zmq.CURVE_SECRETKEY, client_secret)
        if not args.quiet:
            print(f"CURVE encryption enabled", file=sys.stderr)

    for endpoint in args.endpoints:
        sub.bind(endpoint)
        if not args.quiet:
            print(f"Listening on {endpoint}", file=sys.stderr)

    pcap_file = None
    if args.write:
        pcap_file = open(args.write, "wb")
        write_pcap_header(pcap_file, use_ppi=args.gps)
        if not args.quiet:
            dlt_name = "DLT 192 (PPI+GPS)" if args.gps else "DLT 256 (BLE)"
            print(f"Writing PCAP to {args.write} ({dlt_name})", file=sys.stderr)

    pkt_count = 0
    gps_pkt_count = 0
    crc_valid_count = 0
    crc_invalid_count = 0
    start_time = time.time()

    if not args.quiet:
        hdr = f"{'Time':>10s}  {'Freq':>4s}  {'RSSI':>5s}  {'AA':>10s}  {'Len':>4s}  {'CRC':>5s}"
        if args.gps:
            hdr += f"  {'GPS':>20s}"
        print(hdr, file=sys.stderr)
        print("-" * (50 + (22 if args.gps else 0)), file=sys.stderr)

    while running:
        if args.timeout and (time.time() - start_time) > args.timeout:
            break

        try:
            frames = sub.recv_multipart()
        except zmq.Again:
            continue

        # Parse multipart: 1 frame = no GPS, 2 frames = GPS + PCAP record
        gps_info = None
        if len(frames) >= 2 and len(frames[0]) == ZMQ_GPS_FRAME.size:
            lat, lon, alt = ZMQ_GPS_FRAME.unpack(frames[0])
            gps_info = (lat, lon, alt)
            pcap_data = frames[1]
            gps_pkt_count += 1
        else:
            pcap_data = frames[-1]

        # Strip type prefix byte if present (0x00=BLE, 0x01=BT)
        if len(pcap_data) > PCAP_REC_HDR.size and pcap_data[0] <= 0x01:
            pcap_data = pcap_data[1:]

        # Write to PCAP file
        if pcap_file:
            if args.gps:
                write_ppi_pcap_record(pcap_file, pcap_data, gps_info)
            else:
                pcap_file.write(pcap_data)
                pcap_file.flush()

        pkt = parse_ble_packet(pcap_data)
        if pkt is None:
            continue

        pkt_count += 1
        if pkt["crc_checked"]:
            if pkt["crc_valid"]:
                crc_valid_count += 1
            else:
                crc_invalid_count += 1

        if not args.quiet:
            elapsed = time.time() - start_time
            crc_str = "  -  "
            if pkt["crc_checked"]:
                crc_str = " OK  " if pkt["crc_valid"] else " BAD "
            is_adv = " ADV" if pkt["aa"] == BLE_ADV_AA else "DATA"
            line = (f"{elapsed:10.3f}  {pkt['freq_mhz']:4d}  "
                    f"{pkt['signal_power']:5d}  "
                    f"{pkt['aa']:08x}{is_adv}  "
                    f"{pkt['data_len']:4d}  {crc_str}")
            if args.gps and gps_info:
                line += f"  {gps_info[0]:9.5f},{gps_info[1]:10.5f}"
            elif args.gps:
                line += f"  {'no fix':>20s}"
            print(line)

    # Summary
    elapsed = time.time() - start_time
    if not args.quiet:
        print(f"\n--- {pkt_count} packets in {elapsed:.1f}s "
              f"({pkt_count/max(elapsed,0.001):.1f} pkt/s) ---",
              file=sys.stderr)
        if crc_valid_count + crc_invalid_count > 0:
            total = crc_valid_count + crc_invalid_count
            pct = 100.0 * crc_valid_count / total
            print(f"CRC: {crc_valid_count} valid, {crc_invalid_count} invalid "
                  f"({pct:.1f}% valid)", file=sys.stderr)
        if args.gps:
            print(f"GPS: {gps_pkt_count} packets with coordinates", file=sys.stderr)

    if pcap_file:
        pcap_file.close()
        if not args.quiet:
            print(f"PCAP written to {args.write}", file=sys.stderr)

    sub.close()
    ctx.term()


if __name__ == "__main__":
    main()
