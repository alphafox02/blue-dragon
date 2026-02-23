#!/usr/bin/env python3
# Copyright 2025-2026 CEMAXECUTER LLC
"""
PHY-layer sensitivity test bench for Blue Dragon.

Uses a WHAD-compatible device (Butterfly nRF52840) as a controlled BLE
transmitter and sweeps the receiver (bladeRF/USRP) gain to produce a
sensitivity curve: CRC pass rate vs. RX gain at each TX power level.

Requirements:
    - WHAD client installed in a venv at ~/Downloads/whad-client/venv
    - Butterfly dongle connected (detected as uart0)
    - bladeRF or USRP connected
    - SMA cable between Butterfly TX and SDR RX (conducted test)
    - Blue Dragon built with --features "bladerf" (or usrp, etc.)

Usage:
    source ~/Downloads/whad-client/venv/bin/activate
    python3 tools/phy_test_bench.py

    # Custom settings:
    python3 tools/phy_test_bench.py --sdr bladerf0 --duration 15 \\
        --gains "60,50,40,30,20,10,5,0"

Output:
    Prints a results table and writes CSV to phy_test_results.csv
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import threading


# Path to blue-dragon binary
BD_BINARY = os.path.join(os.path.dirname(__file__), "..", "target", "release", "blue-dragon")
BD_BINARY = os.path.normpath(BD_BINARY)


def start_butterfly_tx(interface, name, tx_power):
    """Start the Butterfly as a BLE peripheral (advertising).
    Returns the subprocess handle."""
    script = os.path.join(os.path.dirname(__file__), "ble_test_tx.py")
    cmd = [
        sys.executable, script, interface,
        "--name", name,
        "--tx-power", tx_power,
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Give it time to start advertising
    time.sleep(3)
    if proc.poll() is not None:
        err = proc.stderr.read().decode()
        raise RuntimeError(f"Butterfly TX failed to start: {err}")
    return proc


def run_capture(sdr_interface, gain, duration, center_freq=2426, channels=4):
    """Run Blue Dragon for `duration` seconds and return parsed stats.
    Returns dict with keys: ble, bt, bursts, crc_pct, valid_crc, total_crc, overflow"""
    cmd = [
        BD_BINARY,
        "-l",
        "-i", sdr_interface,
        "-c", str(center_freq),
        "-C", str(channels),
        "--gain", str(gain),
        "--check-crc",
        "--stats",
    ]

    # Capture stderr for stats
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Let it run for the specified duration
    time.sleep(duration)

    # Send SIGINT for clean shutdown with final stats
    proc.send_signal(signal.SIGINT)

    try:
        _, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr = proc.communicate()

    stderr_text = stderr.decode(errors="replace")

    # Parse the final "done" line
    result = {
        "ble": 0, "bt": 0, "bursts": 0,
        "crc_pct": 0.0, "valid_crc": 0, "total_crc": 0,
        "overflow": 0, "raw_output": stderr_text,
    }

    # Try to find the final stats line
    # done (15.0s): BLE: 450 BT: 0 bursts: 1200 CRC: 97.3% (438/450) overflow: 0
    done_pattern = re.compile(
        r"done.*BLE:\s*(\d+)\s+BT:\s*(\d+)\s+bursts:\s*(\d+)\s+"
        r"CRC:\s*([\d.]+)%\s+\((\d+)/(\d+)\).*?overflow:\s*(\d+)"
    )

    # Also try periodic stats (last one before shutdown)
    periodic_pattern = re.compile(
        r"\[[\d.]+s\]\s+BLE:\s*(\d+)\s+BT:\s*(\d+)\s+bursts:\s*(\d+)\s+"
        r"CRC:\s*([\d.]+)%\s+\((\d+)/(\d+)\).*?overflow:\s*(\d+)"
    )

    # Prefer the "done" line, fall back to last periodic
    match = None
    for line in stderr_text.splitlines():
        m = done_pattern.search(line)
        if m:
            match = m
            break

    if match is None:
        # Fall back to last periodic stats line
        for line in stderr_text.splitlines():
            m = periodic_pattern.search(line)
            if m:
                match = m  # keep last match

    if match:
        result["ble"] = int(match.group(1))
        result["bt"] = int(match.group(2))
        result["bursts"] = int(match.group(3))
        result["crc_pct"] = float(match.group(4))
        result["valid_crc"] = int(match.group(5))
        result["total_crc"] = int(match.group(6))
        result["overflow"] = int(match.group(7))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PHY-layer sensitivity test bench for Blue Dragon"
    )
    parser.add_argument(
        "--whad-iface", default="uart0",
        help="WHAD interface for Butterfly TX (default: uart0)"
    )
    parser.add_argument(
        "--sdr", default="bladerf0",
        help="SDR interface for Blue Dragon RX (default: bladerf0)"
    )
    parser.add_argument(
        "--duration", type=int, default=15,
        help="Capture duration per gain step in seconds (default: 15)"
    )
    parser.add_argument(
        "--gains", default="60,50,40,30,20,10,5,0",
        help="Comma-separated RX gain values to sweep (default: 60,50,40,30,20,10,5,0)"
    )
    parser.add_argument(
        "--tx-powers", default="high,medium,low",
        help="Comma-separated TX power levels to test (default: high,medium,low)"
    )
    parser.add_argument(
        "--center-freq", type=int, default=2426,
        help="Center frequency in MHz (default: 2426, advertising ch 38)"
    )
    parser.add_argument(
        "--channels", type=int, default=4,
        help="Number of channels (default: 4)"
    )
    parser.add_argument(
        "--name", default="BD-TEST-01",
        help="BLE device name for test transmitter (default: BD-TEST-01)"
    )
    parser.add_argument(
        "--output", default="phy_test_results.csv",
        help="Output CSV file (default: phy_test_results.csv)"
    )
    args = parser.parse_args()

    gains = [int(g.strip()) for g in args.gains.split(",")]
    tx_powers = [p.strip() for p in args.tx_powers.split(",")]

    # Verify blue-dragon binary exists
    if not os.path.isfile(BD_BINARY):
        print(f"Error: Blue Dragon binary not found at {BD_BINARY}", file=sys.stderr)
        print("Build with: cargo build --release --features bladerf", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("Blue Dragon PHY-Layer Sensitivity Test Bench")
    print("=" * 72)
    print(f"  SDR:          {args.sdr}")
    print(f"  WHAD TX:      {args.whad_iface}")
    print(f"  Center freq:  {args.center_freq} MHz")
    print(f"  Channels:     {args.channels}")
    print(f"  Duration:     {args.duration}s per step")
    print(f"  RX gains:     {gains}")
    print(f"  TX powers:    {tx_powers}")
    print(f"  Output:       {args.output}")
    print("=" * 72)
    print()

    all_results = []

    for tx_power in tx_powers:
        print(f"--- TX Power: {tx_power} ---")
        print(f"Starting Butterfly TX ({args.whad_iface}, name={args.name})...")

        try:
            tx_proc = start_butterfly_tx(args.whad_iface, args.name, tx_power)
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        print(f"  Butterfly advertising (TX power={tx_power})")
        print()

        # Header
        print(f"  {'Gain':>6s}  {'BLE':>6s}  {'Bursts':>8s}  {'CRC%':>7s}  {'Valid/Total':>12s}  {'OVF':>5s}")
        print(f"  {'----':>6s}  {'---':>6s}  {'------':>8s}  {'----':>7s}  {'-----------':>12s}  {'---':>5s}")

        for gain in gains:
            result = run_capture(
                args.sdr, gain, args.duration,
                center_freq=args.center_freq,
                channels=args.channels,
            )

            print(
                f"  {gain:>4d} dB  {result['ble']:>6d}  {result['bursts']:>8d}  "
                f"{result['crc_pct']:>6.1f}%  "
                f"{result['valid_crc']:>5d}/{result['total_crc']:<5d}  "
                f"{result['overflow']:>5d}"
            )

            all_results.append({
                "tx_power": tx_power,
                "rx_gain": gain,
                **result,
            })

            # Brief pause between runs for SDR to reset
            time.sleep(2)

        print()

        # Stop Butterfly
        tx_proc.send_signal(signal.SIGINT)
        try:
            tx_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tx_proc.kill()
            tx_proc.wait()

        print(f"  Butterfly stopped.")
        print()
        time.sleep(2)

    # Write CSV
    with open(args.output, "w") as f:
        f.write("tx_power,rx_gain_db,ble_pkts,bt_pkts,bursts,crc_pct,valid_crc,total_crc,overflow\n")
        for r in all_results:
            f.write(
                f"{r['tx_power']},{r['rx_gain']},{r['ble']},{r['bt']},"
                f"{r['bursts']},{r['crc_pct']:.1f},{r['valid_crc']},{r['total_crc']},"
                f"{r['overflow']}\n"
            )

    print(f"Results written to {args.output}")

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for tx_power in tx_powers:
        rows = [r for r in all_results if r["tx_power"] == tx_power]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["crc_pct"])
        worst_with_pkts = [r for r in rows if r["ble"] > 0]
        if worst_with_pkts:
            weakest = min(worst_with_pkts, key=lambda r: r["rx_gain"])
        else:
            weakest = None

        print(f"\n  TX Power: {tx_power}")
        print(f"    Best CRC:       {best['crc_pct']:.1f}% at RX gain {best['rx_gain']} dB")
        if weakest:
            print(f"    Lowest gain OK: {weakest['rx_gain']} dB ({weakest['crc_pct']:.1f}% CRC, {weakest['ble']} pkts)")
        else:
            print(f"    No packets received at any gain level")

    print()


if __name__ == "__main__":
    main()
