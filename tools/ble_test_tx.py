#!/usr/bin/env python3
# Copyright 2025-2026 CEMAXECUTER LLC
"""
BLE PHY-layer test transmitter using WHAD (Butterfly firmware).

Transmits known BLE advertising packets for receiver sensitivity testing.
Cable the Butterfly dongle SMA to the bladeRF/USRP RX port for conducted
measurements.

Requires: WHAD client installed in a venv
    cd /home/dragon/Downloads/whad-client
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .

Usage:
    # Activate the WHAD venv first:
    source /home/dragon/Downloads/whad-client/venv/bin/activate

    # Transmit on channel 38 (2426 MHz) with default name:
    python3 tools/ble_test_tx.py uart0

    # Custom device name and TX power:
    python3 tools/ble_test_tx.py uart0 --name "BD-TEST-01" --tx-power low

    # Available TX power levels: low, medium (default), high

    # Run blue-dragon on the receiver side:
    blue-dragon -l -i bladerf0 -c 2426 -C 4 --check-crc --stats
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="BLE test transmitter for blue-dragon sensitivity testing"
    )
    parser.add_argument("interface", help="WHAD interface (e.g. uart0)")
    parser.add_argument(
        "--name", default="BD-TEST-01",
        help="BLE device name in advertisements (default: BD-TEST-01)"
    )
    parser.add_argument(
        "--tx-power", choices=["low", "medium", "high"], default="medium",
        help="TX power level (default: medium)"
    )
    args = parser.parse_args()

    try:
        from whad.ble import Peripheral
        from whad.ble.profile.advdata import (
            AdvCompleteLocalName,
            AdvDataFieldList,
            AdvFlagsField,
        )
        from whad.ble.profile import PrimaryService, Characteristic, GenericProfile
        from whad.ble.profile.attribute import UUID
        from whad.device import WhadDevice
        from whad.phy import TxPower
    except ImportError:
        print("Error: WHAD not installed. Activate the venv first:", file=sys.stderr)
        print("  source /home/dragon/Downloads/whad-client/venv/bin/activate", file=sys.stderr)
        sys.exit(1)

    tx_power_map = {
        "low": TxPower.LOW,
        "medium": TxPower.MEDIUM,
        "high": TxPower.HIGH,
    }

    # Simple GATT profile (required for peripheral mode)
    class TestProfile(GenericProfile):
        device = PrimaryService(
            uuid=UUID(0x1800),
            device_name=Characteristic(
                uuid=UUID(0x2A00),
                permissions=["read"],
                value=args.name.encode(),
            ),
        )

    print(f"Opening WHAD device: {args.interface}", file=sys.stderr)
    dev = WhadDevice.create(args.interface)

    profile = TestProfile()
    periph = Peripheral(dev, profile=profile)

    # Set TX power via PHY layer if supported
    tx_level = tx_power_map[args.tx_power]
    try:
        if hasattr(periph, 'set_tx_power'):
            periph.set_tx_power(tx_level)
            print(f"TX power: {args.tx_power}", file=sys.stderr)
        else:
            print(f"TX power control not available via BLE connector, using default",
                  file=sys.stderr)
    except Exception as e:
        print(f"TX power set failed (using default): {e}", file=sys.stderr)

    # Build advertising data
    adv_data = AdvDataFieldList(
        AdvFlagsField(),
        AdvCompleteLocalName(args.name.encode()),
    )

    print(f"Starting BLE peripheral: name='{args.name}'", file=sys.stderr)
    print(f"Advertising on channels 37/38/39 (2402/2426/2480 MHz)", file=sys.stderr)
    print(f"Press Ctrl-C to stop", file=sys.stderr)

    try:
        periph.enable_peripheral_mode(adv_data=adv_data)

        # Keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    finally:
        periph.stop()
        periph.close()


if __name__ == "__main__":
    main()
