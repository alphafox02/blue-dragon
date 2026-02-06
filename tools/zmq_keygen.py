#!/usr/bin/env python3
# Copyright 2025-2026 CEMAXECUTER LLC
"""
Generate CurveZMQ keypair for blue-dragon encrypted streaming.

Creates a server key file (for the sniffer) and prints the server public
key that subscribers need.

Usage:
    python3 zmq_keygen.py server.key

    # The server public key is printed to stdout and also saved in
    # server.key.pub for easy distribution to subscribers.

Requirements:
    pip install pyzmq
"""

import argparse
import os
import sys

try:
    import zmq
except ImportError:
    print("pyzmq is required: pip install pyzmq", file=sys.stderr)
    sys.exit(1)

if not zmq.has("curve"):
    print("ZMQ was built without CURVE support (needs libsodium)", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CurveZMQ keypair for encrypted packet streaming")
    parser.add_argument("keyfile",
                        help="Output server key file (contains secret+public)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite existing key file")
    args = parser.parse_args()

    if os.path.exists(args.keyfile) and not args.force:
        print(f"Key file {args.keyfile} already exists. Use -f to overwrite.",
              file=sys.stderr)
        sys.exit(1)

    public_key, secret_key = zmq.curve_keypair()

    # Write server key file (secret + public, Z85-encoded)
    with open(args.keyfile, "w") as f:
        f.write(f"# CurveZMQ server keypair for blue-dragon\n")
        f.write(f"# Keep this file secret!\n")
        f.write(f"secret_key={secret_key.decode()}\n")
        f.write(f"public_key={public_key.decode()}\n")
    os.chmod(args.keyfile, 0o600)

    # Write public key file for distribution to subscribers
    pubfile = args.keyfile + ".pub"
    with open(pubfile, "w") as f:
        f.write(f"# CurveZMQ server public key - distribute to subscribers\n")
        f.write(f"server_public_key={public_key.decode()}\n")
    os.chmod(pubfile, 0o644)

    print(f"Server key written to: {args.keyfile} (keep secret!)")
    print(f"Public key written to: {pubfile} (give to subscribers)")
    print(f"Server public key: {public_key.decode()}")


if __name__ == "__main__":
    main()
