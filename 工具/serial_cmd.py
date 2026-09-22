#!/usr/bin/env python3
import argparse
import fcntl
import re
import sys
import time

import serial


PROMPT_RE = re.compile(rb"(?:\r?\n)?(?:\d+\|)?console:[^\r\n]* \$ ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--port", default="/dev/cu.usbserial-A10LCBPJ")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    with open("/tmp/codex_qcs8550_serial.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_command(args)


def run_command(args):
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        ser.reset_input_buffer()
        ser.write(b"\r\n")
        time.sleep(0.2)
        ser.read(4096)

        ser.write(args.command.encode("utf-8") + b"\r\n")
        deadline = time.time() + args.timeout
        data = b""
        while time.time() < deadline:
            chunk = ser.read(4096)
            if chunk:
                data += chunk
                if PROMPT_RE.search(data):
                    break
            else:
                time.sleep(0.05)

        text = data.decode("utf-8", "replace")
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
