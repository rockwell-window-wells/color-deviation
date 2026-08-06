"""
Take Lab color measurements using the calibrated TCS3430 sensor node.

Requires calibration.json (produced by calibrate.py) in the same folder.
"""

import json
import numpy as np

from sensor_link import TCS3430Link, find_pico_port
from color_utils import xyz_to_lab

DEFAULT_PORT = "COM5"  # fallback if auto-detect finds nothing


def get_port():
    auto_port = find_pico_port()
    if auto_port:
        print(f"Found Pico on {auto_port}, connecting...")
        return auto_port
    print("Couldn't auto-detect a Pico — check it's plugged in and not held open by Thonny.")
    return input(f"Serial port [{DEFAULT_PORT}]: ").strip() or DEFAULT_PORT


def load_calibration(path="calibration.json"):
    with open(path) as f:
        cal = json.load(f)
    return np.array(cal["matrix"])


def apply_calibration(x, y, z, M):
    raw_aug = np.array([x, y, z, 1.0])
    X, Y, Z = raw_aug @ M
    return X, Y, Z


def main():
    try:
        M = load_calibration()
    except FileNotFoundError:
        print("calibration.json not found — run calibrate.py first.")
        return

    port = get_port()
    link = TCS3430Link(port)

    print("Press Enter to take a measurement, or type 'q' to quit.")
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break

        try:
            x, y, z, ir1 = link.read_sample()
        except (TimeoutError, RuntimeError, ValueError) as e:
            print(f"  Reading failed: {e}")
            continue

        X, Y, Z = apply_calibration(x, y, z, M)
        L, a, b = xyz_to_lab(X, Y, Z)

        print(f"  Raw: X={x:.1f}  Y={y:.1f}  Z={z:.1f}  IR1={ir1:.1f}")
        print(f"  Lab: L*={L:.2f}  a*={a:.2f}  b*={b:.2f}\n")

    link.close()


if __name__ == "__main__":
    main()
