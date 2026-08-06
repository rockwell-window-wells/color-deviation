"""
Calibration routine for the TCS3430 sensor node.

For each color target:
  1. Place the target under the sensor and press Enter to take a reading.
  2. Type in the L*, a*, b* values read off your current reference instrument.

Repeat for at least 3 targets (6+ recommended for a more robust fit), then
type 'done'. This fits a 3x3 correction matrix (plus bias term) mapping the
sensor's raw X,Y,Z channels to CIE XYZ, and saves it to calibration.json.
"""

import json
import numpy as np

from sensor_link import TCS3430Link, find_pico_port
from color_utils import lab_to_xyz, xyz_to_lab

DEFAULT_PORT = "COM5"  # fallback if auto-detect finds nothing


def get_port():
    auto_port = find_pico_port()
    if auto_port:
        print(f"Found Pico on {auto_port}, connecting...")
        return auto_port
    print("Couldn't auto-detect a Pico — check it's plugged in and not held open by Thonny.")
    return input(f"Serial port [{DEFAULT_PORT}]: ").strip() or DEFAULT_PORT


def main():
    port = get_port()
    link = TCS3430Link(port)

    target_names = []
    raw_readings = []
    ref_xyz = []

    print("\nCalibration routine.")
    print("For each target: place it under the sensor, take a reading, then")
    print("type in the L*, a*, b* values from your reference instrument.")
    print("Type 'done' at the name prompt when finished (need 3+ targets).\n")

    count = 0
    while True:
        name = input(f"Target #{count + 1} name (or 'done'): ").strip()
        if name.lower() == "done":
            break

        input("  Place target, then press Enter to read sensor...")
        try:
            x, y, z, ir1 = link.read_sample()
        except (TimeoutError, RuntimeError, ValueError) as e:
            print(f"  Reading failed: {e}. Try this target again.\n")
            continue

        print(f"  Raw reading: X={x:.1f}  Y={y:.1f}  Z={z:.1f}  IR1={ir1:.1f}")

        L = a = b = None
        while True:
            try:
                L = float(input("  Reference L*: ").strip())
                a = float(input("  Reference a*: ").strip())
                b = float(input("  Reference b*: ").strip())
                break
            except ValueError:
                print("  Please enter numeric values.")

        target_names.append(name)
        raw_readings.append([x, y, z])
        ref_xyz.append(lab_to_xyz(L, a, b))
        count += 1
        print()

    link.close()

    if count < 3:
        print(f"Only {count} target(s) collected — need at least 3 to fit a matrix. Nothing saved.")
        return

    A = np.array(raw_readings)                       # (N, 3) raw X,Y,Z
    B = np.array(ref_xyz)                             # (N, 3) reference XYZ
    A_aug = np.hstack([A, np.ones((A.shape[0], 1))])  # add bias column -> (N, 4)

    M, _residuals, rank, _sv = np.linalg.lstsq(A_aug, B, rcond=None)  # (4, 3)

    if rank < 4 and count < 6:
        print(f"Warning: fit is under-determined (rank {rank} with {count} targets).")
        print("Results may be unreliable — consider adding more calibration targets.\n")

    # quick fit-quality check: report per-target error in Lab space, and keep
    # the full per-target breakdown so it can be visualized later
    predicted_xyz = A_aug @ M
    print("Fit check (reference vs. predicted, in Lab space):")
    target_details = []
    for i in range(count):
        ref_L, ref_a, ref_b = xyz_to_lab(*ref_xyz[i])
        pred_L, pred_a, pred_b = xyz_to_lab(*predicted_xyz[i])
        dE = ((ref_L - pred_L) ** 2 + (ref_a - pred_a) ** 2 + (ref_b - pred_b) ** 2) ** 0.5
        print(f"  {target_names[i]}: dE* = {dE:.2f}")
        target_details.append({
            "name": target_names[i],
            "raw_xyz": raw_readings[i],
            "reference_lab": [ref_L, ref_a, ref_b],
            "predicted_lab": [pred_L, pred_a, pred_b],
            "delta_e": dE,
        })

    calibration = {
        "matrix": M.tolist(),  # apply as: [x, y, z, 1] @ M -> [X, Y, Z]
        "white_point": "D65_2deg",
        "n_targets": count,
        "targets": target_details,
    }

    with open("calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to calibration.json using {count} targets.")


if __name__ == "__main__":
    main()
