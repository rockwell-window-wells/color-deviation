"""
Test measurement repeatability of the calibrated TCS3430 sensor against a
known reference (e.g. your colorimeter's L*a*b* reading for the same
target).

Takes N repeated readings — leave the target in place between readings to
test pure repeatability (sensor/electrical noise only), or lift and
reposition it each time to test position-to-position reproducibility
instead. Reports:
  - Mean, std dev, and range for L*, a*, b* across trials
  - dE* of each trial relative to the trial mean (repeatability/spread)
  - dE* of the sensor's mean reading vs. the reference value (accuracy)

Saves raw trial data to repeatability_results.json and a plot to
repeatability_report.png.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from sensor_link import TCS3430Link, find_pico_port
from color_utils import xyz_to_lab
from measure import load_calibration, apply_calibration

DEFAULT_PORT = "COM5"


def get_port():
    auto_port = find_pico_port()
    if auto_port:
        print(f"Found Pico on {auto_port}, connecting...")
        return auto_port
    print("Couldn't auto-detect a Pico — check it's plugged in and not held open by Thonny.")
    return input(f"Serial port [{DEFAULT_PORT}]: ").strip() or DEFAULT_PORT


def delta_e(lab1, lab2):
    return float(np.linalg.norm(np.array(lab1) - np.array(lab2)))


def main():
    try:
        M = load_calibration()
    except FileNotFoundError:
        print("calibration.json not found — run calibrate.py first.")
        return

    name = input("Target name: ").strip() or "target"

    ref_lab = None
    ref_input = input("Reference L*,a*,b* from colorimeter (comma-separated, or blank to skip): ").strip()
    if ref_input:
        try:
            ref_lab = [float(v) for v in ref_input.split(",")]
            if len(ref_lab) != 3:
                raise ValueError
        except ValueError:
            print("Couldn't parse that as three comma-separated numbers — continuing without a reference.")
            ref_lab = None

    try:
        n_trials = int(input("Number of trials [10]: ").strip() or "10")
    except ValueError:
        n_trials = 10

    port = get_port()
    link = TCS3430Link(port)

    print(f"\nTaking {n_trials} readings of '{name}'.")
    print("Leave the target in place for pure repeatability, or lift/reposition")
    print("it between readings to test position reproducibility instead.\n")

    labs = []
    for i in range(n_trials):
        input(f"  Trial {i + 1}/{n_trials} — press Enter to read...")
        try:
            x, y, z, ir1 = link.read_sample()
        except (TimeoutError, RuntimeError, ValueError) as e:
            print(f"  Reading failed: {e}. Skipping this trial.")
            continue
        X, Y, Z = apply_calibration(x, y, z, M)
        L, a, b = xyz_to_lab(X, Y, Z)
        labs.append([L, a, b])
        print(f"    L*={L:.2f}  a*={a:.2f}  b*={b:.2f}")

    link.close()

    if len(labs) < 2:
        print("Need at least 2 successful readings to assess repeatability.")
        return

    labs = np.array(labs)
    mean_lab = labs.mean(axis=0)
    std_lab = labs.std(axis=0, ddof=1)
    range_lab = labs.max(axis=0) - labs.min(axis=0)
    spread_dE = np.array([delta_e(row, mean_lab) for row in labs])

    print(f"\n--- Repeatability report: {name} ({len(labs)} trials) ---")
    print(f"{'':8s}{'L*':>10s}{'a*':>10s}{'b*':>10s}")
    print(f"{'mean':8s}{mean_lab[0]:10.2f}{mean_lab[1]:10.2f}{mean_lab[2]:10.2f}")
    print(f"{'std dev':8s}{std_lab[0]:10.2f}{std_lab[1]:10.2f}{std_lab[2]:10.2f}")
    print(f"{'range':8s}{range_lab[0]:10.2f}{range_lab[1]:10.2f}{range_lab[2]:10.2f}")
    print(f"\nMean per-trial dE* from trial mean (repeatability spread): {spread_dE.mean():.2f}")
    print(f"Max per-trial dE* from trial mean: {spread_dE.max():.2f}")

    accuracy_dE = None
    if ref_lab is not None:
        accuracy_dE = delta_e(mean_lab, ref_lab)
        print(f"\nReference L*a*b*: {ref_lab[0]:.2f}, {ref_lab[1]:.2f}, {ref_lab[2]:.2f}")
        print(f"dE* of sensor mean vs. reference (accuracy): {accuracy_dE:.2f}")

    results = {
        "name": name,
        "reference_lab": ref_lab,
        "trials": labs.tolist(),
        "mean_lab": mean_lab.tolist(),
        "std_lab": std_lab.tolist(),
        "range_lab": range_lab.tolist(),
        "mean_spread_dE": float(spread_dE.mean()),
        "max_spread_dE": float(spread_dE.max()),
        "accuracy_dE": accuracy_dE,
    }
    with open("repeatability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved repeatability_results.json")

    # --- Plot: L* stability over trials, and a*-b* scatter of trials ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Repeatability: {name} ({len(labs)} trials)")

    ax_l = axes[0]
    ax_l.plot(range(1, len(labs) + 1), labs[:, 0], "o-", label="L*")
    ax_l.axhline(mean_lab[0], color="gray", linestyle="--", linewidth=1, label="mean L*")
    if ref_lab is not None:
        ax_l.axhline(ref_lab[0], color="red", linestyle=":", linewidth=1, label="reference L*")
    ax_l.set_xlabel("Trial")
    ax_l.set_ylabel("L*")
    ax_l.set_title("L* across trials")
    ax_l.legend(fontsize=8)

    ax_ab = axes[1]
    ax_ab.scatter(labs[:, 1], labs[:, 2], color="tab:blue", label="trials")
    ax_ab.scatter(mean_lab[1], mean_lab[2], color="black", marker="s", s=60, label="sensor mean")
    if ref_lab is not None:
        ax_ab.scatter(ref_lab[1], ref_lab[2], color="red", marker="*", s=120, label="reference")
    ax_ab.axhline(0, color="lightgray", linewidth=0.8)
    ax_ab.axvline(0, color="lightgray", linewidth=0.8)
    ax_ab.set_xlabel("a*")
    ax_ab.set_ylabel("b*")
    ax_ab.set_title("a*-b* scatter of trials")
    ax_ab.legend(fontsize=8)
    ax_ab.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig("repeatability_report.png", dpi=150)
    print("Saved repeatability_report.png")

    try:
        plt.show()
    except Exception:
        pass  # headless environment — the saved PNG is still there


if __name__ == "__main__":
    main()
