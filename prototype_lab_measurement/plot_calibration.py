"""
Visualize the calibration data and fit quality from calibration.json
(produced by calibrate.py).

Produces:
  1. Parity plots (reference vs. predicted) for L*, a*, b* — points on the
     diagonal line mean a perfect fit; scatter away from it shows error.
  2. A bar chart of per-target dE* (color difference) — the standard
     single-number measure of how far off each target's fit is.
  3. An a*-b* scatter showing where each reference target sits in Lab's
     color plane, with a line to where the fit predicts it — useful for
     seeing whether errors cluster in a particular hue/color region.

Saves everything to calibration_report.png and also opens an interactive
window if your environment supports it.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    try:
        with open("calibration.json") as f:
            cal = json.load(f)
    except FileNotFoundError:
        print("calibration.json not found — run calibrate.py first.")
        sys.exit(1)

    targets = cal.get("targets")
    if not targets:
        print("calibration.json has no per-target data to plot.")
        print("Re-run calibrate.py (the updated version) to regenerate it with target details.")
        sys.exit(1)

    names = [t["name"] for t in targets]
    ref = np.array([t["reference_lab"] for t in targets])   # (N, 3)
    pred = np.array([t["predicted_lab"] for t in targets])  # (N, 3)
    dE = np.array([t["delta_e"] for t in targets])

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Calibration fit report — {len(targets)} targets", fontsize=13)

    # --- Parity plots: reference vs predicted, for each Lab channel ---
    channel_labels = ["L*", "a*", "b*"]
    for i, label in enumerate(channel_labels):
        ax = fig.add_subplot(2, 3, i + 1)
        lo = min(ref[:, i].min(), pred[:, i].min())
        hi = max(ref[:, i].max(), pred[:, i].max())
        pad = (hi - lo) * 0.1 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", linewidth=1, label="perfect fit")
        ax.scatter(ref[:, i], pred[:, i], color="tab:blue")
        for name, x, y in zip(names, ref[:, i], pred[:, i]):
            ax.annotate(name, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"Reference {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label} parity")
        ax.legend(fontsize=8)

    # --- Per-target dE* bar chart ---
    ax_de = fig.add_subplot(2, 3, 4)
    order = np.argsort(dE)[::-1]
    ax_de.barh([names[i] for i in order], dE[order], color="tab:orange")
    ax_de.set_xlabel("dE* (reference vs. predicted)")
    ax_de.set_title(f"Per-target error — mean dE*={dE.mean():.2f}, max={dE.max():.2f}")
    ax_de.axvline(1.0, color="green", linestyle=":", linewidth=1, label="dE*=1 (~imperceptible)")
    ax_de.axvline(3.0, color="red", linestyle=":", linewidth=1, label="dE*=3 (visible)")
    ax_de.legend(fontsize=7)

    # --- a*-b* plane: reference position -> predicted position ---
    ax_ab = fig.add_subplot(2, 3, (5, 6))
    ax_ab.scatter(ref[:, 1], ref[:, 2], color="black", marker="o", label="reference")
    ax_ab.scatter(pred[:, 1], pred[:, 2], color="tab:red", marker="x", label="predicted")
    for i, name in enumerate(names):
        ax_ab.plot([ref[i, 1], pred[i, 1]], [ref[i, 2], pred[i, 2]], color="gray", linewidth=0.8)
        ax_ab.annotate(name, (ref[i, 1], ref[i, 2]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax_ab.axhline(0, color="lightgray", linewidth=0.8)
    ax_ab.axvline(0, color="lightgray", linewidth=0.8)
    ax_ab.set_xlabel("a*")
    ax_ab.set_ylabel("b*")
    ax_ab.set_title("a*-b* plane: reference (o) vs. predicted (x)")
    ax_ab.legend(fontsize=8)
    ax_ab.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig("calibration_report.png", dpi=150)
    print("Saved calibration_report.png")
    print(f"Mean dE*: {dE.mean():.2f}   Max dE*: {dE.max():.2f}   Worst target: {names[int(np.argmax(dE))]}")

    try:
        plt.show()
    except Exception:
        pass  # headless environment — the saved PNG is still there


if __name__ == "__main__":
    main()
