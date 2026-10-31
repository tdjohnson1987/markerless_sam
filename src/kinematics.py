"""SAM kinematics helpers; calculations must use independently verified anatomical definitions.

This module deliberately does NOT invent MHR landmark indices or Vicon angles.
Original video timestamps are preserved; synchronization belongs in validation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LEG_COLORS = {"left": "#D94848", "right": "#24985A"}
JOINTS = ("hip", "knee", "ankle")


def static_offset_calibrate(time_ms, vicon, sam, window_ms=1500):
    """Exploratory calibration only; not part of raw SAM angle generation."""
    t = np.asarray(time_ms, dtype=float)
    v = np.asarray(vicon, dtype=float)
    s = np.asarray(sam, dtype=float)
    if not (t.shape == v.shape == s.shape) or t.ndim != 1 or not len(t):
        raise ValueError("time_ms, vicon and sam must be non-empty equal-length vectors")
    static = ((t <= t[0] + window_ms) & np.isfinite(t) & np.isfinite(v) & np.isfinite(s))
    if static.sum() < 3:
        raise ValueError("Not enough valid static samples")
    offset = float(np.median(v[static]) - np.median(s[static]))
    return s + offset, offset


def plot_bilateral_angles(df: pd.DataFrame, *, method="geo", time_col="time_ms", title=None):
    """Plot independently computed left/right SAM angles (red/green).

    Expected columns: {l,r}_{hip,knee,ankle}_flex_{method}.
    Missing columns are reported explicitly instead of fabricating a side.
    """
    if time_col not in df:
        raise KeyError(f"Missing timestamp column: {time_col}")
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) / 1000
    if len(t) < 2 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite, strictly increasing and in milliseconds")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    available = []
    for ax, joint in zip(axes, JOINTS):
        for side, prefix in (("left", "l"), ("right", "r")):
            col = f"{prefix}_{joint}_flex_{method}"
            if col in df:
                ax.plot(t, pd.to_numeric(df[col], errors="coerce"),
                        color=LEG_COLORS[side], label=f"{side.capitalize()} SAM", linewidth=1.8)
                available.append(col)
        ax.set_ylabel(f"{joint.capitalize()} (°)")
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No calculated angles available", ha="center", va="center", transform=ax.transAxes)
    axes[-1].set_xlabel("Original video time (s) — not synchronized")
    fig.suptitle(title or f"Bilateral SAM joint angles ({method})")
    expected = [f"{p}_{j}_flex_{method}" for p in ("l", "r") for j in JOINTS]
    missing = [col for col in expected if col not in df]
    return fig, axes, missing
