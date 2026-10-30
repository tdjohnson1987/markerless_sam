"""Bilateral SAM/Vicon angle plots. Never alters data or timestamps.

Inputs: a dataframe containing aligned SAM timestamp (`time_ms_aligned`),
SAM joint columns (`l_hip_flex_geo`, `r_hip_flex_geo`, etc.), and optionally
interpolated Vicon columns (`l_vicon_hip_flex`, `r_vicon_hip_flex`).
Legacy `vicon_hip_flex` only maps to the RIGHT side for backward compatibility.
"""
import matplotlib.pyplot as plt
import pandas as pd

JOINTS = ("hip", "knee", "ankle")
METHODS = ("geo", "euler", "smooth")
LEG_COLORS = {"left": "#D94848", "right": "#24985A"}
LEG_PREFIX = {"left": "l", "right": "r"}


def _time(df):
    key = "time_ms_aligned" if "time_ms_aligned" in df else "time_ms"
    if key not in df:
        raise KeyError("Expected time_ms_aligned or time_ms")
    return pd.to_numeric(df[key], errors="coerce") / 1000


def _ref(df, side, joint):
    prefix = LEG_PREFIX[side]
    explicit = f"{prefix}_vicon_{joint}_flex"
    if explicit in df:
        return explicit
    legacy = f"vicon_{joint}_flex"
    return legacy if side == "right" and legacy in df else None


def plot_joint_overlays(df, title="SAM vs Vicon", method="geo", sides=("left", "right")):
    """Solid SAM and dashed Vicon; red left and green right."""
    if method not in METHODS:
        raise ValueError(f"Method must be one of {METHODS}")
    t = _time(df)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for joint, ax in zip(JOINTS, axes):
        for side in sides:
            color = LEG_COLORS[side]
            scol = f"{LEG_PREFIX[side]}_{joint}_flex_{method}"
            vcol = _ref(df, side, joint)
            if scol in df:
                ax.plot(t, pd.to_numeric(df[scol], errors="coerce"), color=color,
                        linestyle="-", label=f"{side.title()} SAM ({method})")
            if vcol:
                ax.plot(t, pd.to_numeric(df[vcol], errors="coerce"), color=color,
                        linestyle="--", label=f"{side.title()} Vicon")
        ax.set_ylabel(f"{joint.title()} (deg)")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(ncol=2)
        else:
            ax.text(.5, .5, "No columns available", transform=ax.transAxes,
                    ha="center", va="center")
    axes[-1].set_xlabel("Time (s; Vicon clock if aligned)")
    fig.suptitle(title)
    return fig


def plot_joint_errors(df, title="SAM − Vicon error", method="geo", sides=("left", "right")):
    t = _time(df)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for joint, ax in zip(JOINTS, axes):
        for side in sides:
            scol = f"{LEG_PREFIX[side]}_{joint}_flex_{method}"
            vcol = _ref(df, side, joint)
            if scol in df and vcol:
                error = (pd.to_numeric(df[scol], errors="coerce") -
                         pd.to_numeric(df[vcol], errors="coerce"))
                ax.plot(t, error, color=LEG_COLORS[side], label=f"{side.title()} ({method})")
        ax.axhline(0, color="black", lw=.8)
        ax.set_ylabel(f"{joint.title()} error (deg)")
        ax.grid(alpha=.2)
        if len(ax.lines) > 1:
            ax.legend()
    axes[-1].set_xlabel("Time (s; Vicon clock if aligned)")
    fig.suptitle(title)
    return fig


def plot_camera_comparison(camera_frames, joint="knee", method="geo", sides=("left", "right")):
    """Compare cameras, each dataframe separately aligned to Vicon time."""
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for cam_idx, (camera, df) in enumerate(camera_frames.items()):
        for side in sides:
            col = f"{LEG_PREFIX[side]}_{joint}_flex_{method}"
            if col in df:
                ax.plot(_time(df), pd.to_numeric(df[col], errors="coerce"),
                        color=LEG_COLORS[side], linestyle="-" if cam_idx == 0 else ":",
                        label=f"{camera} {side.title()} SAM")
    if camera_frames:
        first = next(iter(camera_frames.values()))
        for side in sides:
            ref = _ref(first, side, joint)
            if ref:
                ax.plot(_time(first), pd.to_numeric(first[ref], errors="coerce"),
                        color=LEG_COLORS[side], linestyle="--", label=f"{side.title()} Vicon")
    ax.set(xlabel="Vicon-clock time (s) (if aligned)",
           ylabel=f"{joint.title()} angle (deg)",
           title="Multi-camera comparison — independent camera offsets")
    ax.grid(alpha=.2)
    if ax.lines: ax.legend(ncol=2)
    return fig
