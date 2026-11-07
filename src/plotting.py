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



def _ref(df, side, joint, vicon_reference="original"):
    prefix = LEG_PREFIX[side]

    if vicon_reference not in (
        "original", "harrington", "matched_geometry"
    ):
        raise ValueError(f"Invalid Vicon reference: {vicon_reference}")

    if vicon_reference == "matched_geometry":
        columns = {
            "hip": f"{prefix}_vicon_hip_unsigned_matched",
            "knee": f"{prefix}_vicon_knee_flex_matched",
            "ankle": f"{prefix}_vicon_ankle_df_matched",  # Not yet comparable with old SAM geo
        }

        col = columns[joint]

        if col is not None and col not in df:
            raise KeyError(f"Missing matched Vicon column: {col}")

        return col

    if joint == "hip" and vicon_reference == "harrington":
        col = f"{prefix}_vicon_hip_flex_harrington"

        if col not in df:
            raise KeyError(f"Missing Harrington Vicon column: {col}")

        return col

    explicit = f"{prefix}_vicon_{joint}_flex"
    legacy = f"vicon_{joint}_flex"

    if explicit in df:
        return explicit

    if side == "right" and legacy in df:
        return legacy

    return None



def plot_joint_overlays(df, title="SAM vs Vicon", method="geo", sides=("left", "right"), 
                        vicon_reference="original"):
    """Solid SAM and dashed Vicon; red left and green right."""
    if method not in METHODS:
        raise ValueError(f"Method must be one of {METHODS}")
    t = _time(df)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for joint, ax in zip(JOINTS, axes):
        for side in sides:
            color = LEG_COLORS[side]
            prefix = LEG_PREFIX[side]

            if vicon_reference == "matched_geometry" and joint == "ankle":
                scol = f"{prefix}_ankle_df_matched"
            else:
                scol = f"{prefix}_{joint}_flex_{method}"

            vcol = _ref(df, side, joint, vicon_reference)
      
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


def plot_joint_errors(df, title="SAM − Vicon error", method="geo", sides=("left", "right"), vicon_reference="original"):
    t = _time(df)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for joint, ax in zip(JOINTS, axes):
        for side in sides:
            prefix = LEG_PREFIX[side]

            if vicon_reference == "matched_geometry" and joint == "ankle":
                scol = f"{prefix}_ankle_df_matched"
            else:
                scol = f"{prefix}_{joint}_flex_{method}"

            vcol = _ref(df, side, joint, vicon_reference)
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


def plot_camera_comparison(camera_frames, joint="knee", method="geo", sides=("left", "right"), vicon_reference="original"):
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
            ref = _ref(first, side, joint, vicon_reference)
            if ref:
                ax.plot(_time(first), pd.to_numeric(first[ref], errors="coerce"),
                        color=LEG_COLORS[side], linestyle="--", label=f"{side.title()} Vicon")
    ax.set(xlabel="Vicon-clock time (s) (if aligned)",
           ylabel=f"{joint.title()} angle (deg)",
           title="Multi-camera comparison — independent camera offsets")
    ax.grid(alpha=.2)
    if ax.lines: ax.legend(ncol=2)
    return fig
