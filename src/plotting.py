"""Plot measured SAM vs Vicon angles without modifying them."""
import matplotlib.pyplot as plt
import pandas as pd
JOINTS = ("hip", "knee", "ankle")
METHODS = ("geo", "euler", "smooth")
def _time(df):
    key = "time_ms_aligned" if "time_ms_aligned" in df else "time_ms"
    if key not in df:
        raise KeyError("Expected time_ms_aligned or time_ms")
    return pd.to_numeric(df[key], errors="coerce") / 1000
def plot_joint_overlays(df, title="SAM vs Vicon"):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for joint, ax in zip(JOINTS, axes):
        ref = f"vicon_{joint}_flex"
        if ref in df:
            ax.plot(_time(df), df[ref], label="Vicon", color="black", linewidth=2)
        for method in METHODS:
            col = f"r_{joint}_flex_{method}"
            if col in df:
                ax.plot(_time(df), df[col], label=f"SAM {method}")
        ax.set_ylabel(f"{joint.capitalize()} (deg)")
        ax.grid(alpha=.2)
        if ax.lines: ax.legend()
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    return fig
def plot_joint_errors(df, title="SAM − Vicon error"):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for joint, ax in zip(JOINTS, axes):
        ref = f"vicon_{joint}_flex"
        if ref in df:
            for method in METHODS:
                col = f"r_{joint}_flex_{method}"
                if col in df:
                    err = pd.to_numeric(df[col], errors="coerce") - pd.to_numeric(df[ref], errors="coerce")
                    ax.plot(_time(df), err, label=method)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel(f"{joint.capitalize()} error (deg)")
        ax.grid(alpha=.2)
        if len(ax.lines)>1:ax.legend()
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    return fig
def plot_camera_comparison(camera_frames, joint="knee", method="geo"):
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for camera, df in camera_frames.items():
        col = f"r_{joint}_flex_{method}"
        if col not in df:
            col = f"r_{joint}_flex_smooth"
        if col in df:
            ax.plot(_time(df), df[col], label=f"{camera} SAM {col}")
    if camera_frames:
        df = next(iter(camera_frames.values()))
        ref = f"vicon_{joint}_flex"
        if ref in df:
            ax.plot(_time(df), df[ref], label="Vicon (first file)", color="black", linestyle="--")
    ax.set(xlabel="Aligned time (seconds)", ylabel=f"{joint.capitalize()} angle (deg)", title="Multi-camera comparison (exploratory)")
    ax.grid(alpha=.2)
    if ax.lines:ax.legend()
    return fig
