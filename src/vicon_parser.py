
"""
Vicon C3D Parser & Biomechanical Kinematics Engine.

Exports:
- Legacy marker-derived geometric joint-angle proxies
- Harrington (2007) pelvis-based hip joint centre estimates
- Pelvis-relative sagittal hip flexion proxies

The new angles are anatomical approximations, not exact
Plug-in Gait Cardan outputs.
"""

import c3d
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def unit(v):
    """Normalize vectors along the last dimension."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)

    return np.divide(
        v,
        norm,
        out=np.full_like(v, np.nan),
        where=norm > 1e-8
    )


def build_pelvis_frame(LASI, RASI, LPSI, RPSI):
    """Construct pelvic axes from bilateral ASIS and PSIS markers."""
    asis_mid = (LASI + RASI) / 2
    psis_mid = (LPSI + RPSI) / 2

    pelvis_left = unit(LASI - RASI)

    anterior = asis_mid - psis_mid
    anterior = anterior - (
        np.sum(anterior * pelvis_left, axis=1, keepdims=True)
        * pelvis_left
    )

    pelvis_forward = unit(anterior)
    pelvis_up = unit(np.cross(pelvis_forward, pelvis_left))

    return asis_mid, pelvis_forward, pelvis_left, pelvis_up


def harrington_hjc(
    LASI, RASI, LPSI, RPSI,
    pelvis_forward, pelvis_left, pelvis_up
):
    """
    Harrington et al. (2007) pelvis-only HJC regression.

    Input/output coordinates in millimetres.
    Local axes: anterior +, left +, superior +.
    """
    asis_mid = (LASI + RASI) / 2
    psis_mid = (LPSI + RPSI) / 2

    pw = np.nanmedian(np.linalg.norm(LASI - RASI, axis=1))
    pd = np.nanmedian(np.linalg.norm(asis_mid - psis_mid, axis=1))

    if not np.isfinite(pw) or not np.isfinite(pd) or pw <= 0 or pd <= 0:
        raise ValueError("Invalid pelvic dimensions for HJC regression")

    ap = -0.24 * pd - 9.9
    si = -0.30 * pw - 10.9
    ml = 0.33 * pw + 7.3

    common = (
        asis_mid
        + ap * pelvis_forward
        + si * pelvis_up
    )

    LHJC = common + ml * pelvis_left
    RHJC = common - ml * pelvis_left

    dimensions = {
        "pelvis_width_mm": float(pw),
        "pelvis_depth_mm": float(pd),
        "hjc_ap_mm": float(ap),
        "hjc_si_mm": float(si),
        "hjc_ml_mm": float(ml),
    }

    return LHJC, RHJC, dimensions


def hip_flexion_proxy(hjc, knee_center, pelvis_forward, pelvis_up):
    """
    Signed pelvis-relative sagittal thigh orientation.

    Positive when knee is anterior relative to pelvis.
    Not a full 3D joint coordinate system angle.
    """
    thigh_down = unit(knee_center - hjc)

    forward = np.sum(
        thigh_down * pelvis_forward, axis=1
    )
    downward = -np.sum(
        thigh_down * pelvis_up, axis=1
    )

    return np.degrees(np.arctan2(forward, downward))


def load_vicon_c3d(
    c3d_path: str
) -> Tuple[pd.DataFrame, float, Dict[str, np.ndarray]]:

    with open(c3d_path, "rb") as f:
        reader = c3d.Reader(f)

        frame_rate = float(reader.point_rate)
        labels = [lbl.strip() for lbl in reader.point_labels]
        label_to_idx = {
            label: i for i, label in enumerate(labels) if label
        }

        raw_frames = []

        for _, points, _ in reader.read_frames():
            xyz = np.asarray(points[:, :3], dtype=float).copy()

            # Negative residual indicates an invalid C3D point.
            # Keep it missing rather than interpreting it as anatomy.
            invalid = ~np.isfinite(xyz).all(axis=1)
            if points.shape[1] > 3:
                invalid |= points[:, 3] < 0

            xyz[invalid] = np.nan
            raw_frames.append(xyz)

    raw_frames = np.asarray(raw_frames)
    num_frames = len(raw_frames)

    markers = {
        label: raw_frames[:, idx, :]
        for label, idx in label_to_idx.items()
    }

    def get_marker(name, fallback_names=None):
        for label in [name] + (fallback_names or []):
            if label in markers:
                return markers[label]
        raise KeyError(f"Marker not found: {name}")

    # ------------------------------------------------------
    # 1. Original reference geometry (preserved)
    # ------------------------------------------------------

    l_sh = get_marker(
        "LCLAVLAT", ["LCLAVMID", "C7", "STRN"]
    )
    r_sh = get_marker(
        "RCLAVLAT", ["RCLAVMID", "C7", "STRN"]
    )

    LASI = get_marker("LASI")
    RASI = get_marker("RASI")
    LPSI = get_marker("LPSI")
    RPSI = get_marker("RPSI")

    l_knee = get_marker("LKNE")
    r_knee = get_marker("RKNE")
    l_ankle = get_marker("LANK")
    r_ankle = get_marker("RANK")

    l_heel = get_marker("LHEEL")
    r_heel = get_marker("RHEEL")
    l_toe = get_marker("LTOE")
    r_toe = get_marker("RTOE")

    # Legacy shoulder-based sagittal frame
    mid_hip = (LASI + RASI) / 2
    mid_sh = (l_sh + r_sh) / 2

    left_legacy = unit(LASI - RASI)
    up_provisional = unit(mid_sh - mid_hip)
    fwd_legacy = unit(np.cross(left_legacy, up_provisional))
    up_legacy = unit(np.cross(fwd_legacy, left_legacy))

    def project_legacy(v):
        return np.column_stack((
            np.sum(v * fwd_legacy, axis=1),
            np.sum(v * up_legacy, axis=1)
        ))

    def calc_vicon_angle(p1, p2, p3, joint_type):
        v1_raw = p1 - p2
        v2_raw = p3 - p2

        v1 = project_legacy(v1_raw)
        v2 = project_legacy(v2_raw)

        norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        cosine = np.divide(
            np.sum(v1 * v2, axis=1),
            norms,
            out=np.full(len(v1), np.nan),
            where=norms > 1e-8
        )

        angle = np.degrees(
            np.arccos(np.clip(cosine, -1, 1))
        )

        flex = 180 - angle

        if joint_type == "knee":
            return flex

        if joint_type == "hip":
            is_flexion = np.sum(
                v2_raw * fwd_legacy, axis=1
            ) > 0

            return np.where(is_flexion, flex, -flex)

        raise ValueError(f"Unsupported joint: {joint_type}")

    def calc_ankle_df(knee, ankle, heel, toe):
        shank = project_legacy(knee - ankle)
        foot = project_legacy(toe - heel)

        shank_angle = np.arctan2(shank[:, 1], shank[:, 0])
        foot_angle = np.arctan2(foot[:, 1], foot[:, 0])

        angle = np.degrees(
            np.pi / 2 - (shank_angle - foot_angle)
        )

        result = (angle + 180) % 360 - 180
        valid = (
            np.isfinite(shank).all(axis=1)
            & np.isfinite(foot).all(axis=1)
            & (np.linalg.norm(shank, axis=1) > 1e-8)
            & (np.linalg.norm(foot, axis=1) > 1e-8)
        )
        result[~valid] = np.nan
        return result

    # Original outputs (names and definitions unchanged)
    r_hip_old = calc_vicon_angle(
        r_sh, RASI, r_knee, "hip"
    )
    l_hip_old = calc_vicon_angle(
        l_sh, LASI, l_knee, "hip"
    )

    r_knee_old = calc_vicon_angle(
        RASI, r_knee, r_ankle, "knee"
    )
    l_knee_old = calc_vicon_angle(
        LASI, l_knee, l_ankle, "knee"
    )

    r_ankle_old = calc_ankle_df(
        r_knee, r_ankle, r_heel, r_toe
    )
    l_ankle_old = calc_ankle_df(
        l_knee, l_ankle, l_heel, l_toe
    )

    # ------------------------------------------------------
    # 2. Anatomical pelvic frame
    # ------------------------------------------------------

    (
        asis_mid,
        pelvis_forward,
        pelvis_left,
        pelvis_up,
    ) = build_pelvis_frame(LASI, RASI, LPSI, RPSI)

    # ------------------------------------------------------
    # 3. Harrington HJC estimation
    # ------------------------------------------------------

    LHJC, RHJC, dimensions = harrington_hjc(
        LASI, RASI, LPSI, RPSI,
        pelvis_forward, pelvis_left, pelvis_up
    )

    # Bilateral medial/lateral knee joint-centre proxies
    LKJC = (
        get_marker("LKNE") + get_marker("LKNEMED")
    ) / 2

    RKJC = (
        get_marker("RKNE") + get_marker("RKNEMED")
    ) / 2

    # New pelvis-relative hip flexion
    l_hip_new = hip_flexion_proxy(
        LHJC, LKJC, pelvis_forward, pelvis_up
    )
    r_hip_new = hip_flexion_proxy(
        RHJC, RKJC, pelvis_forward, pelvis_up
    )

    # ------------------------------------------------------
    # 4. Output
    # ------------------------------------------------------

    time_ms = np.arange(num_frames) / frame_rate * 1000

    df = pd.DataFrame({
        "time_ms": time_ms,

        # Original marker-derived proxies
        "r_vicon_hip_flex": r_hip_old,
        "l_vicon_hip_flex": l_hip_old,
        "r_vicon_knee_flex": r_knee_old,
        "l_vicon_knee_flex": l_knee_old,
        "r_vicon_ankle_flex": r_ankle_old,
        "l_vicon_ankle_flex": l_ankle_old,

        # New pelvis-relative HJC-based proxies
        "r_vicon_hip_flex_harrington": r_hip_new,
        "l_vicon_hip_flex_harrington": l_hip_new,
    })

    # Optional geometric outputs for subsequent kinetics
    markers["RHJC_est"] = RHJC
    markers["LHJC_est"] = LHJC
    markers["RKJC_est"] = RKJC
    markers["LKJC_est"] = LKJC

    print("Harrington HJC estimate:")
    for key, value in dimensions.items():
        print(f"  {key}: {value:.2f}")

    return df, frame_rate, markers


if __name__ == "__main__":
    df, rate, markers = load_vicon_c3d(
        "vicon/lunge/s11_lunge_0_1.c3d"
    )
    print(f"Loaded {len(df)} frames at {rate:.1f} Hz")
    print(df.head())
