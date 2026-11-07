
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(
        v, norm,
        out=np.full_like(v, np.nan),
        where=np.isfinite(norm) & (norm > 1e-8)
    )


def included_angle(a, b):
    dot = np.sum(a * b, axis=1)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide(
        dot, norm,
        out=np.full(len(a), np.nan),
        where=np.isfinite(norm) & (norm > 1e-8)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def signed_angle(a, b):
    """Signed 2D angle, with coordinates (forward, up)."""
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = np.sum(a * b, axis=1)

    angle = np.degrees(np.arctan2(cross, dot))

    valid = (
        np.isfinite(a).all(axis=1)
        & np.isfinite(b).all(axis=1)
        & (np.linalg.norm(a, axis=1) > 1e-8)
        & (np.linalg.norm(b, axis=1) > 1e-8)
    )
    return np.where(valid, angle, np.nan)


def calculate_matched_angles(
    left_hip, right_hip,
    left_shoulder, right_shoulder,
    hip, knee, ankle, heel, toe
):
    """
    Shared trunk-derived sagittal geometric definitions.

    All landmarks are arrays of shape (N, 3).
    Inputs should be expressed in a common 3D frame
    within each reconstruction.

    Outputs are geometric angle proxies, not
    calibrated clinical joint angles.
    """
    mid_hip = (left_hip + right_hip) / 2
    mid_sh = (left_shoulder + right_shoulder) / 2

    left = unit(left_hip - right_hip)
    up_temp = unit(mid_sh - mid_hip)

    forward = unit(np.cross(left, up_temp))
    up = unit(np.cross(forward, left))

    def project(v):
        return np.column_stack((
            np.sum(v * forward, axis=1),
            np.sum(v * up, axis=1)
        ))

    trunk = project(mid_sh - hip)
    thigh_down = project(knee - hip)
    thigh_up = project(hip - knee)
    shank_down = project(ankle - knee)
    shank_up = project(knee - ankle)
    foot_forward = project(toe - heel)

    # Unsigned trunk-thigh separation:
    hip_unsigned = 180 - included_angle(
        trunk, thigh_down
    )

    # Signed trunk-relative hip:
    # positive for anterior motion of the thigh
    hip_signed = signed_angle(
        -trunk, thigh_down
    )

    # Unsigned knee flexion proxy:
    knee_flex = 180 - included_angle(
        thigh_up, shank_down
    )

    # Sagittal ankle dorsiflexion proxy:
    shank_angle = np.arctan2(
        shank_up[:, 1], shank_up[:, 0]
    )
    foot_angle = np.arctan2(
        foot_forward[:, 1], foot_forward[:, 0]
    )

    ankle_df = np.degrees(
        np.pi / 2 - (shank_angle - foot_angle)
    )

    ankle_df = (ankle_df + 180) % 360 - 180

    valid_ankle = (
        np.isfinite(shank_up).all(axis=1)
        & np.isfinite(foot_forward).all(axis=1)
        & (np.linalg.norm(shank_up, axis=1) > 1e-8)
        & (np.linalg.norm(foot_forward, axis=1) > 1e-8)
    )
    ankle_df[~valid_ankle] = np.nan

    return {
        "hip_unsigned": hip_unsigned,
        "hip_signed": hip_signed,
        "knee_flex": knee_flex,
        "ankle_df": ankle_df,
    }
