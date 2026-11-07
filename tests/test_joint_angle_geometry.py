
import numpy as np

from src.joint_angle_geometry import calculate_matched_angles


def make_pose(hip_deg=0, knee_deg=0, side="r"):
    """
    Construct one synthetic 3D sagittal pose.

    World axes:
    X = forward, Y = left, Z = up.

    Returns matched-angle outputs for a single frame.
    """
    h = np.deg2rad(hip_deg)
    shank_direction = np.deg2rad(hip_deg + knee_deg)

    l_hip = np.array([[0.0,  0.1, 0.0]])
    r_hip = np.array([[0.0, -0.1, 0.0]])

    l_sh = np.array([[0.0,  0.1, 1.0]])
    r_sh = np.array([[0.0, -0.1, 1.0]])

    hip = l_hip if side == "l" else r_hip

    knee = hip + np.array([[
        np.sin(h), 0.0, -np.cos(h)
    ]])

    ankle = knee + np.array([[
        np.sin(shank_direction),
        0.0,
        -np.cos(shank_direction)
    ]])

    # Level heel-to-toe foot direction
    heel = ankle + np.array([[-0.2, 0.0, 0.0]])
    toe = ankle + np.array([[0.2, 0.0, 0.0]])

    return calculate_matched_angles(
        left_hip=l_hip,
        right_hip=r_hip,
        left_shoulder=l_sh,
        right_shoulder=r_sh,
        hip=hip,
        knee=knee,
        ankle=ankle,
        heel=heel,
        toe=toe,
    )


def check(result, key, expected):
    np.testing.assert_allclose(
        result[key][0],
        expected,
        atol=1e-6,
        err_msg=f"Incorrect {key}"
    )


def test_neutral():
    for side in ("l", "r"):
        result = make_pose(side=side)

        check(result, "hip_unsigned", 0)
        check(result, "hip_signed", 0)
        check(result, "knee_flex", 0)
        check(result, "ankle_df", 0)


def test_hip_flexion():
    for side in ("l", "r"):
        result = make_pose(hip_deg=45, side=side)

        check(result, "hip_unsigned", 45)
        check(result, "hip_signed", 45)


def test_hip_extension():
    for side in ("l", "r"):
        result = make_pose(hip_deg=-15, side=side)

        check(result, "hip_unsigned", 15)
        check(result, "hip_signed", -15)


def test_knee_flexion():
    for side in ("l", "r"):
        result = make_pose(knee_deg=90, side=side)

        check(result, "knee_flex", 90)


def test_ankle_dorsiflexion():
    for side in ("l", "r"):
        # Knee positioned anterior to ankle.
        result = make_pose(knee_deg=-15, side=side)

        check(result, "ankle_df", 15)


def test_ankle_plantarflexion():
    for side in ("l", "r"):
        result = make_pose(knee_deg=15, side=side)

        check(result, "ankle_df", -15)


def test_invalid_landmark():
    # A missing knee must not produce a plausible angle.
    nan = np.array([[np.nan, np.nan, np.nan]])
    result = calculate_matched_angles(
        left_hip=np.array([[0.0, 0.1, 0.0]]),
        right_hip=np.array([[0.0, -0.1, 0.0]]),
        left_shoulder=np.array([[0.0, 0.1, 1.0]]),
        right_shoulder=np.array([[0.0, -0.1, 1.0]]),
        hip=np.array([[0.0, -0.1, 0.0]]),
        knee=nan,
        ankle=np.array([[0.0, -0.1, -2.0]]),
        heel=np.array([[-0.2, -0.1, -2.0]]),
        toe=np.array([[0.2, -0.1, -2.0]]),
    )

    assert np.isnan(result["knee_flex"][0])
    assert np.isnan(result["hip_signed"][0])
    assert np.isnan(result["ankle_df"][0])
