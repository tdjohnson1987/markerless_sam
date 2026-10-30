"""
Vicon C3D Parser & Biomechanical Kinematics Engine (SAM 3D MHR Standard)
Extracts 3D optical marker trajectories from Vicon C3D files and calculates 
clinical joint angles (Knee Flexion, Hip Flexion) using pelvic anatomical frame projection.
"""

import c3d
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def load_vicon_c3d(c3d_path: str) -> Tuple[pd.DataFrame, float, Dict[str, np.ndarray]]:
    """
    Parses a Vicon C3D file and returns:
    1. A pandas DataFrame of calculated joint angles over time.
    2. The sampling frame rate (Hz).
    3. Dictionary of raw 3D marker coordinate arrays shape (N_frames, 3).
    """
    with open(c3d_path, 'rb') as f:
        reader = c3d.Reader(f)
        frame_rate = float(reader.point_rate)
        point_labels = [lbl.strip() for lbl in reader.point_labels]
        label_to_idx = {label: i for i, label in enumerate(point_labels) if label}
        
        raw_frames = []
        for frame_no, points, analog in reader.read_frames():
            # points shape: (N_points, 5) -> (x, y, z, err, cam_cnt)
            raw_frames.append(points[:, :3])
            
    raw_frames = np.array(raw_frames)  # (N_frames, N_points, 3)
    num_frames = len(raw_frames)
    
    markers: Dict[str, np.ndarray] = {}
    for label, idx in label_to_idx.items():
        markers[label] = raw_frames[:, idx, :]

    def get_marker(name: str, fallback_names: list = None) -> np.ndarray:
        if name in markers:
            return markers[name]
        if fallback_names:
            for fname in fallback_names:
                if fname in markers:
                    return markers[fname]
        raise KeyError(f"Marker {name} not found in C3D file.")

    # 1. Primary Vicon Anatomical Markers
    l_sh = get_marker('LCLAVLAT', ['LCLAVMID', 'C7', 'STRN'])
    r_sh = get_marker('RCLAVLAT', ['RCLAVMID', 'C7', 'STRN'])
    l_hip = get_marker('LASI')
    r_hip = get_marker('RASI')
    r_knee = get_marker('RKNE')
    r_ankle = get_marker('RANK')
    
    l_knee = markers.get('LKNE', r_knee)
    l_ankle = markers.get('LANK', r_ankle)

    # Convert shape to (3, N_frames) for vector cross products
    l_sh_t = l_sh.T
    r_sh_t = r_sh.T
    l_hip_t = l_hip.T
    r_hip_t = r_hip.T
    r_knee_t = r_knee.T
    r_ankle_t = r_ankle.T
    l_knee_t = l_knee.T
    l_ankle_t = l_ankle.T

    # 2. Build Anatomical Pelvic Frame (Forward, Up, Left)
    mid_hip = (l_hip_t + r_hip_t) / 2.0
    mid_sh = (l_sh_t + r_sh_t) / 2.0

    left = l_hip_t - r_hip_t
    left_norm = np.linalg.norm(left, axis=0, keepdims=True)
    left_norm = np.where(left_norm == 0, 1e-6, left_norm)
    left /= left_norm

    up_temp = mid_sh - mid_hip
    up_temp_norm = np.linalg.norm(up_temp, axis=0, keepdims=True)
    up_temp_norm = np.where(up_temp_norm == 0, 1e-6, up_temp_norm)
    up_temp /= up_temp_norm

    fwd = np.cross(left, up_temp, axis=0)
    fwd_norm = np.linalg.norm(fwd, axis=0, keepdims=True)
    fwd_norm = np.where(fwd_norm == 0, 1e-6, fwd_norm)
    fwd /= fwd_norm

    up = np.cross(fwd, left, axis=0)
    up_norm = np.linalg.norm(up, axis=0, keepdims=True)
    up_norm = np.where(up_norm == 0, 1e-6, up_norm)
    up /= up_norm

    # 3. Calculate Clinical Projected Joint Angles
    def calc_vicon_angle(p1, p2, p3, joint_type):
        v1_raw = p1 - p2
        v2_raw = p3 - p2
        
        v1 = np.array([np.sum(v1_raw * fwd, axis=0), np.sum(v1_raw * up, axis=0)])
        v2 = np.array([np.sum(v2_raw * fwd, axis=0), np.sum(v2_raw * up, axis=0)])
        
        norm1 = np.linalg.norm(v1, axis=0)
        norm2 = np.linalg.norm(v2, axis=0)
        norm1 = np.where(norm1 == 0, 1e-6, norm1)
        norm2 = np.where(norm2 == 0, 1e-6, norm2)
        
        cosine_angle = np.sum(v1 * v2, axis=0) / (norm1 * norm2)
        abs_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        if joint_type == 'knee':
            return 180.0 - abs_angle
        elif joint_type == 'hip':
            is_flexion = np.sum(v2_raw * fwd, axis=0) > 0
            return np.where(is_flexion, 180.0 - abs_angle, -(180.0 - abs_angle))

    vicon_knee_flex_r = calc_vicon_angle(r_hip_t, r_knee_t, r_ankle_t, 'knee')
    vicon_hip_flex_r = calc_vicon_angle(r_sh_t, r_hip_t, r_knee_t, 'hip')
    vicon_knee_flex_l = calc_vicon_angle(l_hip_t, l_knee_t, l_ankle_t, 'knee')
    vicon_hip_flex_l = calc_vicon_angle(l_sh_t, l_hip_t, l_knee_t, 'hip')
    vicon_ankle_df_r = calc_vicon_angle(r_knee_t, r_ankle_t, r_ankle_t + np.array([[0], [0], [-1]]), 'ankle')
    vicon_ankle_df_l = calc_vicon_angle(l_knee_t, l_ankle_t, l_ankle_t + np.array([[0], [0], [-1]]), 'ankle')

    time_points = np.arange(num_frames) / frame_rate # time in seconds

    df = pd.DataFrame({
        'time': time_points,
        'ts': (time_points * 1000.0).astype(int),
        'r_knee_flex': vicon_knee_flex_r,
        'l_knee_flex': vicon_knee_flex_l,
        'r_hip_flex': vicon_hip_flex_r,
        'l_hip_flex': vicon_hip_flex_l,
        'r_hip_abd': np.zeros(num_frames), 
        'r_knee_valgus': np.zeros(num_frames), #
        'r_ankle_df': vicon_ankle_df_r,
        'l_ankle_df': vicon_ankle_df_l 
    })

    return df, frame_rate, markers


if __name__ == '__main__':
    df, rate, markers = load_vicon_c3d('vicon/lunge/s11_lunge_0_1.c3d')
    print(f"Loaded Vicon SAM 3D anatomical frame file: {len(df)} frames at {rate} Hz")
    print(df[['time', 'r_knee_flex', 'r_hip_flex']].head())
