"""Exploratory Vicon-calibrated SAM CoM/total-GRF analysis.

Positions use row vectors: lab = scale * camera @ R + translation.
No foot-load decomposition or joint inverse dynamics is performed here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, savgol_filter, sosfiltfilt

GRAVITY = 9.80665
SAM_JOINTS = ('l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle')
VICON_JOINTS = ('LHJC_est', 'RHJC_est', 'LKJC_est', 'RKJC_est', 'LAJC_est', 'RAJC_est')
MASS_FRACTIONS = {'HAT': .678, 'l_thigh': .100, 'r_thigh': .100,
                  'l_shank': .0465, 'r_shank': .0465, 'l_foot': .0145, 'r_foot': .0145}


def check_time(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Need at least two finite, strictly increasing timestamps')
    return t


def interp_gap(t, values, query, max_gap=.05):
    """Per-trajectory interpolation; no extrapolation or bridging long gaps.

    An exact observed sample is valid even beside a long missing interval.
    """
    t = check_time(t)
    x = np.asarray(values, dtype=float)
    q = np.asarray(query, dtype=float)
    if len(x) != len(t):
        raise ValueError('Time/value lengths differ')
    shape = x.shape[1:]
    flat = x.reshape(len(t), -1)
    out = np.full((len(q), flat.shape[1]), np.nan)
    # Coordinate-wise interpolation; downstream requires complete XYZ triples.
    for k in range(flat.shape[1]):
        ok = np.isfinite(flat[:, k])
        tv, xv = t[ok], flat[ok, k]
        if not len(tv):
            continue
        right = np.searchsorted(tv, q, side='left')
        rc = np.clip(right, 0, len(tv)-1)
        exact = np.isfinite(q) & np.isclose(tv[rc], q, atol=1e-9, rtol=0)
        out[exact, k] = xv[rc[exact]]
        left = np.clip(right-1, 0, len(tv)-1)
        bracket = (np.isfinite(q) & ~exact & (right > 0) & (right < len(tv))
                   & ((tv[rc]-tv[left]) <= max_gap))
        out[bracket, k] = np.interp(q[bracket], tv, xv)
    return out.reshape((len(q),) + shape)


def fit_registration(X, Y, estimate_scale=False):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3 or len(X) < 6:
        raise ValueError('Need matching (N,3) arrays with >=6 correspondences')
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Registration inputs must be finite')
    mx, my = X.mean(0), Y.mean(0)
    xc, yc = X-mx, Y-my
    U, singular, Vt = np.linalg.svd(xc.T @ yc)
    D = np.eye(3)
    D[-1, -1] = np.linalg.det(U @ Vt)
    R = U @ D @ Vt
    denom = np.sum(xc**2)
    if denom <= 1e-12 or np.linalg.matrix_rank(xc) < 2:
        raise ValueError('Insufficient spatial variation for registration')
    scale = float(np.sum(singular*np.diag(D))/denom) if estimate_scale else 1.
    if scale <= 0:
        raise ValueError('Non-positive fitted scale')
    return {'R': R, 'scale': scale, 'translation': my-scale*mx@R,
            'singular_values': np.linalg.svd(xc, compute_uv=False)}


def apply_registration(x, model):
    return model['scale'] * (np.asarray(x) @ model['R']) + model['translation']


def com_proxy(landmarks):
    """Original illustrative mass model, deliberately unchanged by refactoring.

    Not published full-body anthropometry: HAT midpoint and absent arm/head
    motion are limitations. Vicon uses the same model to compare proxies.
    """
    hip = (landmarks['l_hip']+landmarks['r_hip'])/2
    shoulders = (landmarks['l_sh']+landmarks['r_sh'])/2
    segments = {'HAT': hip+.5*(shoulders-hip)}
    for side in ('l', 'r'):
        h, k, a, toe = (landmarks[f'{side}_{name}'] for name in ('hip', 'knee', 'ankle', 'toe'))
        segments[f'{side}_thigh'] = h+.433*(k-h)
        segments[f'{side}_shank'] = k+.433*(a-k)
        segments[f'{side}_foot'] = a+.5*(toe-a)
    return sum(MASS_FRACTIONS[name]*value for name, value in segments.items())


def sam_landmarks(kp, names):
    d = {name: kp[:, i] for i, name in enumerate(names)}
    for side in ('l', 'r'):
        d[f'{side}_toe'] = (d[f'{side}_big_toe']+d[f'{side}_small_toe'])/2
    return d


def valid_segments(t, values, gap_factor=1.8, dt_tolerance=.02):
    t = check_time(t)
    v = np.asarray(values, float)
    valid = np.isfinite(v.reshape(len(t), -1)).all(1)
    dt = np.diff(t)
    median_dt = np.median(dt)
    cuts = np.flatnonzero((dt > gap_factor*median_dt) | ~valid[:-1] | ~valid[1:])+1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if b-a < 2 or not valid[a:b].all():
            continue
        local_dt = float(np.median(np.diff(t[a:b])))
        if np.max(np.abs(np.diff(t[a:b])-local_dt))/local_dt > dt_tolerance:
            continue
        yield int(a), int(b), local_dt


def sg_kinematics(t, xyz, window_s=.55, polyorder=3):
    """Position, velocity, acceleration; mask SG half-window at every edge.

    Retains original timestamps. Short/irregular/gapped segments remain NaN.
    Returns original-frame stencil bounds to prevent train/test mixing.
    """
    t = check_time(t)
    xyz = np.asarray(xyz, float)
    if xyz.shape != (len(t), 3):
        raise ValueError('Expected (N,3) coordinates')
    outputs = [np.full_like(xyz, np.nan) for _ in range(3)]
    left, right = np.full(len(t), -1, int), np.full(len(t), -1, int)
    for a, b, dt in valid_segments(t, xyz):
        win = max(polyorder+2, int(round(window_s/dt)))
        win += (win % 2 == 0)
        if b-a < win:
            continue
        half = win//2
        sl = slice(a+half, b-half)
        for deriv, out in enumerate(outputs):
            filt = savgol_filter(xyz[a:b], win, polyorder, deriv=deriv, delta=dt, axis=0)
            out[sl] = filt[half:len(filt)-half]
        ids = np.arange(a+half, b-half)
        left[ids], right[ids] = ids-half, ids+half
    return (*outputs, left, right)


def interval_mask(t, intervals):
    t = np.asarray(t, float)
    mask = np.zeros(len(t), bool)
    for start, end in intervals:
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError('Intervals must have finite start < end')
        mask |= (t >= start) & (t < end)
    return mask


def force_reference_at_times(loads, query_t, plate_ids, target_fs):
    """Sum selected plates after zero-phase anti-alias filtering at native rate.

    Never contact-threshold or clip the total force. A missing plate/sample
    remains missing. Filters independently over valid contiguous segments.
    """
    tables = []
    cutoffs = []
    for plate_id in plate_ids:
        tab = loads.loc[loads.plate_id == plate_id].sort_values('time_s')
        if tab.empty:
            raise ValueError(f'Missing measured plate {plate_id}')
        t = check_time(tab.time_s.to_numpy())
        force = tab[['Fx_N', 'Fy_N', 'Fz_N']].to_numpy(float)
        filt = np.full_like(force, np.nan)
        for a, b, dt in valid_segments(t, force):
            cutoff = min(.4*target_fs, .4/dt)
            sos = butter(4, cutoff, fs=1/dt, output='sos')
            pad = 3*(2*len(sos)+1)
            if b-a <= pad+2:
                continue
            temp = sosfiltfilt(sos, force[a:b], axis=0)
            # Filter boundary transients are not eligible reference samples.
            filt[a+pad:b-pad] = temp[pad:-pad]
            cutoffs.append(cutoff)
        tables.append(interp_gap(t, filt, query_t, max_gap=1.8*np.median(np.diff(t))))
    return np.sum(np.stack(tables), axis=0), sorted(set(cutoffs))


def scalar_metrics(estimate, reference, mask, body_weight):
    e, r = np.asarray(estimate), np.asarray(reference)
    valid = np.asarray(mask, bool) & np.isfinite(e) & np.isfinite(r)
    n = int(valid.sum())
    result = {'n': n, 'bias_N': np.nan, 'rmse_N': np.nan, 'rmse_pct_bw': np.nan,
              'correlation': np.nan}
    if n:
        diff = e[valid]-r[valid]
        result.update(bias_N=float(diff.mean()), rmse_N=float(np.sqrt(np.mean(diff**2))))
        result['rmse_pct_bw'] = 100*result['rmse_N']/body_weight
    if n >= 3 and np.std(e[valid]) > 1e-10 and np.std(r[valid]) > 1e-10:
        result['correlation'] = float(np.corrcoef(e[valid], r[valid])[0, 1])
    return result


def load_camera(npz_path, sync_path):
    with np.load(npz_path, allow_pickle=False) as z:
        names = [str(v) for v in z['keypoint_names']]
        t = check_time(np.asarray(z['time_ms'], float)/1000)
        kp = np.asarray(z['keypoints3d'], float)
        translation = np.asarray(z['pred_cam_t'], float)
        frames = np.asarray(z['frame_index'])
    if kp.shape != (len(t), len(names), 3) or translation.shape != (len(t), 3):
        raise ValueError(f'Unexpected SAM dimensions: {npz_path}')
    if frames.shape != (len(t),):
        raise ValueError('Frame-index length differs from timestamps')
    sync = json.loads(Path(sync_path).read_text())
    shift = float(sync['shift_ms'])/1000
    if not np.isfinite(shift):
        raise ValueError('Nonfinite sync offset')
    return {'t_video_s': t, 't_lab_s': t+shift, 'frame_index': frames, 'names': names,
            'joints_camera': kp+translation[:, None, :], 'sync': sync,
            'npz_path': str(npz_path), 'sync_path': str(sync_path)}


def analyse_camera(camera, d, vicon_t, vicon_joints, vicon_com, loads, mass,
                   plate_ids=(1, 2, 3), calibration_fraction=.6, windows=(.55,),
                   support_intervals=(), sync_reviewed=False, clocks_confirmed=False,
                   force_conventions_confirmed=False, full_interpretation_confirmed=False):
    """One fixed rigid/similarity fit per camera, then CoM -> vector total GRF.

    'eligible' means reviewed held-out force comparison, NOT validated kinetics.
    The fixed model is applied to all original frames, never per-frame refit.
    """
    if not np.isfinite(mass) or mass <= 0 or not 0 < calibration_fraction < 1:
        raise ValueError('Invalid mass or calibration fraction')
    t = d['t_lab_s']
    joints = d['joints_camera'][:, [d['names'].index(n) for n in SAM_JOINTS]]
    reference = interp_gap(vicon_t, vicon_joints, t)
    paired = np.isfinite(joints).all((1, 2)) & np.isfinite(reference).all((1, 2))
    ids = np.flatnonzero(paired)
    n_train = int(len(ids)*calibration_fraction)
    if n_train < 3 or len(ids)-n_train < 3:
        raise ValueError(f'{camera}: insufficient calibration/held-out frames')
    train_ids = ids[:n_train]
    split_time = float(t[ids[n_train]])
    split_index = int(ids[n_train])
    com_camera = com_proxy(sam_landmarks(d['joints_camera'], d['names']))
    vcom = interp_gap(vicon_t, vicon_com, t) if vicon_com is not None else np.full_like(com_camera, np.nan)
    fs = 1/np.median(np.diff(d['t_video_s']))
    measured, cutoffs = force_reference_at_times(loads, t, plate_ids, fs)
    support = interval_mask(t, support_intervals)
    gate = bool(sync_reviewed and clocks_confirmed and force_conventions_confirmed)
    models, registration_rows, series, metrics = {}, [], [], []
    for method in ('rigid', 'similarity'):
        model = fit_registration(joints[train_ids].reshape(-1, 3),
                                 reference[train_ids].reshape(-1, 3), method == 'similarity')
        models[method] = model
        prediction = apply_registration(joints, model)
        for partition, ix in [('calibration', train_ids), ('held_out', ids[n_train:])]:
            error = prediction[ix]-reference[ix]
            hip_error = prediction[ix, :2].mean(1)-reference[ix, :2].mean(1)
            registration_rows.append({'camera': camera, 'method': method, 'partition': partition,
                'n_frames': len(ix), 'n_correspondences': len(ix)*6, 'scale': model['scale'],
                'joint_rmse_3d_m': float(np.sqrt(np.mean(np.sum(error**2, axis=2)))),
                'hip_mid_rmse_3d_m': float(np.sqrt(np.mean(np.sum(hip_error**2, axis=1)))),
                'hip_mid_vertical_rmse_m': float(np.sqrt(np.mean(hip_error[:, 2]**2)))})
        com_lab = apply_registration(com_camera, model)
        for window in windows:
            smooth, velocity, acceleration, left, right = sg_kinematics(t, com_lab, window)
            _, _, vicon_acc, _, _ = sg_kinematics(t, vcom, window)
            force = mass*acceleration
            force[:, 2] += mass*GRAVITY
            vforce = mass*vicon_acc
            vforce[:, 2] += mass*GRAVITY
            held = t >= split_time
            stencil_held = held & (left >= split_index)
            eligible = (stencil_held & support & gate & np.isfinite(force).all(1)
                        & np.isfinite(measured).all(1))
            tab = pd.DataFrame({'camera': camera, 'method': method, 'window_s': window,
                'frame_index': d['frame_index'], 'time_video_s': d['t_video_s'], 'time_vicon_s': t,
                'registration_pair_valid': paired, 'held_out': held,
                'derivative_stencil_held_out': stencil_held, 'complete_support_reviewed': support,
                'eligible_force_comparison': eligible, 'sync_reviewed': sync_reviewed,
                'clocks_confirmed': clocks_confirmed,
                'force_conventions_confirmed': force_conventions_confirmed,
                'reconstruction_interpretation_confirmed': full_interpretation_confirmed,
                'estimate_status': 'exploratory_vicon_calibrated_com_proxy'})
            for axis, label in enumerate('xyz'):
                tab[f'com_{label}_m_assumed'] = com_lab[:, axis]
                tab[f'com_smooth_{label}_m_assumed'] = smooth[:, axis]
                tab[f'velocity_{label}_m_s_assumed'] = velocity[:, axis]
                tab[f'accel_{label}_m_s2_assumed'] = acceleration[:, axis]
                tab[f'estimated_F{label}_N'] = force[:, axis]
                tab[f'measured_plate_sum_F{label}_N'] = measured[:, axis]
                tab[f'vicon_com_proxy_{label}_m'] = vcom[:, axis]
            tab['vicon_com_proxy_Fz_N'] = vforce[:, 2]
            series.append(tab)
            for scope, mask in [('diagnostic_held_out_observed_plate_sum', stencil_held),
                                ('reviewed_complete_support_held_out', eligible)]:
                for estimator, fz in [('sam_com_proxy', force[:, 2]), ('vicon_com_proxy', vforce[:, 2])]:
                    metrics.append({'camera': camera, 'method': method, 'window_s': window,
                        'scope': scope, 'estimator': estimator,
                        **scalar_metrics(fz, measured[:, 2], mask, mass*GRAVITY)})
    return {'models': models, 'registration': pd.DataFrame(registration_rows),
            'series': pd.concat(series, ignore_index=True), 'metrics': pd.DataFrame(metrics),
            'split_time_s': split_time, 'train_original_indices': train_ids,
            'held_out_original_indices': ids[n_train:], 'target_fs_hz': fs,
            'force_antialias_cutoffs_hz': cutoffs}

    


def json_ready(obj):
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, np.generic):
        return json_ready(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
