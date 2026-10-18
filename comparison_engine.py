"""
SAM 3D vs. Vicon Motion Capture Comparison & Validation Engine
Handles cross-correlation time synchronization, interpolation, statistical metrics calculation,
and exports high-resolution comparative figures and JSON summaries.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.stats import pearsonr

from vicon_parser import load_vicon_c3d
from sam3d_pipeline import process_sam3d_kinematics

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


def align_and_compare(vicon_c3d_path: str, sam3d_csv_path: str, output_dir: str = 'figures') -> dict:
    """
    Executes full comparative validation between Vicon gold standard and SAM 3D pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Datasets
    vicon_df, vicon_rate, _ = load_vicon_c3d(vicon_c3d_path)
    sam_df = process_sam3d_kinematics(sam3d_csv_path, scale_locked=True)
    
    vicon_t = vicon_df['time'].values
    vicon_knee = vicon_df['r_knee_flex'].values
    vicon_hip = vicon_df['r_hip_flex'].values
    
    sam_t = sam_df['time'].values
    sam_knee_raw = sam_df['r_knee_flex_raw'].values
    sam_knee_kalman = sam_df['r_knee_flex_kalman'].values
    sam_knee_smoothed = sam_df['r_knee_flex_smoothed'].values
    
    # Ensure positive Hip Flexion convention for SAM 3D (flexion = positive angle increase)
    sam_hip_raw = sam_df['r_hip_flex_smoothed'].values
    sam_hip_smoothed = np.abs(sam_hip_raw - sam_hip_raw[0])
    
    # 2. Cross-Correlation Time Alignment (Peak Lunge Matching)
    v_norm = vicon_knee - np.mean(vicon_knee)
    s_norm = sam_knee_smoothed - np.mean(sam_knee_smoothed)
    
    interp_sam_temp = interp1d(sam_t, s_norm, kind='linear', fill_value='extrapolate')
    t_temp = np.arange(0, sam_t[-1], 1.0 / vicon_rate)
    sam_100hz_temp = interp_sam_temp(t_temp)
    
    corr = correlate(v_norm, sam_100hz_temp, mode='full')
    lags = np.arange(-len(sam_100hz_temp) + 1, len(v_norm))
    best_lag = lags[np.argmax(corr)]
    time_shift = best_lag / vicon_rate
    
    sam_t_aligned = sam_t + time_shift
    
    # 3. Create Common Unified Time Grid within Overlapping Window
    t_min = max(vicon_t[0], sam_t_aligned[0])
    t_max = min(vicon_t[-1], sam_t_aligned[-1])
    
    unified_t = np.linspace(t_min, t_max, num=500)
    peak_idx = np.argmax(interp1d(vicon_t, vicon_knee)(unified_t))
    time_rel_peak = unified_t - unified_t[peak_idx]
    
    f_v_knee = interp1d(vicon_t, vicon_knee, kind='linear')
    f_v_hip  = interp1d(vicon_t, vicon_hip, kind='linear')
    
    f_s_knee_raw = interp1d(sam_t_aligned, sam_knee_raw, kind='linear', fill_value='extrapolate')
    f_s_knee_kalman = interp1d(sam_t_aligned, sam_knee_kalman, kind='linear', fill_value='extrapolate')
    f_s_knee_smooth = interp1d(sam_t_aligned, sam_knee_smoothed, kind='linear', fill_value='extrapolate')
    f_s_hip_smooth = interp1d(sam_t_aligned, sam_hip_smoothed, kind='linear', fill_value='extrapolate')
    
    y_vicon_knee = f_v_knee(unified_t)
    y_vicon_hip  = f_v_hip(unified_t)
    
    y_sam_knee_raw = f_s_knee_raw(unified_t)
    y_sam_knee_kalman = f_s_knee_kalman(unified_t)
    y_sam_knee_smooth = f_s_knee_smooth(unified_t)
    y_sam_hip_smooth = f_s_hip_smooth(unified_t)
    
    # 4. Statistical Metrics Calculation
    def calc_metrics(y_true, y_pred):
        diff = y_pred - y_true
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        r_val, _ = pearsonr(y_true, y_pred)
        bias = float(np.mean(diff))
        sd_diff = float(np.std(diff))
        upper_loa = float(bias + 1.96 * sd_diff)
        lower_loa = float(bias - 1.96 * sd_diff)
        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'pearson_r': round(float(r_val), 3),
            'bias': round(bias, 2),
            'sd': round(sd_diff, 2),
            'upper_loa': round(upper_loa, 2),
            'lower_loa': round(lower_loa, 2)
        }

    metrics = {
        'knee_raw': calc_metrics(y_vicon_knee, y_sam_knee_raw),
        'knee_kalman': calc_metrics(y_vicon_knee, y_sam_knee_kalman),
        'knee_smoothed': calc_metrics(y_vicon_knee, y_sam_knee_smooth),
        'hip_smoothed': calc_metrics(y_vicon_hip, y_sam_hip_smooth),
        'time_shift_sec': round(float(time_shift), 3)
    }

    # 5. Generate High-Resolution Figures
    
    # Figure 1: Waveforms & Continuous Error
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(time_rel_peak, y_vicon_knee, label='Vicon Gold Standard', color='#0f172a', linewidth=2.8)
    ax1.plot(time_rel_peak, y_sam_knee_raw, label='SAM 3D (Raw Unfiltered)', color='#ef4444', linestyle=':', alpha=0.5)
    ax1.plot(time_rel_peak, y_sam_knee_kalman, label='SAM 3D (Kalman Filter)', color='#eab308', linestyle='--', linewidth=1.8)
    ax1.plot(time_rel_peak, y_sam_knee_smooth, label='SAM 3D (Smoothed Final)', color='#06b6d4', linewidth=2.5)
    
    ax1.set_ylabel('Knee Flexion (Degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Markerless SAM 3D vs. Vicon Gold Standard: Knee Flexion Waveform', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    ax1.grid(True, alpha=0.4)

    abs_err = np.abs(y_sam_knee_smooth - y_vicon_knee)
    ax2.fill_between(time_rel_peak, abs_err, color='#06b6d4', alpha=0.3, label='Absolute Tracking Error')
    ax2.plot(time_rel_peak, abs_err, color='#0891b2', linewidth=1.5)
    ax2.axhline(metrics['knee_smoothed']['mae'], color='#ef4444', linestyle='--', label=f"Mean Error: {metrics['knee_smoothed']['mae']}°")
    ax2.set_ylabel('Abs Error (°)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time Relative to Peak Lunge (seconds)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'sam3d_vs_vicon_waveforms.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Bland-Altman Agreement Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    def draw_bland_altman(ax, m1, m2, title, col):
        mean_val = (m1 + m2) / 2.0
        diff_val = m1 - m2
        bias = np.mean(diff_val)
        sd_val = np.std(diff_val)
        u_loa = bias + 1.96 * sd_val
        l_loa = bias - 1.96 * sd_val
        
        ax.scatter(mean_val, diff_val, alpha=0.6, color=col, edgecolors='k', s=35)
        ax.axhline(bias, color='#dc2626', linestyle='-', linewidth=2.0, label=f'Bias: {bias:.2f}°')
        ax.axhline(u_loa, color='#1e293b', linestyle='--', linewidth=1.5, label=f'+1.96 SD ({u_loa:.2f}°)')
        ax.axhline(l_loa, color='#1e293b', linestyle='--', linewidth=1.5, label=f'-1.96 SD ({l_loa:.2f}°)')
        ax.set_title(f'Bland-Altman: {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean of Vicon & SAM 3D (°)', fontsize=11)
        ax.set_ylabel('Difference (SAM 3D - Vicon) (°)', fontsize=11)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.4)

    draw_bland_altman(ax1, y_sam_knee_smooth, y_vicon_knee, 'Knee Flexion', '#0284c7')
    draw_bland_altman(ax2, y_sam_hip_smooth, y_vicon_hip, 'Hip Flexion', '#0d9488')
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'bland_altman_plots.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Unity Scatter Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    def draw_unity(ax, vicon_data, sam_data, title, col):
        r_val, _ = pearsonr(vicon_data, sam_data)
        rmse_val = np.sqrt(np.mean((sam_data - vicon_data)**2))
        
        min_v = min(np.min(vicon_data), np.min(sam_data)) - 5
        max_v = max(np.max(vicon_data), np.max(sam_data)) + 5
        
        ax.scatter(vicon_data, sam_data, alpha=0.5, color=col, edgecolors='none', s=25)
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2, label='Unity Line (y = x)')
        
        m, b = np.polyfit(vicon_data, sam_data, 1)
        ax.plot(vicon_data, m * vicon_data + b, color='#dc2626', linewidth=2, label=f'Fit (y = {m:.2f}x + {b:.1f})')
        
        ax.set_title(f'Unity Plot: {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Vicon Gold Standard (°)', fontsize=11)
        ax.set_ylabel('SAM 3D Markerless (°)', fontsize=11)
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        
        props = dict(boxstyle='round', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9)
        ax.text(0.05, 0.92, f"Pearson r: {r_val:.3f}\nRMSE: {rmse_val:.2f}°", transform=ax.transAxes, fontsize=11, bbox=props)
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True, alpha=0.4)

    draw_unity(ax1, y_vicon_knee, y_sam_knee_smooth, 'Knee Flexion', '#2563eb')
    draw_unity(ax2, y_vicon_hip, y_sam_hip_smooth, 'Hip Flexion', '#059669')
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'unity_scatter_plots.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: Kinetics Estimation (Joint Moments & Power)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    interp_moment = interp1d(sam_t_aligned, sam_df['r_knee_moment_nm_kg'], kind='linear', fill_value='extrapolate')(unified_t)
    interp_power  = interp1d(sam_t_aligned, sam_df['r_knee_power_w_kg'], kind='linear', fill_value='extrapolate')(unified_t)
    
    ax1.plot(time_rel_peak, interp_moment, color='#7c3aed', linewidth=2.5, label='Net Knee Extension Moment')
    ax1.axhline(0, color='#64748b', linestyle=':')
    ax1.set_ylabel('Joint Moment (N·m / kg)', fontsize=11, fontweight='bold')
    ax1.set_title('SAM 3D Inverse Dynamics Kinetics: Knee Moment & Joint Power', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)
    
    ax2.plot(time_rel_peak, interp_power, color='#059669', linewidth=2.2, label='Knee Joint Power')
    ax2.axhline(0, color='#64748b', linestyle=':')
    ax2.set_ylabel('Joint Power (W / kg)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time Relative to Peak Lunge (seconds)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, 'kinetics_moments_powers.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Save summary JSON
    results_json = os.path.join(output_dir, 'comparison_summary.json')
    with open(results_json, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Validation complete! Results saved to {output_dir}/")
    return metrics


if __name__ == '__main__':
    metrics = align_and_compare('vicon/lunge/s11_lunge_0_1.c3d', 's11_lunge_cam0_joint_angles.csv')
    print("Metrics summary:")
    print(json.dumps(metrics, indent=2))
