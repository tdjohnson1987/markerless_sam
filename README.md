# Markerless synchronization + dashboard data contract (prototype)

These modules are **additive**, do not modify your GitHub repo, and do not redo SAM inference. They specifically replace the knee-lag alignment for future validation work and give the dashboard a measured-data source.

## Setup

```bash
python -m pip install numpy pandas matplotlib
```

## Independent synchronization

Create `vicon_arm_sync.csv` from **raw Vicon** arm markers, with columns `time_ms` and `arm_raise` (same definition as video, ideally arm elevation relative to pelvis/body-up normalized by torso length). The existing SAM notebook saves `sam3d_sync_signal.csv` with `ts`, `arm_raise`.

```python
from synchronization import synchronize
r = synchronize('sam3d_sync_signal.csv','vicon_arm_sync.csv',
                output_json='results/sync.json', output_plot='results/sync.png')
print(r)
```

Inspect `sync.png`. `status` is intentionally `review`, not automatically `pass`: a single peak cannot independently establish alignment confidence or drift.

## Applying the offset

Only to a **SAM-only angle CSV with raw video time_ms**, not to the notebook's combined, already-aligned `_joint_angles.csv` files:

```python
from synchronization import align_angle_csv
align_angle_csv('sam_angles_raw.csv',r.shift_ms,'results/aligned_angles.csv')
```

If using existing `s11_lunge_cam1_joint_angles.csv`, **do not apply the shift again**. Regenerate the raw angle export from original `sam_df['ts']` and the SAM angles, with no Vicon columns, then run the synchronization step. To include Vicon reference in dashboard exports, interpolate the raw Vicon joint angles *after* independently aligning SAM, and retain NaN outside overlap.

## Dashboard JSON adapter

```python
from dashboard_export import export_dashboard_payload
export_dashboard_payload('results/aligned_angles.csv', 'results/dashboard.json',
                         raw_npz='results/sam_raw.npz',
                         sync_json='results/sync.json',
                         subject='S11',trial='lunge_01',camera='cam1')
```

The NPZ is optional: if omitted, the export contains genuine angle traces only. To render a 3D avatar **without inventing landmarks**, export the true MHR keypoints from your current extraction into NPZ with columns matching `LANDMARKS`. `sam_raw.npz['time_ms']` must be the same sampled frames as the angle CSV. Do not mistake a skeleton from a single 3D coordinate system for a fully calibrated world-space reconstruction.

**Existing Plotly visualizer integration:** its current `export_plotly_visualizer.py` assumes complete `joints3d`, a populated `mesh_vertices` and `kinetics` payload, so it will need guards for absent items and a 3D skeleton mode before reading this JSON. In particular, do not directly call the existing visualizer on a sparsely populated payload without adapting it. It currently expects a synthetic body and demonstrative kinetics. The export format intentionally never invents missing data.

## Next changes in repository
1. Update `comparison_engine.py` to take `sync.json` or aligned CSV and **delete knee-curve lag search**.
2. Refactor `sam3d_pipeline.py` to preserve raw timestamps and choose columns `r_*_flex_geo` explicitly; do not replace missing angle channels with zeros.
3. Adapt `export_plotly_visualizer.py` to render valid available joints/angles and hide kinetics until validated results exist.
4. Keep Vicon marker-derived angles and the angle definitions matched, with raw vs static-calibrated metrics clearly labeled.

Note: the repository's existing generic `vicon_parser.py` does not export the arm-raise synchronization signal, so you must generate it from Vicon `LHAND/RHAND`, `LCLAVLAT/RCLAVLAT`, `LASI/RASI` (or your declared marker set) prior to `synchronize()`.
