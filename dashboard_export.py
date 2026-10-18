"""Measured-data dashboard JSON adapter; never synthesizes joints, forces or moments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Names used by the existing Plotly visualizer's bone groups.
LANDMARKS = {
    'l_sh_3d':'l_shoulder', 'r_sh_3d':'r_shoulder',
    'l_hip_3d':'l_hip','r_hip_3d':'r_hip',
    'r_knee_3d':'r_knee','r_ankle_3d':'r_ankle',
    'r_big_toe_3d':'r_toe'
}
BONES=[['l_shoulder','l_hip'],['r_shoulder','r_hip'],['l_hip','r_hip'],
       ['r_hip','r_knee'],['r_knee','r_ankle'],['r_ankle','r_toe']]
ANGLE_COLS=('r_hip_flex_geo','r_knee_flex_geo','r_ankle_flex_geo',
            'r_hip_flex_euler','r_knee_flex_euler','r_ankle_flex_euler',
            'vicon_hip_flex','vicon_knee_flex','vicon_ankle_flex')

def export_dashboard_payload(aligned_angles_csv, output_json, *, raw_npz=None,
                             sync_json=None, metrics_json=None, subject=None, trial=None, camera=None):
    df=pd.read_csv(aligned_angles_csv)
    if 'time_ms_aligned' not in df: raise KeyError('Input requires time_ms_aligned')
    n=len(df)
    if n==0: raise ValueError('Empty trial')
    times=df['time_ms_aligned'].to_numpy(float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError('Aligned times must be finite and strictly increasing')
    raw=None
    if raw_npz:
        raw=np.load(raw_npz,allow_pickle=False)
        if 'time_ms' not in raw:raise KeyError('raw NPZ needs time_ms')
        rt=np.asarray(raw['time_ms'],float)
        # Strict matching avoids visualizations that are apparently synchronized but are not.
        if len(rt)!=n or not np.allclose(np.diff(rt),np.diff(times),atol=1):
            raise ValueError('Raw 3D output and aligned angle rows are inconsistent; match by raw frame/time before export')
    frames=[]
    for i,t in enumerate(times):
        joints={}
        if raw is not None:
            for raw_key,name in LANDMARKS.items():
                if raw_key in raw:
                    xyz=np.asarray(raw[raw_key][i],float)
                    if xyz.shape==(3,) and np.all(np.isfinite(xyz)):
                        joints[name]=xyz.round(5).tolist()
        angles={c:float(df.at[i,c]) for c in ANGLE_COLS if c in df and pd.notna(df.at[i,c])}
        frames.append({'frame':int(df.at[i,'frame']) if 'frame' in df else i,
                       'time':float(t/1000), 'joints3d':joints,'angles':angles,
                       'kinetics':{},'mesh_vertices':[]})
    def read_json(path):return json.loads(Path(path).read_text()) if path else None
    payload={'metadata':{'engine':'SAM 3D measured reconstruction',
                         'subject':subject,'trial':trial,'camera':camera,
                         'source':'measured reconstruction + processed angles',
                         'has_3d_joints':any(bool(f['joints3d']) for f in frames),
                         'kinetics_available':False,
                         'note':'No synthetic 3D joints, mesh or kinetics are generated.'},
             'skeleton_bones':[b for b in BONES if any(all(k in f['joints3d'] for k in b) for f in frames)],
             'frames':frames, 'sync':read_json(sync_json),'validation_metrics':read_json(metrics_json)}
    p=Path(output_json);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2,allow_nan=False))
    return p
