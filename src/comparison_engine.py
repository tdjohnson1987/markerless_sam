"""Reproducible SAM-vs-Vicon angle comparison with externally determined sync.

No knee/hip/ankle based timing optimization, no fictitious kinetics.
Run `python comparison_engine_v2.py --help` for CLI options.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

JOINTS=('hip','knee','ankle')
METHODS=('geo','euler','smooth')  # 'smooth' supports older dashboard CSVs; convention must be verified


def _load_sync(path):
    data=json.loads(Path(path).read_text())
    if not np.isfinite(float(data['shift_ms'])):
        raise ValueError('Non-finite shift_ms')
    return data


def _time_clean(df, name):
    t=pd.to_numeric(df[name],errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError(f'{name} must be finite and strictly increasing')
    return t


def _interpolate_reference(vicon, t_target, joint):
    key=f'vicon_{joint}_flex'
    if key not in vicon: return np.full(len(t_target),np.nan)
    x=pd.to_numeric(vicon[key],errors='coerce').to_numpy(float)
    t=_time_clean(vicon,'time_ms')
    good=np.isfinite(x)
    if good.sum()<2:return np.full(len(t_target),np.nan)
    return np.interp(t_target,t[good],x[good],left=np.nan,right=np.nan)


def _metric(v,s,t):
    ok=np.isfinite(v)&np.isfinite(s)&np.isfinite(t)
    v,s,t=v[ok],s[ok],t[ok]
    if len(v)<3:return None
    diff=s-v
    cor=float(np.corrcoef(v,s)[0,1]) if np.std(v)>1e-10 and np.std(s)>1e-10 else np.nan
    return dict(n_samples=int(len(v)), rmse_raw_deg=float(np.sqrt(np.mean(diff**2))),
                mae_raw_deg=float(np.mean(np.abs(diff))),bias_deg=float(np.mean(diff)),
                pearson_r=cor,rom_error_deg=float(np.ptp(s)-np.ptp(v)),
                peak_error_deg=float(np.max(s)-np.max(v)),
                peak_time_error_ms=float(t[np.argmax(s)]-t[np.argmax(v)]))


def compare_angles(angle_csv, output_dir, *, sync_json=None, vicon_csv=None,
                   subject='unknown',trial='unknown',camera='unknown',
                   allow_legacy_embedded_reference=False, plot=True):
    """Compare already-synchronized SAM angles OR raw SAM angles with a sync JSON.

    - RAW input must have `time_ms` (video clock); sync_json required.
    - ALIGNED input has `time_ms_aligned`; cannot apply sync again.
    - Independent reference CSV should contain `time_ms` (Vicon clock) and
      vicon_{hip,knee,ankle}_flex columns. No joint-based shifting is done.
    - Legacy pre-aligned joint CSV can be used with explicit opt-in, but cannot
      prove sync independence and must be treated as exploratory.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(angle_csv)
    if len(df)==0:raise ValueError('Empty angle CSV')
    already='time_ms_aligned' in df
    if already:
        if sync_json is not None:
            raise ValueError('Input already has time_ms_aligned. Do not apply a second synchronization.')
        t=_time_clean(df,'time_ms_aligned')
        sync={'method':'prealigned_upstream','shift_ms':None,'status':'not_verified_here'}
    else:
        if 'time_ms' not in df or sync_json is None:
            raise ValueError('Unaligned SAM CSV needs raw time_ms plus --sync-json')
        sync=_load_sync(sync_json)
        t=_time_clean(df,'time_ms')+float(sync['shift_ms'])
        df['time_ms_aligned']=t
    if vicon_csv:
        vicon=pd.read_csv(vicon_csv)
        for joint in JOINTS:
            df[f'vicon_{joint}_flex']=_interpolate_reference(vicon,t,joint)
        reference_origin='independent_vicon_csv'
    else:
        if not allow_legacy_embedded_reference:
            raise ValueError('Provide --vicon-csv, or explicitly use --allow-legacy-embedded-reference for exploratory already-aligned CSV')
        if not already: raise ValueError('Legacy embedded Vicon permitted only for pre-aligned CSVs')
        reference_origin='embedded_preinterpolated_vicon_unverified'
    rows=[]
    for joint in JOINTS:
        vcol=f'vicon_{joint}_flex'
        if vcol not in df:continue
        v=pd.to_numeric(df[vcol],errors='coerce').to_numpy(float)
        for method in METHODS:
            col=f'r_{joint}_flex_{method}'
            if col not in df:continue
            s=pd.to_numeric(df[col],errors='coerce').to_numpy(float)
            m=_metric(v,s,t)
            if m is not None:
                rows.append(dict(subject=subject,trial=trial,camera=camera,joint=joint,
                                 method=method,reference_origin=reference_origin,
                                 sync_method=sync.get('method'),sync_shift_ms=sync.get('shift_ms'),**m))
    if not rows:raise ValueError('No comparable joint/method columns with ≥3 valid overlapping samples')
    metrics=pd.DataFrame(rows)
    angles_path=out/'aligned_angles.csv';metrics_path=out/'validation_metrics.csv'
    df.to_csv(angles_path,index=False);metrics.to_csv(metrics_path,index=False)
    summary={'subject':subject,'trial':trial,'camera':camera,'sync':sync,
             'reference_origin':reference_origin,'metrics':json.loads(metrics.to_json(orient='records'))}
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    if plot:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for ax,joint in zip(axes,JOINTS):
            vcol=f'vicon_{joint}_flex'
            if vcol in df:ax.plot(t/1000,df[vcol],label='Vicon',lw=2.2,c='black')
            for method in METHODS:
                c=f'r_{joint}_flex_{method}'
                if c in df:ax.plot(t/1000,df[c],label=f'SAM {method}',alpha=.85)
            ax.set_ylabel(joint.capitalize()+' (deg)');ax.grid(alpha=.2);ax.legend(loc='best')
        axes[-1].set_xlabel('Vicon-clock time (s)')
        fig.suptitle(f'{subject} / {trial} / {camera} – {reference_origin}')
        fig.tight_layout();fig.savefig(out/'angle_comparison.png',dpi=160);plt.close(fig)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--angles',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--sync-json',default=None)
    p.add_argument('--vicon-csv',default=None)
    p.add_argument('--subject',default='unknown')
    p.add_argument('--trial',default='unknown')
    p.add_argument('--camera',default='unknown')
    p.add_argument('--allow-legacy-embedded-reference',action='store_true')
    args=p.parse_args()
    res=compare_angles(args.angles,args.output,sync_json=args.sync_json,vicon_csv=args.vicon_csv,
                       subject=args.subject,trial=args.trial,camera=args.camera,
                       allow_legacy_embedded_reference=args.allow_legacy_embedded_reference)
    print(res[['joint','method','n_samples','rmse_raw_deg','bias_deg','pearson_r']].to_string(index=False))
    print(f'Outputs saved to {args.output}')

if __name__=='__main__':main()
