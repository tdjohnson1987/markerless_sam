"""Independent video-to-Vicon event synchronization; no joint-angle optimization."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class SyncResult:
    method: str
    shift_ms: float
    video_event_ms: float
    vicon_event_ms: float
    video_prominence: float
    vicon_prominence: float
    status: str
    note: str


def _prepare(df, time_col, signal_col, window_ms):
    if time_col not in df or signal_col not in df:
        raise KeyError(f"Required sync columns: {time_col!r}, {signal_col!r}. Found: {list(df)}")
    t = pd.to_numeric(df[time_col], errors='coerce').to_numpy(float)
    x = pd.to_numeric(df[signal_col], errors='coerce').to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    order = np.argsort(t); t, x = t[order], x[order]
    unique = np.r_[True, np.diff(t)>0] if len(t) else np.array([], dtype=bool)
    t, x = t[unique], x[unique]
    if len(t)<8: raise ValueError('Need at least 8 valid synchronization samples')
    inside = (t>=t[0]) & (t<=t[0]+window_ms)
    t, x = t[inside], x[inside]
    if len(t)<8: raise ValueError('Synchronization window too short')
    return t, x


def _event(t,x,method,min_prominence):
    nbase=max(3,min(len(x)//5,15))
    baseline=float(np.median(x[:nbase]))
    # Moving median removes brief detection outliers without using joint ground truth.
    xs=pd.Series(x).rolling(window=5,center=True,min_periods=1).median().to_numpy()
    peak_i=int(np.argmax(xs)); peak=float(xs[peak_i]); prominence=peak-baseline
    if prominence <= min_prominence: raise ValueError(f'No reliable rising arm event: prominence={prominence:.4g}')
    if method=='peak':
        return float(t[peak_i]),prominence
    if method!='rise50': raise ValueError("method must be 'rise50' or 'peak'")
    threshold=baseline+0.5*prominence
    # Limit to ascent before first major peak.
    crosses=np.where((xs[:peak_i]<threshold)&(xs[1:peak_i+1]>=threshold))[0]
    if not len(crosses): raise ValueError('No 50% rising-edge crossing before peak')
    i=int(crosses[0]); frac=(threshold-xs[i])/max(xs[i+1]-xs[i],1e-12)
    return float(t[i]+np.clip(frac,0,1)*(t[i+1]-t[i])),prominence


def synchronize(video_sync_csv, vicon_sync_csv, *,
                video_time_col='ts', video_signal_col='arm_raise',
                vicon_time_col='time_ms', vicon_signal_col='arm_raise',
                method='rise50', window_ms=4000.0, min_prominence=0.1,
                output_json=None, output_plot=None):
    """Both input time columns must be in milliseconds since recording start.

    Video and Vicon sync signals should be comparable (arm vertical displacement / torso length).
    Time shift is vicon_event_ms - video_event_ms. One shift applies to all joints.
    """
    video=pd.read_csv(video_sync_csv); vicon=pd.read_csv(vicon_sync_csv)
    vt,vx=_prepare(video,video_time_col,video_signal_col,window_ms)
    rt,rx=_prepare(vicon,vicon_time_col,vicon_signal_col,window_ms)
    ve,vp=_event(vt,vx,method,min_prominence)
    re,rp=_event(rt,rx,method,min_prominence)
    result=SyncResult(method, re-ve,ve,re,vp,rp,'review','Confirm event identity visually; no joint-angle fitting performed')
    if output_json:
        p=Path(output_json);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(asdict(result),indent=2))
    if output_plot:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4))
        def norm(x):
            return (x-np.nanmin(x))/max(np.nanmax(x)-np.nanmin(x),1e-12)
        ax.plot(rt/1000,norm(rx),label='Vicon arm rise')
        ax.plot((vt+result.shift_ms)/1000,norm(vx),label='Video arm rise')
        ax.axvline(re/1000,color='k',linestyle=':',label='Event')
        ax.set(xlabel='Vicon-relative time (s)',ylabel='Normalized arm elevation',title=f'Independent synchronization: {result.shift_ms:+.1f} ms')
        ax.legend();fig.tight_layout();p=Path(output_plot);p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=160);plt.close(fig)
    return result


def align_angle_csv(angle_csv, shift_ms, output_csv, *, time_col='time_ms', allow_pre_aligned=False):
    """Align SAM angles only. Do not shift existing Vicon columns or shift twice."""
    df=pd.read_csv(angle_csv)
    if 'time_ms_aligned' in df and not allow_pre_aligned:
        raise ValueError('CSV already contains time_ms_aligned. Supply raw video time_ms or explicitly reconstruct raw timestamps; do not double align.')
    if time_col not in df: raise KeyError(f'Missing raw video timestamp {time_col!r}; available: {list(df)}')
    if any(c.startswith('vicon_') for c in df):
        raise ValueError('This helper expects SAM-only raw angle CSV. Keep Vicon reference in a separate file.')
    df['time_ms_aligned']=pd.to_numeric(df[time_col],errors='raise')+shift_ms
    p=Path(output_csv);p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False)
    return df
