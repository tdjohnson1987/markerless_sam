from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from src.synchronization import synchronize,align_angle_csv
from src.dashboard_export import export_dashboard_payload

def test_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        t=np.arange(0,4000,20,dtype=float)
        def signal(t):return 0.02 + 1/(1+np.exp(-(t-1500)/95))*np.exp(-np.maximum(t-2300,0)/1000)
        pd.DataFrame({'ts':t,'arm_raise':signal(t)}).to_csv(d/'video.csv',index=False)
        pd.DataFrame({'time_ms':t,'arm_raise':signal(t-320)}).to_csv(d/'vicon.csv',index=False)
        r=synchronize(d/'video.csv',d/'vicon.csv',output_json=d/'sync.json',output_plot=d/'sync.png')
        assert abs(r.shift_ms-320)<25,r
        pd.DataFrame({'time_ms':t,'r_knee_flex_geo':30+10*np.sin(t/1000)}).to_csv(d/'angles.csv',index=False)
        a=align_angle_csv(d/'angles.csv',r.shift_ms,d/'aligned.csv')
        np.savez_compressed(d/'raw.npz',time_ms=t,r_hip_3d=np.zeros((len(t),3)),r_knee_3d=np.ones((len(t),3)))
        export_dashboard_payload(d/'aligned.csv',d/'dashboard.json',raw_npz=d/'raw.npz',sync_json=d/'sync.json')
        assert len(__import__('json').loads((d/'dashboard.json').read_text())['frames'])==len(t)
        print('PASS: shift',r.shift_ms,'ms; files + JSON exported')

if __name__=='__main__':test_roundtrip()
