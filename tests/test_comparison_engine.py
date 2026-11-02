import tempfile
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.comparison_engine import compare_angles

def run():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        t=np.arange(0,3000,40,dtype=float)
        shifted=t+240
        def curve(t):return 20+30*np.sin(t/400)
        angles=pd.DataFrame({'time_ms':t,
                             'r_knee_flex_geo':curve(shifted)+2,
                             'r_hip_flex_geo':curve(shifted)/2,
                             'r_ankle_flex_geo':curve(shifted)/4,
                             'r_knee_flex_euler':curve(shifted)+5})
        angles.to_csv(d/'sam.csv',index=False)
        tv=np.arange(0,3401,10,dtype=float)
        vicon=pd.DataFrame({'time_ms':tv,'vicon_knee_flex':curve(tv),
                            'vicon_hip_flex':curve(tv)/2,'vicon_ankle_flex':curve(tv)/4})
        vicon.to_csv(d/'vicon.csv',index=False)
        (d/'sync.json').write_text(json.dumps({'method':'rise50','shift_ms':240,'status':'review'}))
        result=compare_angles(d/'sam.csv',d/'out',sync_json=d/'sync.json',vicon_csv=d/'vicon.csv',plot=False)
        assert len(result)==4
        knee=result[(result.joint=='knee')&(result.method=='geo')].iloc[0]
        assert abs(knee.bias_deg-2)<0.05,knee.bias_deg
        aligned=pd.read_csv(d/'out/aligned_angles.csv')
        assert np.allclose(aligned.time_ms_aligned,t+240)
        try:
            compare_angles(d/'out/aligned_angles.csv',d/'double',sync_json=d/'sync.json',vicon_csv=d/'vicon.csv',plot=False)
            raise AssertionError('Double alignment not rejected')
        except ValueError as e:
            assert 'original-time-confirmed' in str(e)
    print('Comparison engine synthetic tests passed')
if __name__=='__main__':run()
