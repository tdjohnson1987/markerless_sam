import tempfile
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
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

def test_original_time_is_verified_against_sam_sync(tmp_path):
    time_ms = np.arange(0, 400, 100, dtype=float)
    angles_path = tmp_path / 'angles.csv'
    sync_path = tmp_path / 'sam_sync.csv'
    vicon_path = tmp_path / 'vicon.csv'
    sync_json = tmp_path / 'sync.json'
    pd.DataFrame({
        'time_ms': time_ms,
        'r_knee_flex_geo': np.array([0., 10., 20., 30.]),
    }).to_csv(angles_path, index=False)
    pd.DataFrame({'ts': time_ms}).to_csv(sync_path, index=False)
    pd.DataFrame({
        'time_ms': time_ms + 25,
        'vicon_knee_flex': np.array([0., 10., 20., 30.]),
    }).to_csv(vicon_path, index=False)
    sync_json.write_text(json.dumps({
        'method': 'test', 'shift_ms': 25, 'status': 'review',
    }))

    metrics = compare_angles(
        angles_path, tmp_path / 'valid',
        sync_json=sync_json, sam_sync_csv=sync_path,
        vicon_csv=vicon_path, original_time_confirmed=True, plot=False,
    )
    assert len(metrics) == 1
    aligned = pd.read_csv(tmp_path / 'valid' / 'aligned_angles.csv')
    assert np.allclose(aligned['time_ms_aligned'], time_ms + 25)

    wrong_times = pd.read_csv(angles_path)
    wrong_times['time_ms'] += 1
    wrong_times.to_csv(angles_path, index=False)
    with pytest.raises(ValueError, match='does not match SAM synchronization timestamps'):
        compare_angles(
            angles_path, tmp_path / 'invalid',
            sync_json=sync_json, sam_sync_csv=sync_path,
            vicon_csv=vicon_path, original_time_confirmed=True, plot=False,
        )

    legacy_path = tmp_path / 'legacy_angles.csv'
    pd.DataFrame({
        'time_ms_aligned': time_ms - 40,
        'r_knee_flex_geo': np.array([0., 10., 20., 30.]),
    }).to_csv(legacy_path, index=False)
    compare_angles(
        legacy_path, tmp_path / 'legacy',
        sync_json=sync_json, sam_sync_csv=sync_path,
        vicon_csv=vicon_path, original_time_confirmed=True, plot=False,
    )
    recovered = pd.read_csv(tmp_path / 'legacy' / 'aligned_angles.csv')
    assert np.allclose(recovered['time_ms'], time_ms)
    assert np.allclose(recovered['time_ms_aligned'], time_ms + 25)

if __name__=='__main__':run()
