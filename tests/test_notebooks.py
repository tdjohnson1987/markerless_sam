"""Execute notebook code in order against missing and synthetic local inputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_notebook(name, data, results, monkeypatch, cwd=None):
    monkeypatch.chdir(cwd or ROOT)
    namespace = {'__name__': '__main__'}
    notebook = json.loads((ROOT / 'notebooks' / name).read_text())
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        exec(compile(code, f'{name}:cell-{index}', 'exec'), namespace)
        if 'DATA = ROOT' in code:
            namespace.update(DATA=data, RESULTS=results, display=lambda *args: None)
    plt.close('all')
    return namespace


@pytest.mark.parametrize('name', sorted(p.name for p in (ROOT / 'notebooks').glob('*.ipynb')))
def test_notebook_without_recordings(name, tmp_path, monkeypatch):
    run_notebook(name, tmp_path / 'missing', tmp_path / 'results', monkeypatch,
                 cwd=ROOT / 'notebooks')


def test_sync_does_not_display_stale_plot(tmp_path, monkeypatch):
    out = tmp_path / 'results' / 'S11' / 'lunge_01' / 'cam0'
    out.mkdir(parents=True)
    (out / 'sync.png').write_text('invalid stale image')
    ns = run_notebook('01_synchronisation.ipynb', tmp_path / 'missing',
                      tmp_path / 'results', monkeypatch)
    assert ns['sync'] is None


def test_nested_validation_and_distinct_trials(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    t = np.arange(0, 2000, 20, dtype=float)
    for trial in ('lunge_01', 'lunge_02'):
        for camera in ('cam0', 'cam1'):
            directory = data / 'S11' / trial / camera
            directory.mkdir(parents=True)
            pd.DataFrame({'time_ms_aligned': t,
                          'r_knee_flex_geo': np.sin(t / 200) + 2,
                          'r_vicon_knee_flex': np.sin(t / 200)}).to_csv(
                directory / 's11_lunge_cam0_joint_angles.csv', index=False)
    ns = run_notebook('04_validation.ipynb', data, tmp_path / 'results', monkeypatch)
    assert ns['trial'] == 'lunge_01'
    assert set(ns['camera_frames']) == {'cam0', 'cam1'}
    assert len(ns['metrics']) == 1
    assert ns['metrics'].iloc[0].bias_deg == pytest.approx(2)
    assert (ns['out_dir'] / 'validation_summary.json').is_file()


@pytest.mark.parametrize('timestamps', [np.array([0.]), np.array([0., np.nan]),
                                       np.array([1., 0.])])
def test_invalid_npz_timestamps(tmp_path, monkeypatch, timestamps):
    directory = tmp_path / 'data' / 'S11' / 'lunge_01' / 'cam0'
    directory.mkdir(parents=True)
    np.savez(directory / 'sam_raw.npz', time_ms=timestamps)
    with pytest.raises(ValueError, match='finite, increasing timestamps'):
        run_notebook('02_kinematics.ipynb', tmp_path / 'data', tmp_path / 'results', monkeypatch)


def test_landmark_npz_and_nested_csv(tmp_path, monkeypatch):
    directory = tmp_path / 'data' / 'S11' / 'lunge_01' / 'cam0'
    directory.mkdir(parents=True)
    np.savez(directory / 'sam_raw.npz', time_ms=np.arange(5.) * 20,
             r_hip_3d=np.zeros((5, 3)))
    pd.DataFrame({'time_ms': [0, 20], 'r_knee_flex_geo': [1, 2]}).to_csv(
        directory / 'sample_joint_angles.csv', index=False)
    ns = run_notebook('02_kinematics.ipynb', tmp_path / 'data', tmp_path / 'results', monkeypatch)
    assert len(ns['angle_files']) == 1


def test_sync_with_recordings(tmp_path, monkeypatch):
    directory = tmp_path / 'data' / 'S11' / 'lunge_01' / 'cam0'
    directory.mkdir(parents=True)
    t = np.arange(0, 4000, 20, dtype=float)
    def signal(clock):
        return 0.02 + 1 / (1 + np.exp(-(clock - 1500) / 95)) * np.exp(
            -np.maximum(clock - 2300, 0) / 1000)
    pd.DataFrame({'ts': t, 'arm_raise': signal(t)}).to_csv(directory / 'video_sync.csv', index=False)
    pd.DataFrame({'time_ms': t, 'arm_raise': signal(t - 320)}).to_csv(
        directory.parent / 'vicon_sync.csv', index=False)
    ns = run_notebook('01_synchronisation.ipynb', tmp_path / 'data', tmp_path / 'results', monkeypatch)
    assert ns['sync'].shift_ms == pytest.approx(320, abs=25)
    assert (ns['OUT'] / 'sync.png').is_file()


def test_c3d_inspection(tmp_path, monkeypatch):
    import ezc3d
    directory = tmp_path / 'data' / 'S11' / 'lunge_01'
    directory.mkdir(parents=True)
    recording = ezc3d.c3d()
    recording['parameters']['POINT']['RATE']['value'] = [100.]
    recording['parameters']['POINT']['LABELS']['value'] = ['TEST']
    recording['data']['points'] = np.zeros((4, 1, 10))
    recording.write(str(directory / 's11_lunge_0_1.c3d'))
    ns = run_notebook('03_kinetics.ipynb', tmp_path / 'data', tmp_path / 'results', monkeypatch)
    assert ns['c3d']['header']['points']['frame_rate'] == 100
