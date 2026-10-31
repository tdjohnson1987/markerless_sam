"""Notebook execution regressions using temporary synthetic inputs, never patient data.

These checks protect cell order, input validation and module integration; they do
not establish anatomical accuracy or replace review of measured recordings.
"""
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
    namespace = {'__name__': '__main__', 'TEST_DATA': data, 'TEST_RESULTS': results}
    notebook = json.loads((ROOT / 'notebooks' / name).read_text())
    try:
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            code = ''.join(cell['source'])
            # Redirect only dataset/output roots; execute the actual notebook cells
            # and shared modules, retaining all configuration and scientific checks.
            for quote in ('"', "'"):
                code = code.replace(f'ROOT / {quote}lunge_sam_out{quote}', 'TEST_DATA')
                code = code.replace(f'ROOT / {quote}results{quote}', 'TEST_RESULTS')
            exec(compile(code, f'{name}:cell-{index}', 'exec'), namespace)
            namespace['display'] = lambda *args: None
    finally:
        plt.close('all')
    return namespace


@pytest.mark.parametrize('name', ['02_kinematics.ipynb', '03_kinetics.ipynb',
                                 '04_validation.ipynb'])
def test_notebook_without_recordings(name, tmp_path, monkeypatch):
    ns = run_notebook(name, tmp_path / 'missing', tmp_path / 'results', monkeypatch,
                      cwd=ROOT / 'notebooks')
    if name == '04_validation.ipynb':
        assert ns['results'] == []
        assert ns['aligned_by_camera'] == {}


def test_sync_requires_c3d_before_using_stale_results(tmp_path, monkeypatch):
    out = tmp_path / 'results' / 'S11' / 'lunge_01' / 'cam0'
    out.mkdir(parents=True)
    (out / 'sync_candidate.png').write_text('invalid stale image')
    with pytest.raises(FileNotFoundError, match='missing. Available:'):
        run_notebook('01_synchronisation.ipynb', tmp_path / 'missing',
                     tmp_path / 'results', monkeypatch)


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


def arm_signal(t):
    return 0.02 + 1 / (1 + np.exp(-(t - 1500) / 95)) * np.exp(
        -np.maximum(t - 2300, 0) / 1000)


def write_c3d(path):
    """Simple synthetic straight legs plus raised hands for integration checks."""
    import ezc3d
    t = np.arange(0, 4000, 20, dtype=float)
    markers = {}
    for prefix, lateral in (('L', 100), ('R', -100)):
        for suffix, position in {
            'CLAVLAT': (0, lateral, 1500), 'ASI': (0, lateral, 1000),
            'KNE': (0, lateral, 500), 'ANK': (0, lateral, 0),
            'HEEL': (0, lateral, 0), 'TOE': (200, lateral, 0),
            'HAND': (0, lateral, 1500),
        }.items():
            markers[prefix + suffix] = np.tile(np.asarray(position, dtype=float)[:, None], (1, len(t)))
        markers[prefix + 'HAND'][2] += 500 * arm_signal(t)
    recording = ezc3d.c3d()
    recording['parameters']['POINT']['RATE']['value'] = [50.]
    recording['parameters']['POINT']['LABELS']['value'] = list(markers)
    points = np.zeros((4, len(markers), len(t)))
    points[:3] = np.stack(list(markers.values()), axis=1)
    recording['data']['points'] = points
    path.parent.mkdir(parents=True, exist_ok=True)
    recording.write(str(path))
    return t


def test_sync_with_recordings_and_missing_camera(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    t = write_c3d(data / 's11_lunge_0_1.c3d')
    pd.DataFrame({'ts': t, 'right_arm_raise': arm_signal(t - 320)}).to_csv(
        data / 's11_lunge_cam0_sam_sync.csv', index=False)
    ns = run_notebook('01_synchronisation.ipynb', data, tmp_path / 'results', monkeypatch)
    summary = ns['sync_summary'].set_index('camera')
    assert summary.loc['cam0', 'shift_ms'] == pytest.approx(-320, abs=25)
    assert summary.loc['cam0', 'status'] == 'review'
    assert summary.loc['cam1', 'status'] == 'missing_video_sync'
    assert (tmp_path / 'results/S11/lunge_01/cam0/sync_candidate.png').is_file()


def test_current_validation_recovers_camera_clocks(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    results = tmp_path / 'results'
    trial = data / 'S9/lunge_01'
    t = write_c3d(trial / 's9_lunge_0_1.c3d')
    for camera, shift in [('cam0', 240), ('cam1', 320)]:
        pd.DataFrame({'time_ms_aligned': t + 100,
                      'r_knee_flex_geo': np.full(len(t), 2.),
                      'vicon_knee_flex': np.full(len(t), 999.)}).to_csv(
            trial / f's9_lunge_{camera}_joint_angles.csv', index=False)
        pd.DataFrame({'ts': t}).to_csv(trial / f's9_lunge_{camera}_sam_sync.csv', index=False)
        out = results / 'S9/lunge_01' / camera
        out.mkdir(parents=True)
        (out / 'sync_candidate.json').write_text(json.dumps(
            {'method': 'rise50', 'shift_ms': shift, 'status': 'review'}))
    ns = run_notebook('04_validation.ipynb', data, results, monkeypatch)
    assert ns['failed'] == []
    assert len(ns['summary']) == 2
    assert np.allclose(ns['summary'].bias_deg, 2)
    assert set(ns['summary'].reference_origin) == {'independent_vicon_c3d'}
    for camera, shift in [('cam0', 240), ('cam1', 320)]:
        frame = ns['aligned_by_camera'][camera]
        assert np.allclose(frame.time_ms_aligned, t + shift)
        assert 'vicon_knee_flex' not in frame  # Discard stale embedded references.
    assert (results / 'S9/lunge_01/validation_metrics_bilateral.csv').is_file()


def test_c3d_inspection(tmp_path, monkeypatch):
    write_c3d(tmp_path / 'data/S11/lunge_01/s11_lunge_0_1.c3d')
    ns = run_notebook('03_kinetics.ipynb', tmp_path / 'data', tmp_path / 'results', monkeypatch)
    assert ns['c3d']['header']['points']['frame_rate'] == 50
