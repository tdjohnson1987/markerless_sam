import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('force_plate', Path(__file__).parents[1] / 'src/force_plate.py')
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def fixture():
    t = np.arange(10) / 10
    corners = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0.]])
    markers = {key: np.tile(point, (10, 1)).astype(float) for key, point in {
        'LHEEL': [.2, .4, .02], 'LTOE': [.4, .4, .02],
        'RHEEL': [2.2, .4, .02], 'RTOE': [2.4, .4, .02]}.items()}
    plates = [dict(plate_id=1, corners=corners), dict(plate_id=2, corners=corners + np.array([[2], [0], [0]]))]
    events = [dict(plate_id=1, start_s=.1, end_s=.8), dict(plate_id=2, start_s=.1, end_s=.8)]
    tables = []
    for pid, x, weight in [(1, .3, 300.), (2, 2.3, 500.)]:
        loaded = (t >= .1) & (t < .8)
        tables.append(pd.DataFrame(dict(plate_id=pid, time_s=t,
            Fx_N=np.where(loaded, 10., 0.), Fy_N=0., Fz_N=np.where(loaded, weight, 0.),
            CoP_x_m=np.where(loaded, x, np.nan), CoP_y_m=np.where(loaded, .4, np.nan),
            CoP_z_m=np.where(loaded, 0., np.nan), Tz_x_Nm=0., Tz_y_Nm=0., Tz_z_Nm=2.)))
    return t, markers, plates, events, pd.concat(tables, ignore_index=True)


class LoadsTests(unittest.TestCase):
    def setUp(self):
        self.t, self.markers, self.plates, self.events, self.loads = fixture()
        self.assignments = fp.assign_foot_contacts(self.events, self.plates, self.markers, self.t)

    def build(self, loads=None, assignments=None):
        return fp.build_bilateral_external_loads(
            self.loads if loads is None else loads,
            self.assignments if assignments is None else assignments,
            conventions_confirmed=True, surface_z_m=0.)

    def test_spatial_assignment_and_half_open_events(self):
        self.assertEqual(self.assignments.assigned_foot.tolist(), ['L', 'R'])
        self.assertEqual(self.assignments.n_marker_frames.tolist(), [7, 7])
        result = self.build()
        self.assertTrue((result.loc[result.time_s == .8, 'status'] == 'no_assigned_contact').all())

    def test_ambiguous_foot_and_missing_contralateral_markers(self):
        for key in ('RHEEL', 'RTOE'):
            self.markers[key] -= [2, 0, 0]
        assigned = fp.assign_foot_contacts(self.events, self.plates, self.markers, self.t)
        self.assertEqual(assigned.iloc[0].assigned_foot, 'ambiguous')
        self.markers['RTOE'][:] = np.nan
        assigned = fp.assign_foot_contacts(self.events, self.plates, self.markers, self.t)
        self.assertFalse(assigned.assignment_valid.any())

    def test_no_marker_frames_and_low_coverage(self):
        self.markers['LHEEL'][1:7] = np.nan
        assigned = fp.assign_foot_contacts(self.events, self.plates, self.markers, self.t)
        self.assertFalse(assigned.assignment_valid.any())
        assigned = fp.assign_foot_contacts([dict(plate_id=1, start_s=3., end_s=4.)], self.plates, self.markers, self.t)
        self.assertEqual(assigned.iloc[0].assigned_foot, 'unresolved')

    def test_force_conservation_and_single_plate_cop(self):
        result = self.build()
        measured = self.loads.groupby('time_s')[['Fx_N', 'Fy_N', 'Fz_N']].sum()
        assembled = result.groupby('time_s')[['Fx_N', 'Fy_N', 'Fz_N']].sum()
        np.testing.assert_allclose(assembled, measured)
        loaded_l = result[(result.side == 'L') & result.cop_valid]
        np.testing.assert_allclose(loaded_l.CoP_x_m, .3)
        np.testing.assert_allclose(loaded_l.Tz_Nm, 2.)
        self.assertTrue(result.loc[result.status == 'no_assigned_contact', 'CoP_x_m'].isna().all())

    def test_two_plates_one_foot_conserves_wrench(self):
        self.assignments.loc[:, 'assigned_foot'] = 'L'
        result = self.build()
        row = result[(result.time_s == .2) & (result.side == 'L')].iloc[0]
        self.assertAlmostEqual(row.CoP_x_m, (.3*300 + 2.3*500)/800)
        force = row[['Fx_N', 'Fy_N', 'Fz_N']].to_numpy(float)
        cop = row[['CoP_x_m', 'CoP_y_m', 'CoP_z_m']].to_numpy(float)
        torque = row[['Tx_Nm', 'Ty_Nm', 'Tz_Nm']].to_numpy(float)
        expected = np.cross([.3, .4, 0], [10, 0, 300]) + np.cross([2.3, .4, 0], [10, 0, 500]) + [0, 0, 4]
        np.testing.assert_allclose(np.cross(cop, force) + torque, expected, atol=1e-12)
        self.assertEqual(row.contributing_plate_ids, '1;2')

    def test_unresolved_load_invalidates_both_feet(self):
        self.assignments.loc[0, ['assigned_foot', 'assignment_valid']] = ['unresolved', False]
        result = self.build()
        rows = result[result.time_s == .2]
        self.assertTrue(rows.Fz_N.isna().all())
        self.assertTrue((rows.unresolved_plate_ids == '1').all())

    def test_unassigned_loaded_plate_is_visible(self):
        result = self.build(assignments=self.assignments.iloc[:1])
        self.assertTrue(result.loc[result.time_s == .2, 'Fz_N'].isna().all())

    def test_missing_cop_and_wrong_surface_flag_invalid(self):
        self.loads.loc[(self.loads.plate_id == 1) & (self.loads.time_s == .2), 'CoP_x_m'] = np.nan
        result = self.build()
        self.assertFalse(result[(result.side == 'L') & (result.time_s == .2)].iloc[0].assignment_valid)
        self.loads.loc[self.loads.plate_id == 2, 'CoP_z_m'] = .1
        result = self.build()
        self.assertFalse(result[(result.side == 'R') & (result.time_s == .2)].iloc[0].assignment_valid)

    def test_overlap_and_mismatched_clocks_rejected(self):
        duplicate = self.assignments.iloc[[0]].copy()
        duplicate['event_id'] = 'overlap'
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            self.build(assignments=pd.concat([self.assignments, duplicate]))
        self.loads.loc[self.loads.plate_id == 2, 'time_s'] += .01
        with self.assertRaisesRegex(ValueError, 'timebases'):
            self.build()

    def test_requires_explicit_conventions(self):
        with self.assertRaisesRegex(ValueError, 'Verify'):
            fp.build_bilateral_external_loads(self.loads, self.assignments)

    def test_empty_events_keep_loaded_unknown_and_unloaded_zero(self):
        result = self.build(assignments=[])
        self.assertTrue(result.loc[result.time_s == .2, 'Fz_N'].isna().all())
        self.assertTrue((result.loc[result.time_s == 0, 'Fz_N'] == 0).all())

    def test_export_retains_flags_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = fp.export_external_loads(self.build(), Path(folder)/'loads.csv', assignments=self.assignments)
            self.assertTrue(all(p.exists() for p in paths.values()))
            table = pd.read_csv(paths['loads'])
            self.assertEqual(len(table), 20)
            self.assertIn('unresolved_plate_ids', table)


if __name__ == '__main__':
    unittest.main()
