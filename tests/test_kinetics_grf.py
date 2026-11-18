"""Synthetic numerical checks; no participant data or validation claims."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('kinetics_grf', Path(__file__).resolve().parents[1]/'src/kinetics_grf.py')
grf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grf)


def fixture():
    t = np.arange(0, 10, 1/12)
    # All points undergo a common smooth translation; offsets preserve anatomy.
    names = list(grf.SAM_JOINTS) + ['l_sh','r_sh','l_big_toe','l_small_toe','r_big_toe','r_small_toe']
    offsets = np.array([[-.15,0,.9],[.15,0,.9],[-.15,0,.5],[.15,0,.5],
                        [-.15,0,.1],[.15,0,.1],[-.2,0,1.4],[.2,0,1.4],
                        [-.15,.1,0],[-.14,.1,0],[.15,.1,0],[.14,.1,0]])
    translation = np.c_[0*t, .2*t, .025*np.sin(2*np.pi*.4*t)]
    positions = offsets[None]+translation[:,None]
    # Camera coords inverse of a known proper transform.
    theta=.6
    R = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    camera_positions=(positions-np.array([1,2,3]))@R.T/1.03
    d={'t_video_s':t,'t_lab_s':t,'frame_index':np.arange(len(t))*5,'names':names,
       'joints_camera':camera_positions}
    tr=np.arange(0,10,.001)
    loadrows=[]
    total=76*(grf.GRAVITY-.025*(2*np.pi*.4)**2*np.sin(2*np.pi*.4*tr))
    for p,fraction in [(1,.5),(2,.5),(3,0)]:
        loadrows.append(pd.DataFrame({'plate_id':p,'time_s':tr,'Fx_N':0.,'Fy_N':0.,'Fz_N':fraction*total}))
    return t,positions,d,pd.concat(loadrows,ignore_index=True)


class NumericalChecks(unittest.TestCase):
    def test_known_similarity_and_proper_rotation(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(100,3))
        R=np.array([[0,1,0],[-1,0,0],[0,0,1]])
        y=1.03*x@R+np.array([2,3,4])
        model=grf.fit_registration(x[:60],y[:60],True)
        np.testing.assert_allclose(grf.apply_registration(x[60:],model),y[60:],atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(model['R']),1)
        self.assertAlmostEqual(model['scale'],1.03)

    def test_tilted_camera_recovers_lab_vertical_acceleration(self):
        t=np.arange(0,5,1/12)
        lab=np.c_[.2*t, .1*t, .5*1.5*t**2]
        a=.7
        R=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        model={'scale':1.03,'R':R,'translation':np.array([2,3,4])}
        camera=(lab-model['translation'])@R.T/model['scale']
        _,_,acc,_,_=grf.sg_kinematics(t,grf.apply_registration(camera,model))
        ok=np.isfinite(acc).all(1)
        np.testing.assert_allclose(acc[ok],np.broadcast_to([0,0,1.5],acc[ok].shape),atol=1e-10)

    def test_no_interpolation_across_gaps_but_exact_samples_survive(self):
        t=np.array([0.,.01,1.,1.01])
        q=np.array([-.01,.005,.5,1.,1.005,2.])
        result=grf.interp_gap(t,t,q,.05)
        np.testing.assert_allclose(result[[1,3,4]],[.005,1,1.005])
        self.assertTrue(np.isnan(result[[0,2,5]]).all())

    def test_derivatives_in_seconds_and_edge_masks(self):
        t=np.arange(0,5,1/12)
        xyz=np.c_[t*0,t*0,2+.5*1.5*t**2]
        _,vel,acc,left,right=grf.sg_kinematics(t,xyz,.55)
        ok=np.isfinite(acc[:,2])
        np.testing.assert_allclose(acc[ok,2],1.5,atol=1e-10)
        np.testing.assert_allclose(vel[ok,2],1.5*t[ok],atol=1e-10)
        self.assertTrue(np.isnan(acc[:3]).all())
        self.assertTrue(np.all(left[ok]>=0) and np.all(right[ok]<len(t)))

    def test_gap_and_irregular_segments_stay_missing(self):
        t=np.arange(0,5,1/12)
        xyz=np.c_[t,t,t]
        xyz[25]=np.nan
        _,_,a,_,_=grf.sg_kinematics(t,xyz,.55)
        self.assertTrue(np.isnan(a[22:29]).all())
        jitter=t.copy();jitter[4]+=.01
        _,_,a,_,_=grf.sg_kinematics(jitter,np.c_[t,t,t],.55)
        self.assertTrue(np.isnan(a).all())

    def test_static_grf_bodyweight_and_heldout_stencils(self):
        t,p,d,loads=fixture()
        p[:,:,2]-=.025*np.sin(2*np.pi*.4*t)[:,None]
        d['joints_camera'][:,:,2]-=.025*np.sin(2*np.pi*.4*t)[:,None]/1.03
        loads['Fz_N']=np.where(loads.plate_id<3,76*grf.GRAVITY*.5,0.)
        com=grf.com_proxy(grf.sam_landmarks(p,d['names']))
        result=grf.analyse_camera('cam0',d,t,p[:,:6],com,loads,76,windows=(.55,),
            support_intervals=((0.,10.),),sync_reviewed=True,clocks_confirmed=True,
            force_conventions_confirmed=True)
        tab=result['series'].query("method == 'similarity'")
        ok=np.isfinite(tab.estimated_Fz_N)
        np.testing.assert_allclose(tab.loc[ok,'estimated_Fz_N'],76*grf.GRAVITY,atol=1e-8)
        eligible=tab.eligible_force_comparison
        self.assertTrue(eligible.any())
        first_test=result['held_out_original_indices'][0]
        self.assertFalse(eligible.iloc[first_test:first_test+3].any())

    def test_pipeline_keeps_frames_and_gates_reviewed_metrics(self):
        t,p,d,loads=fixture()
        com=grf.com_proxy(grf.sam_landmarks(p,d['names']))
        result=grf.analyse_camera('cam1',d,t,p[:,:6],com,loads,76,windows=(.35,.55,.75))
        self.assertEqual(len(result['series']),len(t)*2*3)
        self.assertFalse(result['series'].eligible_force_comparison.any())
        reviewed=result['metrics'].query("scope == 'reviewed_complete_support_held_out'")
        self.assertTrue((reviewed.n==0).all())
        self.assertTrue(reviewed.rmse_N.isna().all())
        self.assertEqual(len(result['registration']),4)
        sim=result['registration'].query("method == 'similarity'")
        self.assertTrue((sim.joint_rmse_3d_m<1e-12).all())
        baseline=result['metrics'].query("estimator == 'vicon_com_proxy' and scope == 'diagnostic_held_out_observed_plate_sum'")
        self.assertTrue((baseline.rmse_N < 4).all())

    def test_absent_reference_plate_fails(self):
        t,_,_,loads=fixture()
        with self.assertRaises(ValueError):
            grf.force_reference_at_times(loads,t,(4,),12)

    def test_empty_evaluation_is_nan_not_zero(self):
        out=grf.scalar_metrics([1.],[1.],[False],745.)
        self.assertEqual(out['n'],0)
        self.assertTrue(np.isnan(out['rmse_N']))


if __name__=='__main__':
    unittest.main()
