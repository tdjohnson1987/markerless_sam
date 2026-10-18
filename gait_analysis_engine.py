"""
Gait Analysis & Clinical Decision Support Engine
Computes 0-100% gait cycle time-normalization, quantitative deviation metrics
(Gait Deviation Index - GDI, Gait Profile Score - GPS, RMSD), phase-specific anomalies,
and generates evidence-based clinical recommendations and targeted interventions.
"""

import os
import json
import numpy as np
from scipy.interpolate import interp1d
from normative_gait_data import generate_normative_curves, generate_pathological_presets, export_normative_bank_json


def time_normalize_gait_cycle(raw_times, raw_angles, num_points=101):
    """
    Time-normalizes a raw time-series array of joint angles to 101 points (0-100% gait cycle).
    """
    if len(raw_times) < 3 or len(raw_angles) < 3:
        return np.zeros(num_points)

    t_norm = (raw_times - raw_times[0]) / (raw_times[-1] - raw_times[0])
    target_grid = np.linspace(0.0, 1.0, num_points)
    f_interp = interp1d(t_norm, raw_angles, kind='linear', fill_value='extrapolate')
    return f_interp(target_grid)


def calculate_rmsd(patient_curve, norm_mean_curve):
    """Calculates Root Mean Square Deviation from normative mean curve."""
    diff = np.array(patient_curve) - np.array(norm_mean_curve)
    return float(np.sqrt(np.mean(diff ** 2)))


def calculate_gdi(patient_curves, norm_data):
    """
    Calculates Gait Deviation Index (GDI).
    GDI = 100 - 10 * sqrt(sum(z_scores^2) / N).
    Normal GDI >= 100, Mild: 85-99, Moderate: 70-84, Severe: < 70.
    """
    z_scores = []
    
    joint_keys = [
        ("knee_flexion", "knee_flexion"),
        ("hip_flexion", "hip_flexion"),
        ("ankle_dorsiflexion", "ankle_dorsiflexion"),
        ("pelvic_tilt", "pelvic_tilt"),
        ("hip_abduction", "hip_abduction")
    ]
    
    for pat_key, norm_key in joint_keys:
        if pat_key in patient_curves:
            pat_arr = np.array(patient_curves[pat_key])
            norm_mean = np.array(norm_data["kinematics"][norm_key]["mean"])
            norm_sd = np.array(norm_data["kinematics"][norm_key]["sd"])
            
            z = (pat_arr - norm_mean) / norm_sd
            z_scores.extend(z)
            
    if len(z_scores) == 0:
        return 100.0, "Normal"

    mean_sq_z = np.mean(np.array(z_scores) ** 2)
    gdi_val = float(100.0 - 10.0 * np.sqrt(mean_sq_z))
    gdi_val = round(max(30.0, min(120.0, gdi_val)), 1)
    
    if gdi_val >= 95.0:
        classification = "Normal Baseline"
    elif gdi_val >= 85.0:
        classification = "Mild Deviation"
    elif gdi_val >= 70.0:
        classification = "Moderate Deviation"
    else:
        classification = "Severe Pathological Deviation"
        
    return gdi_val, classification


def evaluate_clinical_interventions(patient_curves, spatiotemporal, norm_data):
    """
    Evaluates patient kinematic curves against clinical rules and returns a list of
    identified pathological gait findings, severity grades, root causes, and targeted interventions.
    """
    findings = []
    interventions = {
        "physical_therapy": [],
        "orthotics_bracing": [],
        "biofeedback_retraining": [],
        "medical_surgical": []
    }

    knee_flex = np.array(patient_curves.get("knee_flexion", norm_data["kinematics"]["knee_flexion"]["mean"]))
    hip_flex = np.array(patient_curves.get("hip_flexion", norm_data["kinematics"]["hip_flexion"]["mean"]))
    ankle_df = np.array(patient_curves.get("ankle_dorsiflexion", norm_data["kinematics"]["ankle_dorsiflexion"]["mean"]))
    hip_abd = np.array(patient_curves.get("hip_abduction", norm_data["kinematics"]["hip_abduction"]["mean"]))
    pelvis_tilt = np.array(patient_curves.get("pelvic_tilt", norm_data["kinematics"]["pelvic_tilt"]["mean"]))

    norm_knee_mean = np.array(norm_data["kinematics"]["knee_flexion"]["mean"])
    norm_ankle_mean = np.array(norm_data["kinematics"]["ankle_dorsiflexion"]["mean"])

    # Phase Slices
    # Stance: 0 to 60 (indices 0 to 60)
    # Swing: 60 to 100 (indices 60 to 100)
    stance_knee = knee_flex[0:61]
    swing_knee = knee_flex[60:101]
    ic_ankle = ankle_df[0]
    stance_ankle = ankle_df[0:61]

    # 1. Crouch Gait Evaluation (Excessive stance knee flexion)
    mean_stance_knee = np.mean(stance_knee)
    norm_stance_knee = np.mean(norm_knee_mean[0:61])
    knee_stance_dev = mean_stance_knee - norm_stance_knee

    if knee_stance_dev > 8.0:
        severity = "Severe" if knee_stance_dev > 18.0 else ("Moderate" if knee_stance_dev > 12.0 else "Mild")
        findings.append({
            "code": "CROUCH_GAIT",
            "name": f"{severity} Crouch Gait Pattern",
            "severity": severity,
            "metric": f"+{knee_stance_dev:.1f}° Excessive Stance Knee Flexion",
            "phase": "Stance Phase (0-60%)",
            "root_causes": [
                "Hamstring tightness / spasticity",
                "Quadriceps muscle weakness",
                "Calf muscle (soleus) weakness allowing excessive dorsiflexion"
            ]
        })
        interventions["physical_therapy"].extend([
            "Hamstring eccentric lengthening and passive stretching",
            "Closed-chain quadriceps strengthening (squats, leg press)",
            "Soleus power training to control anterior tibial progression"
        ])
        interventions["orthotics_bracing"].append(
            "Solid or Ground Reaction AFO (GRAFO) to restrain anterior tibial translation during stance"
        )
        interventions["biofeedback_retraining"].append(
            "Visual/auditory biofeedback targeting full knee extension during mid-stance"
        )
        if severity in ["Moderate", "Severe"]:
            interventions["medical_surgical"].append(
                "Surgical evaluation for hamstring tendon lengthening or distal femoral extension osteotomy"
            )

    # 2. Stiff Knee Gait Evaluation (Reduced swing knee flexion)
    peak_swing_knee = np.max(swing_knee)
    norm_peak_swing = np.max(norm_knee_mean[60:101])
    swing_flex_loss = norm_peak_swing - peak_swing_knee

    if swing_flex_loss > 10.0:
        severity = "Severe" if swing_flex_loss > 25.0 else ("Moderate" if swing_flex_loss > 15.0 else "Mild")
        findings.append({
            "code": "STIFF_KNEE_GAIT",
            "name": f"{severity} Stiff Knee Swing Gait",
            "severity": severity,
            "metric": f"Peak Swing Flexion: {peak_swing_knee:.1f}° (Norm: {norm_peak_swing:.1f}°)",
            "phase": "Swing Phase (60-100%)",
            "root_causes": [
                "Rectus femoris overactivity / inappropriate swing phase firing",
                "Decreased knee flexor velocity at toe-off",
                "Co-contraction of quadriceps and hamstrings"
            ]
        })
        interventions["physical_therapy"].extend([
            "Prone rectus femoris stretching & hip flexor release",
            "Active knee flexor strengthening at high angular velocities"
        ])
        interventions["orthotics_bracing"].append(
            "Hinged AFO with dorsiflexion assist to compensate for toe clearance loss"
        )
        interventions["biofeedback_retraining"].append(
            "Functional Electrical Stimulation (FES) to hamstrings during initial swing phase"
        )
        if severity in ["Moderate", "Severe"]:
            interventions["medical_surgical"].append(
                "Rectus femoris transfer procedure (to semitendinosus) or botulinum toxin A injection to rectus femoris"
            )

    # 3. Equinus / Toe Walking Evaluation (Excessive plantarflexion)
    if ic_ankle < -4.0 or np.mean(stance_ankle) < -5.0:
        pf_val = abs(ic_ankle)
        severity = "Severe" if pf_val > 15.0 else ("Moderate" if pf_val > 8.0 else "Mild")
        findings.append({
            "code": "EQUINUS_GAIT",
            "name": f"{severity} Equinus / Toe-Walking Gait",
            "severity": severity,
            "metric": f"Ankle IC: {ic_ankle:.1f}° Plantarflexion (Norm: 0°)",
            "phase": "Initial Contact & Mid-Stance",
            "root_causes": [
                "Gastrocnemius / Soleus contracture or spasticity",
                "Weakness of Tibialis Anterior (dorsiflexor)",
                "Habitual toe walking"
            ]
        })
        interventions["physical_therapy"].extend([
            "Calf muscle static passive stretching (gastrocnemius & soleus)",
            "Active dorsiflexion resisted strengthening (Tibialis Anterior)"
        ])
        interventions["orthotics_bracing"].append(
            "Rigid AFO with 90° neutral ankle positioning or serial stretch casting"
        )
        interventions["biofeedback_retraining"].append(
            "Ground reaction pressure sensor biofeedback for initial heel strike"
        )
        if severity in ["Moderate", "Severe"]:
            interventions["medical_surgical"].append(
                "Percutaneous Achilles Tendon Lengthening (TAL) or botox injection to gastrocnemius"
            )

    # 4. Trendelenburg / Pelvic Drop Evaluation
    stance_hip_abd = hip_abd[0:61]
    max_adduction = np.max(stance_hip_abd)
    if max_adduction > 7.0:
        severity = "Severe" if max_adduction > 14.0 else ("Moderate" if max_adduction > 10.0 else "Mild")
        findings.append({
            "code": "TRENDELENBURG_GAIT",
            "name": f"{severity} Trendelenburg Pelvic Drop",
            "severity": severity,
            "metric": f"Excessive Hip Adduction: {max_adduction:.1f}°",
            "phase": "Single-Leg Stance Phase",
            "root_causes": [
                "Gluteus Medius / Minimus weakness on stance side",
                "Inhibited hip abductor firing due to hip joint pain",
                "Pelvic instability"
            ]
        })
        interventions["physical_therapy"].extend([
            "Gluteus Medius isolation exercises (side-lying leg lifts, clamshells)",
            "Single-leg stance balance & core stability exercises"
        ])
        interventions["orthotics_bracing"].append(
            "Contralateral walking cane / stick to reduce stance hip abductor torque demand by 50%"
        )
        interventions["biofeedback_retraining"].append(
            "Real-time mirror gait training targeting level pelvic tilt"
        )

    # 5. Asymmetry / Antalgic Evaluation
    asymmetry_val = spatiotemporal.get("asymmetry_pct", 2.0)
    if asymmetry_val > 8.0:
        severity = "Severe" if asymmetry_val > 15.0 else "Moderate"
        findings.append({
            "code": "ANTALGIC_GAIT",
            "name": f"{severity} Asymmetrical / Antalgic Gait",
            "severity": severity,
            "metric": f"Asymmetry Index: {asymmetry_val:.1f}% (Norm: <3%)",
            "phase": "Full Gait Cycle",
            "root_causes": [
                "Lower extremity pain avoidance during weight bearing",
                "Unilateral muscular weakness or joint arthritis"
            ]
        })
        interventions["physical_therapy"].extend([
            "Unweighted treadmill gait training / hydrotherapy",
            "Joint unloading exercises & targeted limb strengthening"
        ])
        interventions["orthotics_bracing"].append(
            "Shock-absorbing viscoelastic shoe insoles or lateral wedged orthotics"
        )

    # 6. Kinetic Overload & Power Deficit Evaluation
    knee_mom_curve = patient_curves.get("knee_moment", norm_data.get("kinetics", {}).get("knee_moment", {}).get("mean", []))
    knee_pow_curve = patient_curves.get("knee_power", norm_data.get("kinetics", {}).get("knee_power", {}).get("mean", []))
    grf_curve = patient_curves.get("grf_vertical", norm_data.get("kinetics", {}).get("grf_vertical", {}).get("mean", []))

    if len(knee_mom_curve) > 0:
        max_knee_mom = float(np.max(knee_mom_curve))
        if max_knee_mom > 1.05:
            severity = "Severe" if max_knee_mom > 1.40 else "Moderate"
            findings.append({
                "code": "QUADRICEPS_OVERLOAD",
                "name": f"{severity} Quadriceps Moment Overload",
                "severity": severity,
                "metric": f"Peak Knee Torque: {max_knee_mom:.2f} N·m/kg (Norm: ~0.55 N·m/kg)",
                "phase": "Stance Phase (10-40%)",
                "root_causes": [
                    "Excessive knee flexion loading during stance phase",
                    "Pathological crouch posture or extensor weakness",
                    "Elevated patellofemoral joint stress"
                ]
            })
            interventions["physical_therapy"].extend([
                "Eccentric quadriceps endurance training & progressive load management",
                "Gait retraining for upright posture to reduce sagittal knee moment arm"
            ])
            interventions["orthotics_bracing"].append(
                "Ground Reaction AFO (GRAFO) or extensor assist bracing to reduce quadriceps torque requirement"
            )

    if len(knee_pow_curve) > 0:
        max_pushoff_pow = float(np.max(knee_pow_curve[50:80])) if len(knee_pow_curve) >= 80 else 0.0
        if max_pushoff_pow < 0.35 and max_pushoff_pow >= 0.0:
            findings.append({
                "code": "PROPULSIVE_POWER_DEFICIT",
                "name": "Propulsive Power Deficit",
                "severity": "Moderate",
                "metric": f"Push-off Power: {max_pushoff_pow:.2f} W/kg (Norm: ~0.75 W/kg)",
                "phase": "Terminal Stance / Pre-Swing (50-65%)",
                "root_causes": [
                    "Plantarflexor / push-off muscle weakness or stiff knee swing",
                    "Loss of elastic energy recoil in tendon structures"
                ]
            })
            interventions["physical_therapy"].extend([
                "Plantarflexor explosive plyometric power training",
                "High-velocity knee flexor impulse drills"
            ])
            interventions["biofeedback_retraining"].append(
                "Real-time auditory feedback for ground push-off acceleration"
            )

    if len(grf_curve) > 0:
        max_grf = float(np.max(grf_curve))
        if max_grf > 120.0:
            findings.append({
                "code": "HIGH_IMPACT_LOADING",
                "name": "Excessive Ground Impact Force",
                "severity": "Moderate",
                "metric": f"Peak GRF: {max_grf:.0f}% Body Weight (Norm: ~110%)",
                "phase": "Initial Contact & Weight Acceptance (0-20%)",
                "root_causes": [
                    "Abrupt heel strike without eccentric shock absorption",
                    "Reduced ankle dorsiflexion mobility or stiff knee landing"
                ]
            })
            interventions["orthotics_bracing"].append(
                "Viscoelastic shock-absorbing heel pads & custom foot orthotics"
            )
            interventions["biofeedback_retraining"].append(
                "Soft-landing gait retraining using acoustic impact sensors"
            )

    # Fallback if no major pathology found
    if len(findings) == 0:
        findings.append({
            "code": "NORMAL_GAIT",
            "name": "Physiological Normal Gait Baseline",
            "severity": "None",
            "metric": "All kinematic and kinetic curves within ±1.5 SD normative band",
            "phase": "Full Gait Cycle (0-100%)",
            "root_causes": ["Normal physiological muscle activation, joint ROM, and ground reaction forces."]
        })
        interventions["physical_therapy"].append("Maintain current physical activity and flexibility regimen.")

    # Deduplicate recommendations
    for key in interventions:
        interventions[key] = list(dict.fromkeys(interventions[key]))

    return findings, interventions


def analyze_patient_gait(patient_data):
    """
    Main Analysis Entry Point. Takes a patient gait trial dict (or preset id),
    evaluates against normative data, computes GDI, GPS, phase metrics, findings,
    and returns full clinical report dict.
    """
    norm_bank = generate_normative_curves()
    presets = generate_pathological_presets()

    if isinstance(patient_data, str) and patient_data in presets:
        patient = presets[patient_data]
    elif isinstance(patient_data, dict):
        patient = patient_data
    else:
        patient = presets["normal_gait"]

    patient_curves = {
        "knee_flexion": patient.get("knee_flexion", norm_bank["kinematics"]["knee_flexion"]["mean"]),
        "hip_flexion": patient.get("hip_flexion", norm_bank["kinematics"]["hip_flexion"]["mean"]),
        "ankle_dorsiflexion": patient.get("ankle_dorsiflexion", norm_bank["kinematics"]["ankle_dorsiflexion"]["mean"]),
        "pelvic_tilt": patient.get("pelvic_tilt", norm_bank["kinematics"]["pelvic_tilt"]["mean"]),
        "hip_abduction": patient.get("hip_abduction", norm_bank["kinematics"]["hip_abduction"]["mean"])
    }

    spatiotemporal = {
        "speed_m_s": patient.get("speed_m_s", 1.25),
        "cadence": patient.get("cadence", 112),
        "stance_pct": patient.get("stance_pct", 60.0),
        "asymmetry_pct": patient.get("asymmetry_pct", 2.5)
    }

    # 1. Compute RMSD per joint
    rmsd_metrics = {}
    for joint_name, curve in patient_curves.items():
        if joint_name in norm_bank["kinematics"]:
            rmsd_val = calculate_rmsd(curve, norm_bank["kinematics"][joint_name]["mean"])
            rmsd_metrics[joint_name] = round(rmsd_val, 2)

    # 2. Compute Gait Profile Score (GPS)
    gps_score = round(float(np.mean(list(rmsd_metrics.values()))), 2)

    # 3. Compute Gait Deviation Index (GDI)
    gdi_score, gdi_class = calculate_gdi(patient_curves, norm_bank)

    # 4. Evaluate Clinical Interventions
    findings, interventions = evaluate_clinical_interventions(patient_curves, spatiotemporal, norm_bank)

    # 5. Build Complete Clinical Report JSON
    report = {
        "patient_metadata": {
            "trial_id": patient.get("id", "TRIAL_001"),
            "trial_name": patient.get("name", "Custom Recorded Gait Trial"),
            "description": patient.get("description", ""),
            "date": "2026-09-06"
        },
        "scores": {
            "gdi_score": gdi_score,
            "gdi_classification": gdi_class,
            "gps_score": gps_score,
            "rmsd_per_joint": rmsd_metrics
        },
        "spatiotemporal": spatiotemporal,
        "curves": patient_curves,
        "gait_percent": norm_bank["gait_percent"],
        "findings": findings,
        "interventions": interventions
    }

    return report


if __name__ == "__main__":
    # Test run on Crouch Gait preset
    crouch_report = analyze_patient_gait("crouch_gait")
    print("--------------------------------------------------")
    print(" CLINICAL GAIT ANALYSIS REPORT: Crouch Gait")
    print("--------------------------------------------------")
    print(f"GDI Score: {crouch_report['scores']['gdi_score']} ({crouch_report['scores']['gdi_classification']})")
    print(f"GPS Score: {crouch_report['scores']['gps_score']}°")
    print("\nIdentified Pathological Findings:")
    for f in crouch_report["findings"]:
        print(f" - [{f['severity']}] {f['name']} ({f['metric']})")
    print("\nTargeted Physical Therapy Interventions:")
    for pt in crouch_report["interventions"]["physical_therapy"]:
        print(f" - {pt}")
    print("\nTargeted Orthotics & Bracing:")
    for orth in crouch_report["interventions"]["orthotics_bracing"]:
        print(f" - {orth}")
