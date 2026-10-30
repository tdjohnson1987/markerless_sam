"""SAM–Vicon angle validation, with independent synchronization and optional diagnostics.

Run: python -m src.comparison_engine --help

A diagnostic timing shift is never applied to the primary aligned angles, and
metrics calculated after a diagnostic shift are clearly labelled exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ("hip", "knee", "ankle")
METHODS = ("geo", "euler", "smooth")
SIDES = {"right": "r", "left": "l"}


def _load_sync(path):
    data = json.loads(Path(path).read_text())
    if not np.isfinite(float(data["shift_ms"])):
        raise ValueError("Non-finite shift_ms")
    return data


def _time_clean(df, column):
    t = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"{column} must contain at least two finite, increasing timestamps")
    return t


def _reference_column(df, joint, side):
    explicit = f"{side}_vicon_{joint}_flex"
    legacy = f"vicon_{joint}_flex"
    if explicit in df:
        return explicit
    if side == "r" and legacy in df:
        return legacy
    return None


def _interpolate_reference(ref, t_target, column):
    if column not in ref:
        return np.full(len(t_target), np.nan)
    t = _time_clean(ref, "time_ms")
    x = pd.to_numeric(ref[column], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.full(len(t_target), np.nan)
    return np.interp(t_target, t[good], x[good], left=np.nan, right=np.nan)


def _rmse(s, v):
    return float(np.sqrt(np.mean((s - v) ** 2)))


def _metric(v, s, t, *, static_window_ms=None, diagnostic_shift_ms=None):
    ok = np.isfinite(v) & np.isfinite(s) & np.isfinite(t)
    v, s, t = v[ok], s[ok], t[ok]
    if len(v) < 3:
        return None
    if np.any(np.diff(t) <= 0):
        raise ValueError("Aligned timestamps must be increasing")
    diff = s - v
    corr = float(np.corrcoef(v, s)[0, 1]) if np.std(v) > 1e-10 and np.std(s) > 1e-10 else np.nan
    result = dict(
        n_samples=int(len(v)), rmse_raw_deg=_rmse(s, v),
        mae_raw_deg=float(np.mean(np.abs(diff))), bias_deg=float(np.mean(diff)),
        pearson_r=corr, rom_error_deg=float(np.ptp(s) - np.ptp(v)),
        peak_error_deg=float(np.max(s) - np.max(v)),
        peak_time_error_ms=float(t[np.argmax(s)] - t[np.argmax(v)]),
        baseline_offset_deg=np.nan, vicon_baseline_deg=np.nan,
        sam_baseline_deg=np.nan, rmse_neutral_deg=np.nan,
        rmse_neutral_same_overlap_deg=np.nan,
        rmse_neutral_shifted_deg=np.nan,
        rmse_improvement_pct=np.nan,
        diagnostic_shift_ms=np.nan, calibration_window_ms=np.nan,
        n_shift_overlap=0,
        calibration_status="not_requested", shift_status="not_requested",
    )
    if static_window_ms is None:
        return result
    if static_window_ms <= 0:
        raise ValueError("static_window_ms must be positive")
    static = t <= t[0] + static_window_ms
    if np.count_nonzero(static) < 3:
        raise ValueError("At least three valid samples required in the static window")
    v0, s0 = float(np.median(v[static])), float(np.median(s[static]))
    vn, sn = v - v0, s - s0
    result.update(
        vicon_baseline_deg=v0, sam_baseline_deg=s0,
        baseline_offset_deg=v0 - s0,
        rmse_neutral_deg=_rmse(sn, vn),
        calibration_window_ms=float(static_window_ms),
        calibration_status="trial_start_assumed_static_exploratory",
    )
    if diagnostic_shift_ms is None:
        return result
    if not np.isfinite(diagnostic_shift_ms):
        raise ValueError("diagnostic_shift_ms must be finite")
    # SAM sample originally at t is displayed at t + shift. Compare at Vicon t
    # using SAM evaluated at t - shift. Out-of-domain points are excluded.
    shifted = np.interp(t - diagnostic_shift_ms, t, sn, left=np.nan, right=np.nan)
    overlap = np.isfinite(shifted)
    result.update(diagnostic_shift_ms=float(diagnostic_shift_ms),
                  n_shift_overlap=int(overlap.sum()),
                  shift_status="diagnostic_joint_derived_not_validated")
    if overlap.sum() >= 3:
        before = _rmse(sn[overlap], vn[overlap])
        after = _rmse(shifted[overlap], vn[overlap])
        result.update(rmse_neutral_same_overlap_deg=before,
                      rmse_neutral_shifted_deg=after,
                      rmse_improvement_pct=100 * (before-after)/before if before > 0 else np.nan)
    return result


def compare_angles(angle_csv, output_dir, *, sync_json=None, sam_sync_csv=None, vicon_csv=None, vicon_c3d=None,
                   subject="unknown", trial="unknown", camera="unknown",
                   side="right", allow_legacy_embedded_reference=False,
                   static_window_ms=None, diagnostic_shift_ms=None, plot=True,
                   original_time_confirmed=False):
    """Compare SAM angles to Vicon; optionally report *exploratory* calibration/shift metrics.

    Raw SAM CSV: video `time_ms` plus reviewed sync_json and independent Vicon.
    Legacy recovery: verified per-row `sam_sync_csv` timestamps, explicitly acknowledged.
    Legacy CSV: pre-existing `time_ms_aligned` and embedded Vicon, requires opt-in.
    The diagnostic shift never changes the original timestamps or primary RMSE.
    """
    if vicon_csv is not None and vicon_c3d is not None:
        raise ValueError("Supply either vicon_csv or vicon_c3d, not both")
    if side not in SIDES:
        raise ValueError("side must be 'right' or 'left'")
    if diagnostic_shift_ms is not None and static_window_ms is None:
        raise ValueError("diagnostic_shift_ms requires static_window_ms")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(angle_csv)
    if df.empty:
        raise ValueError("Empty angle CSV")
    already = "time_ms_aligned" in df
    if sam_sync_csv is not None:
        if sync_json is None:
            raise ValueError("--sam-sync-csv requires --sync-json")
        if not original_time_confirmed:
            raise ValueError(
                "Legacy time recovery needs confirmation that the SAM angle values "
                "remain in their original frame order and were not resampled. "
                "After checking export provenance, pass --original-time-confirmed."
            )
        sam_sync = pd.read_csv(sam_sync_csv)
        video_t = _time_clean(sam_sync, "ts")
        if len(video_t) != len(df):
            raise ValueError(
                f"SAM sync/angles row count differs: {len(video_t)} vs {len(df)}; "
                "do not match rows by index"
            )
        if not already:
            raise ValueError("Legacy time recovery expects time_ms_aligned for shift verification")
        legacy_t = _time_clean(df, "time_ms_aligned")
        differences = legacy_t - video_t
        legacy_shift = float(np.median(differences))
        max_deviation = float(np.max(np.abs(differences - legacy_shift)))
        if max_deviation > 1e-3:
            raise ValueError(
                f"Non-constant legacy shift (maximum deviation {max_deviation:.6f} ms); "
                "cannot establish matching row timestamps"
            )
        if "time_ms" in df and not np.allclose(
            _time_clean(df, "time_ms"), video_t, rtol=0, atol=1e-3
        ):
            raise ValueError("Existing time_ms does not match SAM synchronization timestamps")
        df["time_ms"] = video_t
        print(f"Recovered video timestamps from {sam_sync_csv}; verified old shift "
              f"{legacy_shift:+.3f} ms, max deviation {max_deviation:.6g} ms")
    if sync_json is not None:
        if "time_ms" not in df:
            raise ValueError("Independent synchronization requires original time_ms or --sam-sync-csv")
        if already and not original_time_confirmed:
            raise ValueError(
                "CSV contains time_ms_aligned. Confirm time_ms is ORIGINAL VIDEO TIME "
                "and SAM rows were NOT resampled; then pass --original-time-confirmed."
            )
        # Never read the old aligned times or the embedded Vicon reference.
        t = _time_clean(df, "time_ms") + float(_load_sync(sync_json)["shift_ms"])
        sync = _load_sync(sync_json)
        if sync.get("status") not in ("approved", "accepted", "validated"):
            print(f"WARNING: sync status={sync.get('status', 'missing')} — exploratory results only; manually review event correspondence")
        df = df.drop(columns=[c for c in df.columns if c == "time_ms_aligned" or c.startswith("vicon_") or "_vicon_" in c])
        df["time_ms_aligned"] = t
    elif already:
        t = _time_clean(df, "time_ms_aligned")
        sync = {"method": "prealigned_upstream", "shift_ms": None,
                "status": "not_verified_here"}
    else:
        raise ValueError("Provide --sync-json and original time_ms for independent validation")
    prefix = SIDES[side]
    if vicon_c3d is not None:
        from src.vicon_parser import load_vicon_c3d
        vicon, _, _ = load_vicon_c3d(str(vicon_c3d))
        # Accept both current parser's field names and proposed validation names.
        if "time_ms" not in vicon:
            if "ts" in vicon:
                vicon["time_ms"] = pd.to_numeric(vicon["ts"], errors="raise")
            elif "time" in vicon:
                vicon["time_ms"] = pd.to_numeric(vicon["time"], errors="raise") * 1000.0
            else:
                raise ValueError("Vicon parser must return time_ms, ts, or time")
        for prefix0 in ("r", "l"):
            for joint in JOINTS:
                dst = f"{prefix0}_vicon_{joint}_flex"
                src = f"{prefix0}_{joint}_flex" if joint != "ankle" else f"{prefix0}_ankle_df"
                if dst not in vicon and src in vicon:
                    vicon[dst] = vicon[src]
        reference_origin = "independent_vicon_c3d"
    elif vicon_csv is not None:
        vicon = pd.read_csv(vicon_csv)
        reference_origin = "independent_vicon_csv"
    else:
        vicon = None

    if vicon is not None:
        for joint in JOINTS:
            explicit = f"{prefix}_vicon_{joint}_flex"
            legacy = f"vicon_{joint}_flex"
            source_col = explicit if explicit in vicon else legacy if side == "right" and legacy in vicon else None
            if source_col is not None:
                df[explicit] = _interpolate_reference(vicon, t, source_col)
    else:
        if not allow_legacy_embedded_reference or not already:
            raise ValueError("Provide --vicon-csv or opt into pre-aligned embedded Vicon legacy mode")
        reference_origin = "embedded_preinterpolated_vicon_unverified"
    rows = []
    for joint in JOINTS:
        vcol = _reference_column(df, joint, prefix)
        if vcol is None:
            continue
        v = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=float)
        for method in METHODS:
            col = f"{prefix}_{joint}_flex_{method}"
            if col not in df:
                continue
            s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            m = _metric(v, s, t, static_window_ms=static_window_ms,
                        diagnostic_shift_ms=diagnostic_shift_ms)
            if m is not None:
                rows.append(dict(subject=subject, trial=trial, camera=camera,
                                 side=side, joint=joint, method=method,
                                 reference_origin=reference_origin,
                                 sync_method=sync.get("method"),
                                 sync_shift_ms=sync.get("shift_ms"), **m))
    if not rows:
        raise ValueError("No comparable joint/method columns with >=3 samples")
    metrics = pd.DataFrame(rows)
    df.to_csv(out / "aligned_angles.csv", index=False)
    metrics.to_csv(out / "validation_metrics.csv", index=False)
    summary = {"subject": subject, "trial": trial, "camera": camera,
               "side": side, "sync": sync, "reference_origin": reference_origin,
               "metrics": json.loads(metrics.to_json(orient="records"))}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for ax, joint in zip(axes, JOINTS):
            vcol = _reference_column(df, joint, prefix)
            if vcol is not None:
                ax.plot(t/1000, df[vcol], label="Vicon", lw=2.2,
                        color={"right": "#24985A", "left": "#D94848"}[side], linestyle="--")
            for method in METHODS:
                col = f"{prefix}_{joint}_flex_{method}"
                if col in df:
                    ax.plot(t/1000, df[col], label=f"SAM {method}", alpha=.85,
                            color={"right": "#24985A", "left": "#D94848"}[side])
            ax.set_ylabel(f"{side.capitalize()} {joint} (deg)")
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend()
        axes[-1].set_xlabel("Vicon-clock time (s)")
        fig.suptitle(f"{subject} / {trial} / {camera} — {reference_origin}")
        fig.tight_layout()
        fig.savefig(out / "angle_comparison.png", dpi=160)
        plt.close(fig)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--angles", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sync-json")
    p.add_argument("--sam-sync-csv", help="Original video ts from corresponding SAM synchronization CSV; verifies constant legacy offset")
    p.add_argument("--vicon-csv")
    p.add_argument("--vicon-c3d", help="Use src.vicon_parser directly; no Vicon angle CSV required")
    p.add_argument("--original-time-confirmed", action="store_true",
                   help="Permit old exports containing time_ms_aligned ONLY when time_ms is original video time and SAM was not resampled")
    p.add_argument("--subject", default="unknown")
    p.add_argument("--trial", default="unknown")
    p.add_argument("--camera", default="unknown")
    p.add_argument("--side", choices=SIDES, default="right")
    p.add_argument("--allow-legacy-embedded-reference", action="store_true")
    p.add_argument("--static-window-ms", type=float)
    p.add_argument("--diagnostic-shift-ms", type=float)
    args = p.parse_args()
    res = compare_angles(args.angles, args.output, sync_json=args.sync_json,
                         sam_sync_csv=args.sam_sync_csv,
                         vicon_csv=args.vicon_csv, vicon_c3d=args.vicon_c3d,
                         original_time_confirmed=args.original_time_confirmed, subject=args.subject,
                         trial=args.trial, camera=args.camera, side=args.side,
                         allow_legacy_embedded_reference=args.allow_legacy_embedded_reference,
                         static_window_ms=args.static_window_ms,
                         diagnostic_shift_ms=args.diagnostic_shift_ms)
    print(res[["side", "joint", "method", "rmse_raw_deg", "rmse_neutral_deg",
               "rmse_neutral_shifted_deg"]].to_string(index=False))
    print(f"Outputs saved to {args.output}")


if __name__ == "__main__":
    main()
