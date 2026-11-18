
from pathlib import Path
import numpy as np


def load_force_plates(c3d_path):
    """
    Load calibrated force-platform outputs from Vicon C3D.

    Returns a dictionary containing native analog timestamps,
    per-platform force, moment, CoP and free moment.

    Coordinate frames and units must be verified before
    using these outputs for inverse dynamics.
    """
    import ezc3d

    c3d_path = Path(c3d_path)

    if not c3d_path.is_file():
        raise FileNotFoundError(c3d_path)

    c3d = ezc3d.c3d(
        str(c3d_path),
        extract_forceplat_data=True
    )

    analog_rate = float(
        c3d["header"]["analogs"]["frame_rate"]
    )
    marker_rate = float(
        c3d["header"]["points"]["frame_rate"]
    )

    if analog_rate <= 0 or marker_rate <= 0:
        raise ValueError("Invalid acquisition rates")

    platforms = []

    def samples_by_3(values):
        arr = np.asarray(values, dtype=float)

        if arr.ndim != 2:
            raise ValueError(f"Unexpected shape: {arr.shape}")

        if arr.shape[0] == 3:
            return arr.T

        if arr.shape[1] == 3:
            return arr

        raise ValueError(f"Expected 3-axis data: {arr.shape}")

    for plate_id, plate in enumerate(
        c3d["data"].get("platform", []),
        start=1
    ):
        force = samples_by_3(plate["force"])
        moment = samples_by_3(plate["moment"])
        cop = samples_by_3(plate["center_of_pressure"])
        free_moment = samples_by_3(plate["Tz"])

        n = len(force)

        if not all(
            len(arr) == n
            for arr in (moment, cop, free_moment)
        ):
            raise ValueError(
                f"Plate {plate_id}: sample count mismatch"
            )

        platforms.append({
            "plate_id": plate_id,
            "time_s": np.arange(n) / analog_rate,
            "force": force,
            "moment": moment,
            "cop": cop,
            "free_moment": free_moment,
            "corners": np.asarray(
                plate["corners"], dtype=float
            ),
            "unit_force": plate["unit_force"],
            "unit_moment": plate["unit_moment"],
            "unit_position": plate["unit_position"],
        })

    return {
        "source": str(c3d_path),
        "analog_rate_hz": analog_rate,
        "marker_rate_hz": marker_rate,
        "platforms": platforms,
        "time_origin": "relative_sample_index_provisional",

    }

def validate_force_plate_metadata(data):
    required = (
        "analog_rate_hz",
        "marker_rate_hz",
        "platforms",
    )

    for key in required:
        if key not in data:
            raise ValueError(f"Missing metadata: {key}")

    if not data["platforms"]:
        raise ValueError("No force platforms found")

    for plate in data["platforms"]:
        n = len(plate["time_s"])

        for key in ("force", "moment", "cop", "free_moment"):
            if plate[key].shape != (n, 3):
                raise ValueError(
                    f"Plate {plate['plate_id']}: invalid {key} shape"
                )

        if not np.all(np.isfinite(plate["force"])):
            raise ValueError(
                f"Plate {plate['plate_id']}: non-finite force data"
            )

        if np.any(np.diff(plate["time_s"]) <= 0):
            raise ValueError("Non-increasing timestamps")

    return True



def detect_contact_events(
    plate,
    threshold_n=20.0,
    min_duration_s=0.10,
    vertical_axis=2,
    vertical_sign=1
):
    """
    Detect ground-contact intervals from vertical force.

    Assumes vertical_axis and vertical_sign have been
    independently verified against laboratory conventions.
    """
    time = np.asarray(plate["time_s"], dtype=float)
    force = np.asarray(plate["force"], dtype=float)

    if len(time) < 2 or force.shape != (len(time), 3):
        raise ValueError("Invalid force/time arrays")

    if vertical_axis not in (0, 1, 2):
        raise ValueError("Invalid vertical axis")

    if vertical_sign not in (-1, 1):
        raise ValueError("vertical_sign must be -1 or 1")

    fz = vertical_sign * force[:, vertical_axis]

    contact = np.isfinite(fz) & (fz > threshold_n)

    transitions = np.diff(
        np.r_[False, contact, False].astype(int)
    )

    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    dt = np.median(np.diff(time))
    events = []

    for start, end in zip(starts, ends):
        duration_s = (end - start) * dt

        if duration_s < min_duration_s:
            continue

        events.append({
            "plate_id": plate["plate_id"],
            "start_s": float(time[start]),
            "end_s": float(
                time[end] if end < len(time)
                else time[-1] + dt
            ),
            "duration_s": float(duration_s),
            "peak_fz_n": float(np.max(fz[start:end])),
        })

    return events

def _strict_time(values, name):
    t = np.asarray(values, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: expected finite, strictly increasing timestamps")
    return t


def _inside_plate(xy, corners):
    """Convex footprint test, including its boundary; coordinates share units."""
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (3, 4) or not np.isfinite(corners).all():
        raise ValueError("Plate corners must have shape (3, 4), in the marker frame/units")
    p = corners[:2].T
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0]))]
    edge = np.roll(p, -1, axis=0) - p
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    if abs(area2) <= np.finfo(float).eps:
        raise ValueError("Degenerate plate footprint")
    d = np.asarray(xy)[:, None, :] - p[None, :, :]
    cross = edge[None, :, 0] * d[:, :, 1] - edge[None, :, 1] * d[:, :, 0]
    tolerance = 1e-10 * np.max(np.linalg.norm(edge, axis=1)) ** 2
    return np.all(cross >= -tolerance, axis=1) | np.all(cross <= tolerance, axis=1)


def assign_foot_contacts(
    contact_events, platforms, foot_markers, marker_time_s,
    threshold=0.75, min_valid_fraction=0.75, min_valid_frames=3,
):
    """Assign each [start_s, end_s) plate event to L/R using foot-centre XY.

    platforms: dictionaries with plate_id and corners (3, 4).
    foot_markers: LHEEL/LTOE/RHEEL/RTOE arrays (n_marker_frames, 3).
    Markers and corners MUST share a laboratory frame and length units.
    Marker times and event times MUST share the same recording origin.

    Occupancy is the fraction of valid heel/toe midpoint samples in the plate.
    Both feet need sufficient observations; missing contralateral data cannot
    rule out a shared/ambiguous plate. This is assignment evidence, not proof
    that the entire supporting foot is on a plate or all support is measured.
    Returns one DataFrame row per event, preserving original event fields.
    """
    import pandas as pd
    if not 0 < threshold <= 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError("Fractions must be in (0, 1]")
    if not isinstance(min_valid_frames, (int, np.integer)) or min_valid_frames < 1:
        raise ValueError("min_valid_frames must be a positive integer")
    t = _strict_time(marker_time_s, "marker_time_s")
    lookup = {p["plate_id"]: p for p in platforms}
    if len(lookup) != len(platforms):
        raise ValueError("Duplicate plate IDs")
    centers = {}
    for side in ("L", "R"):
        heel = np.asarray(foot_markers[f"{side}HEEL"], dtype=float)
        toe = np.asarray(foot_markers[f"{side}TOE"], dtype=float)
        if heel.shape != (len(t), 3) or toe.shape != heel.shape:
            raise ValueError("Each foot marker must have shape (n_marker_frames, 3)")
        centers[side] = (heel + toe) / 2
    events = contact_events.to_dict("records") if hasattr(contact_events, "to_dict") else contact_events
    rows = []
    for index, event in enumerate(events):
        start, end = float(event["start_s"]), float(event["end_s"])
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError("Invalid event interval")
        plate = lookup[event["plate_id"]]
        mask = (t >= start) & (t < end)
        row = dict(event)
        row.update(event_id=event.get("event_id", f"event_{index + 1:04d}"),
                   assigned_foot="unresolved", assignment_valid=False,
                   assignment_reason="insufficient_marker_coverage", n_marker_frames=int(mask.sum()))
        reliable = {}
        for side, label in (("L", "left"), ("R", "right")):
            xyz = centers[side][mask]
            valid = np.isfinite(xyz).all(axis=1)
            count = int(valid.sum())
            coverage = count / len(xyz) if len(xyz) else 0.0
            fraction = float(_inside_plate(xyz[valid, :2], plate["corners"]).mean()) if count else np.nan
            row[f"{label}_fraction_inside"] = fraction
            row[f"{label}_valid_fraction"] = coverage
            row[f"{label}_valid_frames"] = count
            reliable[side] = count >= min_valid_frames and coverage >= min_valid_fraction
        if all(reliable.values()) and start >= t[0] and end <= t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 0):
            candidates = [s for s, label in (("L", "left"), ("R", "right"))
                          if row[f"{label}_fraction_inside"] >= threshold]
            if len(candidates) == 1:
                row.update(assigned_foot=candidates[0], assignment_valid=True,
                           assignment_reason="unique_spatial_candidate")
            elif len(candidates) == 2:
                row.update(assigned_foot="ambiguous", assignment_reason="both_feet_inside")
            else:
                row["assignment_reason"] = "neither_foot_meets_threshold"
        rows.append(row)
    columns = ["plate_id", "event_id", "start_s", "end_s", "assigned_foot",
               "assignment_valid", "assignment_reason", "n_marker_frames",
               "left_fraction_inside", "right_fraction_inside", "left_valid_fraction",
               "right_valid_fraction", "left_valid_frames", "right_valid_frames"]
    result = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    if result["event_id"].duplicated().any():
        raise ValueError("event_id values must be unique")
    return result


def build_bilateral_external_loads(
    measured_loads, assignments, *, conventions_confirmed=False,
    contact_threshold_n=20.0, surface_z_m=0.0, surface_tolerance_m=0.002,
):
    """Assemble measured L/R loads on the unchanged force-platform timebase.

    Input: notebook's SI, long-form measured_external_loads DataFrame:
      plate_id, time_s, Fx_N/Fy_N/Fz_N, CoP_x_m/CoP_y_m/CoP_z_m,
      Tz_x_Nm/Tz_y_Nm/Tz_z_Nm (a free-moment VECTOR at each plate CoP).
    Raw Mx/My/Mz columns are intentionally NOT used: their reference point
    may differ. Confirm that force is ground-on-body, lab Z is upward,
    all vectors/CoPs share the lab frame and clocks, and free moments are
    represented at CoP. This function does not infer or fix conventions.

    For each assigned contact: M_origin = CoP cross F + T_free.
    Sum forces and origin moments. On the declared common horizontal surface,
    use Fz-weighted CoP, then T_at_CoP = M_origin - CoP cross F_total.
    T_at_CoP is a 3-vector and is retained without discarding components.

    Unknown/ambiguous loaded plates invalidate BOTH sides conservatively.
    Outside assigned events: zero *assembled* force, unavailable CoP; this
    does not assert swing or zero real-world force. No support-completeness
    claim is made. All event intervals are half open [start_s, end_s).
    """
    import pandas as pd
    if conventions_confirmed is not True:
        raise ValueError("Verify SI units, lab Z-up frame, force/free-moment conventions and clocks first")
    if (not np.isfinite([contact_threshold_n, surface_z_m, surface_tolerance_m]).all()
            or contact_threshold_n <= 0 or surface_tolerance_m < 0):
        raise ValueError("Invalid force threshold or support surface")
    force_cols = ["Fx_N", "Fy_N", "Fz_N"]
    cop_cols = ["CoP_x_m", "CoP_y_m", "CoP_z_m"]
    free_cols = ["Tz_x_Nm", "Tz_y_Nm", "Tz_z_Nm"]
    required = ["plate_id", "time_s"] + force_cols + cop_cols + free_cols
    missing = set(required) - set(measured_loads.columns)
    if missing:
        raise ValueError(f"Missing measured-load columns: {sorted(missing)}")
    data = measured_loads.copy()
    if data.empty or data["plate_id"].isna().any():
        raise ValueError("No measured plates or missing plate IDs")
    groups = {pid: group.sort_values("time_s") for pid, group in data.groupby("plate_id")}
    time = None
    for pid, group in groups.items():
        t = _strict_time(group["time_s"], f"plate {pid}")
        if time is None:
            time = t
        elif not np.array_equal(t, time):
            raise ValueError("Plate timebases must match exactly; do not interpolate here")
    n = len(time)
    events = assignments.to_dict("records") if hasattr(assignments, "to_dict") else list(assignments)
    ids = [e["event_id"] for e in events]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate event IDs")
    owner = {pid: np.full(n, "", dtype=object) for pid in groups}
    event_at = {pid: np.full(n, "", dtype=object) for pid in groups}
    for e in events:
        pid = e["plate_id"]
        if pid not in groups:
            raise ValueError(f"Assignment references absent plate {pid}")
        start, end = float(e["start_s"]), float(e["end_s"])
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError("Invalid event interval")
        mask = (time >= start) & (time < end)
        if np.any(event_at[pid][mask] != ""):
            raise ValueError(f"Overlapping events on plate {pid}")
        side = e["assigned_foot"]
        if side not in ("L", "R", "ambiguous", "unresolved"):
            raise ValueError(f"Unknown assigned_foot: {side}")
        accepted = e.get("assignment_valid", side in ("L", "R"))
        if not isinstance(accepted, (bool, np.bool_)):
            raise ValueError("assignment_valid must contain booleans (parse CSV values explicitly)")
        owner[pid][mask] = side if accepted and side in ("L", "R") else "unresolved"
        event_at[pid][mask] = str(e["event_id"])
    sums = {s: dict(force=np.zeros((n, 3)), moment=np.zeros((n, 3)),
                    weighted_cop=np.zeros((n, 3)), weight=np.zeros(n),
                    bad=np.zeros(n, bool), plates=[[] for _ in range(n)],
                    events=[[] for _ in range(n)]) for s in ("L", "R")}
    unresolved = [[] for _ in range(n)]
    for pid, group in groups.items():
        f = group[force_cols].to_numpy(float)
        p = group[cop_cols].to_numpy(float)
        free = group[free_cols].to_numpy(float)
        finite_force = np.isfinite(f).all(axis=1)
        unresolved_mask = (owner[pid] == "unresolved") | (
            (owner[pid] == "") & (~finite_force | (np.abs(f[:, 2]) > contact_threshold_n)))
        for i in np.flatnonzero(unresolved_mask):
            unresolved[i].append(str(pid))
        # Finite CoP/free moment required only for samples being assigned.
        for side in ("L", "R"):
            selected = owner[pid] == side
            if not selected.any():
                continue
            good = (finite_force & np.isfinite(p).all(axis=1) & np.isfinite(free).all(axis=1)
                    & (f[:, 2] >= 0) & (np.abs(p[:, 2] - surface_z_m) <= surface_tolerance_m))
            dest = sums[side]
            dest["bad"] |= selected & ~good
            valid = selected & good
            dest["force"][valid] += f[valid]
            dest["moment"][valid] += np.cross(p[valid], f[valid]) + free[valid]
            dest["weighted_cop"][valid] += p[valid] * f[valid, 2, None]
            dest["weight"][valid] += f[valid, 2]
            for i in np.flatnonzero(selected):
                dest["plates"][i].append(str(pid))
                dest["events"][i].append(event_at[pid][i])
    unknown = np.array([bool(v) for v in unresolved])
    tables = []
    for side in ("L", "R"):
        d = sums[side]
        bad = d["bad"] | unknown
        f, m0 = d["force"], d["moment"]
        cop = np.full((n, 3), np.nan)
        loaded = (d["weight"] > contact_threshold_n) & ~bad
        cop[loaded] = d["weighted_cop"][loaded] / d["weight"][loaded, None]
        # Use the declared common surface for the resultant application point.
        cop[loaded, 2] = surface_z_m
        torque = np.full((n, 3), np.nan)
        torque[loaded] = m0[loaded] - np.cross(cop[loaded], f[loaded])
        f[bad] = np.nan
        m0[bad] = np.nan
        status = np.full(n, "no_assigned_contact", dtype=object)
        has_event = np.array([bool(v) for v in d["plates"]])
        status[has_event] = "below_cop_threshold"
        status[loaded] = "assigned_contact"
        status[d["bad"]] = "invalid_assigned_load"
        status[unknown] = "unresolved_plate_load"
        table = pd.DataFrame({"time_s": time, "side": side,
            **{c: f[:, j] for j, c in enumerate(force_cols)},
            **{c: cop[:, j] for j, c in enumerate(cop_cols)},
            **{f"T{axis}_Nm": torque[:, j] for j, axis in enumerate("xyz")},
            **{f"M0{axis}_Nm": m0[:, j] for j, axis in enumerate("xyz")},
            "assignment_valid": ~bad, "cop_valid": loaded, "status": status,
            "contributing_plate_ids": [";".join(v) for v in d["plates"]],
            "event_ids": [";".join(v) for v in d["events"]],
            "unresolved_plate_ids": [";".join(v) for v in unresolved]})
        tables.append(table)
    result = pd.concat(tables, ignore_index=True).sort_values(["time_s", "side"]).reset_index(drop=True)
    result.attrs.update(coordinate_frame="laboratory_Z_up", force_unit="N", position_unit="m",
                        moment_unit="Nm", moment_reference="T at resultant CoP; M0 at lab origin",
                        surface_z_m=float(surface_z_m), contact_threshold_n=float(contact_threshold_n),
                        complete_support_confirmed=False)
    return result


def export_external_loads(bilateral_loads, output_path, *, assignments=None):
    """Write combined CSV, optional assignment CSV, and convention metadata JSON.

    output_path is the combined CSV filename, not a directory. Returns paths.
    Invalid samples stay NaN; full native timestamps and status flags are kept.
    """
    import json
    import pandas as pd
    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must be a .csv filename")
    required = {"time_s", "side", "assignment_valid", "cop_valid", "status",
                "contributing_plate_ids", "event_ids", "unresolved_plate_ids",
                "Fx_N", "Fy_N", "Fz_N", "CoP_x_m", "CoP_y_m", "CoP_z_m",
                "Tx_Nm", "Ty_Nm", "Tz_Nm", "M0x_Nm", "M0y_Nm", "M0z_Nm"}
    if not required <= set(bilateral_loads.columns):
        raise ValueError("Expected build_bilateral_external_loads output")
    if not bilateral_loads.attrs:
        raise ValueError("Missing convention metadata; export the builder output directly")
    path.parent.mkdir(parents=True, exist_ok=True)
    bilateral_loads.to_csv(path, index=False)
    metadata = path.with_suffix(".metadata.json")
    metadata.write_text(json.dumps(bilateral_loads.attrs, indent=2) + "\n")
    outputs = {"loads": path, "metadata": metadata}
    if assignments is not None:
        assignment_path = path.with_name(path.stem + "_assignments.csv")
        pd.DataFrame(assignments).to_csv(assignment_path, index=False)
        outputs["assignments"] = assignment_path
    return outputs


def plot_force_plate_data(force_plate_data):
    """
    Plot the force plate data for visualization.

    Parameters:
    - force_plate_data: dict, containing time, force, and moment data.
    """
    import matplotlib.pyplot as plt

    time = force_plate_data['time']
    force = force_plate_data['force']
    moment = force_plate_data['moment']

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, force[:, 0], label='Force X')
    plt.plot(time, force[:, 1], label='Force Y')
    plt.plot(time, force[:, 2], label='Force Z')
    plt.title('Force Plate Forces')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, moment[:, 0], label='Moment X')
    plt.plot(time, moment[:, 1], label='Moment Y')
    plt.plot(time, moment[:, 2], label='Moment Z')
    plt.title('Force Plate Moments')
    plt.xlabel('Time (s)')
    plt.ylabel('Moment (Nm)')
    plt.legend()

    plt.tight_layout()
    plt.show()
