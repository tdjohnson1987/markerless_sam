
from pathlib import Path
import numpy as np
import ezc3d


def load_force_plates(c3d_path):
    """
    Load calibrated force-platform outputs from Vicon C3D.

    Returns a dictionary containing native analog timestamps,
    per-platform force, moment, CoP and free moment.

    Coordinate frames and units must be verified before
    using these outputs for inverse dynamics.
    """
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

# Placeholder for assigning foot contacts to vicon foot trajectories, replace with spatial assignment logic 
def assign_foot_contacts(contact_events, joint_data, joint_names):
    """
    Assign detected foot contact events to vicon foot trajectories
    """
    foot_contacts = {joint: [] for joint in joint_names}

    for event in contact_events:
        start_time = event["start_s"]
        end_time = event["end_s"]
        for joint in joint_names:
            joint_times = joint_data[joint]['time']
            if np.any((joint_times >= start_time) & (joint_times <= end_time)):
                foot_contacts[joint].append((start_time, end_time))

    return foot_contacts

#Placeholder for exporting external loads to CSV files
def export_external_loads(foot_contacts, output_path):
    """
    Save 3D GRF, CoP and free moment per plate/contact to CSV files for each foot contact event."""

    for joint, contacts in foot_contacts.items():
        for i, (start_time, end_time) in enumerate(contacts):
            filename = f"{output_path}/{joint}_contact_{i+1}.csv"
            # Implement the logic to save the relevant data to CSV

            # For example, extract the force and moment data for the time range
            # and save it to the file. 
            print(f"Exporting contact event for {joint} from {start_time} to {end_time} to {filename}")


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
