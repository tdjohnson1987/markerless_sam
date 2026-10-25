#Angle Generation and Kinematics
def static_offset_calibrate(
    time_ms, vicon, sam, window_ms=1500
):
    time_ms = np.asarray(time_ms)
    vicon = np.asarray(vicon)
    sam = np.asarray(sam)

    static = (
        (time_ms <= time_ms[0] + window_ms)
        & np.isfinite(vicon)
        & np.isfinite(sam)
    )

    if static.sum() < 3:
        raise ValueError("Not enough valid static samples")

    offset = np.median(vicon[static]) - np.median(sam[static])

    return sam + offset, offset

