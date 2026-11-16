
from pathlib import Path
import pandas as pd


def load_subject_metadata(subject, root=None):
    """
    Load anthropometric metadata for a subject.

    Example:
        load_subject_metadata("S11")
    """

    if root is None:
        root = Path(__file__).resolve().parents[1]

    metadata_path = (
        Path(root) / "data" / "metadata" / "subjects.csv"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    df = pd.read_csv(metadata_path)

    required = {"subject", "mass_kg", "height_m"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    matches = df.loc[
        df["subject"].astype(str).str.strip() == subject
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one metadata entry for {subject}, "
            f"found {len(matches)}"
        )

    row = matches.iloc[0]

    mass = float(row["mass_kg"])
    height = float(row["height_m"])

    if not (20 <= mass <= 300):
        raise ValueError(f"Invalid mass for {subject}")

    if not (1.0 <= height <= 2.5):
        raise ValueError(f"Invalid height for {subject}")

    return {
        "subject": subject,
        "mass_kg": mass,
        "height_m": height,
        "body_weight_n": mass * 9.80665,
    }
