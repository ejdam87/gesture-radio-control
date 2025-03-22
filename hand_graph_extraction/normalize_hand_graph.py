import sys
from pathlib import Path 

import pandas as pd

from constants import LANDMARKS_COUNT


def get_landmarks_list(row: pd.Series) -> tuple[ list[float], list[float] ]:
    "Extract hand's x and y coordinates from dataframe."
    row = row.filter(regex=r'^[xy]\d')
    xs = [row[i * 2] for i in range(LANDMARKS_COUNT)]
    ys = [row[i * 2 + 1] for i in range(LANDMARKS_COUNT)]

    return xs, ys


def find_corners(row: pd.Series) -> pd.Series:
    """Find corners of a set of hand's landmarks."""
    xs, ys = get_landmarks_list(row)
    return pd.Series({"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)})


def rescale_landmarks(row: pd.Series) -> pd.Series:
    """Scale landmarks to bounding box."""
    xs, ys = get_landmarks_list(row)
    rescaled_landmarks = {}
    for i in range(LANDMARKS_COUNT):
        x = (xs[i] - row["x_min"]) / (row["x_max"] - row["x_min"])
        y = (ys[i] - row["y_min"]) / (row["y_max"] - row["y_min"])
        rescaled_landmarks[f"x{i}"] = x
        rescaled_landmarks[f"y{i}"] = y
    return pd.Series(rescaled_landmarks)


def normalize_hand():
    """Normalize hand landmarks.

    Normalize hand landmarks to [0,1]. Former format should be
    also in [0,1], but normalized to the whole source image,
    not to the bounding box of the hand.

    Example usage: python ./normalize_hand_graph.py ./non_normalized_dataframe.parquet ./normalized_dataframe.parquet
    """
    if len(sys.argv) != 3:
        raise ValueError("Enter input and output parquet paths.")

    df_path = Path(sys.argv[1])
    rescaled_df_path = Path(sys.argv[2])
    df = pd.read_parquet(df_path)
    df[["x_min", "x_max", "y_min", "y_max"]] = df.apply(find_corners, axis=1)
    columns = [f"{axis}{i}" for i in range(LANDMARKS_COUNT) for axis in "xy"]
    df[columns] = df.apply(rescale_landmarks, axis=1)
    df.to_parquet(rescaled_df_path)


if __name__ == "__main__":
    normalize_hand()
