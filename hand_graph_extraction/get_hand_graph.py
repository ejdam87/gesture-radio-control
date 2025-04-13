import sys
from pathlib import Path
from typing import Any
from multiprocessing import Pool

from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd


def process_image_from_path(img_path: Path) -> list[Any]:
    img = mp.Image.create_from_file(str(img_path))
    return process_image(img)

def process_image(img: mp.Image, img_path: Path | None=None) -> list[Any]:
    base_options = python.BaseOptions(model_asset_path='./hand_graph_extraction/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=1,
                                        min_hand_presence_confidence=0,
                                        min_tracking_confidence=0,
                                        min_hand_detection_confidence=0.1)
    detector = vision.HandLandmarker.create_from_options(options)
    results = detector.detect(img)

    if results.hand_landmarks:
        hand = 0 if results.handedness[0][0].category_name == "Left" else 1
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            confidence = results.hand_world_landmarks[i][0].z
            row = [str(img_path.name) if img_path else "", hand]

            for item in hand_landmarks:
                row.extend([item.x, item.y, item.z, confidence])
        return row


def get_landmarks() -> None:
    """Extract hand graph from images in a directory.

    Example usage: python ./get_hand_graph.py directory ./output_dataframe.parquet
    """
    if len(sys.argv) != 3:
        raise ValueError("Specify input and output paths!")

    path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    img_paths = list(path.glob("*"))

    with Pool(processes=2) as pool:  # Adjust based on CPU cores
        results = list(tqdm(pool.imap(process_image_from_path, img_paths), total=len(img_paths)))

    results = [x for x in results if x is not None]

    columns = ["path", "hand"] + [f"{axis}{i}" for i in range(21) for axis in "xyzc"]
    df = pd.DataFrame(results, columns=columns)

    df.to_parquet(out, index=False)
    print(f"Finished. Recognized {len(df)} hands from {len(img_paths)} images.")


if __name__ == "__main__":
    get_landmarks()
