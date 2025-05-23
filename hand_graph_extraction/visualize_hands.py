""" Visualization of the hand graphs. """
import sys
import cv2
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def visualize():
    """Visualize hand nodes on an imgages.

    Dataframe should have non-normalized coords.

    Example usage:

    python ./visualize_hands ./directory/of/images ./dataframe.parquet

    """
    if len(sys.argv) != 3:
        raise ValueError("Enter input image path and dataframe path.")

    img_paths = Path(sys.argv[1])
    df_path = Path(sys.argv[2])

    for img_path in img_paths.glob("*"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        df = pd.read_parquet(df_path)
        landmarks = df[df["path"] == str(img_path.name)]
        if len(landmarks) == 0:
            continue

        landmarks_list = list(landmarks.iloc[0])[2:]
        points_x = []
        points_y = []
        for i in range(21):
            points_x.append(landmarks_list[i * 4] * img.shape[1])
            points_y.append(landmarks_list[i * 4 + 1] * img.shape[0])

        plt.imshow(img)
        plt.scatter(points_x, points_y, c='green', s=40, marker='o')

        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    visualize()
