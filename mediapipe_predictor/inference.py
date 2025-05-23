""" Single image prediction. """
from mediapipe_predictor.models.linear_net import LinearNet

from hand_graph_extraction.get_hand_graph import process_image_with_detector, get_detector
from hand_graph_extraction.normalize_hand_graph import rescale_landmarks, find_corners
from hand_graph_extraction.constants import LANDMARKS_COUNT

from utils.persistency import load_model
from utils.pred_vis import visualize

from numpy.typing import NDArray
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
import torch
import mediapipe as mp
import pandas as pd
import cv2

import sys
from pathlib import Path


def mediapipe_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = LinearNet(14).to(device)
    load_model(model, model_path)
    model.eval()
    return model

def mediapipe_pred(model: torch.nn.Module, frame: NDArray, device: torch.device, detector: HandLandmarker) -> int:
    im = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    raw_landmarks = process_image_with_detector(im, detector)
    if not raw_landmarks:
        return 5

    columns = ["path", "hand"] + [f"{axis}{i}" for i in range(21) for axis in "xyzc"]
    df = pd.DataFrame([raw_landmarks], columns=columns)
    df[["x_min", "x_max", "y_min", "y_max"]] = df.apply(find_corners, axis=1)
    columns = [f"{axis}{i}" for i in range(LANDMARKS_COUNT) for axis in "xy"]
    df[columns] = df.apply(rescale_landmarks, axis=1)
    norm_landmarks = df.iloc[0]
    norm_landmarks = norm_landmarks.filter(regex=r'^[xyzc]\d|hand')
    norm_landmarks = torch.tensor(list(norm_landmarks), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(norm_landmarks)

    class_id = pred.argmax(1).item()
    return class_id


def run_model(model_path: str, im_path: str, device: torch.device) -> int:
    model = mediapipe_model(model_path, device)
    frame = cv2.imread(im_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = get_detector()
    return mediapipe_pred(model, frame, device, detector)


if __name__ == "__main__":
    model_path = sys.argv[1]
    im_dir = sys.argv[2]
    device = torch.device("cpu")

    for im_path in Path(im_dir).glob("*"):
        pred = run_model(model_path, im_path, device)
        visualize(im_path, pred)
