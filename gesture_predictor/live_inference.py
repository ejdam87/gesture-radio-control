import time
import json
import sys

import cv2
import torch
import numpy as np
import pandas as pd
import mediapipe as mp

from hand_graph_extraction.get_hand_graph import get_detector, process_image_with_detector
from hand_graph_extraction.normalize_hand_graph import rescale_landmarks, find_corners
from hand_graph_extraction.constants import LANDMARKS_COUNT

from gesture_predictor.models.stupid_net import StupidNet

from utils.persistency import load_model

FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_model(model_path: str, device) -> torch.nn.Module:
    model = StupidNet(14).to(device)
    load_model(model, model_path)
    model.eval()
    return model


def get_pred(model: torch.nn.Module, im: mp.Image, device, detector) -> int:
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


def run(source, model_path, labels_path, device) -> None:
    cap = cv2.VideoCapture(source)
    t1 = frame_cnt = 0

    detector = get_detector()
    model = get_model(model_path, device)

    with open(labels_path, "r") as f:
        labels = json.load(f)["inv_labels"]

    while True:
        delta = time.time() - t1
        t1 = time.time()

        ret, frame = cap.read()
        if ret:

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            frame = frame_rgb
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            class_id = get_pred(model, im, device, detector)

            cv2.putText(
                frame, labels[str(class_id)], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3
            )
            fps = 1 / delta
            frame_cnt += 1
            cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {frame_cnt}", (30, 30), FONT, 1, (255, 0, 255), 2)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                return
        else:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    source = sys.argv[1]
    model = sys.argv[2]
    labels = sys.argv[3]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run(source, model, labels, device)
