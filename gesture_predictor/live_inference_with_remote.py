import time
import json
import sys
import spidev

import cv2
import torch
import numpy as np
import pandas as pd
import mediapipe as mp
from picamera2 import Picamera2

from hand_graph_extraction.get_hand_graph import get_detector, process_image_with_detector
from hand_graph_extraction.normalize_hand_graph import rescale_landmarks, find_corners
from hand_graph_extraction.constants import LANDMARKS_COUNT

from gesture_predictor.models.stupid_net import StupidNet

from utils.persistency import load_model

FONT = cv2.FONT_HERSHEY_SIMPLEX

def set_resistor_remote(value):
    if not (0 <= value <= 255):
        raise ValueError("Value must be 0-255")
    command = [0x11, value]  # MCP41100: write to pot0
    spi.xfer2(command)

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
    t1 = frame_cnt = 0

    # Initialize SPI
    spi = spidev.SpiDev()
    spi.open(0, 0)  # Bus 0, Device 0 (CE0 = GPIO8)
    spi.max_speed_hz = 1000000  # 1 MHz

    # Initialize detector model
    detector = get_detector()
    model = get_model(model_path, device)

    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start()

    with open(labels_path, "r") as f:
        labels = json.load(f)["inv_labels"]

    commands = {
        "source/off": 3, # 1.2kΩ
        "att/mute": 8, # 3.3kΩ
        "display": 15, # 5.57kΩ
        "tune_up": 21, # 8kΩ
        "tune_down": 29, # 11.25kΩ
        "volume_up": 62, # 24kΩ
        "volume_down": 160, # 62.5kΩ
        "band/escape": 230, # 90kΩ
        "no_command": 0 # 0kΩ
    }

    gestures_mapping = {
        "palm": "source/off",
        "ok": "att/mute",
        "palm": "display",
        "peace_inv": "tune_up",
        "two_up_inv": "tune_down",
        "like": "volume_up",
        "rock": "volume_down",
        "one": "band/escape",
        "no_gesture": "no_command"
    }

    last_gestures = []

    while True:
        delta = time.time() - t1
        t1 = time.time()

        frame = picam2.capture_array()

        im = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        class_id = get_pred(model, im, device, detector)

        # Most common gesture over last 10 detections (~1.4 seconds to recognize a gesture)
        last_gestures.append(class_id)
        if len(last_gestures) == 10:
            most_common_class = max(set(last_gestures), key=last_gestures.count)
            # Goes through the whole mapping path - just for demonstration purposes,
            # mapping class ids directly to resistance values is of course more streamlined
            resistance = commands[gestures_mapping[labels[str(class_id)]]]
            set_resistor_remote(resistance)
            last_gestures = []

        cv2.putText(
            frame, labels[str(class_id)], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3
        )
        fps = 1 / delta
        frame_cnt += 1
        cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {frame_cnt}", (30, 30), FONT, 1, (255, 0, 255), 2)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            spi.close()
            return


if __name__ == "__main__":
    source = sys.argv[1]
    model = sys.argv[2]
    labels = sys.argv[3]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run(source, model, labels, device)
