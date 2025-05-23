""" Radio control (designed for raspberry device). """
import time
import json
import sys
from typing import Any

import cv2
import torch
from torchvision.transforms import v2

# uncomment when on raspberry
# from picamera2 import Picamera2
# import spidev

from cnn_predictor.inference import cnn_model, cnn_pred
from mediapipe_predictor.inference import mediapipe_model, mediapipe_pred

from hand_graph_extraction.get_hand_graph import get_detector


FONT = cv2.FONT_HERSHEY_SIMPLEX
MEDIAPIPE = True # whether to run mediapipe or CNN (False means CNN)
PICAM_SOURCE = False # where to get video from (Picam -> True, Video file -> False)
GRAYSCALE = True # grayscale inference ?


def set_resistor_remote(spi: Any, value: int) -> None:
    if not (0 <= value <= 255):
        raise ValueError("Value must be 0-255")
    command = [0x11, value]  # MCP41100: write to pot0
    spi.xfer2(command)


def run(
        source: str, model_path: str, stats_path: str,
        labels_path: str, device: torch.device
    ) -> None:

    t1 = frame_cnt = 0
    model = mediapipe_model(model_path, device) if MEDIAPIPE else cnn_model(model_path, device)
    if MEDIAPIPE:
        detector = get_detector()

    if not MEDIAPIPE:
        with open(stats_path, "r") as f:
            stats = json.load(f)
            means = stats["means"]
            stds = stats["stds"]

        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=means, std=stds)
            ]
        )

    # # Initialize SPI
    # spi = spidev.SpiDev()
    # spi.open(0, 0)  # Bus 0, Device 0 (CE0 = GPIO8)
    # spi.max_speed_hz = 1000000  # 1 MHz

    if PICAM_SOURCE:
        # Initialize camera
        # picam2 = Picamera2()
        # picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
        # picam2.start()
        ...
    else:
        # if classifying the video
        cap = cv2.VideoCapture(source)

    # --- Radio commands mappings
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
    # ---

    last_gestures = []

    with open(labels_path, "r") as f:
        labels = json.load(f)["inv_labels"]

    while True:
        delta = time.time() - t1
        t1 = time.time()

        if PICAM_SOURCE:
            # frame = picam2.capture_array()
            ...
        else:
            _, frame = cap.read()
            if GRAYSCALE:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        class_id = mediapipe_pred(model, frame, device, detector) if MEDIAPIPE else cnn_pred(model, frame, transforms, device)
        # Most common gesture over last 10 detections (~1.4 seconds to recognize a gesture)
        last_gestures.append(class_id)
        if len(last_gestures) == 10:
            most_common_class = max(set(last_gestures), key=last_gestures.count)
            if most_common_class not in commands:
                continue
            # Goes through the whole mapping path - just for demonstration purposes,
            # mapping class ids directly to resistance values is of course more streamlined
            resistance = commands[gestures_mapping[labels[str(class_id)]]]
            # set_resistor_remote(spi, resistance)
            print("resistance set:", resistance)
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
            # spi.close()
            return


if __name__ == "__main__":
    source = sys.argv[1]
    model = sys.argv[2]
    stats = sys.argv[3] # if not present, pass ""
    labels = sys.argv[4]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run(source, model, stats, labels, device)
