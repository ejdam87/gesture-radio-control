import cv2
import torch
import time
import json
import sys
from torchvision.transforms import v2
from torchsummary import summary

from cnn_predictor.models.classifier import Classifier
from cnn_predictor.models.feature_extractor import ResNet18
from cnn_predictor.models.gesture_cnn import GestureCNN
from utils.persistency import load_model

FONT = cv2.FONT_HERSHEY_SIMPLEX

def run(source, model_path, stats_path, labels_path, device) -> None:
    cap = cv2.VideoCapture(source)
    t1 = frame_cnt = 0
    model = GestureCNN(ResNet18(), Classifier(5))
    load_model(model, model_path)
    model = model.to(device)
    model.eval()
    summary(model, (3,320,240))

    while True:
        delta = time.time() - t1
        t1 = time.time()

        ret, frame = cap.read()
        if ret:
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

            im = transforms(frame)

            with torch.no_grad():
                pred = model(im.unsqueeze(0).to(device))
            
            class_id = pred.argmax(1).item()

            with open(labels_path, "r") as f:
                labels = json.load(f)["inv_labels"]

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
    stats = sys.argv[3]
    labels = sys.argv[4]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run(source, model, stats, labels, device)