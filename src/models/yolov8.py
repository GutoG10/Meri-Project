import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from core.metrics import is_stopped, compute_metrics

_ROOT = Path(__file__).parents[2]


def run(video_path: str, gt_binary: list, conf_threshold: float = 0.5, skip_frames: int = 1) -> dict:
    from ultralytics import YOLO

    print("  Carregando YOLOv8...")
    weights = _ROOT / "weights" / "yolov8s.pt"
    if not weights.exists():
        weights = _ROOT / "yolov8s.pt"
    model = YOLO(str(weights))

    history: dict = {}
    cap = cv2.VideoCapture(video_path)
    frame_times, confidences, preds = [], [], []

    frame_idx = 0
    while True:
        if skip_frames > 1 and frame_idx % skip_frames != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        results = model.track(frame, classes=[0], persist=True, verbose=False)
        frame_times.append(time.perf_counter() - t0)

        frame_stopped = False
        if results[0].boxes.id is not None:
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for track_id, box, score in zip(ids, boxes, confs):
                if score >= conf_threshold:
                    confidences.append(float(score))
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    if is_stopped(history, track_id, (cx, cy)):
                        frame_stopped = True

        preds.append(1 if frame_stopped else 0)

    cap.release()
    return compute_metrics(gt_binary, preds, frame_times, confidences, "YOLOv8")
