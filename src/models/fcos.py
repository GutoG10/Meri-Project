import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from core.metrics import is_stopped, compute_metrics
from core.tracker import CentroidTracker


def run(video_path: str, gt_binary: list, conf_threshold: float = 0.5, skip_frames: int = 1) -> dict:
    import torch
    from torchvision.models.detection import (
        FCOS_ResNet50_FPN_Weights,
        fcos_resnet50_fpn,
    )
    from torchvision.transforms import functional as TF

    print("  Carregando FCOS...")
    weights = FCOS_ResNet50_FPN_Weights.COCO_V1
    model = fcos_resnet50_fpn(weights=weights).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history: dict = {}
    tracker = CentroidTracker()
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

        tensor = TF.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model([tensor])[0]
        frame_times.append(time.perf_counter() - t0)

        centroids, centroid_scores = [], {}
        for label, score, box in zip(pred["labels"].cpu().numpy(),
                                     pred["scores"].cpu().numpy(),
                                     pred["boxes"].cpu().numpy()):
            if label == 1 and score >= conf_threshold:
                x1, y1, x2, y2 = box.astype(int)
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                centroids.append(c)
                centroid_scores[c] = float(score)

        tracked = tracker.update(centroids)

        frame_stopped = False
        for track_id, centroid in tracked.items():
            confidences.append(centroid_scores.get(centroid, conf_threshold))
            if is_stopped(history, track_id, centroid):
                frame_stopped = True

        preds.append(1 if frame_stopped else 0)

    cap.release()
    return compute_metrics(gt_binary, preds, frame_times, confidences, "FCOS")
