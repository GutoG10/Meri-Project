"""
EfficientDet-D0 via biblioteca effdet.

Instalacao:
    pip install effdet timm
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from core.metrics import is_stopped, compute_metrics
from core.tracker import CentroidTracker

_INPUT_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run(video_path: str, gt_binary: list, conf_threshold: float = 0.5, skip_frames: int = 1) -> dict:
    import torch
    from effdet import create_model

    print("  Carregando EfficientDet-D0...")
    model = create_model("efficientdet_d0", bench_task="predict", pretrained=True)
    model = model.eval()
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

        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(tensor)
        frame_times.append(time.perf_counter() - t0)

        # output: [1, max_det, 6] -> [y1, x1, y2, x2, score, class_id (1-indexed COCO)]
        dets = output[0].cpu().numpy()

        centroids, centroid_scores = [], {}
        for det in dets:
            y1, x1, y2, x2, score, class_id = det
            if score < conf_threshold:
                continue
            if int(round(class_id)) == 1:  # person = 1 (1-indexed COCO)
                xi1 = int(x1 * orig_w / _INPUT_SIZE)
                xi2 = int(x2 * orig_w / _INPUT_SIZE)
                yi1 = int(y1 * orig_h / _INPUT_SIZE)
                yi2 = int(y2 * orig_h / _INPUT_SIZE)
                c = ((xi1 + xi2) // 2, (yi1 + yi2) // 2)
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
    return compute_metrics(gt_binary, preds, frame_times, confidences, "EfficientDet")
