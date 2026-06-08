"""
Conditional DETR via Hugging Face Transformers.

Melhoria sobre o DETR original: spatial queries condicionais que aceleram
a convergencia em ~10x mantendo a arquitetura encoder-decoder.

Instalacao: pip install transformers Pillow  (ja instalado)
"""
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
    from PIL import Image
    from transformers import (
        ConditionalDetrForObjectDetection,
        ConditionalDetrImageProcessor,
    )

    print("  Carregando Conditional DETR (microsoft/conditional-detr-resnet-50)...")
    processor = ConditionalDetrImageProcessor.from_pretrained(
        "microsoft/conditional-detr-resnet-50"
    )
    model = ConditionalDetrForObjectDetection.from_pretrained(
        "microsoft/conditional-detr-resnet-50"
    ).eval()
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

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = {k: v.to(device) for k, v in
                  processor(images=pil_img, return_tensors="pt").items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        frame_times.append(time.perf_counter() - t0)

        target_sizes = torch.tensor([pil_img.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=conf_threshold, target_sizes=target_sizes
        )[0]

        centroids, centroid_scores = [], {}
        for score, label, box in zip(results["scores"].cpu(),
                                     results["labels"].cpu(),
                                     results["boxes"].cpu()):
            if label.item() == 1:  # person = 1 em COCO
                x1, y1, x2, y2 = box.tolist()
                c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
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
    return compute_metrics(gt_binary, preds, frame_times, confidences, "Conditional DETR")
