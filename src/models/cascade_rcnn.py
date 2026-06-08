"""
Cascade R-CNN via MMDetection (OpenMMLab).

Instalacao:
    pip install torch torchvision  # ja instalado
    pip install mmengine
    pip install mmcv -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    pip install mmdet

Baixar pesos automaticamente via mim:
    pip install openmim
    mim download mmdet --config cascade-rcnn_r50_fpn_1x_coco --dest weights/

Ou manualmente:
    Config:      https://github.com/open-mmlab/mmdetection/blob/main/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py
    Checkpoint:  https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56dde.pth
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from core.metrics import is_stopped, compute_metrics
from core.tracker import CentroidTracker

_ROOT = Path(__file__).parents[2]
_CONFIG     = _ROOT / "weights" / "cascade-rcnn_r50_fpn_1x_coco.py"
_CHECKPOINT = _ROOT / "weights" / "cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56dde.pth"


def run(video_path: str, gt_binary: list, conf_threshold: float = 0.5, skip_frames: int = 1) -> dict:
    import torch
    from mmdet.apis import init_detector, inference_detector

    if not _CONFIG.exists() or not _CHECKPOINT.exists():
        raise FileNotFoundError(
            "Pesos do Cascade R-CNN nao encontrados em weights/.\n"
            "Execute: mim download mmdet --config cascade-rcnn_r50_fpn_1x_coco --dest weights/"
        )

    print("  Carregando Cascade R-CNN...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(str(_CONFIG), str(_CHECKPOINT), device=device)

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

        t0 = time.perf_counter()
        result = inference_detector(model, frame)
        frame_times.append(time.perf_counter() - t0)

        # mmdet 3.x: result.pred_instances (labels sao 0-indexed, pessoa = 0)
        instances = result.pred_instances
        labels = instances.labels.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        bboxes = instances.bboxes.cpu().numpy()

        centroids, centroid_scores = [], {}
        for label, score, box in zip(labels, scores, bboxes):
            if label == 0 and score >= conf_threshold:  # person = 0 (COCO 80 classes)
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
    return compute_metrics(gt_binary, preds, frame_times, confidences, "Cascade R-CNN")
