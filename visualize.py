"""
Visualiza deteccoes frame a frame e salva screenshots anotados.

Uso:
    python visualize.py videos/mulher.mp4 yolov8
    python visualize.py videos/mulher.mp4 faster_rcnn --gt videos/mulher_gt.json
    python visualize.py videos/mulher.mp4 yolov8 --every-n 20 --output capturas/
    python visualize.py videos/mulher.mp4 yolov8 --video-out capturas/mulher_yolo.mp4

Modelos: yolov8  ssd  faster_rcnn  retinanet  fcos  efficientdet
         detr  rt_detr  conditional_detr  deformable_detr
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).parent / "src"
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from core.metrics import is_stopped, load_gt
from core.tracker import CentroidTracker

FONT     = cv2.FONT_HERSHEY_SIMPLEX
CLR_STOP = (0, 220, 0)      # verde  - pedestre parado
CLR_WALK = (0, 165, 255)    # laranja - detectado, andando
CLR_TEXT = (255, 255, 255)  # branco
CLR_PAN  = (25, 25, 25)     # fundo do painel


# --------------------------------------------------------------------------- #
# Desenho                                                                       #
# --------------------------------------------------------------------------- #

def _draw_box(frame, x1, y1, x2, y2, conf, stopped):
    color = CLR_STOP if stopped else CLR_WALK
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"PARADO {conf:.2f}" if stopped else f"pessoa {conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                FONT, 0.5, color, 1, cv2.LINE_AA)


def _draw_panel(frame, frame_idx, total, model_name, pred_stopped, gt_val):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), CLR_PAN, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, f"Frame {frame_idx}/{total}", (8, 32),
                FONT, 0.6, CLR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, model_name, (w // 2 - 80, 32),
                FONT, 0.6, CLR_TEXT, 1, cv2.LINE_AA)

    if gt_val is not None:
        gt_color = CLR_STOP if gt_val == 1 else (150, 150, 150)
        gt_text  = "GT: PARADO" if gt_val == 1 else "GT: NEGATIVO"
        cv2.putText(frame, gt_text, (w - 195, 18),
                    FONT, 0.52, gt_color, 1, cv2.LINE_AA)

    sem_color = CLR_STOP if pred_stopped else (60, 60, 220)
    sem_text  = "SEMAFORO: VERDE" if pred_stopped else "SEMAFORO: VERMELHO"
    y_sem = 44 if gt_val is not None else 32
    cv2.putText(frame, sem_text, (w - 212, y_sem),
                FONT, 0.52, sem_color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Detectores                                                                    #
# --------------------------------------------------------------------------- #

def _ultralytics_detector(model_cls, weights_path, conf):
    model = model_cls(str(weights_path))
    history: dict = {}

    def detect(frame):
        results = model.track(frame, classes=[0], persist=True, verbose=False)
        boxes, pred_stopped = [], False
        if results[0].boxes.id is not None:
            ids  = results[0].boxes.id.cpu().numpy().astype(int)
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for tid, box, sc in zip(ids, xyxy, confs):
                if sc < conf:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                stopped = is_stopped(history, int(tid), (cx, cy))
                if stopped:
                    pred_stopped = True
                boxes.append((x1, y1, x2, y2, float(sc), stopped))
        return boxes, pred_stopped

    return detect


def _torchvision_detector(model_factory, weights_factory, conf):
    import torch
    from torchvision.transforms import functional as TF

    weights = weights_factory.COCO_V1
    model   = model_factory(weights=weights).eval()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history: dict = {}
    tracker = CentroidTracker()

    def detect(frame):
        tensor = TF.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
        with torch.no_grad():
            pred = model([tensor])[0]

        centroids, scores_map, raw_boxes = [], {}, {}
        for label, score, box in zip(pred["labels"].cpu().numpy(),
                                     pred["scores"].cpu().numpy(),
                                     pred["boxes"].cpu().numpy()):
            if label == 1 and score >= conf:
                x1, y1, x2, y2 = box.astype(int)
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                centroids.append(c)
                scores_map[c] = float(score)
                raw_boxes[c]  = (x1, y1, x2, y2)

        tracked      = tracker.update(centroids)
        boxes        = []
        pred_stopped = False
        for tid, centroid in tracked.items():
            stopped = is_stopped(history, tid, centroid)
            if stopped:
                pred_stopped = True
            if centroid in raw_boxes:
                x1, y1, x2, y2 = raw_boxes[centroid]
                boxes.append((x1, y1, x2, y2, scores_map.get(centroid, conf), stopped))
        return boxes, pred_stopped

    return detect


def _efficientdet_detector(conf):
    import torch
    from effdet import create_model

    _IN  = 512
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model  = create_model("efficientdet_d0", bench_task="predict", pretrained=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history: dict = {}
    tracker = CentroidTracker()

    def detect(frame):
        oh, ow = frame.shape[:2]
        img = cv2.resize(frame, (_IN, _IN))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
        dets = output[0].cpu().numpy()

        centroids, scores_map, raw_boxes = [], {}, {}
        for det in dets:
            y1, x1, y2, x2, score, cls = det
            if score < conf or int(round(cls)) != 1:
                continue
            xi1 = int(x1 * ow / _IN); xi2 = int(x2 * ow / _IN)
            yi1 = int(y1 * oh / _IN); yi2 = int(y2 * oh / _IN)
            c   = ((xi1 + xi2) // 2, (yi1 + yi2) // 2)
            centroids.append(c)
            scores_map[c] = float(score)
            raw_boxes[c]  = (xi1, yi1, xi2, yi2)

        tracked      = tracker.update(centroids)
        boxes        = []
        pred_stopped = False
        for tid, centroid in tracked.items():
            stopped = is_stopped(history, tid, centroid)
            if stopped:
                pred_stopped = True
            if centroid in raw_boxes:
                x1, y1, x2, y2 = raw_boxes[centroid]
                boxes.append((x1, y1, x2, y2, scores_map.get(centroid, conf), stopped))
        return boxes, pred_stopped

    return detect


def _hf_detr_detector(processor_cls, model_cls, model_id, conf):
    import torch
    from PIL import Image

    processor = processor_cls.from_pretrained(model_id)
    model     = model_cls.from_pretrained(model_id).eval()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history: dict = {}
    tracker = CentroidTracker()

    def detect(frame):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs  = {k: v.to(device) for k, v in
                   processor(images=pil_img, return_tensors="pt").items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([pil_img.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=conf, target_sizes=target_sizes
        )[0]

        centroids, scores_map, raw_boxes = [], {}, {}
        for score, label, box in zip(results["scores"].cpu(),
                                     results["labels"].cpu(),
                                     results["boxes"].cpu()):
            if label.item() == 1:
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                centroids.append(c)
                scores_map[c] = float(score)
                raw_boxes[c]  = (x1, y1, x2, y2)

        tracked      = tracker.update(centroids)
        boxes        = []
        pred_stopped = False
        for tid, centroid in tracked.items():
            stopped = is_stopped(history, tid, centroid)
            if stopped:
                pred_stopped = True
            if centroid in raw_boxes:
                x1, y1, x2, y2 = raw_boxes[centroid]
                boxes.append((x1, y1, x2, y2, scores_map.get(centroid, conf), stopped))
        return boxes, pred_stopped

    return detect


def build_detector(model_id: str, conf: float):
    if model_id == "yolov8":
        from ultralytics import YOLO
        w = _ROOT / "weights" / "yolov8s.pt"
        if not w.exists():
            w = _ROOT / "yolov8s.pt"
        return "YOLOv8", _ultralytics_detector(YOLO, w, conf)

    if model_id == "rt_detr":
        from ultralytics import RTDETR
        w = _ROOT / "weights" / "rtdetr-l.pt"
        if not w.exists():
            w = _ROOT / "rtdetr-l.pt"
        return "RT-DETR", _ultralytics_detector(RTDETR, w, conf)

    if model_id == "ssd":
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights)
        return "SSD", _torchvision_detector(
            ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights, conf)

    if model_id == "faster_rcnn":
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)
        return "Faster R-CNN", _torchvision_detector(
            fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, conf)

    if model_id == "retinanet":
        from torchvision.models.detection import (
            retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights)
        return "RetinaNet", _torchvision_detector(
            retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights, conf)

    if model_id == "fcos":
        from torchvision.models.detection import (
            fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights)
        return "FCOS", _torchvision_detector(
            fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights, conf)

    if model_id == "efficientdet":
        return "EfficientDet", _efficientdet_detector(conf)

    if model_id == "detr":
        from transformers import DetrImageProcessor, DetrForObjectDetection
        return "DETR", _hf_detr_detector(
            DetrImageProcessor, DetrForObjectDetection,
            "facebook/detr-resnet-50", conf)

    if model_id == "conditional_detr":
        from transformers import (ConditionalDetrImageProcessor,
                                   ConditionalDetrForObjectDetection)
        return "Conditional DETR", _hf_detr_detector(
            ConditionalDetrImageProcessor, ConditionalDetrForObjectDetection,
            "microsoft/conditional-detr-resnet-50", conf)

    if model_id == "deformable_detr":
        from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
        return "Deformable DETR", _hf_detr_detector(
            AutoImageProcessor, DeformableDetrForObjectDetection,
            "SenseTime/deformable-detr", conf)

    raise ValueError(f"Modelo desconhecido: {model_id}")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Visualiza deteccoes e salva screenshots anotados"
    )
    parser.add_argument("video", help="Caminho para o video")
    parser.add_argument("model", choices=[
        "yolov8", "ssd", "faster_rcnn", "retinanet", "fcos", "efficientdet",
        "detr", "rt_detr", "conditional_detr", "deformable_detr",
    ])
    parser.add_argument("--gt", help="Arquivo JSON de ground truth (opcional)")
    parser.add_argument("--output", "-o", default="capturas",
                        help="Pasta raiz para salvar screenshots (padrao: capturas/)")
    parser.add_argument("--every-n", "-n", type=int, default=30, dest="every_n",
                        help="Salvar 1 frame a cada N (padrao: 30). 0 = apenas mudancas de estado")
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                        help="Limiar de confianca (padrao: 0.5)")
    parser.add_argument("--video-out", dest="video_out",
                        help="Salvar video anotado (ex: capturas/mulher_yolo.mp4)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir {args.video}")
        sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gt_binary = None
    if args.gt:
        gt_raw    = load_gt(args.gt, total)
        gt_binary = [1 if v == 2 else 0 for v in gt_raw]

    video_stem = Path(args.video).stem
    out_dir    = Path(args.output) / f"{video_stem}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Screenshots -> {out_dir}/")

    video_writer = None
    if args.video_out:
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.video_out, fourcc, fps_in, (w_vid, h_vid))
        print(f"Video anotado -> {args.video_out}")

    print(f"Carregando {args.model}...")
    model_name, detector = build_detector(args.model, args.conf)
    print(f"Processando {total} frames...\n")

    frame_idx    = 0
    prev_stopped = None
    saved_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, pred_stopped = detector(frame)
        gt_val = gt_binary[frame_idx] if gt_binary and frame_idx < len(gt_binary) else None

        for (x1, y1, x2, y2, sc, stopped) in boxes:
            _draw_box(frame, x1, y1, x2, y2, sc, stopped)
        _draw_panel(frame, frame_idx + 1, total, model_name, pred_stopped, gt_val)

        if video_writer:
            video_writer.write(frame)

        state_changed = pred_stopped != prev_stopped
        periodic_save = args.every_n > 0 and frame_idx % args.every_n == 0

        if periodic_save or state_changed:
            tag = "PARADO" if pred_stopped else "neg"
            fname = out_dir / f"frame_{frame_idx:05d}_{tag}.jpg"
            cv2.imwrite(str(fname), frame)
            saved_count += 1

        prev_stopped = pred_stopped
        frame_idx   += 1

        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames...", end="\r")

    cap.release()
    if video_writer:
        video_writer.release()

    print(f"\nConcluido! {saved_count} screenshots em {out_dir}/")


if __name__ == "__main__":
    main()
