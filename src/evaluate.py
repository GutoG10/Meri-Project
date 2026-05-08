"""
Avaliacao comparativa de YOLOv8, SSD e Faster R-CNN para deteccao de pedestres.

Uso:
    python evaluate.py video.mp4 ground_truth.json [resultados.csv]

O arquivo ground_truth.json deve ser gerado pelo annotate.py.
Gera um CSV com todas as metricas e imprime a tabela comparativa no terminal.
"""

import csv
import json
import sys
import threading
import time

import cv2
import numpy as np
import psutil



def load_gt(gt_path: str, total_frames: int) -> list[int]:
    with open(gt_path) as f:
        raw = json.load(f)
    return [int(raw.get(str(i), 0)) for i in range(total_frames)]


def compute_metrics(y_true, y_pred, frame_times, confidences,
                    cpu_samples, mem_samples, model_name) -> dict:
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / n if n > 0 else 0.0

    return {
        "Modelo":           model_name,
        "FPS_medio":        round(1.0 / np.mean(frame_times), 2) if frame_times else 0,
        "Tempo_medio_ms":   round(np.mean(frame_times) * 1000, 2) if frame_times else 0,
        "Confianca_media":  round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "CPU_medio_pct":    round(float(np.mean(cpu_samples)), 1) if cpu_samples else 0.0,
        "Memoria_media_MB": round(float(np.mean(mem_samples)), 1) if mem_samples else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precisao":  round(precision, 4),
        "Recall":    round(recall, 4),
        "F1":        round(f1, 4),
        "Acuracia":  round(accuracy, 4),
        "Total_frames": n,
    }


class ResourceSampler:
    """Amostra CPU e memoria em background durante a inferencia."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._cpu: list[float] = []
        self._mem: list[float] = []
        self._running = False
        self._process = psutil.Process()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._cpu.clear()
        self._mem.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[list[float], list[float]]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self._cpu, self._mem

    def _run(self) -> None:
        while self._running:
            self._cpu.append(psutil.cpu_percent(interval=None))
            self._mem.append(self._process.memory_info().rss / 1024 / 1024)
            time.sleep(self._interval)


# ── execucao por modelo ───────────────────────────────────────────────────────

def run_yolo(video_path: str, gt: list[int], conf_threshold: float = 0.5) -> dict:
    from ultralytics import YOLO

    print("  Carregando YOLOv8...")
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    sampler = ResourceSampler()
    frame_times, confidences, preds = [], [], []

    sampler.start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model(frame, classes=[0], verbose=False)
        frame_times.append(time.perf_counter() - t0)

        detected = False
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for score in boxes.conf.cpu().numpy():
                if score >= conf_threshold:
                    detected = True
                    confidences.append(float(score))

        preds.append(1 if detected else 0)

    cap.release()
    cpu_s, mem_s = sampler.stop()
    return compute_metrics(gt, preds, frame_times, confidences, cpu_s, mem_s, "YOLOv8")


def run_ssd(video_path: str, gt: list[int], conf_threshold: float = 0.5) -> dict:
    import torch
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
    from torchvision.transforms import functional as TF

    print("  Carregando SSD...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    sampler = ResourceSampler()
    frame_times, confidences, preds = [], [], []

    sampler.start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = TF.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model([tensor])[0]
        frame_times.append(time.perf_counter() - t0)

        detected = False
        for label, score in zip(pred["labels"].cpu().numpy(), pred["scores"].cpu().numpy()):
            if label == 1 and score >= conf_threshold:   # classe 1 = person no COCO
                detected = True
                confidences.append(float(score))

        preds.append(1 if detected else 0)

    cap.release()
    cpu_s, mem_s = sampler.stop()
    return compute_metrics(gt, preds, frame_times, confidences, cpu_s, mem_s, "SSD")


def run_frcnn(video_path: str, gt: list[int], conf_threshold: float = 0.5) -> dict:
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.transforms import functional as TF

    print("  Carregando Faster R-CNN...")
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    sampler = ResourceSampler()
    frame_times, confidences, preds = [], [], []

    sampler.start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = TF.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model([tensor])[0]
        frame_times.append(time.perf_counter() - t0)

        detected = False
        for label, score in zip(pred["labels"].cpu().numpy(), pred["scores"].cpu().numpy()):
            if label == 1 and score >= conf_threshold:
                detected = True
                confidences.append(float(score))

        preds.append(1 if detected else 0)

    cap.release()
    cpu_s, mem_s = sampler.stop()
    return compute_metrics(gt, preds, frame_times, confidences, cpu_s, mem_s, "Faster R-CNN")


# ── saida ─────────────────────────────────────────────────────────────────────

HEADERS = [
    "Modelo", "FPS_medio", "Tempo_medio_ms", "Confianca_media",
    "CPU_medio_pct", "Memoria_media_MB",
    "TP", "FP", "FN", "TN",
    "Precisao", "Recall", "F1", "Acuracia", "Total_frames",
]


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Modelo':<16} {'FPS':>6} {'ms/f':>7} {'Conf.':>6} {'CPU%':>5} {'Mem(MB)':>8}"
          f" {'Acur.':>6} {'Prec.':>6} {'Rec.':>6} {'F1':>6}")
    print("-" * 90)
    for r in results:
        print(f"{r['Modelo']:<16} {r['FPS_medio']:>6.1f} {r['Tempo_medio_ms']:>7.1f}"
              f" {r['Confianca_media']:>6.3f} {r['CPU_medio_pct']:>5.1f}"
              f" {r['Memoria_media_MB']:>8.1f}"
              f" {r['Acuracia']:>6.4f} {r['Precisao']:>6.4f}"
              f" {r['Recall']:>6.4f} {r['F1']:>6.4f}")
    print("=" * 90)

    print("\nMatriz de confusao:")
    print(f"{'Modelo':<16} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print("-" * 40)
    for r in results:
        print(f"{r['Modelo']:<16} {r['TP']:>6} {r['FP']:>6} {r['FN']:>6} {r['TN']:>6}")


def save_csv(results: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in HEADERS})
    print(f"\nResultados salvos em: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 3:
        print("Uso: python evaluate.py video.mp4 ground_truth.json [resultados.csv]")
        sys.exit(1)

    video_path  = sys.argv[1]
    gt_path     = sys.argv[2]
    output_csv  = sys.argv[3] if len(sys.argv) > 3 else "resultados.csv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir {video_path}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gt = load_gt(gt_path, total)
    present = sum(gt)
    print(f"\nVideo: {total} frames | com pedestre: {present} | sem pedestre: {total - present}")

    results = []
    for label, fn in [("YOLOv8", run_yolo), ("SSD", run_ssd), ("Faster R-CNN", run_frcnn)]:
        print(f"\n[{label}]")
        results.append(fn(video_path, gt))

    print_table(results)
    save_csv(results, output_csv)


if __name__ == "__main__":
    main()
