import csv
import json

import numpy as np

LIMIAR_PARADO = 10
FRAMES_PARADO = 45

HEADERS = [
    "Modelo", "FPS_medio", "Tempo_medio_ms", "Confianca_media",
    "TP", "FP", "FN", "TN",
    "Precisao", "Recall", "F1", "Acuracia", "Total_frames",
]


def is_stopped(history: dict, track_id, new_pos) -> bool:
    if track_id not in history:
        history[track_id] = []
    history[track_id].append(new_pos)

    if len(history[track_id]) < FRAMES_PARADO:
        return False

    last = np.array(history[track_id][-FRAMES_PARADO:])
    mov = np.mean(np.linalg.norm(np.diff(last, axis=0), axis=1))
    return mov < LIMIAR_PARADO


def load_gt(gt_path: str, total_frames: int) -> list:
    with open(gt_path) as f:
        raw = json.load(f)
    return [int(raw.get(str(i), 0)) for i in range(total_frames)]


def compute_metrics(y_true, y_pred, frame_times, confidences, model_name) -> dict:
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
        "Modelo":          model_name,
        "FPS_medio":       round(1.0 / np.mean(frame_times), 2) if frame_times else 0,
        "Tempo_medio_ms":  round(np.mean(frame_times) * 1000, 2) if frame_times else 0,
        "Confianca_media": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precisao":     round(precision, 4),
        "Recall":       round(recall, 4),
        "F1":           round(f1, 4),
        "Acuracia":     round(accuracy, 4),
        "Total_frames": n,
    }


def print_table(results: list) -> None:
    print("\n" + "=" * 85)
    print(f"{'Modelo':<20} {'FPS':>6} {'ms/f':>7} {'Conf.':>6}"
          f" {'Acur.':>6} {'Prec.':>6} {'Rec.':>6} {'F1':>6}")
    print("-" * 85)
    for r in results:
        print(f"{r['Modelo']:<20} {r['FPS_medio']:>6.1f} {r['Tempo_medio_ms']:>7.1f}"
              f" {r['Confianca_media']:>6.3f}"
              f" {r['Acuracia']:>6.4f} {r['Precisao']:>6.4f}"
              f" {r['Recall']:>6.4f} {r['F1']:>6.4f}")
    print("=" * 85)

    print("\nMatriz de confusao (pedestre parado = positivo):")
    print(f"{'Modelo':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print("-" * 44)
    for r in results:
        print(f"{r['Modelo']:<20} {r['TP']:>6} {r['FP']:>6} {r['FN']:>6} {r['TN']:>6}")


def save_csv(results: list, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in HEADERS})
    print(f"\nResultados salvos em: {path}")
