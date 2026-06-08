
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.metrics import is_stopped
from core.tracker import CentroidTracker

_ROOT = Path(__file__).parents[2]
# Cascade R-CNN requer arquivos de configuração e checkpoint do mmdetection;
# devem ser baixados com: mim download mmdet --config cascade-rcnn_r50_fpn_1x_coco --dest weights/
_CONFIG     = _ROOT / "weights" / "cascade-rcnn_r50_fpn_1x_coco.py"
_CHECKPOINT = _ROOT / "weights" / "cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56dde.pth"


def main():
    import torch
    from mmdet.apis import init_detector, inference_detector

    if not _CONFIG.exists() or not _CHECKPOINT.exists():
        raise FileNotFoundError(
            "Pesos nao encontrados em weights/.\n"
            "Execute: mim download mmdet --config cascade-rcnn_r50_fpn_1x_coco --dest weights/"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # init_detector: carrega modelo e pesos via API do mmdetection
    model = init_detector(str(_CONFIG), str(_CHECKPOINT), device=device)

    history: dict = {}
    tracker = CentroidTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inference_detector: executa a detecção e retorna um objeto DetDataSample
        result = inference_detector(model, frame)
        instances = result.pred_instances
        labels = instances.labels.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        bboxes = instances.bboxes.cpu().numpy()

        # No mmdetection, label==0 corresponde à classe "pessoa" (índice base-0, diferente do COCO padrão)
        centroids, centroid_scores, centroid_boxes = [], {}, {}
        for lbl, score, box in zip(labels, scores, bboxes):
            if lbl == 0 and score >= 0.5:
                x1, y1, x2, y2 = box.astype(int)
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                centroids.append(c)
                centroid_scores[c] = float(score)
                centroid_boxes[c] = (x1, y1, x2, y2)

        tracked = tracker.update(centroids)

        annotated = frame.copy()
        estado = "vermelho"
        for track_id, centroid in tracked.items():
            stopped = is_stopped(history, track_id, centroid)
            if stopped:
                estado = "verde"
            box = centroid_boxes.get(centroid)
            if box:
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if stopped else (0, 0, 255)
                label = "Aguardando travessia" if stopped else "Andando"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Semáforo virtual: círculo verde se algum pedestre está parado, vermelho caso contrário
        semaforo = np.zeros((300, 200, 3), dtype=np.uint8)
        cor = (0, 255, 0) if estado == "verde" else (0, 0, 255)
        cv2.circle(semaforo, (100, 150), 70, cor, -1)
        cv2.putText(semaforo, estado.upper(), (45, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Cascade R-CNN — Deteccao", annotated)
        cv2.imshow("Semaforo Virtual", semaforo)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
