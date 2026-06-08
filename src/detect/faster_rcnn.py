import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.metrics import is_stopped
from core.tracker import CentroidTracker


def main():
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.transforms import functional as TF

    # Carrega o Faster R-CNN com backbone ResNet-50 + FPN, pré-treinado no COCO
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history: dict = {}
    # Faster R-CNN não possui rastreamento nativo; CentroidTracker associa detecções
    # de frames consecutivos por proximidade de centróide para manter IDs estáveis
    tracker = CentroidTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converte BGR→RGB e normaliza para [0,1] conforme esperado pelo torchvision
        tensor = TF.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
        with torch.no_grad():
            pred = model([tensor])[0]

        # Filtra apenas label==1 (pessoa no COCO) com score acima do limiar
        centroids, centroid_scores, centroid_boxes = [], {}, {}
        for label, score, box in zip(pred["labels"].cpu().numpy(),
                                     pred["scores"].cpu().numpy(),
                                     pred["boxes"].cpu().numpy()):
            if label == 1 and score >= 0.5:
                x1, y1, x2, y2 = box.astype(int)
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                centroids.append(c)
                centroid_scores[c] = float(score)
                centroid_boxes[c] = (x1, y1, x2, y2)

        # Atualiza o tracker com os centróides detectados e obtém IDs consistentes
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

        cv2.imshow("Faster R-CNN — Deteccao", annotated)
        cv2.imshow("Semaforo Virtual", semaforo)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
