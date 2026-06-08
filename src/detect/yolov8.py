import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.metrics import is_stopped

_ROOT = Path(__file__).parents[2]


def main():
    from ultralytics import YOLO

    # Carrega os pesos do YOLOv8s; procura primeiro em weights/, depois na raiz do projeto
    weights = _ROOT / "weights" / "yolov8s.pt"
    if not weights.exists():
        weights = _ROOT / "yolov8s.pt"
    model = YOLO(str(weights))

    # history acumula as posições anteriores de cada pedestre para detectar parada
    history: dict = {}
    # YOLOv8 usa rastreamento nativo via model.track(), dispensando CentroidTracker externo
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # classes=[0] restringe a detecção apenas à classe "pessoa" do COCO;
        # persist=True mantém IDs consistentes entre frames consecutivos
        results = model.track(frame, classes=[0], persist=True, verbose=False)
        annotated = frame.copy()
        estado = "vermelho"  # padrão: semáforo fechado para pedestres

        if results[0].boxes.id is not None:
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = map(int, box)
                # Usa o centróide da bounding box como posição do pedestre no frame
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                stopped = is_stopped(history, track_id, (cx, cy))
                # Se qualquer pedestre está parado, o semáforo muda para verde
                if stopped:
                    estado = "verde"
                color = (0, 255, 0) if stopped else (0, 0, 255)
                label = "Aguardando travessia" if stopped else "Andando"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Renderiza o semáforo virtual como uma imagem separada (círculo colorido)
        semaforo = np.zeros((300, 200, 3), dtype=np.uint8)
        cor = (0, 255, 0) if estado == "verde" else (0, 0, 255)
        cv2.circle(semaforo, (100, 150), 70, cor, -1)
        cv2.putText(semaforo, estado.upper(), (45, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 — Deteccao", annotated)
        cv2.imshow("Semaforo Virtual", semaforo)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
