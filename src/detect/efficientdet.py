
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.metrics import is_stopped
from core.tracker import CentroidTracker

# EfficientDet-D0 espera entrada 512×512 normalizada com média/desvio da ImageNet
_INPUT_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    import torch
    from effdet import create_model

    # Cria EfficientDet-D0 no modo de predição com pesos pré-treinados no COCO
    model = create_model("efficientdet_d0", bench_task="predict", pretrained=True)
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history: dict = {}
    tracker = CentroidTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Guarda dimensões originais para reprojetar as caixas após o resize
        orig_h, orig_w = frame.shape[:2]
        # Redimensiona, converte para RGB, normaliza com média/desvio da ImageNet
        img = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

        # effdet retorna detecções como array (N, 6): [y1, x1, y2, x2, score, class_id]
        dets = output[0].cpu().numpy()

        centroids, centroid_scores, centroid_boxes = [], {}, {}
        for det in dets:
            y1, x1, y2, x2, score, class_id = det
            if score < 0.5:
                continue
            # class_id==1 corresponde à classe "pessoa" no COCO (índice base-1)
            if int(round(class_id)) == 1:
                # Reprojetar coordenadas do espaço 512×512 para o tamanho original do frame
                xi1 = int(x1 * orig_w / _INPUT_SIZE)
                xi2 = int(x2 * orig_w / _INPUT_SIZE)
                yi1 = int(y1 * orig_h / _INPUT_SIZE)
                yi2 = int(y2 * orig_h / _INPUT_SIZE)
                c = ((xi1 + xi2) // 2, (yi1 + yi2) // 2)
                centroids.append(c)
                centroid_scores[c] = float(score)
                centroid_boxes[c] = (xi1, yi1, xi2, yi2)

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

        cv2.imshow("EfficientDet — Deteccao", annotated)
        cv2.imshow("Semaforo Virtual", semaforo)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
