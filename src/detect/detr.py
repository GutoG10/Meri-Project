
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.metrics import is_stopped
from core.tracker import CentroidTracker


def main():
    import torch
    from PIL import Image
    from transformers import DetrForObjectDetection, DetrImageProcessor

    # DETR (Detection Transformer): usa atenção multi-head em vez de âncoras ou NMS,
    # carregado do HuggingFace Hub com backbone ResNet-50
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history: dict = {}
    # CentroidTracker provê rastreamento frame a frame via associação por distância
    tracker = CentroidTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # O processador do HuggingFace espera PIL Image em RGB; converte de BGR
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = {k: v.to(device) for k, v in
                  processor(images=pil_img, return_tensors="pt").items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # post_process_object_detection redimensiona as caixas de volta ao tamanho original
        target_sizes = torch.tensor([pil_img.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        # Filtra apenas label==1 (pessoa no COCO)
        centroids, centroid_scores, centroid_boxes = [], {}, {}
        for score, label, box in zip(results["scores"].cpu(),
                                     results["labels"].cpu(),
                                     results["boxes"].cpu()):
            if label.item() == 1:
                x1, y1, x2, y2 = map(int, box.tolist())
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

        cv2.imshow("DETR — Deteccao", annotated)
        cv2.imshow("Semaforo Virtual", semaforo)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
