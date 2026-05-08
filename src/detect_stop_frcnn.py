import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

LIMIAR_PARADO = 10
FRAMES_PARADO = 45
CONFIANCA_MIN = 0.5
PERSON_CLASS = 1  # classe "person" no COCO

history = {}


class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def update(self, centroids):
        if not centroids:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return {}

        if not self.objects:
            for c in centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return dict(zip(range(self.next_id - len(centroids), self.next_id), centroids))

        obj_ids = list(self.objects)
        obj_cents = np.array(list(self.objects.values()), dtype=float)
        new_cents = np.array(centroids, dtype=float)

        D = np.linalg.norm(obj_cents[:, None] - new_cents[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        current_frame_ids = {}

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = obj_ids[r]
            self.objects[oid] = centroids[c]
            self.disappeared[oid] = 0
            current_frame_ids[oid] = centroids[c]
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(obj_ids))) - used_rows:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                del self.objects[oid]
                del self.disappeared[oid]

        for c in set(range(len(centroids))) - used_cols:
            self.objects[self.next_id] = centroids[c]
            self.disappeared[self.next_id] = 0
            current_frame_ids[self.next_id] = centroids[c]
            self.next_id += 1

        return current_frame_ids


def is_stopped(track_id, new_pos):
    if track_id not in history:
        history[track_id] = []
    history[track_id].append(new_pos)

    if len(history[track_id]) < FRAMES_PARADO:
        return False

    last_positions = np.array(history[track_id][-FRAMES_PARADO:])
    mov = np.mean(np.linalg.norm(np.diff(last_positions, axis=0), axis=1))
    return mov < LIMIAR_PARADO


def main():
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tracker = CentroidTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(device)

        with torch.no_grad():
            predictions = model([img_tensor])[0]

        boxes_det = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        person_boxes = []
        centroids = []
        for box, label, score in zip(boxes_det, labels, scores):
            if label == PERSON_CLASS and score >= CONFIANCA_MIN:
                x1, y1, x2, y2 = box.astype(int)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                person_boxes.append((x1, y1, x2, y2))
                centroids.append((cx, cy))

        centroid_to_box = {c: b for b, c in zip(person_boxes, centroids)}
        tracked = tracker.update(centroids)

        annotated = frame.copy()
        estado_pedestre = "vermelho"

        for track_id, centroid in tracked.items():
            box = centroid_to_box.get(centroid)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            stopped = is_stopped(track_id, centroid)

            if stopped:
                estado_pedestre = "verde"

            color = (0, 255, 0) if stopped else (0, 0, 255)
            text = "Aguardando travessia" if stopped else "Andando"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print("ESTADO ATUAL:", estado_pedestre)

        semaforo = np.zeros((300, 200, 3), dtype=np.uint8)
        if estado_pedestre == "verde":
            cv2.circle(semaforo, (100, 150), 70, (0, 255, 0), -1)
        else:
            cv2.circle(semaforo, (100, 150), 70, (0, 0, 255), -1)
        cv2.putText(semaforo, estado_pedestre.upper(), (45, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Deteccao de Pedestres - Faster R-CNN", annotated)
        cv2.imshow("Semaforo Virtual", semaforo)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
