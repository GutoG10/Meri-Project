from ultralytics import YOLO
import cv2
import numpy as np

history = {}
LIMIAR_PARADO = 20  

def is_stopped(track_id, new_pos):
    if track_id not in history:
        history[track_id] = []
    history[track_id].append(new_pos)

    if len(history[track_id]) < 10:
        return False

    last_positions = np.array(history[track_id][-10:])
    mov = np.mean(np.linalg.norm(np.diff(last_positions, axis=0), axis=1))

    return mov < LIMIAR_PARADO


def main():
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(0)

    estado_pedestre = "vermelho"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, classes=[0], persist=True)
        annotated = frame.copy()

        estado_pedestre = "vermelho"

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                stopped = is_stopped(track_id, (cx, cy))

                if stopped:
                    estado_pedestre = "verde"

                color = (0, 255, 0) if stopped else (0, 0, 255)
                text = "Aguardando travessia" if stopped else "Andando"

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, text, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print("ESTADO ATUAL:", estado_pedestre)

        
        semaforo = np.zeros((300, 200, 3), dtype=np.uint8)

        if estado_pedestre == "verde":
            cv2.circle(semaforo, (100, 150), 70, (0, 255, 0), -1)
        else:
            cv2.circle(semaforo, (100, 150), 70, (0, 0, 255), -1)

        cv2.putText(semaforo, estado_pedestre.upper(), (45, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Deteccao de Pedestres", annotated)
        cv2.imshow("Semaforo Virtual", semaforo)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
