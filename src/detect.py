from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8s.pt")  # pré-treinado no COCO

    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("Erro ao acessar câmera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0])  # apenas pessoas

        annotated = results[0].plot()

        cv2.imshow("Detecção de Pedestres - YOLO COCO", annotated)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
