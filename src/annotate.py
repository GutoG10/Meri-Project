"""
Ferramenta de anotacao de ground truth para videos de teste.

Uso:
    python annotate.py video.mp4 ground_truth.json

Controles durante a reproducao:
    1     -> marcar "pedestre presente" a partir deste frame
    0     -> marcar "sem pedestre" a partir deste frame
    SPACE -> pausar / continuar
    ESC   -> salvar e encerrar
"""

import cv2
import json
import sys


def annotate_video(video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video: {total} frames | {fps_video:.0f} FPS")
    print("Controles: 1 = pedestre | 0 = sem pedestre | SPACE = pausar | ESC = salvar")

    # cada item: (frame_inicio, rotulo)
    segments = [(0, 0)]
    paused = False
    frame_idx = 0
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        if frame is None:
            break

        current_label = segments[-1][1]
        display = frame.copy()
        color = (0, 255, 0) if current_label == 1 else (0, 0, 255)
        label_text = "PEDESTRE PRESENTE" if current_label == 1 else "SEM PEDESTRE"

        cv2.rectangle(display, (0, 0), (420, 100), (0, 0, 0), -1)
        cv2.putText(display, f"Frame {frame_idx}/{total}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"GT: {label_text}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        estado = "[ PAUSADO ]" if paused else "[ REPRODUZINDO ]"
        cv2.putText(display, estado, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow("Anotacao - Ground Truth", display)

        delay = 100 if paused else max(1, int(1000 / fps_video))
        key = cv2.waitKey(delay) & 0xFF

        if key == 27:   # ESC
            break
        elif key == ord('1') and current_label != 1:
            segments.append((frame_idx, 1))
        elif key == ord('0') and current_label != 0:
            segments.append((frame_idx, 0))
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    # expande segmentos para rotulo por frame
    gt: dict[str, int] = {}
    seg_sorted = sorted(segments, key=lambda x: x[0])
    for i, (start, lbl) in enumerate(seg_sorted):
        end = seg_sorted[i + 1][0] if i + 1 < len(seg_sorted) else total
        for f in range(start, end):
            gt[str(f)] = lbl

    with open(output_path, "w") as fp:
        json.dump(gt, fp)

    present = sum(1 for v in gt.values() if v == 1)
    print(f"\nSalvo em: {output_path}")
    print(f"Frames com pedestre: {present}/{total} ({100 * present / total:.1f}%)")
    print(f"Frames sem pedestre: {total - present}/{total} ({100 * (total - present) / total:.1f}%)")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    output = sys.argv[2] if len(sys.argv) > 2 else "ground_truth.json"
    annotate_video(video, output)
