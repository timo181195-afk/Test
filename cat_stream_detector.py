#!/usr/bin/env python3
"""Live-Katzen-Erkennung in einem Kamera-Stream mit YOLOv8."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Sequence, Union


@dataclass
class DetectionStats:
    total_frames: int = 0
    cat_frames: int = 0


def parse_source(raw_source: str) -> Union[int, str]:
    """Interpretiert numerische Quellen als Kamera-Index, sonst als Pfad/URL."""
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def draw_cat_annotations(cv2_module: Any, frame: Any, boxes: Sequence[Sequence[float]], confidences: Sequence[float]) -> None:
    """Zeichnet Bounding-Boxen und Konfidenz für erkannte Katzen."""
    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 40), 2)
        label = f"Katze {confidence:.2f}"
        cv2_module.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 200, 40),
            2,
            cv2_module.LINE_AA,
        )


def extract_cat_boxes(result: Any) -> tuple[list[list[float]], list[float]]:
    """Extrahiert Katzen-Detektionen aus einem Ultralytics-Resultat."""
    cat_boxes: list[list[float]] = []
    cat_confidences: list[float] = []

    for det in result.boxes:
        class_id = int(det.cls[0])
        if result.names.get(class_id) != "cat":
            continue
        cat_boxes.append(det.xyxy[0].tolist())
        cat_confidences.append(float(det.conf[0]))

    return cat_boxes, cat_confidences


def run_detector(source: Union[int, str], model_path: str, conf_threshold: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV (cv2) ist nicht installiert. Bitte `pip install -r requirements.txt` ausführen."
        ) from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics ist nicht installiert. Bitte `pip install -r requirements.txt` ausführen."
        ) from exc

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Kamera/Stream konnte nicht geöffnet werden: {source}")

    print("Live-Erkennung gestartet. Drücke 'q' zum Beenden.")
    stats = DetectionStats()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kein Frame mehr empfangen. Stream wird beendet.")
                break

            stats.total_frames += 1
            result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            cat_boxes, cat_confidences = extract_cat_boxes(result)

            if cat_boxes:
                stats.cat_frames += 1

            draw_cat_annotations(cv2, frame, cat_boxes, cat_confidences)

            info = f"Frames: {stats.total_frames} | Cat-Frames: {stats.cat_frames}"
            cv2.putText(
                frame,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Katzen-Erkennung", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    ratio = (stats.cat_frames / stats.total_frames * 100) if stats.total_frames else 0
    print(
        "Beendet. "
        f"Verarbeitete Frames: {stats.total_frames}, "
        f"Frames mit Katze: {stats.cat_frames} ({ratio:.1f}%)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Erkennt Katzen in einem Live-Kamerastream (Webcam oder RTSP/HTTP)."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Kameraquelle: Index (z. B. 0), Videodatei oder URL (RTSP/HTTP).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO-Modellpfad (Standard: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Konfidenz-Schwelle zwischen 0 und 1 (Standard: 0.4).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not 0 <= args.conf <= 1:
        print("Fehler: --conf muss zwischen 0 und 1 liegen.", file=sys.stderr)
        return 2

    source = parse_source(args.source)

    try:
        run_detector(source, model_path=args.model, conf_threshold=args.conf)
    except Exception as exc:  # Falls Kamera/Modell nicht erreichbar sind.
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
