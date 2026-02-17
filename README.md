# Live-Katzen-Erkennung aus Kamera-Stream

Dieses Tool nimmt einen Kamera-Stream (Webcam, RTSP, HTTP oder Datei) und erkennt im Live-Feed Katzen mit einem vortrainierten YOLO-Modell.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start

### Webcam (Standardkamera)

```bash
python cat_stream_detector.py --source 0
```

### Andere Kamera (z. B. Index 1)

```bash
python cat_stream_detector.py --source 1
```

### RTSP-Stream

```bash
python cat_stream_detector.py --source "rtsp://user:pass@ip:554/stream"
```

## Optionen

- `--source`: Kameraindex, Videodatei oder Stream-URL.
- `--model`: Modellpfad, Standard ist `yolov8n.pt`.
- `--conf`: Konfidenzschwelle von `0` bis `1`, Standard `0.4`.

## Bedienung

- Ein Fenster `Katzen-Erkennung` zeigt den Live-Feed.
- Erkannte Katzen werden mit Bounding-Box und Konfidenz markiert.
- Mit `q` beendest du den Lauf.

## Fehlerbehebung

- Hilfe anzeigen funktioniert immer ohne installierte Pakete:
  ```bash
  python cat_stream_detector.py --help
  ```
- Falls beim Start `cv2` oder `ultralytics` fehlt:
  ```bash
  pip install -r requirements.txt
  ```

## Hinweise

- Beim ersten Start lädt `ultralytics` ggf. das Modell `yolov8n.pt` aus dem Internet.
- Für RTSP/Netzwerkquellen muss der Stream vom System erreichbar sein.
