# Erg Video Analysis (Local + In‑Browser)

This repo provides two ways to analyze indoor rowing technique from a side profile:

- Local Python CLI (OpenCV + MediaPipe) for analyzing webcam or video files, entirely offline.
- GitHub Pages web app that runs all processing in your browser (no uploads, no server).

Both approaches compute simple rowing metrics like stroke rate (SPM), stroke count, back angle, shin angle at the catch, and drive/recovery timing ratio.

## 1) Local Python Analyzer

Requirements:

- Python 3.9+
- `opencv-python`, `mediapipe`, `numpy`

Install deps:

```
pip install opencv-python mediapipe numpy
```

Run with webcam (either form works):

```
python -m src.python.erg_analyzer --source webcam
# or
python src/python/erg_analyzer.py --source webcam
```

Analyze a video file:

```
python -m src.python.erg_analyzer --source path/to/rowing_side_view.mp4 --annotate out.mp4
```

The CLI prints metrics to the console and optionally writes an annotated video if `--annotate` is provided.

## 2) In-Browser (GitHub Pages)

The web app runs fully on-device using MediaPipe Pose in the browser. It supports webcam streaming and file upload.

To host on GitHub Pages:

1. Push this repo to GitHub.
2. In your repo settings, set GitHub Pages Source to `docs/`.
3. Visit the published URL (e.g., `https://<your-username>.github.io/<repo-name>/`).

Local preview without a server: simply open `docs/index.html` in a modern browser (Chrome/Edge recommended). Webcam access requires `https` or `localhost`, but file upload will work from a file URL.

## Notes & Tips

- Best results come from a true side profile with the camera roughly hip height.
- Ensure good lighting and the athlete occupies a reasonable portion of the frame.
- The analysis is approximate and intended for training feedback, not medical use.


This tool is designed to provide real-time feedback on a rowers technique. This utilizes 
