# Video Research Tool

Desktop application for preprocessing, synchronising, labelling, and reviewing
multi-camera DJI video recordings alongside physiological signals.

## Requirements

- Python 3.11+
- FFmpeg (must be on PATH or configured inside the app)

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Workflow

1. **Preprocessing** — select video segments per camera, concatenate, audio-sync,
   and trim to common start/duration.
2. **Signal synchronisation** — import timestamped CSV signals (e.g. heart-rate),
   align with video timeline, optionally average dual sensors.
3. **Labelling** — drag intervals on the timeline, assign labels, export each
   labelled segment as a folder with trimmed video + signal files.
4. **Review** — load a manifest CSV to browse through exported segments.

## Adding a signal loader

Place a new Python module in `app/signals/`.  Implement the `SignalLoader` ABC
with `can_load(path) -> bool` and `load(path) -> pd.DataFrame`.  The module is
auto-discovered at startup via `pkgutil`.

## Project files

Projects are saved as `.vrt` JSON files with relative paths for portability.
