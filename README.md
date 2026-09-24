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
   trim to common start/duration, and generate the project metadata file. To use
   videos that are already prepared, generate their metadata before loading the
   project in another section.
2. **Signal synchronisation** — load the project metadata, import HR and ECG CSV files separately, align
   them with the video timeline, and write one CSV per sensor directly in the
   project output folder (for example, `polar_1.csv`, `polar_2.csv`,
   `zephyr_1.csv`, or `ecg_1.csv`). Sensor numbering always starts at `_1`,
   even when only one sensor of that type is loaded. Select the plot type to view one signal kind at
   a time. Legacy projects containing combined synchronized CSVs remain
   supported.
3. **Labelling** — drag intervals on the timeline, assign labels, export each
   labelled segment as a folder with trimmed video plus one CSV per sensor. To
   reuse existing segment folders, load the project metadata,
   import their `manifest.csv`, then use **Export ECG to existing segments** or
   **Export HR to existing segments** after adding or synchronizing the respective
   signal. These exports use the imported intervals directly without asking for
   the manifest again.
4. **Review** — load a manifest CSV to browse through exported segments.

Signal synchronisation and labelled export provide an optional cleanup checkbox
for removing legacy combined CSV files after replacement per-sensor files are
written successfully. Enable **Options > Archive removed files instead of
deleting** to move cleaned files into `obsolete_files/` in the project output
directory instead of deleting them. Their original project-relative folder
structure is preserved for recovery.

Metadata, synchronized signals, and imported labelled-segment manifests are
shared between Signal sync, Labelling, and Review. Switching sections refreshes
the destination view from the same project state, regardless of which section
performed the import.

## Adding a signal loader

Place a new Python module in `app/signals/`.  Implement the `SignalLoader` ABC
with `can_load(path) -> bool` and `load(path) -> pd.DataFrame`.  The module is
auto-discovered at startup via `pkgutil`.

## Project files

Projects are saved as `.vrt` JSON files with relative paths for portability.
