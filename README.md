# Video Research Tool

Desktop application for preprocessing, synchronising, labelling, and reviewing
multi-camera DJI video recordings alongside physiological signals.

For a detailed, publication-oriented description of ECG peak detection,
synthetic PPG generation, HR/HRV estimation, design decisions, validation, and
limitations, see [`docs/SYNTHETIC_PPG_METHODS.md`](docs/SYNTHETIC_PPG_METHODS.md).

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
3. **Synthetic PPG (optional)** — choose one of the ECG files produced by Signal
   sync, detect R peaks, and generate an absolute-timestamped synthetic PPG at
   125 Hz. Only samples in that synchronized ECG are used; the generator does
   not load the original ECG or look for peaks outside the imported interval.
   Output spans from the first detected peak through the last detected peak. It
   also exports detected RR intervals, one-second HR estimates from overlapping
   10-second FFT windows, rolling 300-beat SDNN, and RMSSD. The HR estimator uses
   sampling-rate-scaled smoothness-prior detrending, a 0.6-3.3 Hz band-pass, and
   continuity-aware fundamental selection to avoid mistaking the synthetic
   waveform's second harmonic for heart rate. Spectral peaks are refined between
   FFT bins to avoid stair-step BPM values. When synchronized Zephyr HR exists,
   generated HR is evaluated on those exact timestamps and therefore has the same
   point count; timestamps near either edge use the nearest complete 10-second
   PPG window. Views compare synthetic HR and HRV with synchronized Zephyr data
   and provide a normalized ECG/PPG overlay.
4. **Labelling** — drag intervals on the timeline, assign labels, export each
   labelled segment as a folder with trimmed video plus one CSV per sensor. To
   reuse existing segment folders, load the project metadata,
   import their `manifest.csv`, then use **Export ECG to existing segments** or
   **Export HR to existing segments** after adding or synchronizing the respective
   signal. These exports use the imported intervals directly without asking for
   the manifest again.
5. **Review** — load a manifest CSV to browse through exported segments,
   including synthetic PPG when it was generated and exported.

Synthetic outputs are grouped in the `synthetic_ppg/` directory inside the
project output directory:

- `synthetic_ppg/synthetic_ppg.csv`
- `synthetic_ppg/synthetic_ppg_rr.csv`
- `synthetic_ppg/synthetic_ppg_hr.csv`
- `synthetic_ppg/synthetic_ppg_hrv.csv`

When a project created by an earlier version is regenerated, the corresponding
root-level files are removed or moved to `obsolete_files/` according to the
configured cleanup policy.

The PPG model changes RR values at exact cumulative interval boundaries. This
avoids the growing timing dilation caused by independently rounding every RR
interval to a whole number of samples.

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

Enable **Options > Blur faces for privacy** to pixelate detected faces in every
camera preview, synchronized frame capture, mosaic video, and exported labelled
camera clip. The setting persists between application sessions. Original input
and preprocessing working videos are left unchanged so analysis data is not
destroyed; use only privacy-enabled exports when sharing material. Automatic
face detection can miss obscured or unusually angled faces, so review exported
material before distribution.

## Adding a signal loader

Place a new Python module in `app/signals/`.  Implement the `SignalLoader` ABC
with `can_load(path) -> bool` and `load(path) -> pd.DataFrame`.  The module is
auto-discovered at startup via `pkgutil`.

## Project files

Projects are saved as `.vrt` JSON files with relative paths for portability.

## PPGSynth license

The synthetic PPG dynamical model is adapted from PPGSynth by Tang et al. and
is distributed under GNU GPL v3. See `THIRD_PARTY_NOTICES.md` and
`LICENSES/PPGSynth-GPL-3.0.txt`. Distribution of this combined application must
comply with the GPL; consult qualified counsel for legal advice about a specific
distribution model.
