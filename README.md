# Video Research Tool

Desktop application for preprocessing, synchronising, labelling, and reviewing
multi-camera DJI video recordings alongside physiological signals.

For a detailed, publication-oriented description of ECG peak detection,
synthetic PPG generation, HR/HRV estimation, design decisions, validation, and
limitations, see [`docs/SYNTHETIC_PPG_METHODS.md`](docs/SYNTHETIC_PPG_METHODS.md).

## Requirements

- Python 3.11+
- FFmpeg (must be on PATH or configured inside the app)
- PyQt6 6.8 or newer for video playback with sound. Older PyQt6 wheels on
  Windows do not ship Qt's FFmpeg playback backend; the app then falls back to
  slow still-frame previews without sound and shows a warning in the status bar.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Screen layout

Every section uses the same layout: a control panel on the left (collapsible
sections; hide the whole panel with **Ctrl+B**) and a work area on the right
with the cameras, the signal graphs and, in Labelling, the timeline. Drag the
splitter handles to resize them; sizes are remembered per section
(**View > Reset layout of this section** restores the defaults).

## Video playback

All cameras play together with hardware decoding (4K HEVC included) and sound.
One camera provides the sound (sound selector in the transport bar, or
right-click a camera); the others follow its clock and are kept in sync
automatically. Seeking and frame stepping are frame-accurate.

- **Space / K** play-pause, **Left / Right** one frame, **Shift+Left/Right** 1 s,
  **Ctrl+Left/Right** 10 s, **J / L** 5 s, **Home / End**, **[ / ]** speed
  (0.1x to 4x), **M** mute.
- Type a time (`1:02:03.5`, `02:03`, `75.2`) or a frame (`f4500`) in **Go to**.
- The transport bar shows video time, frame number and, when the signal anchor
  is known, the corresponding UTC clock time.
- **Layout** arranges cameras automatically (largest picture), in a row, in
  two columns, or with one camera enlarged (double-click a camera).
- The window button (or **Ctrl+Shift+W**) moves the cameras to a separate window,
  for example on a second screen; the controls stay in the main window.

## Renaming cameras

After preprocessing or importing project metadata, choose **File > Rename cameras…**
from any section. Edit the camera names and click **Rename**. The application
renames final videos, existing labelled camera clips and individual captured-frame
filenames, updates their JSON
metadata and path references, and saves the current project if it already has a
project file. Original recordings, sensor CSVs and interval labels are unchanged.
Imported metadata is updated in place, even if it has a custom filename.

Names must be unique and valid Windows filename text. Existing destination files
are never overwritten; failed operations restore the previous files and metadata.
Wait for preprocessing and exports to finish before renaming. Camera previews are
paused during renaming. A progress dialog shows the current file and stage; **Cancel**
restores previous filenames and metadata before closing. File operations run in
the background; existing plots, signal data and playback position are retained
without rebuilding the section or re-probing videos. Text already rendered into mosaic videos or captured images
is not editable; re-export those visualizations to use the new names.

## Dataset templates

Save the setup you repeat for every dataset and fill it in with one click.
A template can hold:

- **Cameras** — number of cameras, their names, **Not limiting size** and the
  per-camera **Audio sync start offset**;
- **Labels** — the label library used in Labelling;
- **Audio sync & cleanup options** — audio sync duration, pairwise refinement
  and automatic deletion of intermediate files.

The **Template** row is at the top of **Preprocessing > Setup** (applies
cameras, options and labels) and in **Labelling > Label library** (applies
labels only). Pick a template and click **Apply**, or click **Save…** to store
the current setup as a new template or update an existing one. When updating,
parts you leave unchecked keep their saved values, so you can, for example,
update only the labels of a template from the Labelling section. Applying labels
replaces the library; labels still used by intervals are kept, and **Ctrl+Z**
undoes it in Labelling.

The **⋯** button renames or deletes a template, or marks it **Use for new
projects** (shown with ★): new projects then start with that template's
cameras, labels and options. Opened projects are never changed automatically.
Templates are stored per user in the application settings and shared by all
projects. Segments, output folders and signal files are not part of templates.

## Graphs and timeline

Signals are shown as stacked panels (for example heart rate above ECG) sharing
one time axis with the timeline, and each panel has its own y-axis.

- Mouse wheel zooms time, clicking moves the playhead, and the red cursor can be
  dragged to scrub the videos.
- With **Select zoom** on (the default), drag on a graph to select the region
  to focus on; with it off, dragging pans and **Ctrl+drag** selects. A mostly
  horizontal drag sets only the time window (its start and duration show below
  the graph while dragging); a taller box also zooms that panel's y-axis.
  **Undo zoom** steps back through previous focus regions, and **Full range**
  shows the whole signal and restores automatic y-scaling for box-zoomed panels.
- The graphs and the timeline stay aligned and follow the playhead while zoomed
  (**Follow playhead**).
- Hovering a graph shows the time, the value of each visible signal, and the
  label at that time.
- Samples are placed at `timestamp - signal anchor`, i.e. exactly where the
  labelled-segment export cuts them. Earlier versions positioned each signal
  relative to its own first sample, which could shift a sensor that started
  recording after the videos.

## Labelling

- Pick the label with **1-9** (or the label box under the transport bar), press
  **I** at the start and **O** at the end. **Esc** cancels a pending start.
- On the timeline: drag on empty space to create an interval, drag edges to
  resize, drag a selected interval to move it. Edges snap to the playhead and to
  neighbouring intervals (hold Shift to disable snapping); intervals cannot
  overlap. **Shift+drag** on a graph also creates an interval.
- Right-click an interval to relabel it, go to its start/end, set its start/end
  to the playhead, split it at the playhead, subdivide it, or export a figure or
  mosaic for it.
- **S** splits at the playhead, **R** applies the current label to the selected
  interval, **N / P** jump to the next/previous interval, **Delete** removes the
  selected interval, **Ctrl+Z / Ctrl+Y** undo and redo.
- The Intervals table lists every interval; double-click a row to jump to it.
- Exporting labelled segments runs in the background (progress and Cancel);
  camera clips of one interval are cut in parallel. Without face blurring the
  clips are cut with stream copy exactly as before (no re-encoding).

Press **F1** for the full list of shortcuts.

## Exporting visualizations

- **Graph > Export > Figure** opens a dialog with a live preview: current view,
  full session, selected interval, a custom range, or one figure per labelled
  interval (batch). Choose the panels, a label-timeline track, interval shading,
  playhead, legends, title, size presets (slide, single or double column),
  DPI, font size, time-axis format, and PNG, SVG, PDF or TIFF output.
  **Label timeline only** produces an overview strip of all labels.
- **Graph > Export > Plotted data (CSV)** writes the samples shown in the
  current view, with video time, UTC timestamp and the label at each sample.
- **Capture** (transport bar) exports the exact current frame of each camera as
  PNG, or a composite snapshot: all camera frames at the playhead above the
  signals around it, with the label track.
- Mosaic videos can now include the sound of a chosen camera.

## Workflow

1. **Preprocessing** — select video segments per camera, concatenate, audio-sync,
   trim to common start/duration, and generate the project metadata file. You
   can set a global **Audio sync duration (s)** plus a per-camera
   **Audio sync start offset (s)** to skip idle time before synchronization
   analysis starts for each camera. To use videos that are already prepared,
   generate their metadata before loading the project in another section.
2. **Signal synchronisation** — load the project metadata, import HR and ECG CSV files separately, align
   them with the video timeline, and write one CSV per sensor directly in the
   project output folder (for example, `polar_1.csv`, `polar_2.csv`,
   `zephyr_1.csv`, or `ecg_1.csv`). Sensor numbering always starts at `_1`,
   even when only one sensor of that type is loaded. Select the plot type to view one signal kind at
   a time. Legacy projects containing combined synchronized CSVs remain
   supported. To replace source signals, select files in the Heart rate or ECG
   source list and click **Remove selected**, then add the new files and run
   **Load & synchronise HR / ECG** again. Multiple sources can be selected with
   Ctrl or Shift. Removal only unlinks sources from the project; it does not
   delete the original files or existing synchronized CSVs from disk.
3. **Synthetic PPG (optional)** — choose one of the ECG files produced by Signal
   sync, detect R peaks, and generate an absolute-timestamped synthetic PPG at
   125 Hz. Only samples in that synchronized ECG are used; the generator does
   not load the original ECG or look for peaks outside the imported interval.
   Output spans from the first detected peak through the last detected peak. It
   also exports detected RR intervals, one-second HR estimates from overlapping
   10-second FFT windows, and SDNN/RMSSD over the latest 30-300 accepted beats. The HR estimator uses
   sampling-rate-scaled smoothness-prior detrending, a 0.6-3.3 Hz band-pass, and
   continuity- and autocorrelation-supported fundamental selection to avoid
   mistaking the synthetic waveform's second harmonic for heart rate. Spectral
   peaks are refined between FFT bins to avoid stair-step BPM values. When synchronized Zephyr HR exists,
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

Enable **Options > Blur faces for privacy** to anonymize detected faces in every
camera preview, synchronized frame capture, mosaic video, and exported labelled
camera clip. The setting persists between application sessions. Original input
and preprocessing working videos are left unchanged so analysis data is not
destroyed; use only privacy-enabled exports when sharing material. Automatic
face detection can miss obscured or unusually angled faces, so review exported
material before distribution.

**Options > Face privacy settings** controls the style (smooth blur, pixelate,
solid mask), strength, detection quality, sensitivity, margin around the face,
and how long a face stays masked after detection is lost; it can test the
settings on the current frame of each camera. Compared with earlier versions:

- detection runs at up to 1920 px instead of 960 px (faces down to about 60 px
  wide in 4K footage), or with full-resolution tiles in *Thorough* mode
  (down to about 35 px, slower);
- faces are tracked between frames, so a briefly missed face stays masked
  instead of flickering;
- the masked area is larger (including the forehead and hair) and uses a
  feathered elliptical blur instead of coarse blocks;
- exported clips keep the original audio stream (copied, not re-encoded).

Blurred clips are necessarily re-encoded (H.264, CRF 20). The on-screen preview
samples a few frames per second, so its masks can trail very fast motion
slightly; exports are processed frame by frame.

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
