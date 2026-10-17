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
metadata and path references. Original recordings, sensor CSVs and interval
labels are unchanged.
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

The **⋯** button renames or deletes a template, or marks it **Use at
startup** (shown with ★): the application then starts with that template's
labels, and Preprocessing with its cameras and options while no cameras are
loaded. Loaded metadata is never changed automatically.
Templates are stored per user in the application settings and shared by all
datasets. Segments, output folders and signal files are not part of templates.

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
5. **PPG alignment** — measure and correct the remaining offset between the
   videos and the signals (see [PPG alignment](#ppg-alignment)).
6. **Review** — load a manifest CSV to browse through exported segments,
   including synthetic PPG when it was generated and exported.
7. **Artifacts** (tab on the right of the toolbar) — once the dataset is
   reviewed, delete the processing artifacts or move them out of the dataset,
   and move them back if needed
   (see [Cleaning up artifacts](#cleaning-up-artifacts)).

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

## PPG alignment

Camera start times come from the file names (1 s resolution) plus manual time
corrections, so the signals can be several seconds early or late relative to
the video. For waveform-level work (training rPPG models on the synthetic PPG)
the pulse must line up within a fraction of a heartbeat. This section measures
the offset from the video itself.

1. Pick a labelled segment where the face is clearly visible and the
   participant is still (a resting or baseline segment), and the camera to use
   (the front camera by default).
2. **Estimate this segment** tracks the face (YuNet), averages the skin-coloured
   pixels of the inner face box in every frame and turns the colour into a pulse
   waveform with POS, CHROM or Green. A 60 s clip takes about 10 s with hardware
   decoding. Results are cached in `<output>/ppg_alignment/cache/`.
   **Estimate all** does the same for every segment not estimated yet, or only
   for the segments of one label when one is chosen in **Estimate all for**.
3. The graph overlays the estimated rPPG (blue) and the synthetic PPG (orange)
   shifted by the current offset. Below it, the correlation for every offset of
   the search range (default ±10 s) is computed on the shaded window; drag its
   edges (grab them anywhere near the dashed line, or by the round handle
   pointing outward), or use **Window: visible range**, to keep a calm part of
   the segment.
4. Choose the offset:
   - **Best in range** jumps to the highest correlation;
   - **◀ Previous peak / Next peak ▶** move to the neighbouring correlation peak
     (the pulse is nearly periodic, so peaks repeat about one heartbeat apart);
     **Nearest** snaps to the closest peak;
   - switch the graph's **Mouse drag** toggle to **Shift PPG** and drag the
     curves (or Shift+drag in either mode), drag the marker or click a peak in
     the correlation plot, or type a value.

   The toggle starts in **Zoom**: dragging selects the region to focus on, as in
   the other graphs (a taller box also zooms the y-axis), **Undo zoom** steps
   back and **Full range** shows everything. Ctrl+drag pans and the wheel zooms
   time in both modes.
5. Check other segments: the offset is kept when you change segment, so you can
   see whether it also aligns them. **Pool estimated segments** (Search)
   replaces the correlation with its average over every 30 s window of the
   segments estimated for the camera (**Estimate all** first). While pooling,
   each segment of the list has a tick box: untick a segment to leave it out
   (**Pooled segments: All / None** ticks or unticks them all; right-click a
   segment to tick or untick every segment with its label, or tick only those).
   It leaves out
   windows where the face was lost or where the rPPG and PPG heart rates differ
   by more than **Max HR difference** (3 bpm by default; changing it takes
   effect at once). If no window passes, the graph pools every window where the
   face was found instead, marked **unreliable**, and each segment lists the
   video / PPG heart rate of its windows. That means the rPPG is not following
   the pulse: try another method or camera before trusting the offset. In a single window the correlation peaks every heartbeat
   with almost equal heights. Segments at different heart rates put those
   neighbouring peaks at different offsets but share the true one, so pooling
   makes it stand out; the peak buttons then work on the pooled curve. Below the
   summary, each segment lists its own peak nearest the offset: they should
   agree within about 0.1 s.
6. **Apply to project …** stores the offset and rewrites the HR, ECG and
   synthetic PPG slices and the timing (`signal_anchor_datetime_utc`,
   `segment_start/end_datetime_utc`, `video_signal_offset_sec`) of every
   exported segment, updates the metadata sidecar and appends the change to
   `<output>/ppg_alignment/alignment.json`. Videos are not modified. Applying
   again replaces the offset (it does not add up).
7. **Remove offset …** returns to the original camera timing (it applies 0 s).
   Every apply cuts the slices again from the project-level signals, so the
   result never depends on earlier applications: removing the offset restores
   the segments exactly, and to change the offset you simply apply the new value.

**Import offset …** reads the offset applied in another project, for example
the same participant's baseline session (its `ppg_alignment/alignment.json`,
metadata sidecar or a segment `meta.json`), so you can inspect it on
this project's segments before applying it. Sessions recorded the same day do
not always share an offset — each project's camera time correction is set
separately — so always check the correlation plot of the target session:
**Nearest** refines an imported value that is close, and a different best peak
means the sessions differ.

Conventions and notes:

- A positive offset places the signals later on the video timeline; the
  corrected anchor is `camera anchor − offset`. Signal sync, Labelling exports
  and the sidecar all include it.
- Every method returns a waveform with the polarity of a contact PPG (it rises
  with blood volume). Raw POS has the opposite sign and is inverted; aligning to
  raw POS would place the PPG half a heartbeat off.
- The **Show** row under the transport bar selects the cameras and the face ROI
  video (the region used, with non-skin pixels dimmed) to play.
- The heart rate of the rPPG and of the synthetic PPG are shown for the window;
  when they differ, the estimate is unreliable there (motion, lighting, face
  lost) and the window or segment should be changed.
- The synthetic PPG places each pulse at a fixed fraction of its RR interval, so
  its delay after the R peak shortens at high heart rates. Together with real
  changes in pulse transit time, segments of one session can differ by up to
  about 0.1 s; prefer calm (resting) segments.
- Synthetic PPGs generated before 2026-10-15 drift against their own ECG peaks
  (by about 0.3 s over a 35-minute session), so no single offset aligns every
  segment. The section shows a warning for such a file: regenerate it in
  **Synthetic PPG**, then apply the offset here, which re-slices every exported
  segment.

## Cleaning up artifacts

The last section, **7. Artifacts**, removes the files the pipeline needed but
the finished dataset does not. It works on the project output folder (the one
that contains `labelled_segments/`); choose another folder with **…**.

- **Full-length videos** — the synchronised, trimmed full-session videos
  (`cam1_<label>_final.mp4`) in the root of the output folder. The clips inside
  `labelled_segments/` are never touched.
- **PPG alignment files** — everything in `ppg_alignment/` (face-trace cache,
  ROI videos, `alignment.json`). The applied offset stays in every segment's
  `meta.json` and in the metadata sidecar.
- **Intermediate videos** — `cam*_concat` / `cam*_trimstart` videos, when
  intermediates were kept during preprocessing.

Choose the categories, then either **Move to an artifacts folder** or **Delete
permanently**. Moving recreates the dataset layout in the artifacts folder:
videos at its root and `ppg_alignment/...` in `ppg_alignment/`. The folder is
created in **Location** (by default next to the dataset folder; the last
location chosen is remembered) and is named `<dataset folder>_artifacts`
unless you type another **Folder name**. It must lie outside the dataset.
Existing files there are only replaced after confirmation. Moving within one
drive is instant; to another drive the files are copied with progress and the
originals are removed once the copy is complete (**Cancel** leaves the current
file in the dataset). Folders emptied by the clean-up are removed.

A move also writes `artifacts_manifest.json` at the root of the artifacts
folder: the dataset the files came from, and each file's size and category.

The right-hand pane lists every file concerned with its size and destination;
files that already exist at the destination are marked.
Labelled segments, signals, synthetic PPG and metadata always
stay in the dataset. Once the full-length videos are gone, Preprocessing,
Labelling and PPG alignment can no longer play the full session, and the
labelled segments cannot be exported again, so clean up after the review.
To import an alignment offset from a cleaned-up project, pick the
`alignment.json` in its artifacts folder or a segment `meta.json`.

### Restoring moved artifacts

Choose **Restore from an artifacts folder (move back)**. **Location** and
**Folder name** then designate the artifacts folder to read — by default the one
a move from this dataset would have used — or pick it with **Choose artifacts
folder …**. The category boxes now count the files in the artifacts folder, and
each file goes back to the same place in the dataset. Files that are not
artifacts (and leftovers of an interrupted copy) are listed as *left in place*
and are not moved. Validation, shown above the list and again before starting:

- errors (restore refused): the dataset or the artifacts folder is missing, one
  folder lies inside the other, nothing is selected, or the dataset drive lacks
  the free space for the files that must be copied;
- warnings (confirmation required): the manifest is missing, the files were moved
  out of another dataset (or the dataset has moved since), a file's size changed
  since it was moved, files recorded in the manifest are gone, or a full-length
  video does not belong to the open project;
- files that already exist in the dataset: **Replace** them or **Skip existing**.

Nothing is ever restored into `labelled_segments/`. Restored files are removed
from the manifest; when the artifacts folder ends up empty it is deleted.
Camera views that lost their video in a move made during the same session show
it again after the restore; otherwise load the project metadata again.

## Adding a signal loader

Place a new Python module in `app/signals/`.  Implement the `SignalLoader` ABC
with `can_load(path) -> bool` and `load(path) -> pd.DataFrame`.  The module is
auto-discovered at startup via `pkgutil`.

## New project and switching datasets

**File > New project** (Ctrl+N) clears every section — cameras, signals,
intervals, plots and players — so another set of videos can be processed.
Loading the metadata of another dataset in any section does the same, then
loads that dataset everywhere; loading the metadata of the dataset already open
only refreshes it. New project asks for confirmation when something is loaded;
switching datasets asks only when intervals drawn in Labelling were never
exported. Files on disk are never changed.

## What is saved

There is no project file. Each dataset is described by `project_meta.json` in
its output folder, written by Preprocessing and updated by Signal sync,
Synthetic PPG and PPG alignment; load it in any section with **Load project
metadata**. Exported labelled segments are listed in their `manifest.csv`.
Intervals drawn in Labelling are kept only until the application closes, so
export them before quitting.

## PPGSynth license

The synthetic PPG dynamical model is adapted from PPGSynth by Tang et al. and
is distributed under GNU GPL v3. See `THIRD_PARTY_NOTICES.md` and
`LICENSES/PPGSynth-GPL-3.0.txt`. Distribution of this combined application must
comply with the GPL; consult qualified counsel for legal advice about a specific
distribution model.
