# Third-Party Notices

## PPGSynth

The synthetic PPG generator in `app/synthetic_ppg.py` adapts the dynamical PPG
model from PPGSynth:

- Q. Tang, Z. Chen, R. Ward et al., "Synthetic photoplethysmogram generation
  using two Gaussian functions," Scientific Reports 10, 13883 (2020).
- https://doi.org/10.1038/s41598-020-69076-x
- Upstream source supplied in this repository under `PPG_generator/PPG-Synthesis`.

The upstream project is licensed under the GNU General Public License, version
3. A complete copy is provided in `LICENSES/PPGSynth-GPL-3.0.txt`.

### Modifications

This application contains a Python adaptation integrated into the synchronized
video workflow. It accepts detected ECG R-peak timestamps, applies RR changes at
exact cumulative time boundaries, emits absolute UTC timestamps, crops output to
the first and last peaks detected within the synchronized ECG, estimates HR with
windowed frequency analysis, and derives SDNN and RMSSD from detected synthetic
pulses. Detailed methods and modifications are documented in
`docs/SYNTHETIC_PPG_METHODS.md`.

The software is provided without warranty, including without implied warranties
of merchantability or fitness for a particular purpose.

## OpenCV Zoo YuNet

Face privacy detection uses the `face_detection_yunet_2023mar.onnx` model from
OpenCV Zoo. YuNet is distributed under the MIT License. A complete copy is
provided in `LICENSES/OpenCV-Zoo-YuNet-MIT.txt`.

- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet