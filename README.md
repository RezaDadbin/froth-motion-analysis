# Froth Motion and Stability Analysis

Froth Motion and Stability Analysis is a Python pipeline for estimating motion and stability of flotation froth between consecutive frames. The project combines multiscale phase correlation, optical flow, and block matching to deliver global translation estimates, dense local motion fields, and tile-wise stability metrics.

## Research Status and Data Availability

This repository forms part of broader froth-image analysis research. Experimental work completed; a data-paper manuscript is currently in preparation.

The research imagery is not distributed with this repository. Users should provide their own authorized consecutive froth frames. Code is provided for research and reproducibility with compatible, independently supplied data. Exact reproduction of the private experiments also requires their data, splits, configuration, and checkpoints.

## Authors

- Sina Lotfi
- Reza Dadbin

## Features

- Multiscale phase-correlation registration with residual diagnostics.
- Fusion of optical flow, local phase correlation, and block matching into smoothed motion vectors.
- RANSAC-based global translation estimation and tile-wise stability metrics.
- Visualization outputs: arrow overlays, heatmaps, and CSV summaries.

## Repository Structure

```
froth-motion-analysis/
├── README.md
├── src/
│   ├── phase_correlation_registration.py
│   └── froth_motion_and_stability.py
├── data/
│   └── input/
│       ├── .gitkeep
│       ├── frame1.tiff   # User-supplied frame; gitignored
│       └── frame2.tiff   # User-supplied consecutive frame; gitignored
└── outputs/        # Generated figures and CSVs (gitignored)
```

- `data/input/` is the expected location for your own authorized consecutive froth frames. Only the directory placeholder is tracked; TIFF/TIF images anywhere under `data/` are ignored by Git.
- `outputs/` is used by the scripts to store generated visualizations and CSV files.

## Installation

Create a Python 3 virtual environment and install the dependencies used by the scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python matplotlib
```

On Windows, activate with `.venv\Scripts\activate`. There is no `requirements.txt` in the current source tree; record installed versions when reproducing a run.

The project primarily relies on:

- `numpy`
- `opencv-python`
- `matplotlib`
- `pandas` (optional, useful for further CSV analysis)

## Usage

### 1. Global Multiscale Phase Correlation

File: `src/phase_correlation_registration.py`

Estimate global translation between two grayscale froth images via a coarse-to-fine phase-correlation pyramid, then warp image 1 into image 2 and report residual statistics.

**Example:**

```bash
python src/phase_correlation_registration.py data/input/frame1.tiff data/input/frame2.tiff \
    --out-dir outputs/phase_correlation
```

**Key Arguments:**

- `img1`, `img2`: Paths to the source and target grayscale images.
- `--levels`: Number of pyramid levels (default: 4).
- `--out-dir`: Directory for saved outputs (default: `phase_correlation_output`).
- `--no-show`: Disable interactive Matplotlib display.

**Outputs:**

- `aligned_image1.png`: Warped version of image 1 aligned to image 2.
- `residual_heatmap.png`: Heatmap of absolute residuals.
- `side_by_side.png`: Composite view of image 2, aligned image 1, and the residual heatmap.
- `shift_arrow_overlay.png`: Image 2 annotated with the estimated global shift.

The script also prints the estimated shift `(dx, dy)` and residual statistics (mean, median, max).

### 2. Froth Motion Fusion and Stability Analysis

File: `src/froth_motion_and_stability.py`

Compute dense local motion vectors and tile-wise stability metrics by fusing optical flow, local phase correlation, and block matching cues. Includes RANSAC-based global translation estimation and visualizations.

**Example:**

```bash
python src/froth_motion_and_stability.py data/input/frame1.tiff data/input/frame2.tiff \
    --out-dir outputs/froth_motion --dt 1.5
```

**Key Arguments:**

- `img1`, `img2`: Paths to the consecutive froth images.
- `--out-dir`: Output directory for figures and CSV files (default: `froth_motion_output`).
- `--dt`: Time interval between frames in seconds (default: 1.0).

**Outputs:**

- `fused_vectors.csv`: Fused, smoothed motion vectors at tile centers.
- `tile_stability_values.csv`: Tile-wise stability metrics.
- `motion_vectors_overlay.png`: Color-coded arrows showing local motion.
- `motion_vectors_quiver.png`: Matplotlib quiver plot of motion vectors.
- `tile_stability_heatmap.png`: Colorized stability heatmap.
- `tile_stability_overlay.png`: Stability heatmap blended with the froth image.
- `motion_and_stability_overlay.png`: Combined motion and stability visualization.

The console reports global phase-correlation shift, RANSAC translation statistics, average stability metrics, and global speed estimates derived from the phase-correlation displacement and provided `dt`.

## Interpreting Results

- **Stable regions**: low `motion_variance`, high `smoothness_ratio`, high `directional_stability`.
- **Unstable regions**: high `motion_variance`, low `smoothness_ratio`, low `directional_stability`.

Use the overlays and CSV outputs to locate areas of interest on the froth surface.

## Customization Tips

Key parameters such as tile sizing, motion thresholds, pyramid levels, and optical flow window sizes are defined near the top of `froth_motion_and_stability.py`. Adjust them to match image resolution, froth texture, and process conditions.


## Reproducibility and Scope

Record the input frame pair, acquisition interval, parameter settings, dependency versions, and repository commit. Use imagery you are authorized to process and redistribute. Motion and stability outputs depend on texture, imaging conditions, and parameter choices; this README does not claim validated production performance.
