"""
Froth motion fusion + tile-wise stability analysis.

Combines:
  - multiscale phase correlation (global alignment cue)
  - dense Farneback optical flow
  - normalized cross-correlation block matching

Then:
  - fuses motion cues into block-wise motion vectors
  - smooths the motion field
  - estimates a global translation via RANSAC
  - computes tile-wise stability metrics:
      * motion_variance
      * smoothness_ratio
      * directional_stability
  - saves CSVs and visualizations (arrows, heatmaps, overlays)
"""

import argparse
import csv
import math
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------- Tuned parameters (edit here if needed) ----------
TILE_SIZE = 64
TILE_STEP = 32

OF_MAG_THRESHOLD = 0.2     # min |OF| mag to trust optical flow
PC_RESP_THRESHOLD = 0.02   # min phase correlation response
BM_CORR_THRESHOLD = 0.3    # min NCC score for block match

GAUSS_K = (7, 7)           # smoothing kernel for motion field
ARROW_SCALE = 8.0          # visual arrow length scaling

RANSAC_ITERS = 2500
RANSAC_THRESH = 12.0       # inlier threshold (pixels)

PYR_LEVELS = 4             # multiscale phase correlation levels
FARNEBACK_WINSIZE = 9      # Farneback neighborhood window size
# ------------------------------------------------------------


def preprocess(im: np.ndarray) -> np.ndarray:
    """Gaussian blur + histogram equalization, returned as float32."""
    im = cv2.GaussianBlur(im, (5, 5), 0)
    im = cv2.equalizeHist(im)
    return im.astype(np.float32)


def calculate_stability_from_block_matching(vx_list, vy_list):
    """Compute motion_variance, smoothness_ratio, directional_stability."""
    vx_array = np.array(vx_list, dtype=np.float32)
    vy_array = np.array(vy_list, dtype=np.float32)

    if vx_array.size == 0:
        return {
            "motion_variance": float("nan"),
            "smoothness_ratio": float("nan"),
            "directional_stability": float("nan"),
        }

    motion_variance = float(np.var(vx_array) + np.var(vy_array))

    displacement = np.sqrt(vx_array ** 2 + vy_array ** 2)
    smoothness = float(
        np.mean(displacement) / (np.std(displacement) + 1e-5)
    )

    angles = np.arctan2(vy_array, vx_array)
    direction_variance = float(np.var(angles))
    directional_stability = 1.0 / (direction_variance + 1e-5)

    return {
        "motion_variance": motion_variance,
        "smoothness_ratio": smoothness,
        "directional_stability": directional_stability,
    }


def multiscale_phase_correlation(a: np.ndarray, b: np.ndarray, levels: int):
    """Global translation via coarse-to-fine phase correlation."""
    pyr_a = [a]
    pyr_b = [b]
    for _ in range(1, levels):
        pyr_a.append(cv2.pyrDown(pyr_a[-1]))
        pyr_b.append(cv2.pyrDown(pyr_b[-1]))

    current_shift = np.array([0.0, 0.0], dtype=np.float32)

    for L in reversed(range(levels)):
        A = pyr_a[L]
        B = pyr_b[L]

        scale = 1.0 / (2 ** L)
        shift_level = current_shift * scale

        M = np.array(
            [[1.0, 0.0, shift_level[0]], [0.0, 1.0, shift_level[1]]],
            dtype=np.float32,
        )
        Aw = cv2.warpAffine(
            A,
            M,
            (A.shape[1], A.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        win_y = np.hanning(Aw.shape[0])
        win_x = np.hanning(Aw.shape[1])
        window = np.outer(win_y, win_x).astype(np.float32)

        try:
            shift_res, resp = cv2.phaseCorrelate(Aw, B, window)
        except cv2.error:
            shift_res = (0.0, 0.0)

        shift_res = np.array(shift_res, dtype=np.float32)
        current_shift += shift_res * (2 ** L)

    return current_shift


def run_froth_motion_analysis(
    img1_path: str,
    img2_path: str,
    out_dir: str,
    frame_interval: float,
):
    """
    Main analysis pipeline.

    Parameters
    ----------
    img1_path : str
        Path to first grayscale froth image (time t1).
    img2_path : str
        Path to second grayscale froth image (time t2).
    out_dir : str
        Output directory for CSVs and visualizations.
    frame_interval : float
        Time between frames in seconds (used for speed estimate).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load images -----
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Failed to load images. Check paths.")
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")

    h, w = img1.shape
    print(f"Loaded images with shape: {w} x {h}")

    # ----- Preprocess -----
    p1 = preprocess(img1)
    p2 = preprocess(img2)

    # ----- Global multiscale phase-correlation -----
    global_shift = multiscale_phase_correlation(p1, p2, levels=PYR_LEVELS)
    print(
        f"Global phase-correlation shift (px): "
        f"dx={global_shift[0]:.3f}, dy={global_shift[1]:.3f}"
    )

    # ----- Dense Farneback optical flow -----
    flow = cv2.calcOpticalFlowFarneback(
        p1.astype(np.uint8),
        p2.astype(np.uint8),
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=FARNEBACK_WINSIZE,
        iterations=7,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    u = flow[..., 0]
    v = flow[..., 1]

    # ----- Local phase-correlation + normalized cross-correlation block matching -----
    pc_vectors = []
    bm_vectors = []
    for y in range(0, h - TILE_SIZE + 1, TILE_STEP):
        for x in range(0, w - TILE_SIZE + 1, TILE_STEP):
            a = p1[y : y + TILE_SIZE, x : x + TILE_SIZE]
            b = p2[y : y + TILE_SIZE, x : x + TILE_SIZE]

            win_y = np.hanning(a.shape[0])
            win_x = np.hanning(a.shape[1])
            window = np.outer(win_y, win_x).astype(np.float32)

            try:
                sres, resp = cv2.phaseCorrelate(a, b, window)
            except cv2.error:
                continue

            dx_pc, dy_pc = float(sres[0]), float(sres[1])
            cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            pc_vectors.append((cx, cy, dx_pc, dy_pc, float(resp)))

            # Block matching via normalized cross-correlation
            sr = int(TILE_SIZE * 0.6)
            x0 = max(0, x - sr)
            x1 = min(w, x + TILE_SIZE + sr)
            y0 = max(0, y - sr)
            y1 = min(h, y + TILE_SIZE + sr)

            window_big = p2[y0:y1, x0:x1].astype(np.uint8)
            block = p1[y : y + TILE_SIZE, x : x + TILE_SIZE].astype(np.uint8)

            # skip nearly flat tiles
            if block.std() < 8:
                continue

            res = cv2.matchTemplate(window_big, block, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            best_x = x0 + max_loc[0]
            best_y = y0 + max_loc[1]
            dx_bm = best_x - x
            dy_bm = best_y - y
            score = float(max_val)
            bm_vectors.append((cx, cy, dx_bm, dy_bm, score))

    # ----- Aggregate per tile center -----
    center_to_data = {}
    for cy in range(TILE_SIZE // 2, h - TILE_SIZE // 2 + 1, TILE_STEP):
        for cx in range(TILE_SIZE // 2, w - TILE_SIZE // 2 + 1, TILE_STEP):
            center_to_data[(cx, cy)] = {"of": None, "pc": None, "bm": None}

    # Optical flow at centers
    for (cx, cy) in list(center_to_data.keys()):
        fx = int(np.clip(cx, 0, w - 1))
        fy = int(np.clip(cy, 0, h - 1))
        dx_of = float(u[fy, fx])
        dy_of = float(v[fy, fx])
        center_to_data[(cx, cy)]["of"] = (
            dx_of,
            dy_of,
            math.hypot(dx_of, dy_of),
        )

    # Phase correlation at centers
    for cx, cy, dx, dy, resp in pc_vectors:
        key = (int(cx), int(cy))
        if key in center_to_data:
            center_to_data[key]["pc"] = (dx, dy, resp)

    # Block matching at centers
    for cx, cy, dx, dy, score in bm_vectors:
        key = (int(cx), int(cy))
        if key in center_to_data:
            center_to_data[key]["bm"] = (dx, dy, score)

    # ----- Fuse motion cues with weights -----
    fused_pts = []
    for (cx, cy), d in center_to_data.items():
        of = d["of"]
        pc = d["pc"]
        bm = d["bm"]
        vecs = []

        # OF contribution
        if of is not None and of[2] >= OF_MAG_THRESHOLD:
            w_of = (of[2] + 0.1) * 0.8
            vecs.append((of[0], of[1], w_of))

        # Phase correlation contribution
        if pc is not None and pc[2] >= PC_RESP_THRESHOLD:
            w_pc = max(pc[2], 0.001) * 1.4
            vecs.append((pc[0], pc[1], w_pc))

        # Block-matching contribution
        if bm is not None and bm[2] >= BM_CORR_THRESHOLD:
            w_bm = max(0.001, bm[2]) * 1.8
            vecs.append((bm[0], bm[1], w_bm))

        if not vecs:
            continue

        ws = np.array([v[2] for v in vecs], dtype=np.float32)
        vals_dx = np.array([v[0] for v in vecs], dtype=np.float32)
        vals_dy = np.array([v[1] for v in vecs], dtype=np.float32)

        fused_dx = float((vals_dx * ws).sum() / (ws.sum() + 1e-8))
        fused_dy = float((vals_dy * ws).sum() / (ws.sum() + 1e-8))
        conf = float(np.mean(ws))
        fused_pts.append((cx, cy, fused_dx, fused_dy, conf))

    # ----- Smooth motion field -----
    dx_map = np.full((h, w), np.nan, dtype=np.float32)
    dy_map = np.full((h, w), np.nan, dtype=np.float32)
    for cx, cy, dx, dy, conf in fused_pts:
        dx_map[cy, cx] = dx
        dy_map[cy, cx] = dy

    dx_map_f = dx_map.copy()
    dy_map_f = dy_map.copy()
    nan_mask = np.isnan(dx_map_f)
    dx_map_f[nan_mask] = 0.0
    dy_map_f[nan_mask] = 0.0

    dx_s = cv2.GaussianBlur(dx_map_f, GAUSS_K, 0)
    dy_s = cv2.GaussianBlur(dy_map_f, GAUSS_K, 0)

    smoothed_vectors = []
    for cx, cy, _, _, conf in fused_pts:
        sdx = float(dx_s[cy, cx])
        sdy = float(dy_s[cy, cx])
        smoothed_vectors.append((cx, cy, sdx, sdy, conf))

    # ----- Save fused vectors CSV -----
    vectors_csv = out_dir / "fused_vectors.csv"
    with open(vectors_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "dx_px", "dy_px", "conf"])
        for cx, cy, dx, dy, conf in smoothed_vectors:
            writer.writerow([int(cx), int(cy), float(dx), float(dy), float(conf)])
    print(f"Saved fused vectors CSV: {vectors_csv}")

    # ----- RANSAC translation-only model (from fused vectors) -----
    best_tx = 0.0
    best_ty = 0.0
    best_mask = None
    best_in = 0

    if len(fused_pts) >= 3:
        src = np.array([[p[0], p[1]] for p in fused_pts], dtype=np.float32)
        dst = np.array([[p[0] + p[2], p[1] + p[3]] for p in fused_pts], dtype=np.float32)
        N = len(src)
        for _ in range(RANSAC_ITERS):
            i = random.randrange(N)
            tx_try = dst[i, 0] - src[i, 0]
            ty_try = dst[i, 1] - src[i, 1]
            preds = src + np.array([tx_try, ty_try], dtype=np.float32)
            dists = np.linalg.norm(dst - preds, axis=1)
            mask = dists < RANSAC_THRESH
            cnt = int(mask.sum())
            if cnt > best_in:
                best_in = cnt
                best_tx = float(tx_try)
                best_ty = float(ty_try)
                best_mask = mask.copy()

        if best_mask is not None:
            inlier_src = src[best_mask]
            inlier_dst = dst[best_mask]
            in_disp = inlier_dst - inlier_src
            mean_dx = float(np.mean(in_disp[:, 0]))
            mean_dy = float(np.mean(in_disp[:, 1]))
        else:
            mean_dx = mean_dy = 0.0

        print(
            f"RANSAC translation (local vectors): "
            f"tx={best_tx:.3f}, ty={best_ty:.3f}, inliers={best_in}/{N}"
        )
        print(
            f"Mean inlier displacement: "
            f"dx={mean_dx:.3f}, dy={mean_dy:.3f}"
        )
    else:
        print("Not enough fused points for RANSAC translation.")

    # ----- High-visibility arrow overlay on froth image -----
    vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for cx, cy, dx, dy, conf in smoothed_vectors:
        start = (int(cx), int(cy))
        end = (int(cx + dx * ARROW_SCALE), int(cy + dy * ARROW_SCALE))
        mag = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        hue = int(((ang + math.pi) / (2 * math.pi)) * 179)
        val = min(255, int(60 + mag * 12))
        color_hsv = np.uint8([[[hue, 200, val]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.arrowedLine(vis, start, end, color_bgr, 2, tipLength=0.35)

    # Draw global translation (from RANSAC) as blue arrow at top
    mid = (w // 2, h // 10)
    glob_end = (int(mid[0] + best_tx * ARROW_SCALE), int(mid[1] + best_ty * ARROW_SCALE))
    cv2.arrowedLine(vis, mid, glob_end, (255, 0, 0), 3, tipLength=0.35)

    overlay_vectors_path = out_dir / "motion_vectors_overlay.png"
    cv2.imwrite(str(overlay_vectors_path), vis)
    print(f"Saved arrow overlay: {overlay_vectors_path}")

    # ----- Quiver plot (matplotlib) -----
    qx = [p[0] for p in smoothed_vectors]
    qy = [p[1] for p in smoothed_vectors]
    qu = [p[2] * ARROW_SCALE for p in smoothed_vectors]
    qv = [p[3] * ARROW_SCALE for p in smoothed_vectors]

    plt.figure(figsize=(10, 8))
    plt.imshow(img2, cmap="gray")
    plt.quiver(qx, qy, qu, qv, angles="xy", scale_units="xy", scale=1, width=0.005)
    plt.gca().invert_yaxis()
    plt.axis("off")
    quiver_path = out_dir / "motion_vectors_quiver.png"
    plt.savefig(str(quiver_path), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved quiver plot: {quiver_path}")

    # ----- Tile-wise stability heatmap -----
    tiles = []
    stability_vals = []

    for y in range(0, h - TILE_SIZE + 1, TILE_STEP):
        for x in range(0, w - TILE_SIZE + 1, TILE_STEP):
            vx_list = []
            vy_list = []
            for sx, sy, sdx, sdy, conf in smoothed_vectors:
                if (x <= sx < x + TILE_SIZE) and (y <= sy < y + TILE_SIZE):
                    vx_list.append(sdx)
                    vy_list.append(sdy)
            st = calculate_stability_from_block_matching(vx_list, vy_list)
            tiles.append(((x + TILE_SIZE // 2, y + TILE_SIZE // 2), st))
            ds_val = st["directional_stability"]
            stability_vals.append(0.0 if math.isnan(ds_val) else ds_val)

    arr_ds = np.array(stability_vals, dtype=np.float32)
    arr_ds = np.nan_to_num(arr_ds)
    if arr_ds.size > 0:
        lo = float(np.percentile(arr_ds, 5))
        hi = float(np.percentile(arr_ds, 95))
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-6:
        hi = lo + 1.0

    norm_ds = np.clip((arr_ds - lo) / (hi - lo), 0.0, 1.0)

    rows = (h - TILE_SIZE) // TILE_STEP + 1
    cols = (w - TILE_SIZE) // TILE_STEP + 1
    grid = np.zeros((rows, cols), dtype=np.float32)
    idx = 0
    for ry in range(rows):
        for rx in range(cols):
            grid[ry, rx] = norm_ds[idx] if idx < len(norm_ds) else 0.0
            idx += 1

    heat_up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_up = cv2.GaussianBlur(heat_up, (31, 31), 0)
    heat_u8 = (np.clip(heat_up, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    heatmap_path = out_dir / "tile_stability_heatmap.png"
    heat_overlay_path = out_dir / "tile_stability_overlay.png"
    combo_path = out_dir / "motion_and_stability_overlay.png"

    base_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(str(heatmap_path), heat_color)
    cv2.imwrite(str(heat_overlay_path), overlay)

    combo = cv2.addWeighted(vis, 0.7, heat_color, 0.3, 0)
    cv2.imwrite(str(combo_path), combo)

    print(f"Saved stability heatmap: {heatmap_path}")
    print(f"Saved stability overlay: {heat_overlay_path}")
    print(f"Saved motion+stability overlay: {combo_path}")

    # ----- Save tile CSV -----
    tile_csv = out_dir / "tile_stability_values.csv"
    with open(tile_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            [
                "tile_cx",
                "tile_cy",
                "directional_stability",
                "motion_variance",
                "smoothness_ratio",
            ]
        )
        for (cx, cy), st in tiles:
            ds = st["directional_stability"]
            mv = st["motion_variance"]
            sm = st["smoothness_ratio"]
            wcsv.writerow(
                [
                    int(cx),
                    int(cy),
                    float(0.0 if math.isnan(ds) else ds),
                    float(0.0 if math.isnan(mv) else mv),
                    float(0.0 if math.isnan(sm) else sm),
                ]
            )
    print(f"Saved tile stability CSV: {tile_csv}")

    # ----- Average stability metrics (no pandas) -----
    motion_var_vals = []
    smoothness_vals = []
    directional_vals = []
    for _, st in tiles:
        mv = st["motion_variance"]
        sm = st["smoothness_ratio"]
        ds = st["directional_stability"]
        if not (math.isnan(mv) or math.isinf(mv)):
            motion_var_vals.append(mv)
        if not (math.isnan(sm) or math.isinf(sm)):
            smoothness_vals.append(sm)
        if not (math.isnan(ds) or math.isinf(ds)):
            directional_vals.append(ds)

    avg_motion_var = (
        sum(motion_var_vals) / len(motion_var_vals)
        if motion_var_vals
        else float("nan")
    )
    avg_smoothness = (
        sum(smoothness_vals) / len(smoothness_vals)
        if smoothness_vals
        else float("nan")
    )
    avg_directional = (
        sum(directional_vals) / len(directional_vals)
        if directional_vals
        else float("nan")
    )

    print("\nAverage stability metrics:")
    print(f"  motion_variance     = {avg_motion_var:.4f}")
    print(f"  smoothness_ratio    = {avg_smoothness:.4f}")
    print(f"  directional_stability = {avg_directional:.4f}")

    # ----- Global speed estimate from phase-correlation shift -----
    dx, dy = float(global_shift[0]), float(global_shift[1])
    displacement = math.hypot(dx, dy)
    if frame_interval > 0.0:
        speed = displacement / frame_interval
        speed_x = dx / frame_interval
        speed_y = dy / frame_interval
    else:
        speed = speed_x = speed_y = float("nan")

    print("\nGlobal displacement / speed (using phase correlation):")
    print(f"  dx = {dx:.3f} px, dy = {dy:.3f} px")
    print(f"  |disp| = {displacement:.3f} px over dt = {frame_interval:.3f} s")
    print(f"  speed magnitude = {speed:.3f} px/s")
    print(f"  speed components: vx = {speed_x:.3f}, vy = {speed_y:.3f} px/s")

    # Return a summary dict (useful if imported as a library)
    return {
        "global_shift": (dx, dy),
        "frame_interval": frame_interval,
        "speed": {
            "vx": speed_x,
            "vy": speed_y,
            "v": speed,
        },
        "avg_stability": {
            "motion_variance": avg_motion_var,
            "smoothness_ratio": avg_smoothness,
            "directional_stability": avg_directional,
        },
        "paths": {
            "fused_vectors_csv": str(vectors_csv),
            "tile_stability_csv": str(tile_csv),
            "overlay_vectors": str(overlay_vectors_path),
            "quiver": str(quiver_path),
            "heatmap": str(heatmap_path),
            "heat_overlay": str(heat_overlay_path),
            "motion_stability_overlay": str(combo_path),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Froth motion fusion + tile-wise stability analysis."
    )
    parser.add_argument("img1", help="Path to first grayscale image (t1).")
    parser.add_argument("img2", help="Path to second grayscale image (t2).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="froth_motion_output",
        help="Output directory (default: froth_motion_output).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time between frames in seconds (default: 1.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_froth_motion_analysis(
        img1_path=args.img1,
        img2_path=args.img2,
        out_dir=args.out_dir,
        frame_interval=args.dt,
    )
