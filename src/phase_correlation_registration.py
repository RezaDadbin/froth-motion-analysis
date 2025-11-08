"""
Multiscale phase-correlation registration with visualization.

Aligns image1 to image2 using a coarse-to-fine phase-correlation scheme and
saves:
  - aligned version of image1
  - residual heatmap
  - side-by-side visualization with residual
  - arrow overlay indicating the estimated global shift
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess(im: np.ndarray) -> np.ndarray:
    """Apply mild blur + histogram equalization."""
    im = cv2.GaussianBlur(im, (5, 5), 0)
    im = cv2.equalizeHist(im)
    return im


def build_pyramid(im: np.ndarray, levels: int) -> list:
    """Build Gaussian pyramid: level 0 = original, higher = downsampled."""
    pyr = [im.astype(np.float32)]
    for _ in range(1, levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def multiscale_phase_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    levels: int = 4,
):
    """
    Coarse-to-fine phase correlation to estimate translation (image1 -> image2).

    Returns
    -------
    shift : np.ndarray, shape (2,)
        (dx, dy) in original-image pixels.
    responses : list of tuples
        [(level, resp, dx_level, dy_level), ...]
    """
    pyr1 = build_pyramid(img1, levels)
    pyr2 = build_pyramid(img2, levels)

    current_shift = np.array([0.0, 0.0], dtype=np.float32)
    responses = []

    for L in reversed(range(levels)):  # from coarsest to finest
        im1_l = pyr1[L]
        im2_l = pyr2[L]

        scale_factor = 1.0 / (2 ** L)
        shift_level = current_shift * scale_factor

        # warp im1_l by current estimate at this scale
        M = np.array(
            [[1.0, 0.0, shift_level[0]], [0.0, 1.0, shift_level[1]]],
            dtype=np.float32,
        )
        im1_w = cv2.warpAffine(
            im1_l,
            M,
            (im1_l.shape[1], im1_l.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Hanning window to reduce edge effects
        win_y = np.hanning(im1_w.shape[0])
        win_x = np.hanning(im1_w.shape[1])
        window = np.outer(win_y, win_x).astype(np.float32)

        try:
            shift_res, resp = cv2.phaseCorrelate(im1_w, im2_l, window)
        except cv2.error:
            shift_res, resp = (0.0, 0.0), 0.0

        shift_res = np.array(shift_res, dtype=np.float32)  # (dx, dy) in this level
        current_shift += shift_res * (2 ** L)  # back to original-pixel units

        responses.append(
            (L, float(resp), float(shift_res[0]), float(shift_res[1]))
        )
        print(
            f"Level {L}: shift_res(level)=({shift_res[0]:.4f}, {shift_res[1]:.4f}), "
            f"resp={resp:.4f}, current_shift(orig)=({current_shift[0]:.4f},{current_shift[1]:.4f})"
        )

    return current_shift, responses


def align_and_compute_residual(
    img1: np.ndarray,
    img2: np.ndarray,
    shift: np.ndarray,
):
    """Warp img1 into img2 coordinates and compute residual + basic stats."""
    h, w = img1.shape
    dx, dy = float(shift[0]), float(shift[1])

    M_final = np.array(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        dtype=np.float32,
    )
    img1_warp = cv2.warpAffine(
        img1.astype(np.float32),
        M_final,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    img1_warp_u8 = np.clip(img1_warp, 0, 255).astype(np.uint8)

    residual = cv2.absdiff(img2, img1_warp_u8)
    mean_resid = float(np.mean(residual))
    median_resid = float(np.median(residual))
    max_resid = int(np.max(residual))

    stats = {
        "mean_residual": mean_resid,
        "median_residual": median_resid,
        "max_residual": max_resid,
    }
    return img1_warp_u8, residual, stats


def save_visualizations(
    img2: np.ndarray,
    img1_warp_u8: np.ndarray,
    residual: np.ndarray,
    shift: np.ndarray,
    out_dir: Path,
    show: bool = True,
):
    """Save aligned image, residual heatmap, side-by-side, and arrow overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = img2.shape
    dx, dy = float(shift[0]), float(shift[1])

    aligned_path = out_dir / "aligned_image1.png"
    residual_path = out_dir / "residual_heatmap.png"
    side_path = out_dir / "side_by_side.png"
    overlay_path = out_dir / "shift_arrow_overlay.png"

    # Arrow overlay on target image
    vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    start = (w // 2, h // 8)
    end = (int(start[0] + dx * 2.0), int(start[1] + dy * 2.0))
    cv2.arrowedLine(vis, start, end, (255, 0, 0), 3, tipLength=0.3)
    cv2.putText(
        vis,
        f"dx={dx:.2f}, dy={dy:.2f}",
        (start[0] - 120, start[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    # Residual as heatmap
    res_col = cv2.applyColorMap(
        np.clip((residual * 2).astype(np.uint8), 0, 255),
        cv2.COLORMAP_JET,
    )

    # Side-by-side: arrow overlay (target), aligned1, residual heatmap
    side = np.concatenate(
        [
            vis,
            cv2.cvtColor(img1_warp_u8, cv2.COLOR_GRAY2BGR),
            res_col,
        ],
        axis=1,
    )

    cv2.imwrite(str(aligned_path), img1_warp_u8)
    cv2.imwrite(str(residual_path), res_col)
    cv2.imwrite(str(side_path), side)
    cv2.imwrite(str(overlay_path), vis)

    print("\nOutputs saved:")
    print(f" - Aligned image1:      {aligned_path}")
    print(f" - Residual heatmap:    {residual_path}")
    print(f" - Side-by-side view:   {side_path}")
    print(f" - Shift arrow overlay: {overlay_path}")

    # Optional inline visualization
    if show:
        plt.figure(figsize=(14, 6))
        plt.imshow(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(
            "Left: image2 with shift arrow | "
            "Middle: image1 warped | Right: residual heatmap"
        )
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multiscale phase-correlation registration between two grayscale images."
        )
    )
    parser.add_argument("img1", help="Path to first image (source).")
    parser.add_argument("img2", help="Path to second image (target).")
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Number of pyramid levels (default: 4).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="phase_correlation_output",
        help="Output directory for images (default: phase_correlation_output).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display matplotlib figure.",
    )
    args = parser.parse_args()

    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Failed to load one or both images.")
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")

    h, w = img1.shape
    print(f"Image size: {w} x {h}")

    pimg1 = preprocess(img1)
    pimg2 = preprocess(img2)

    shift, responses = multiscale_phase_correlation(
        pimg1, pimg2, levels=args.levels
    )
    dx, dy = float(shift[0]), float(shift[1])
    print(
        f"\nFINAL estimated shift (pixels, image1 -> image2): "
        f"dx = {dx:.4f}, dy = {dy:.4f}"
    )

    img1_warp_u8, residual, stats = align_and_compute_residual(
        img1, img2, shift
    )
    print(
        "Residual stats after alignment: "
        f"mean={stats['mean_residual']:.2f}, "
        f"median={stats['median_residual']:.2f}, "
        f"max={stats['max_residual']}"
    )

    save_visualizations(
        img2,
        img1_warp_u8,
        residual,
        shift,
        Path(args.out_dir),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
