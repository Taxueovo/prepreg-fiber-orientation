"""Gradient Structure Tensor baseline for fiber-orientation estimation.

This classical computer-vision baseline provides a reference point for
FiberAngleNet.

References:
- Jähne, B. (1993). Spatio-Temporal Image Processing. Springer.
- Bigun, J. & Granlund, G. (1987). Optimal Orientation Detection of Linear Symmetry.
"""

import os
import csv
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def gst_orientation(image, sigma=3.0, rho=15.0, patch_size=32):
    """Estimate local dominant fiber orientation with a structure tensor.

    Args:
        image: Grayscale image with shape ``(H, W)``.
        sigma: Gaussian standard deviation for gradient smoothing.
        rho: Gaussian standard deviation for tensor smoothing.
        patch_size: Block size; one angle is returned per block.

    Returns:
        angles: Per-block acute orientations in degrees, in ``[0, 90)``.
        coherence: Per-block orientation coherence in ``[0, 1]``.
    """
    H, W = image.shape

    # Step 1: estimate image gradients.
    from scipy.ndimage import sobel

    # Approximate gradients with Sobel operators.
    Gx = sobel(image, axis=1)  # Horizontal gradient.
    Gy = sobel(image, axis=0)  # Vertical gradient.

    # Optionally smooth the gradients.
    if sigma > 0:
        Gx = gaussian_filter(Gx, sigma=sigma)
        Gy = gaussian_filter(Gy, sigma=sigma)

    # Step 2: construct structure-tensor components.
    Jxx = Gx * Gx
    Jyy = Gy * Gy
    Jxy = Gx * Gy

    # Step 3: aggregate the tensor over a Gaussian neighborhood.
    Jxx = gaussian_filter(Jxx, sigma=rho)
    Jyy = gaussian_filter(Jyy, sigma=rho)
    Jxy = gaussian_filter(Jxy, sigma=rho)

    # Step 4: aggregate blocks and compute the eigensystem.
    n_h = H // patch_size
    n_w = W // patch_size
    angles = np.zeros((n_h, n_w))
    coherence = np.zeros((n_h, n_w))

    for i in range(n_h):
        for j in range(n_w):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size

            # Sum tensor components within the block.
            jxx = np.sum(Jxx[y0:y1, x0:x1])
            jyy = np.sum(Jyy[y0:y1, x0:x1])
            jxy = np.sum(Jxy[y0:y1, x0:x1])

            # Eigenvalues of the 2x2 matrix [jxx, jxy; jxy, jyy].
            # λ = (jxx + jyy ± sqrt((jxx - jyy)² + 4*jxy²)) / 2
            trace = jxx + jyy
            det = jxx * jyy - jxy * jxy

            if trace == 0:
                angles[i, j] = 0.0
                coherence[i, j] = 0.0
                continue

            discriminant = trace * trace - 4 * det
            if discriminant < 0:
                discriminant = 0

            sqrt_disc = np.sqrt(discriminant)
            lambda1 = (trace + sqrt_disc) / 2   # Larger eigenvalue.
            lambda2 = (trace - sqrt_disc) / 2   # Smaller eigenvalue.

            # Coherence: 1 is perfectly oriented; 0 is isotropic.
            if lambda1 + lambda2 > 0:
                coherence[i, j] = (lambda1 - lambda2) / (lambda1 + lambda2)
            else:
                coherence[i, j] = 0.0

            # Principal direction: eigenvector of the larger eigenvalue.
            # Eigenvector for lambda1: [lambda1 - jyy, jxy].
            vx = lambda1 - jyy
            vy = jxy

            # Convert the orientation to an acute angle in [0, 90).
            if np.abs(vx) > 1e-10:
                angle_rad = np.arctan(np.abs(vy / vx))
            else:
                angle_rad = np.pi / 2

            angles[i, j] = np.degrees(angle_rad)

    return angles, coherence


def estimate_fiber_angle(image, sigma=3.0, rho=15.0, patch_size=32):
    """Estimate whole-image dominant angle and mean coherence."""
    angles, coherence = gst_orientation(image, sigma=sigma, rho=rho, patch_size=patch_size)

    # Estimate the dominant direction with coherence-weighted voting.
    if np.sum(coherence) > 0:
        # Weighted orientation histogram.
        hist_bins = 180
        hist = np.zeros(hist_bins)
        bin_edges = np.linspace(0, 90, hist_bins + 1)

        for i in range(angles.shape[0]):
            for j in range(angles.shape[1]):
                ang = angles[i, j]
                coh = coherence[i, j]
                if coh < 0.1:  # Ignore weakly oriented blocks.
                    continue
                bin_idx = min(int(ang / 90 * hist_bins), hist_bins - 1)
                hist[bin_idx] += coh

        if np.sum(hist) > 0:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            dominant_angle = np.average(bin_centers, weights=hist)
        else:
            dominant_angle = np.median(angles)
    else:
        dominant_angle = np.median(angles)

    mean_coherence = np.mean(coherence)

    return dominant_angle, mean_coherence


def run_gst_baseline(csv_path, image_root, sigma=3.0, rho=15.0, patch_size=32):
    """Run the GST baseline over a labeled CSV/image-root pair."""
    results = []
    ground_truth = []
    filenames = []
    coherences = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} images...")

    for row in tqdm(rows):
        filename = row['patch_filename']
        gt_angle = float(row['angle'])

        img_path = os.path.join(image_root, filename)
        if not os.path.exists(img_path):
            print(f"Warning: file not found: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.float32)

            pred_angle, coherence = estimate_fiber_angle(
                img_array, sigma=sigma, rho=rho, patch_size=patch_size
            )

            results.append(pred_angle)
            ground_truth.append(gt_angle)
            filenames.append(filename)
            coherences.append(coherence)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue

    results = np.array(results)
    ground_truth = np.array(ground_truth)
    if not len(results):
        raise ValueError("No images were evaluated; check the CSV and --image_root.")

    # Compute regression metrics.
    errors = np.abs(results - ground_truth)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # R²
    ss_res = np.sum((ground_truth - results) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Standard deviation of absolute errors.
    std = np.std(errors)

    print(f"\n{'='*50}")
    print(f"GST baseline (sigma={sigma}, rho={rho}, patch={patch_size})")
    print(f"{'='*50}")
    print(f"Samples:      {len(results)}")
    print(f"MAE:          {mae:.4f}°")
    print(f"RMSE:         {rmse:.4f}°")
    print(f"R²:           {r2:.4f}")
    print(f"Std:          {std:.4f}°")
    print(f"Error range:  [{np.min(errors):.4f}°, {np.max(errors):.4f}°]")
    print(f"Median error: {np.median(errors):.4f}°")
    print(f"Mean Coherence: {np.mean(coherences):.4f}")
    print(f"{'='*50}")

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'std': std,
        'n_samples': len(results),
        'errors': errors,
        'predictions': results,
        'ground_truth': ground_truth,
        'filenames': filenames,
        'coherences': np.asarray(coherences),
    }


def grid_search(csv_path, image_root):
    """Grid-search GST hyperparameters by MAE."""
    sigma_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho_list = [5.0, 10.0, 15.0, 20.0, 25.0]
    patch_list = [16, 32, 64]

    best_mae = float('inf')
    best_params = {}

    print("GST hyperparameter grid search...")

    for sigma in sigma_list:
        for rho in rho_list:
            for patch_size in patch_list:
                if rho < sigma:
                    continue  # Tensor smoothing should not be narrower than gradient smoothing.

                print(f"  sigma={sigma}, rho={rho}, patch={patch_size} ...", end=" ")
                result = run_gst_baseline(csv_path, image_root,
                                         sigma=sigma, rho=rho, patch_size=patch_size)
                mae = result['mae']
                print(f"MAE={mae:.4f}°")

                if mae < best_mae:
                    best_mae = mae
                    best_params = {
                        'sigma': sigma,
                        'rho': rho,
                        'patch_size': patch_size,
                        'mae': mae,
                        'rmse': result['rmse'],
                        'r2': result['r2'],
                    }

    print(f"\nBest parameters: {best_params}")
    return best_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GST fiber-orientation baseline')
    parser.add_argument('--csv', type=str, default='database/test/test.csv', help='Label CSV path')
    parser.add_argument('--image_root', type=str, default='database/test/images', help='Image directory')
    parser.add_argument('--sigma', type=float, default=3.0, help='Gradient smoothing sigma (default: 3.0)')
    parser.add_argument('--rho', type=float, default=15.0, help='Tensor smoothing rho (default: 15.0)')
    parser.add_argument('--patch_size', type=int, default=32, help='Block size (default: 32)')
    parser.add_argument('--grid_search', action='store_true', help='Run the hyperparameter grid search')
    parser.add_argument('--output', type=str, default=None, help='Detailed prediction CSV path')

    args = parser.parse_args()

    if args.grid_search:
        best = grid_search(args.csv, args.image_root)
        print(f"\nRecommended: sigma={best['sigma']}, rho={best['rho']}, patch_size={best['patch_size']}")
        print(f"Validation MAE: {best['mae']:.4f}°")
    else:
        result = run_gst_baseline(
            args.csv, args.image_root,
            sigma=args.sigma, rho=args.rho, patch_size=args.patch_size
        )

        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "gt_angle", "gst_angle", "error", "coherence"])
                writer.writerows(zip(
                    result["filenames"], result["ground_truth"], result["predictions"],
                    result["errors"], result["coherences"],
                ))
            print(f"\nDetailed results saved to: {args.output}")
