"""
GST (Gradient Structure Tensor) 纤维取向基线方法
用于与 FiberAngleNet 对比，计算传统 CV 方法的纤维剪切角检测精度

参考:
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
    """
    梯度结构张量法估计纤维主方向

    参数:
        image: 输入灰度图像 (H, W)
        sigma: 梯度平滑的高斯标准差
        rho: 结构张量平滑的高斯标准差
        patch_size: 分块大小（每块输出一个角度）

    返回:
        angles: 每个 patch 的估计角度 (度, [0, 90) 锐角)
        coherence: 每个 patch 的相干性 (0-1, 越高表示方向越明确)
    """
    H, W = image.shape

    # Step 1: 计算高斯梯度
    # 先用高斯平滑，再求梯度
    from scipy.ndimage import sobel

    # Sobel 算子近似梯度
    Gx = sobel(image, axis=1)  # x方向梯度
    Gy = sobel(image, axis=0)  # y方向梯度

    # 可选：用高斯平滑后的梯度
    if sigma > 0:
        Gx = gaussian_filter(Gx, sigma=sigma)
        Gy = gaussian_filter(Gy, sigma=sigma)

    # Step 2: 构建结构张量分量
    Jxx = Gx * Gx
    Jyy = Gy * Gy
    Jxy = Gx * Gy

    # Step 3: 高斯平滑结构张量（邻域聚合）
    Jxx = gaussian_filter(Jxx, sigma=rho)
    Jyy = gaussian_filter(Jyy, sigma=rho)
    Jxy = gaussian_filter(Jxy, sigma=rho)

    # Step 4: 分块聚合 + 特征分解
    n_h = H // patch_size
    n_w = W // patch_size
    angles = np.zeros((n_h, n_w))
    coherence = np.zeros((n_h, n_w))

    for i in range(n_h):
        for j in range(n_w):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size

            # 块内张量累加
            jxx = np.sum(Jxx[y0:y1, x0:x1])
            jyy = np.sum(Jyy[y0:y1, x0:x1])
            jxy = np.sum(Jxy[y0:y1, x0:x1])

            # 特征值分解: 2x2 矩阵 [jxx, jxy; jxy, jyy]
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
            lambda1 = (trace + sqrt_disc) / 2   # 较大特征值
            lambda2 = (trace - sqrt_disc) / 2   # 较小特征值

            # 相干性: 1 表示完美方向, 0 表示各向同性
            if lambda1 + lambda2 > 0:
                coherence[i, j] = (lambda1 - lambda2) / (lambda1 + lambda2)
            else:
                coherence[i, j] = 0.0

            # 主方向: 较大特征值对应的特征向量
            # 特征向量: [λ1 - jyy, jxy] (对应 λ1)
            vx = lambda1 - jyy
            vy = jxy

            # 角度(弧度), 取锐角 [0, 90)
            if np.abs(vx) > 1e-10:
                angle_rad = np.arctan(np.abs(vy / vx))
            else:
                angle_rad = np.pi / 2

            angles[i, j] = np.degrees(angle_rad)

    return angles, coherence


def estimate_fiber_angle(image, sigma=3.0, rho=15.0, patch_size=32):
    """
    估计整张图像的主导纤维角度

    返回:
        dominant_angle: 主导纤维角度 (度, 0-90 锐角)
        mean_coherence: 平均相干性
    """
    angles, coherence = gst_orientation(image, sigma=sigma, rho=rho, patch_size=patch_size)

    # 用相干性加权取主导方向
    if np.sum(coherence) > 0:
        # 加权直方图
        hist_bins = 180
        hist = np.zeros(hist_bins)
        bin_edges = np.linspace(0, 90, hist_bins + 1)

        for i in range(angles.shape[0]):
            for j in range(angles.shape[1]):
                ang = angles[i, j]
                coh = coherence[i, j]
                if coh < 0.1:  # 过滤低相干性块
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
    """
    对整个数据集运行 GST 基线

    参数:
        csv_path: CSV文件路径 (含 patch_filename, angle 列)
        image_root: 图像根目录
        sigma, rho, patch_size: GST 参数
    """
    results = []
    ground_truth = []
    filenames = []
    coherences = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"处理 {len(rows)} 张图像...")

    for row in tqdm(rows):
        filename = row['patch_filename']
        gt_angle = float(row['angle'])

        img_path = os.path.join(image_root, filename)
        if not os.path.exists(img_path):
            print(f"警告: 文件不存在 {img_path}")
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
            print(f"处理 {filename} 时出错: {e}")
            continue

    results = np.array(results)
    ground_truth = np.array(ground_truth)
    if not len(results):
        raise ValueError("没有可评估的图像；请检查 CSV 和 --image_root。")

    # 计算指标
    errors = np.abs(results - ground_truth)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # R²
    ss_res = np.sum((ground_truth - results) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 标准差
    std = np.std(errors)

    print(f"\n{'='*50}")
    print(f"GST 基线结果 (sigma={sigma}, rho={rho}, patch={patch_size})")
    print(f"{'='*50}")
    print(f"样本数:       {len(results)}")
    print(f"MAE:          {mae:.4f}°")
    print(f"RMSE:         {rmse:.4f}°")
    print(f"R²:           {r2:.4f}")
    print(f"Std:          {std:.4f}°")
    print(f"误差范围:      [{np.min(errors):.4f}°, {np.max(errors):.4f}°]")
    print(f"中位误差:      {np.median(errors):.4f}°")
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
    """超参数网格搜索，找到最优 GST 参数"""
    sigma_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho_list = [5.0, 10.0, 15.0, 20.0, 25.0]
    patch_list = [16, 32, 64]

    best_mae = float('inf')
    best_params = {}

    print("GST 超参数网格搜索...")

    for sigma in sigma_list:
        for rho in rho_list:
            for patch_size in patch_list:
                if rho < sigma:
                    continue  # rho 应该 >= sigma

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

    print(f"\n最优参数: {best_params}")
    return best_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GST 纤维取向基线')
    parser.add_argument('--csv', type=str, default='database/test/test.csv', help='CSV标注文件路径')
    parser.add_argument('--image_root', type=str, default='database/test/images', help='图像目录')
    parser.add_argument('--sigma', type=float, default=3.0, help='梯度平滑sigma (默认3.0)')
    parser.add_argument('--rho', type=float, default=15.0, help='张量平滑rho (默认15.0)')
    parser.add_argument('--patch_size', type=int, default=32, help='分块大小 (默认32)')
    parser.add_argument('--grid_search', action='store_true', help='运行超参数网格搜索')
    parser.add_argument('--output', type=str, default=None, help='保存详细结果的CSV路径')

    args = parser.parse_args()

    if args.grid_search:
        best = grid_search(args.csv, args.image_root)
        print(f"\n推荐参数: sigma={best['sigma']}, rho={best['rho']}, patch_size={best['patch_size']}")
        print(f"预期 MAE: {best['mae']:.4f}°")
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
            print(f"\n详细结果已保存至: {args.output}")
