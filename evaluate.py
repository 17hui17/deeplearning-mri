"""
评估脚本（对应 PPT 第 10 页 "The Big Jump" 的四行对比表）

在测试集上对比四种结果：
    1. 零填充输入 z
    2. 网络输出 x_hat = clip(z + U(z), 0, 1)（用 EMA 权重）
    3. x_hat + 数据一致性 DC（把真实测量的 k 空间点贴回去）
    4. x_hat + DC + TTA（4 次翻转各跑一次取平均，再 DC）

输出：终端打印对比表 + results.csv + comparison.png（前几例的可视化拼图）

用法：
    python evaluate.py --data_dir data --checkpoint runs/default/best.pt
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import MRIDataset, collect_image_paths, split_paths
from model import UNet
from utils import (
    apply_data_consistency,
    make_cartesian_mask,
    psnr,
    simulate_undersampling,
    ssim,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MRI 重建评估（DC + TTA）")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_show", type=int, default=4,
                        help="可视化拼图里展示的样本数")
    return parser.parse_args()


@torch.no_grad()
def run_network(model, z):
    """网络前向 + 残差连接，返回 clip 后的图像。"""
    return (z + model(z)).clamp(0.0, 1.0)


@torch.no_grad()
def run_network_tta(model, z):
    """4 次水平翻转组合的 TTA：原向 + 水平翻转 + 垂直翻转 + 双向翻转，取平均。"""
    outputs = []
    for flip_h, flip_w in ((False, False), (True, False), (False, True), (True, True)):
        z_t = z
        if flip_h:
            z_t = torch.flip(z_t, dims=[-2])
        if flip_w:
            z_t = torch.flip(z_t, dims=[-1])

        out = run_network(model, z_t)

        if flip_w:
            out = torch.flip(out, dims=[-1])
        if flip_h:
            out = torch.flip(out, dims=[-2])
        outputs.append(out)

    return torch.stack(outputs).mean(dim=0)


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
    config = checkpoint["config"]
    print(f"加载 checkpoint: {args.checkpoint} "
          f"(验证 PSNR {checkpoint.get('val_psnr', 0):.2f} dB)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 用训练时保存的 config 复原掩膜、模型、数据划分 —— 保证和训练完全一致
    model = UNet(in_channels=1, base_channels=config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["ema_state"]["shadow"])
    model.eval()

    mask = torch.from_numpy(
        make_cartesian_mask(config["image_size"], config["image_size"],
                            ratio=config["mask_ratio"], seed=config["seed"])
    ).to(device)

    paths = collect_image_paths(args.data_dir)
    if config.get("limit", 0) > 0:
        paths = paths[:config["limit"]]
    _, _, test_paths = split_paths(paths, seed=config["seed"])

    test_set = MRIDataset(test_paths, config["image_size"], augment=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    print(f"测试集: {len(test_set)} 张")

    methods = ["zero_filled", "network", "network_dc", "network_dc_tta"]
    scores = {name: {"psnr": [], "ssim": []} for name in methods}
    shown = 0

    fig, axes = plt.subplots(args.num_show, 5,
                             figsize=(15, 3.2 * args.num_show))

    for x_gt in test_loader:
        x_gt = x_gt.to(device)
        k_meas, z = simulate_undersampling(x_gt, mask)

        x_net = run_network(model, z)
        x_dc = apply_data_consistency(x_net, k_meas, mask)
        x_tta = run_network_tta(model, z)
        x_dc_tta = apply_data_consistency(x_tta, k_meas, mask)

        for name, image in zip(methods, (z, x_net, x_dc, x_dc_tta)):
            scores[name]["psnr"].append(psnr(image, x_gt))
            scores[name]["ssim"].append(ssim(image, x_gt).item())

        # 可视化前几个 batch
        if shown < args.num_show:
            for i in range(min(args.batch_size, args.num_show - shown)):
                row = shown
                images = [z, x_net, x_dc_tta, x_gt]
                # 图题用英文，避免 Kaggle/Linux 缺中文字体
                titles = ["Zero-filled input", "Network", "Network + DC + TTA",
                          "Ground truth"]
                row_psnr = [psnr(img[i:i+1], x_gt[i:i+1]) for img in images]
                for col, (img, title, p) in enumerate(zip(images, titles, row_psnr)):
                    ax = axes[row, col] if args.num_show > 1 else axes[col]
                    ax.imshow(img[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
                    ax.set_title(f"{title}\n{p:.2f} dB", fontsize=10)
                    ax.axis("off")
                # 最后一列放掩膜说明
                shown += 1

    # 掩膜画在第一行最后一列
    ax = axes[0, 4] if args.num_show > 1 else axes[4]
    ax.imshow(mask.cpu(), cmap="gray_r")
    ax.set_title(f"Sampling mask ({config['mask_ratio']:.0%})", fontsize=10)
    ax.axis("off")
    for r in range(1, args.num_show):
        axes[r, 4].axis("off")

    # ---------------- 汇总输出 ----------------
    print("\n===== 测试集结果（对应 PPT 第 10 页的表） =====")
    print(f"{'方法':<18}{'PSNR (dB)':>12}{'SSIM':>10}")
    rows = []
    for name in methods:
        mean_psnr = sum(scores[name]["psnr"]) / len(scores[name]["psnr"])
        mean_ssim = sum(scores[name]["ssim"]) / len(scores[name]["ssim"])
        label = {
            "zero_filled": "零填充输入",
            "network": "网络输出",
            "network_dc": "网络 + DC",
            "network_dc_tta": "网络 + DC + TTA",
        }[name]
        print(f"{label:<18}{mean_psnr:>12.2f}{mean_ssim:>10.4f}")
        rows.append((label, round(mean_psnr, 3), round(mean_ssim, 4)))

    out_dir = Path(args.checkpoint).parent
    with open(out_dir / "test_results.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "psnr_db", "ssim"])
        writer.writerows(rows)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=150)
    print(f"\n已保存: {out_dir / 'test_results.csv'}")
    print(f"已保存: {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
