"""
训练脚本（对应 PPT 第 5-9 页）

流程：
    原图 x_gt ──FFT + 25% 掩膜──► k_meas ──IFFT──► 零填充 z
    z ──U-Net──► 修正量 r；x_hat = clip(z + r, 0, 1)
    损失 = 0.5*(1 - SSIM(x_hat, x_gt)) + 0.5*L1(x_hat, x_gt)

特性：
- EMA（默认 decay 0.999），验证和保存都用 EMA 权重（PPT：EMA 曲线更平滑）
- GPU 上自动启用 bf16 混合精度（PPT 第 7 页 "bf16 AMP"）；CPU 自动关闭
- 每个 epoch 在验证集上算 PSNR，保存最优 checkpoint 到输出目录

示例（本地 CPU 冒烟测试）：
    python train.py --data_dir data --image_size 128 --base_channels 16 ^
        --batch_size 8 --epochs 3 --limit 500

示例（Kaggle/Colab GPU 正式训练，对齐 PPT）：
    python train.py --data_dir /kaggle/input/brain-tumor-mri-dataset ^
        --image_size 256 --base_channels 64 --batch_size 16 --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import MRIDataset, collect_image_paths, split_paths
from model import UNet, count_parameters
from utils import (
    EMA,
    combined_loss,
    make_cartesian_mask,
    psnr,
    simulate_undersampling,
    ssim,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MRI 残差 U-Net 训练")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="图片根目录（递归读取）")
    parser.add_argument("--out_dir", type=str, default="runs/default",
                        help="输出目录（checkpoint、日志）")

    parser.add_argument("--image_size", type=int, default=128,
                        help="图像边长（GPU 上建议 256，与 PPT 一致）")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="U-Net 基础通道数（64 = PPT 的 7.7M 参数）")
    parser.add_argument("--mask_ratio", type=float, default=0.25,
                        help="k 空间采样率（PPT 用 25%%）")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--limit", type=int, default=0,
                        help="只用前 N 张图（0 = 全部，冒烟测试用）")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Windows 下保持 0；Linux/Kaggle 可设 2")
    parser.add_argument("--no_residual", action="store_true",
                        help="关闭残差连接（教学用：复现 PPT v1 的失败）")

    return parser.parse_args()


@torch.no_grad()
def validate(model, loader, mask, device, no_residual=False):
    """用 EMA 权重在验证集上算 PSNR / SSIM。"""
    model.eval()
    psnr_list, ssim_list = [], []

    for x_gt in loader:
        x_gt = x_gt.to(device)
        _, z = simulate_undersampling(x_gt, mask)
        r = model(z)
        x_hat = (r if no_residual else z + r).clamp(0.0, 1.0)

        psnr_list.append(psnr(x_hat, x_gt))
        ssim_list.append(ssim(x_hat, x_gt).item())

    model.train()
    return sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # bf16 AMP 只在 GPU 上开
    print(f"设备: {device}（AMP: {use_amp}）")

    # ---------------- 数据 ----------------
    paths = collect_image_paths(args.data_dir)
    if args.limit > 0:
        paths = paths[:args.limit]
    if len(paths) < 10:
        raise SystemExit(f"只找到 {len(paths)} 张图，请检查 --data_dir")

    train_paths, val_paths, test_paths = split_paths(paths, seed=args.seed)
    print(f"数据: 共 {len(paths)} 张 -> "
          f"训练 {len(train_paths)} / 验证 {len(val_paths)} / 测试 {len(test_paths)}")

    train_set = MRIDataset(train_paths, args.image_size, augment=True, seed=args.seed)
    val_set = MRIDataset(val_paths, args.image_size, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # ---------------- 掩膜（固定，评估时用同样参数复现） ----------------
    mask = torch.from_numpy(
        make_cartesian_mask(args.image_size, args.image_size,
                            ratio=args.mask_ratio, seed=args.seed)
    ).to(device)

    # ---------------- 模型 ----------------
    model = UNet(in_channels=1, base_channels=args.base_channels).to(device)
    print(f"模型参数量: {count_parameters(model):,}")

    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---------------- 训练循环 ----------------
    best_val_psnr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_time = 0.0, time.time()

        for x_gt in train_loader:
            x_gt = x_gt.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                _, z = simulate_undersampling(x_gt, mask)
                r = model(z)
                x_hat = (r if args.no_residual else z + r).clamp(0.0, 1.0)
                loss = combined_loss(x_hat, x_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()

        scheduler.step()

        # 用 EMA 权重验证（先备份原始权重，验证完恢复）
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val_psnr, val_ssim = validate(model, val_loader, mask, device,
                                       no_residual=args.no_residual)
        model.load_state_dict(backup)

        avg_loss = epoch_loss / len(train_loader)
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "val_psnr": round(val_psnr, 3),
            "val_ssim": round(val_ssim, 4),
        })
        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"loss {avg_loss:.4f} | val PSNR {val_psnr:.2f} dB | "
              f"val SSIM {val_ssim:.4f} | {time.time() - epoch_time:.0f}s")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                "ema_state": ema.state_dict(),
                "config": vars(args),
                "val_psnr": val_psnr,
            }, out_dir / "best.pt")

    # 保存训练日志和最后一轮的 EMA 权重
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save({
        "ema_state": ema.state_dict(),
        "config": vars(args),
        "val_psnr": val_psnr,
    }, out_dir / "last.pt")

    print(f"\n训练完成。最优验证 PSNR = {best_val_psnr:.2f} dB")
    print(f"checkpoint 保存在: {out_dir.resolve()}")
    print("下一步: python evaluate.py --data_dir <同上> "
          f"--checkpoint {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
