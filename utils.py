"""
工具函数：k 空间采样模拟、SSIM、EMA、PSNR

对应 PPT：
- 第 5 页：k_meas = M ⊙ F(x_gt)，z = |F^{-1}(k_meas)|，
  DC：k_DC = (1-M) ⊙ F(x_hat) + k_meas
- 第 7 页：L = 0.5*(1-SSIM) + 0.5*L1；SSIM 用 7x7 高斯窗，sigma=1.5，
  C1=0.01^2，C2=0.03^2；EMA decay 0.999
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 1. 采样掩膜：25% Cartesian 线采样（中心低频采满 + 外围随机）
# ============================================================

def make_cartesian_mask(height, width, ratio=0.25, center_fraction=0.08, seed=42):
    """
    生成固定的线采样掩膜，形状 (height, width)，值为 0/1。

    采整列（phase-encode 方向），中心 center_fraction 的列全部保留
    （低频决定图像主体），剩余列均匀随机抽取，使总采样率 = ratio。

    掩膜只由 (height, width, ratio, seed) 决定，训练和评估用同样参数
    生成的掩膜完全一致。
    """
    rng = np.random.default_rng(seed)

    num_sampled = int(round(width * ratio))
    num_center = int(round(width * center_fraction))

    center_start = (width - num_center) // 2
    center_cols = list(range(center_start, center_start + num_center))

    outer_cols = [c for c in range(width) if c not in center_cols]
    num_outer = num_sampled - num_center
    if num_outer > 0:
        chosen_outer = rng.choice(outer_cols, size=num_outer, replace=False)
    else:
        chosen_outer = []

    sampled = sorted(center_cols + list(chosen_outer))

    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, sampled] = 1.0
    return mask


# ============================================================
# 2. MRI 采样模拟：图像 -> 欠采样 k 空间 -> 零填充图像
# ============================================================

def simulate_undersampling(x, mask):
    """
    x    : (B, 1, H, W)，取值 [0, 1] 的原图（ground truth）
    mask : (H, W)，0/1

    返回 (k_meas, z)：
    k_meas : (B, 1, H, W) 复数，测得的 k 空间（未采位置为 0）
    z      : (B, 1, H, W)，零填充图像 |F^{-1}(k_meas)|，clip 到 [0, 1]
    """
    k = torch.fft.fft2(x, norm="ortho")
    k = torch.fft.fftshift(k, dim=(-2, -1))

    k_meas = k * mask

    z = torch.fft.ifft2(torch.fft.ifftshift(k_meas, dim=(-2, -1)), norm="ortho")
    z = torch.abs(z).clamp(0.0, 1.0)
    return k_meas, z


def apply_data_consistency(x_hat, k_meas, mask):
    """
    硬数据一致性（PPT 第 5、10 页，一行核心代码）：
        k_DC = (1 - M) ⊙ F(x_hat) + k_meas
    已测量的点用真实值，未测量的点用网络的猜测。
    """
    k_hat = torch.fft.fft2(x_hat, norm="ortho")
    k_hat = torch.fft.fftshift(k_hat, dim=(-2, -1))

    k_dc = (1.0 - mask) * k_hat + k_meas

    out = torch.fft.ifft2(torch.fft.ifftshift(k_dc, dim=(-2, -1)), norm="ortho")
    return torch.abs(out).clamp(0.0, 1.0)


# ============================================================
# 3. SSIM（7x7 高斯窗，sigma=1.5，C1=0.01^2，C2=0.03^2）
# ============================================================

def _gaussian_window(window_size=7, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float64) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window.float()


def ssim(a, b, window_size=7, sigma=1.5):
    """
    两幅 (B, 1, H, W)、取值 [0,1] 图像的平均 SSIM。越高越像。
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    window = _gaussian_window(window_size, sigma).to(a.device, a.dtype)
    window = window.expand(a.shape[1], 1, window_size, window_size)

    pad = window_size // 2

    mu_a = F.conv2d(a, window, padding=pad, groups=a.shape[1])
    mu_b = F.conv2d(b, window, padding=pad, groups=b.shape[1])

    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a = F.conv2d(a * a, window, padding=pad, groups=a.shape[1]) - mu_a2
    sigma_b = F.conv2d(b * b, window, padding=pad, groups=b.shape[1]) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=a.shape[1]) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a2 + mu_b2 + C1) * (sigma_a + sigma_b + C2))

    return ssim_map.mean()


def combined_loss(x_hat, x_gt):
    """PPT 第 7 页：L = 0.5*(1 - SSIM) + 0.5*|x_hat - x_gt|_1"""
    return 0.5 * (1.0 - ssim(x_hat, x_gt)) + 0.5 * F.l1_loss(x_hat, x_gt)


# ============================================================
# 4. PSNR
# ============================================================

def psnr(a, b, max_val=1.0):
    """两幅图像的平均 PSNR（dB）。"""
    mse = ((a - b) ** 2).flatten(1).mean(dim=1)
    mse = mse.clamp_min(1e-12)
    return (10.0 * torch.log10(max_val ** 2 / mse)).mean().item()


# ============================================================
# 5. EMA（指数滑动平均，PPT 第 7、9 页：decay 0.999）
# ============================================================

class EMA:
    """对模型参数做滑动平均；评估时用平滑后的权重。"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            shadow_value = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow_value.mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                # BatchNorm 的 num_batches_tracked 等整数量直接复制
                shadow_value.copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
