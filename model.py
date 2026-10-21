"""
残差 U-Net（对应 PPT 第 6 页 "Residual U-Net Architecture (v6)"）

- 输入/输出：单通道图像（默认 128x128 或 256x256）
- 3 次 MaxPool 下采样，bottleneck 通道数 = base * 8
- 每个 ConvBlock = 两个 3x3 卷积 + BatchNorm + ReLU
- 转置卷积上采样，编码器特征通过 skip connection 拼接
- base=64 时参数量约 7.70M（与 PPT 的 7,699,009 一致）

注意：网络只预测"修正量" r = U(z)，
真正的输出在 train.py / evaluate.py 里做 x_hat = clip(z + r, 0, 1)。
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """两个 3x3 卷积 + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """3 级下采样的 U-Net。"""

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        b = base_channels

        # 编码器
        self.enc1 = ConvBlock(in_channels, b)        # H
        self.enc2 = ConvBlock(b, b * 2)              # H/2
        self.enc3 = ConvBlock(b * 2, b * 4)          # H/4
        self.bottleneck = ConvBlock(b * 4, b * 8)    # H/8

        self.pool = nn.MaxPool2d(2)

        # 解码器
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)

        # 输出修正量（1x1 卷积，不加激活——残差可正可负）
        self.head = nn.Conv2d(b, in_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自测：base=64 应约为 7.70M（PPT 是 7,699,009）
    for base in (16, 32, 64):
        net = UNet(base_channels=base)
        n = count_parameters(net)
        out = net(torch.zeros(1, 1, 128, 128))
        print(f"base={base:3d}  params={n:,}  out_shape={tuple(out.shape)}")
