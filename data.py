"""
数据集加载（对应 PPT 第 5、7 页）

- 递归读取文件夹里的图片（png/jpg/jpeg/bmp/tif），转灰度、resize、归一化到 [0,1]
- 训练集做 8 倍二面体群增强（4 个旋转 x 2 个翻转，PPT 第 7 页 "8x dihedral augmentation"）
- 训练/验证/测试按 85/5/10 比例、固定种子划分，保证 train.py 和 evaluate.py 的
  划分完全一致

Kaggle 的 Brain Tumor MRI Dataset 本身有 Training/Testing 两个文件夹；
这里不区分原始目录，把所有图收集后重新随机划分（简单且够用）。
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_image_paths(data_dir):
    """递归收集所有图片路径，排序保证跨平台一致。"""
    data_dir = Path(data_dir)
    paths = [
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def split_paths(paths, train_ratio=0.85, val_ratio=0.05, seed=42):
    """固定种子随机划分为 train / val / test 三份。"""
    indices = np.random.default_rng(seed).permutation(len(paths))

    n_train = int(len(paths) * train_ratio)
    n_val = int(len(paths) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    paths = np.array(paths)
    return (
        [str(p) for p in paths[train_idx]],
        [str(p) for p in paths[val_idx]],
        [str(p) for p in paths[test_idx]],
    )


class MRIDataset(Dataset):
    """
    每条样本就是一张 [0,1] 灰度图（ground truth）。
    k 空间欠采样在训练循环里在线模拟（见 utils.simulate_undersampling），
    这样同一张图每个 epoch 都用，且评估时可以复现完全相同的掩膜。
    """

    def __init__(self, paths, image_size=128, augment=False, seed=42):
        self.paths = paths
        self.image_size = image_size
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path):
        image = Image.open(path).convert("L")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.LANCZOS,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)  # (1, H, W)

    def _dihedral_augment(self, x):
        """8 倍二面体群增强：随机选一个 k in [0,8)，旋转 + 可选水平翻转。"""
        k = self._rng.randrange(8)
        if k >= 4:
            x = torch.flip(x, dims=[-1])
            k -= 4
        x = torch.rot90(x, k, dims=(-2, -1))
        return x

    def __getitem__(self, index):
        x = self._load_image(self.paths[index])
        if self.augment:
            x = self._dihedral_augment(x)
        return x
