# 深度学习 MRI 重建 —— 训练指南

复刻 PPT 第 3 部分：残差 U-Net 从 25% k 空间采样重建脑部 MRI。
整体方案见上一级的 `PLAN.md`，这里只讲怎么跑。

---

## 第 0 步：拿到数据

**推荐数据集：Kaggle「Brain Tumor MRI Dataset」（约 7023 张脑 MRI）**

- 本地训练：到 kaggle.com 注册 → 搜索 "Brain Tumor MRI Dataset"
  （作者 masoudnickparvar）→ Download → 解压到 `deep_learning_mri/data/`
  （目录里会出现 `Training/`、`Testing/` 子文件夹，脚本会递归读取，不用管结构）
- Kaggle 在线训练：不用下载，新建 Notebook 后点右侧 "Add Data" 搜同名数据集即可

## 第 1 步：装依赖（本地）

你的 anaconda Python 已有 torch 2.8.0（CPU 版），只需确认 pillow/matplotlib：

```bash
D:\Users\diannao\anaconda\python.exe -m pip install pillow matplotlib
```

（Kaggle Notebook 里这些都预装好了，跳过。）

## 第 2 步：本地 CPU 冒烟测试（强烈建议先跑这个）

目的不是出结果，是确认代码在你机器上能完整跑通。用 500 张图、小网络、3 个 epoch：

```bash
cd D:\diannao\Desktop\xuexi\MRI-Cambridge\deep_learning_mri
D:\Users\diannao\anaconda\python.exe train.py --data_dir data --out_dir runs/smoke ^
    --image_size 128 --base_channels 16 --batch_size 8 --epochs 3 --limit 500 ^
    --ema_decay 0.9
```

> 注意：冒烟测试图少、步数少，EMA（默认 0.999）跟不上权重的变化，
> 所以小跑时要加 `--ema_decay 0.9`；**正式训练时去掉它，用默认 0.999**。

正常的话你会看到每个 epoch 打印 loss 和验证 PSNR，loss 应逐步下降。
跑完 `runs/smoke/` 里会有 `best.pt`、`last.pt`、`history.json`。

## 第 3 步：正式训练（Kaggle 免费 GPU，推荐）

1. 打开 kaggle.com → 右上角头像需要先绑定手机号才能用 GPU（Settings → Phone Verification）
2. 左侧 "Code" → "New Notebook" → 右侧 "Add Data" → 搜 **Brain Tumor MRI Dataset** 添加
3. Notebook 右上角 Settings → Accelerator 选 **GPU T4**
4. 把本项目 5 个 .py 文件的内容粘进 Notebook 的单元格（或 File → Import Notebook
   上传）。最省事的方式：依次建 6 个单元格，前 5 个分别写
   `%%writefile model.py`、`%%writefile data.py`、`%%writefile utils.py`、
   `%%writefile train.py`、`%%writefile evaluate.py` 然后粘贴对应文件内容
5. 最后一个单元格运行（注意叹号开头）：

```
!python train.py --data_dir /kaggle/input/brain-tumor-mri-dataset \
    --out_dir /kaggle/working/runs/v6 --image_size 256 --base_channels 64 \
    --batch_size 16 --epochs 30 --num_workers 2
```

- T4 上预计 1–2 小时（7000 张 256×256、30 epoch）
- 中途可以去睡觉，Kaggle 会保持运行；完成后 Output 区的 `runs/v6/` 可下载

## 第 4 步：评估（复刻 PPT 的四行对比表）

本地：

```bash
D:\Users\diannao\anaconda\python.exe evaluate.py --data_dir data ^
    --checkpoint runs/smoke/best.pt
```

Kaggle（把 best.pt 留在 /kaggle/working 即可）：

```
!python evaluate.py --data_dir /kaggle/input/brain-tumor-mri-dataset \
    --checkpoint /kaggle/working/runs/v6/best.pt
```

会得到：
- 终端四行对比表：零填充 / 网络 / 网络+DC / 网络+DC+TTA（对应 PPT 第 10 页）
- `test_results.csv` 和 `comparison.png`（重建效果拼图，可直接放进报告）

## 判断复现成功的标准

1. 网络输出 PSNR > 零填充基线（说明网络确实学到了东西）
2. +DC 后 PSNR 再涨 3~4 dB（PPT 最大的"白送"提升）
3. 数据量/epoch 增加后曲线变平（PPT 第 8 页"more of the same data won't help"）

## 常见问题

| 问题 | 解决 |
|---|---|
| Kaggle 提示要绑手机才能开 GPU | Settings → Phone Verification，支持国内手机号 |
| 本地训练很慢 | 正常，CPU 就是慢；正式训练请用 Kaggle GPU |
| 显存不够 (CUDA out of memory) | `--batch_size 8` 或降到 4 |
| 想看训练曲线 | `runs/*/history.json` 里有每个 epoch 的 loss/PSNR，可画图 |
| 小数据测试时验证 PSNR 不涨 | EMA 滞后，加 `--ema_decay 0.9`（正式训练改回默认） |
| 想完全对齐 PPT 的 7.7M 参数 | 确认 `--base_channels 64`（train.py 启动时会打印参数量 7,699,009） |
