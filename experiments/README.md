# DiT 动量精度对照

使用独立的 `viola.optim.FP8Muon`，单卡直接运行，多卡使用 PyTorch DDP。
此目录的诊断会额外保存 BF16 shadow 动量，不能用其显存测量代表优化器生产占用。

| 设置 | 默认 |
| --- | --- |
| 模型 | DiT-B/4，latent patch4 等价于图像 patch32 |
| 数据 | ImageNet SD-VAE latent，训练前转换为 mmap |
| 目标 | v-prediction FM，logit-normal 时间采样 |
| 全局 batch | 256，每卡 microbatch 上限 64 |
| 更新 | FP32 主权重与梯度，BF16 autocast，独立 FP8 动量 |
| 正交化 | standard BF16 NS，5 步 |
| 训练 | 10 epochs，学习率 1e-4，预热 1K 步 |

```bash
PYTHONPATH=. python experiments/prepare_latents.py --source /path/to/latents

PYTHONPATH=. python experiments/train_dit.py \
  --data /path/to/latents/mmap --output outputs/fp8 --momentum-dtype fp8

PYTHONPATH=. python experiments/train_dit.py \
  --data /path/to/latents/mmap --output outputs/bf16 --momentum-dtype bf16

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 experiments/train_dit.py \
  --data /path/to/latents/mmap --output outputs/fp8-ddp --batch 256

PYTHONPATH=. python experiments/compare_runs.py \
  --fp8 outputs/fp8 --bf16 outputs/bf16 --output outputs/comparison
```

BF16 对照只替换动量存储和更新 kernel，其余 NS、权重更新与训练流程相同。
训练日志记录逐步 loss、固定 probe loss、耗时与动量误差；逐矩阵指标保存为 Parquet，
包括相对 L2、cosine、最大绝对误差、尺度比和尺度对齐后的误差。每 100 步额外比较 NS 更新。
每个 rank 都记录本地完整矩阵，汇总时按 `(step, name)` 去重。

`checkpoint.pt` 包含普通模型与优化器 state_dict，以及主权重名称。
对照脚本直接读取其中的主权重，比较逐层及全模型误差，无需任何分片恢复工具。
独立训练最终的权重差异包含训练轨迹分离，不能当成单次量化误差。

当前脚本与旧版本结果采用不同的并行、保存路径，实验队列写入新的 `standalone` 目录。
不会自动续接旧实验或覆盖旧结果。
