# DiT-B 动量精度实验

使用完整 ImageNet-256 SD-VAE posterior mean 缓存，保留原始 FP32 数据。原图及翻转
图像各一个 latent，训练时随机选择。数据来源：
[yuanchenyang/imagenet-256-sd-vae-ft-mse-latents](https://huggingface.co/datasets/yuanchenyang/imagenet-256-sd-vae-ft-mse-latents)。
下载使用 HF 镜像，revision 与文件大小写入服务器上的下载 manifest。

| 设置 | 值 |
| --- | --- |
| 模型 | DiT-B，12 层、宽 768、12 heads，标准 adaLN-Zero |
| Patch | latent patch4，原图侧等价 patch32；每图 64 tokens |
| 训练目标 | FM v-prediction，MSE；t 服从 logit-normal(0, 1) |
| 优化器 | Viola FP8 Muon + 辅助 AdamW，match_rms_adamw |
| 动量 | beta=0.95，E4M3、tile256、nearest，无高精度残差 |
| 非有限值 | 不扫描/跳步，让 Inf/NaN 正常传播；动量仅单份、原地更新 |
| 正交化 | DeepSpeed standard BF16 NS，5 步 |
| 学习率 | 1e-4，1K 步线性预热后恒定 |
| 训练计算 | BF16 模型，FP32 梯度累积、主权重；无梯度裁剪 |
| Token budget | 全局每次 optimizer step 4K / 8K / 16K tokens |
| Global batch | 64 / 128 / 256；每个设置训练 10 ep |
| 顺序 | 当前单卡 16K FP8 → 相同设置的独立 BF16 对照 → 其余单卡 budget → 两卡三个 budget |
| 数据尾批 | 每个 epoch 丢弃不足 global batch 的最后一批 |

不同 DP 数量使用相同初始化、全局图像顺序、翻转选择、噪声、时间和类别 dropout。
不同 budget 的 epoch 相同、optimizer step 数不同：学习率预热按 step，因此不能将
最终误差差异完全归因为 token 数。日志同时保存 step、epoch 与完整配置。

每一步、每个 Muon 矩阵维护相同真实梯度驱动的 **BF16 诊断对照动量**。
对照先用 FP32 临时计算，再舍入为 BF16 保存。NS 使用当前步舍入前的新动量构造输入，
与 FP8 路径一致，不在构造 NS 输入前额外舍入一次。
对照状态不参与参数更新；这是同梯度数值对照，不是两组独立训练的收敛比较。
正在运行的早期快照仍保留原先的 FP32/BF16/FP16 诊断，汇报选择其中的直接 BF16 对照；
早期快照的 NS 参考为 FP32，不能混入新版本的 BF16 NS 指标。每组实验保存源码哈希。

新增选项可分别做舍入和分组方式的消融，默认不改变原来的量化配方：

```bash
# 连续 128 元素分块，随机舍入
python experiments/train_dit.py --data /path/to/mmap --output /path/to/run \
  --rounding stochastic --rounding-seed 42 --granularity tile --tile-size 128

# 二维 16×16 分块，随机舍入
python experiments/train_dit.py --data /path/to/mmap --output /path/to/run \
  --rounding stochastic --rounding-seed 42 --granularity block --block-shape 16 16
```

分布式启动时将 `python` 替换为 `torchrun --standalone --nproc_per_node=2`。
这些选项不自动替换正在运行的实验配置，也不会自动重复已完成的训练。

tile-wise 表示 1D 连续分组，block-wise 表示 2D 矩形分组。不对矩阵强制对齐，
初始化时 rank 0 汇总未对齐矩阵的 warning。早期日志使用相反的名称；原始记录保留，
新 checkpoint 的量化布局版本为 2，读取旧状态时按旧含义迁移标签，不重新量化。

独立 BF16 对照使用 `--momentum-dtype bf16`：只将 Muon 的历史动量保存为 BF16，
保持同一个 FP32 临时 EMA、BF16 NS、ZeRO-2、辅助 AdamW 与训练设置。
使用相同 seed、数据顺序、噪声、时间采样、batch256、1K warmup 和 50,040 次更新，
从相同初始化重新训练。当前 FP8 训练继续运行，队列接管不重启模型。

两组完成后，`compare_runs.py` 导出逐步 loss、固定探测 loss 和最终权重误差：

```bash
python experiments/compare_runs.py --fp8 /path/to/fp8-run --bf16 /path/to/bf16-run \
  --output /path/to/comparison
```

权重从 ZeRO checkpoint 恢复为 FP32 主权重，记录全模型与逐矩阵的相对 L2、cosine、
RMSE、平均/最大绝对误差。最终权重差异包含两条训练轨迹长期累积的偏离，
与同梯度 shadow 动量的量化误差分别报告。

`metrics/momentum-rank*-*.parquet` 保存全部矩阵每一步的数据：

- FP8 相对 BF16 的相对 L2、cosine、最大绝对误差、均值偏差。
- 误差平方和与参考平方和，用于正确聚合各矩阵，而不是直接平均相对误差。
- 本步量化误差与累计历史误差，梯度/动量 RMS、最大值、scale 范围。
- 零值比例、非有限值比例、量化码保持不变比例、编码达到端点的比例。
- NS 输入方向误差；NS 输出误差默认每 100 step 检查全部本地 Muon 矩阵。
- 动量范数比，以及拟合单一整体缩放后剩余误差占 FP8 动量范数的比例。

端点比例不等于溢出比例；每块 absmax 缩放通常会让最大元素落在 FP8 端点。
两边都为零时 cosine 定义为 1；相对 L2 分母的平方和下限为 1e-30。
跨 ZeRO 分界的完整矩阵可能在多个 rank 记录；汇总时按 `(step, name)` 去重。
每个 rank 的 JSONL 汇总只覆盖该 rank 的矩阵，不应当成整个模型的汇总。

日志中的显存峰值包含诊断状态及临时量，不能作为纯 FP8 优化器的显存成绩。
诊断状态占用另外记录，缓存分配与 kernel 临时量仍应单独做无诊断测试。
固定 1,024 条训练分布样本监测 FM loss；这不是 held-out 验证，也不提供 gFID 结论。
