# Viola · FP8 Muon

通过一行代码，为 DeepSpeed Muon 启用 FP8 动量存储。

```python
import viola

viola.enable_fp8_muon()

# 随后按原来的方式初始化 DeepSpeed 或 Accelerate。
engine, optimizer, _, _ = deepspeed.initialize(...)
```

支持 **ZeRO-1、ZeRO-2**。导入模块不会修改 DeepSpeed；显式启用时输出一次 warning，
并覆盖当前 Python 进程中的 Muon 和 ZeRO 更新路径。不会改写 site-packages 源码。
原来的 DeepSpeed 优化器名称仍然使用 `Muon`。
多卡时仅 rank 0 输出启用 warning，每个 rank 都执行同样的覆盖。

## 主权重、梯度与优化器状态

三个精度分别配置，在每个 rank 初始化 DeepSpeed / Accelerate **之前**调用：

```python
viola.set_master_weight_dtype("bf16")  # fp32 / bf16 / fp16；存储和更新写回精度
viola.set_gradient_dtype("fp32")       # fp32 / bf16 / fp16；参数梯度链路
viola.enable_fp8_muon(adam_state_dtype="fp32")  # Muon 动量为 FP8，辅助 Adam 状态独立设置

engine, optimizer, _, _ = deepspeed.initialize(...)
```

`set_gradient_dtype` 作用于**参数梯度接收与累积 → ZeRO 通信桶 → 分片缓冲 → 优化器输入**。
它不修改模型参数精度、激活梯度或各个算子的 forward/backward 内部计算。算子仍可使用
BF16 输入、FP32 内部累加，或者在归一化等环节使用 FP32。模型计算精度沿用 DeepSpeed 的
`bf16` / `fp16` 配置。各设置只影响随后创建的引擎。

参数入口使用 PyTorch 的叶子张量 `grad_dtype`；ZeRO 通信桶和分片存储也使用指定精度，
不会在中途按模型 dtype 分配再降精度。手写 backward 直接返回的 FP32 权重梯度可保留到
FP32 参数梯度链路；普通算子若已经产生 BF16 结果，随后转 FP32 不能恢复舍入掉的信息。
如果自定义 kernel 绕过 autograd、直接写自己的 `main_grad`，需要另外接入 ZeRO reducer；
这两个设置不会自动接管任意第三方缓冲区。

主权重更新和辅助 Adam 的运算使用 FP32 临时值，完成后写回所选 dtype；不保存额外的
FP32 主权重副本。BF16/FP16 写回可能丢掉较小更新，可独立开启主权重随机舍入：

```python
viola.set_master_weight_dtype("bf16", rounding="stochastic", seed=42)
```

它与 `enable_fp8_muon(rounding=...)` 的**动量量化舍入**分开，两者都不保存逐元素残差。
Muon 的 NS 计算精度仍由其算法实现决定，不随梯度存储 dtype 一起切换。
不调用两个 setter 时，沿用原来的 DeepSpeed 精度路径；默认辅助 Adam 状态为 FP32。
这些覆盖针对 GPU 上的 Viola Muon + ZeRO-1/2，使用 DeepSpeed 的 `bf16` / `fp16` 模式。

## 速度与精度配置

```python
viola.enable_fp8_muon(mode="balanced")  # 默认，与初版相同
# viola.enable_fp8_muon(mode="fast")
# viola.enable_fp8_muon(mode="stable")
```

| 预设 | tile_size | NS 步数 | 取舍 |
| --- | ---: | --- | --- |
| `fast` | 256 | 3 | 减少 NS 矩阵乘法，正交化近似会改变 |
| `balanced` | 256 | 沿用优化器参数组，未指定时为 5 | 保持原实现行为 |
| `stable` | 128 | 5 | 更细的缩放粒度，减少块内离群值的影响，增加少量 scale 存储 |

所有预设都只持久保存 FP8 动量和每块 FP32 scale，不保存逐元素高精度残差。
`stable` 是配置取舍，不保证所有数据分布下误差都更小；仍需要实际训练对照。

可以覆盖单项，NS 默认继续使用优化器参数组中的设置：

```python
viola.enable_fp8_muon(
    mode="stable",
    granularity="tile",
    tile_size=256,       # 64 / 128 / 256 / 512 / 1024
    rounding="nearest",  # nearest / stochastic
    rounding_seed=0,
    ns_steps=5,
    ns_method="standard",  # standard / gram
    skip_nonfinite=False,  # 默认：不检查/跳步，让 Inf/NaN 传播到参数和 loss
)
```

`ns_steps` 越少通常越省计算，但会改变正交化近似，不能保证与原 Muon 等价。
`ns_method` 未指定时沿用参数组设置，预设不会自动切换算法。
默认 `nearest` 是就近舍入、正好居中时取偶数；`stochastic` 按到相邻两个 E4M3
数值的距离随机选择，减小反复舍入产生的系统性偏差，但单次误差不保证更小。
两者都不保存高精度残差。数值诊断以 **BF16 动量**为参考，分别观察历史动量与 NS 更新误差。

## 量化粒度与舍入

每个 scale 覆盖的元素由 `granularity` 决定，和 CUDA kernel 的执行 tile 分开配置：

这里统一使用 **tile-wise = 1D 连续分组，block-wise = 2D 矩形分组**。

| 配置 | 一个 scale 覆盖的元素 |
| --- | --- |
| `granularity="tile", tile_size=256` | 矩阵展平后连续 256 个元素；可能跨行 |
| `granularity="block", block_shape=(16, 16)` | 原矩阵中一个二维 16×16 区域 |

两种分组都是 256 个元素，但取到的元素通常不同。`tile_size` 用于连续分块，
`block_shape` 用于二维分块，不同时设置。二维 block 每条边支持 8、16、32、64。

```python
# 连续分块 + 随机舍入
viola.enable_fp8_muon(
    granularity="tile", tile_size=128,
    rounding="stochastic", rounding_seed=42,
)

# 或者：二维分块 + 随机舍入（在另一训练进程启用）
viola.enable_fp8_muon(
    granularity="block", block_shape=(16, 16),
    rounding="stochastic", rounding_seed=42,
)
```

随机舍入使用 Philox4x32-10，以 seed、全局参数顺序、动量更新计数与元素索引生成随机数。
同一矩阵在 ZeRO 分片边界被多个 rank 持有时，得到相同的舍入结果，不消耗训练使用的
PyTorch 随机数流。checkpoint 保存更新计数和舍入配置；恢复时保持配置与参数分组一致。
随机数在寄存器中生成，不分配随机数矩阵，不增加第二份 FP8 动量。

二维模式的 FP8 数据仍按原矩阵行优先连续存放，scale 单独按 block 排列；不转置、重排或
填充整个动量矩阵。尾部 block 只统计有效元素。卷积先展开为 `[Cout, Cin * Kh * Kw]`，
再按该二维矩阵分块；batched expert 矩阵在各自最后两个维度上分块。

不强制矩阵尺寸对齐，也不修改模型权重形状。初始化时 rank 0 汇总一次 warning：
1D tile 检查行宽是否为 `tile_size` 的倍数，未对齐时连续分组可能跨行；2D block 检查
行、列是否分别对齐 `block_shape`。边缘不足一组时只计算有效元素，训练 step 内不重复警告。
checkpoint 写入量化布局版本 2；读取旧 checkpoint 时只迁移反用的名称，保留数据和分组含义。

## 使用

需要 PyTorch 2.11 或更新版本（参数梯度配置使用 `Tensor.grad_dtype`）。
验证环境使用 PyTorch 2.11 / 2.15、DeepSpeed 0.19.7、TileLang 0.1.15。
需要 NVIDIA GPU 和可用的 CUDA 编译工具链。安装依赖后，在本仓库目录中可以直接导入
`viola`，无需安装 Viola 本身；从别的训练目录运行时，将本仓库加入 `PYTHONPATH`。

```bash
pip install -r requirements.txt
```

DeepSpeed 配置示例：

```json
{
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 1,
  "bf16": {"enabled": true},
  "data_types": {"grad_accum_dtype": "fp32"},
  "gradient_clipping": 0.0,
  "zero_optimization": {"stage": 2},
  "optimizer": {
    "type": "Muon",
    "params": {
      "lr": 0.001,
      "momentum": 0.95,
      "weight_decay": 0.01,
      "ns_method": "standard",
      "adjust_lr_fn": "match_rms_adamw"
    }
  },
  "zero_allow_untested_optimizer": true
}
```

将 `stage` 改为 `1` 即使用 ZeRO-1。插件会覆盖 ZeRO-1 在 BF16 参数、FP32 梯度累积下
选择 `BF16_Optimizer` 的路径，让完整矩阵的 Muon 更新继续在 ZeRO 封装内执行。
已有代码显式构造 `MuonWithAuxAdam` 时也可以使用。

Accelerate 示例：

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 examples/train.py --zero-stage 2
```

## 实现

- 默认每连续 256 个元素共享一个 FP32 scale，可配置分块形状和舍入方式。
- TileLang 在寄存器内恢复并更新 FP32 动量，同时输出正交化输入和下一步 FP8 状态。
- 保存**量化前的新动量所构造的本步更新**供 NS 使用，历史动量再压缩保存。
- 正交化调用 DeepSpeed 的 `ns_method`：`standard` 使用 BF16，`gram` 沿用官方实现。
- 卷积权重展开为 `[Cout, Cin * Kh * Kw]`；量化块不改变正交化的矩阵边界。
- 完整矩阵正交化后立即复制本 rank 所需的切片，不保存所有矩阵的完整更新。
- 权重衰减和主权重更新由 TileLang kernel 融合；FP32 临时计算后按主权重 dtype 写回。
- 使用独立精度配置时，辅助 Adam/AdamW 也通过融合 kernel 分开处理权重、梯度和状态精度。
  未配置时沿用原辅助优化器。参数同步沿用 DeepSpeed。

`match_rms_adamw` 使用 `0.2 * sqrt(max(rows, cols))` 的更新缩放。
也可以在参数组或 DeepSpeed 配置中指定 `adjust_lr_fn="original"`。

```python
viola.enable_fp8_muon(tile_size=256)  # 可选：64、128、256、512、1024
```

## 显存与 checkpoint

动量只保存一份，存储成本约为 `1 + 4 / tile_size` 字节/元素，默认约 **1.016 字节**。
相对单份 FP32 动量，状态节省约 74.6%；相对单份 FP16/BF16 动量，节省约 49.2%。
backward 期间只记录已有梯度的引用，不更新动量；进入 optimizer step 后原地更新。
默认 `skip_nonfinite=False`，覆盖 DeepSpeed 的自动溢出跳步行为，不做梯度非有限值扫描
或溢出标志同步，让 Inf/NaN 传播到动量、参数和 loss。
如果显式设置 `skip_nonfinite=True`，才会在写状态之前检查原始已规约梯度并同步标志，
发现非有限值后跳过该步；仍只需要一份动量。原生 step 复用检查结果，不重复检查。
这不包含模型权重、主权重、梯度、激活及正交化工作区。
`stable` 的 tile size 128 使每份状态变为约 1.031 字节/元素。
二维 block 使用一字节/有效元素，加上每个 block 的四字节 scale；矩阵尺寸能整除
16×16 时，与 tile256 的存储量相同。边缘 block 只增加 scale 数量，不填充 FP8 数据。
跨越分片边界的矩阵沿用 DeepSpeed 的完整矩阵动量布局，相关 rank 会分别保存该矩阵的
动量，因此不能把状态总量简单按 GPU 数量均分来估算。

checkpoint 使用 `uint8` 保存 FP8 的原始位模式，避免优化器加载时将浮点状态自动升为
FP32；插件的加载钩子在 PyTorch 自动转换之前提取这些数据，按原 dtype 恢复。
在 kernel 入口进行零拷贝的 FP8 视图转换。scale 保持 FP32。
不存在待提交动量缓冲；临时工作区不进入 checkpoint。

使用常规 `save_checkpoint` / `load_checkpoint`，恢复时保持相同的参数分组、DP 数量、
分区布局、精度设置、量化分组形状及舍入方式/seed。首次加载原 Muon 的完整精度动量时，逐矩阵转换为 FP8。
当前范围为 GPU 上的普通 ZeRO-1/2；不包含 CPU offload、ZeRO-3 或 elastic 重分片恢复。

未调用 `set_gradient_dtype`、ZeRO-1 使用独立 FP32 梯度累积缓冲时，插件会显式同步该缓冲，保证正交化读取的是
跨 rank 平均后的梯度。这复用已有梯度存储，但会增加该路径的通信量；不能只切换
优化器封装后继续使用未同步的局部梯度。
调用 `set_gradient_dtype` 后，参数梯度本身已使用目标 dtype，直接走相同 dtype 的
ZeRO reducer，不再需要上述额外的 ZeRO-1 累积缓冲同步。

## 验证

```bash
PYTHONPATH=. python checks/numerics.py
PYTHONPATH=. python checks/quantization.py
PYTHONPATH=. python checks/diagnostics.py
PYTHONPATH=. python checks/precision_kernels.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 checks/precision.py --stage 2 --master-dtype bf16 --gradient fp32
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 checks/zero.py --stage 1
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 checks/zero.py --stage 2
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 checks/zero.py --stage 2 --granularity block --rounding stochastic
```

数值检查对照 PyTorch 运算，覆盖量化尾块、Nesterov、loss unscale 和权重更新。
多卡检查让权重矩阵跨越分片边界，对照完整矩阵计算，并检查梯度累积、溢出跳过及
checkpoint 恢复后的连续更新。这里验证实现正确性，长期收敛效果仍需实际模型实验。

2026-10-29 的实测结果：

| 项目 | 结果 |
| --- | --- |
| RTX 5060 Ti / RTX 5090 融合 kernel | 数值对照、尾块与非对齐视图检查通过 |
| 两张 RTX 5090，ZeRO-1 / ZeRO-2 | 10 次更新窗口、梯度累积 2、跨分片权重矩阵检查通过 |
| 非有限值 | 默认传播 Inf/NaN；显式 skip 模式才检查并跳过更新 |
| checkpoint | FP8 原始字节及 scale 恢复完全一致，恢复后更新通过 |
| 显式传入 `MuonWithAuxAdam` | 两卡 ZeRO-2 检查通过 |
| Accelerate | 两种 ZeRO stage 均完成两卡运行 |
| 连续分块 / 二维 block，nearest / stochastic | 两卡 ZeRO-1、ZeRO-2 各 10 次更新窗口通过；重复矩阵状态一致 |
| 随机舍入恢复 | checkpoint 原始状态、更新计数恢复一致；恢复后的更新继续通过完整矩阵对照 |
| 随机舍入概率 | 设定 27% 向上舍入，实测 26.994%；Philox 已知答案检查通过 |
| BF16 诊断 | 动量存储与量化前 NS 输入检查通过，DiT-B 二维随机舍入完成 6 步训练 |

新增量化配置的检查记录在 `results/quantization-5060ti.log`、
`results/quantization-zero/` 与 `results/quantization-dit/`。
这些短检查验证实现与恢复路径，不代表随机舍入或二维分块已取得长期收敛收益。

4194304 个动量元素、tile size 256 的单份 FP8 状态合计 **4.0625 MiB**，单份 BF16
为 8 MiB，单份 FP32 为 16 MiB。此前独立计时的融合动量 kernel 在 RTX 5090 上平均约
**0.041 ms**（50 次 CUDA event 计时），在 RTX 5060 Ti 上约 0.175 ms。
该计时只包含动量更新和量化，不包含 NS、权重更新或通信。
该 kernel 的 PyTorch allocator 额外峰值为 0；这不等于整个优化器没有临时显存。

完整 DiT-B 的对照使用 `checks/memory.py`：FP8、BF16、FP16 都保存单份动量，采用相同
BF16 模型计算、FP32 梯度累积、主权重、ZeRO-2 与 NS 设置，不开启高精度 shadow state。
同时记录动量状态字节数和训练峰值显存，不能把动量节省比例当成整个训练的节省比例。

RTX 5090，DiT-B/4（130.38M 参数），batch64，ZeRO-2，各跑 16 次更新的实测：

| 动量格式 | 常驻动量 | 训练峰值 allocated |
| --- | ---: | ---: |
| FP8 E4M3 | 123.40 MiB | 2583.70 MiB |
| BF16 | 243.00 MiB | 2703.30 MiB |
| FP16 | 243.00 MiB | 2703.30 MiB |

FP8 的动量状态减少 **49.22%**，该配置下训练峰值减少 **119.60 MiB（4.42%）**。
这三组对照都只保存一份动量；16bit 对照使用同样的融合 FP32 临时运算与 BF16 NS。
这是一项显存验证，不是 10 epoch 的收敛结论。主权重、梯度、激活等仍占据显存。

参考：[8-bit Muon 研究](https://arxiv.org/abs/2509.23106)、
[DeepSpeed Muon](https://github.com/deepspeedai/DeepSpeed/tree/v0.19.7/deepspeed/runtime/zero/muon)、
[TileLang](https://github.com/tile-ai/tilelang)。该研究的整数线性量化与本实现的 E4M3 是不同格式。
