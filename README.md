# Viola · FP8 Linear & Muon

用 TileLang 实现低精度算子与优化器，优先减少训练中的常驻显存。

```python
import viola

model = viola.nn.Linear(1024, 4096, device="cuda")
optimizer = viola.optim.FP8Muon(model.parameters(), lr=1e-3)
viola.optim.bind_optimizer(model, optimizer)
```

| 模块 | 职责 |
| --- | --- |
| `viola.nn` | 网络算子，目前提供 FP8 `Linear` |
| `viola.optim` | 独立的 `FP8Muon` 优化器与 `bind_optimizer` 主权重绑定 |
| `viola.precision` | 主权重、参数梯度的独立精度策略 |

使用普通 PyTorch `backward()` / `optimizer.step()`。不依赖或兼容 DeepSpeed，
没有全局优化器替换、ZeRO 钩子或旧插件入口。

## FP8 Muon

```python
optimizer = viola.optim.FP8Muon(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    mode="balanced",            # fast / balanced / stable
    rounding="nearest",         # nearest / stochastic
    ns_method="standard",       # standard / gram
    adjust_lr_fn="match_rms_adamw",
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

默认矩阵参数使用 Muon，向量参数使用辅助 AdamW。Embedding、输出层等矩阵若要使用
AdamW，可显式分组；各组可以分别指定学习率：

```python
optimizer = viola.optim.FP8Muon([
    {"params": hidden_weights, "use_muon": True, "lr": 1e-3},
    {"params": other_parameters, "use_muon": False, "lr": 1e-4},
], adam_state_dtype="fp32")
```

卷积权重展开为 `[Cout, Cin * Kh * Kw]` 进行正交化。专家矩阵参数组设置 `batched=True`，
则保持最后两维为矩阵，对每个专家分别计算。参数梯度由训练框架完成累积和规约，
优化器每次 `step()` 才更新一次动量，不自行发起分布式通信。

| 预设 | 默认 tile_size | 默认 NS 步数 |
| --- | ---: | ---: |
| `fast` | 256 | 3 |
| `balanced` | 256 | 5 |
| `stable` | 128 | 5 |

预设可用单独参数覆盖。减少 NS 步数会改变正交化近似；更细量化也不保证所有任务误差更小。
`standard` 在 BF16 中做 Newton–Schulz；`gram` 在 FP16 中对较小的 Gram 矩阵迭代，
两步后重建一次矩阵。两种实现均直接使用 PyTorch，无外部优化器依赖。
`match_rms_adamw` 使用 `0.2 * sqrt(max(rows, cols))` 缩放更新，权重衰减仍单独按原学习率计算。

### 量化与舍入

```python
# tile-wise：展平后每 256 个连续元素共享 scale。
optimizer = viola.optim.FP8Muon(
    model.parameters(), granularity="tile", tile_size=256,
    rounding="stochastic", rounding_seed=42,
)

# block-wise：原矩阵每个二维区域共享 scale。
optimizer = viola.optim.FP8Muon(
    model.parameters(), granularity="block", block_shape=(16, 16),
)
```

两种布局均只保存一份 E4M3 动量与每组 FP32 scale，没有逐元素残差或第二份动量。
默认存储约 `1 + 4 / 256 = 1.016` 字节/元素，相比一份 BF16 动量减少约 49.2%。
不对齐的矩阵保留原形状，初始化时 warning；一维尾部填充不足一组的部分，二维边缘只
增加 scale 数量，不填充整个矩阵。

TileLang 在寄存器内以 FP32 恢复和更新动量，产生本步 BF16 正交化输入，然后原地压缩
历史状态。不同参数复用一块最大矩阵大小的 BF16 工作区；NS 的临时张量不属于动量状态。
随机舍入使用 Philox，按 seed、参数顺序、更新步和元素索引生成，不分配随机数矩阵，
不消耗模型训练的随机数流。

默认 `skip_nonfinite=False`，不额外扫描梯度、不自动跳步，Inf/NaN 正常传播。
显式设置 `True` 才检查并跳过整个更新；它仅检查当前进程收到的梯度。

## FP8 Linear

```python
layer = viola.nn.Linear(
    1024, 4096, device="cuda", dtype=torch.bfloat16,
    mode="memory",           # memory / fast
    weight_dtype="fp8",      # fp8 / bf16 / fp16，独立选择权重存储
    forward_dtype="e4m3",    # e4m3 / e5m2
    backward_dtype="e5m2",   # e4m3 / e5m2，独立选择
)
```

前向 `X @ W.T` 与反向 `dY @ W`、`dY.T @ X` 使用真正的 FP8 Tensor Core GEMM，
默认使用 FP32 内部累加。默认前向权重、输入为 E4M3，反向 dY 为 E5M2，直接执行混合
E5M2 × E4M3 乘法，复用前向保存的权重和输入。`backward_dtype` 不控制最终参数梯度的存储。
量化、转置、GEMM 都由 TileLang 实现，scale 与 bias 在 GEMM 尾部融合。

| 权重存储 | 预设 | 常驻模型权重 | 权重转置 |
| --- | --- | --- | --- |
| `fp8` | `memory` | FP8 W | dX 时生成，随后释放 |
| `fp8` | `fast` | FP8 W + W.T | 跨 microbatch 复用，优化后刷新 |
| `bf16` / `fp16` | `memory` | BF16 / FP16 W | backward 从参数量化生成，用后释放 |
| `bf16` / `fp16` | `fast` | BF16 / FP16 W | forward 生成临时 FP8 W.T，仅保存至本次 backward |

三种权重存储都使用 FP8 GEMM。BF16/FP16 存储示例：

```python
layer = viola.nn.Linear(
    1024, 4096, device="cuda", dtype=torch.bfloat16,
    weight_dtype="bf16", mode="memory",  # 也可设为 fp16
    forward_dtype="e4m3", backward_dtype="e5m2",
    scaling="grouped",             # current / delayed / grouped / mxfp8
    weight_group_shape=(32, 32),
)
```

`weight_dtype="bf16"/"fp16"` 的参数和模型 checkpoint 存对应精度的真实权重。forward 临时量化 W，
GEMM 后释放 FP8 W；默认 backward 再从这些权重量化 W.T，不保留跨 microbatch 的
FP8 权重缓存。`cache_weight_transpose=True` 可保存当前 forward 的 W.T 到 backward，
其生命周期与该次计算图相同。`delayed` 模式的 backward 使用对应 forward 保存的 scale，
不重复推进权重统计。分组模式直接从 BF16/FP16 W 分别量化两个方向。

W、X 使用 `forward_dtype`；dY 使用 `backward_dtype`，默认反向执行 E5M2 × E4M3 GEMM。
输入默认保存 FP8 X.T，也可以用 `saved_input_dtype="bf16"` 在 backward 再量化 X.T。
这些选项和现有累加策略、autotune 一起使用；旋转仍适用于 `current/delayed` 缩放。
`dtype` 控制 bias 等浮点张量，`weight_dtype` 单独控制权重存储。

可独立覆盖 `cache_weight_transpose=True/False`、`saved_input_dtype="fp8"/"bf16"/"fp16"/"fp32"`、
`scaling="current"/"delayed"`、`history_len=16`。默认 per-tensor current scaling。
per-tensor 模式的 W 与 W.T 共享一个 FP32 scale。已有层用
`viola.nn.Linear.from_float(nn_linear, weight_dtype="bf16", ...)` 转换；默认存储仍为 `fp8`。

需要局部缩放时，可使用普通 FP8 MMA 配合 FP32 scale，或 RTX Blackwell 的原生 MXFP8：

```python
layer = viola.nn.Linear(
    4096, 16384,
    scaling="grouped",                 # grouped / mxfp8
    weight_group_shape=(32, 32),        # W 的二维分块
    input_group_shape=(1, 32),          # X 沿 GEMM K 维的一维分组
    grad_group_shape=(1, 32),           # dY 的分组
    autotune=True,
    device="cuda", dtype=torch.bfloat16,
)
```

三项分组形状可分别设置。`(1, 64)` 表示每行连续 64 个元素共享 scale；`(64, 64)`
表示一个二维区域共享 scale。边长为 2 的幂，第二维至少为 32；尾部不足一组正常处理。
分组大小描述量化共享范围，独立于 GEMM 的计算 tile。前向和反向均使用 TileLang kernel。

| `scaling` | scale | 计算方式 |
| --- | --- | --- |
| `current` / `delayed` | 整张量一个 FP32 scale | GEMM 尾部恢复尺度 |
| `grouped` | 每组一个自由 FP32 scale | 普通 FP8 MMA，局部和恢复尺度后合并 |
| `mxfp8` | 每组一个 UE8M0 scale，值为 2 的幂 | SM120/SM121 原生 block-scaled MMA，FP32 累加 |

两种分组模式均使用当前分组的 amax，不使用 LS 修正或历史统计。MXFP8 以 32 个 K
维元素为硬件缩放单元；更大的分组和二维分块通过复用对应的 scale 实现。该后端要求
FP32 累加，不与 `rotation_size` 组合；`grouped` 支持 FP16 局部累加和既有范围限制。

X.T 和 dY.T 直接从浮点输入独立量化，FP8 输入保存模式只保留 dW 所需的 X.T。
W 的二维分块两边均至少为 32 时，转置仅调整 FP8 数据和 scale 的布局；一维 W
或行分组小于 32 时，需要在转置方向重新分组量化。默认的二维权重分块可避免这次额外量化。
FP8 权重存储时，`memory` 在 dX 时生成 W.T 并释放，`fast` 缓存 W.T 和对应 scale。
BF16/FP16 权重存储时，W.T 直接从参数量化。恢复 checkpoint 时使用相同的
weight_dtype、scaling、分组形状和 FP8 格式。

分组后端的 `autotune=True` 搜索计算 tile、K 分块、线程数、流水级数、scale 的 shared
memory 缓存和异步搬运配置。候选先验证数值和重复执行的一致性，再计时；结果按 GPU、形状、精度、分组、
累加配置和 kernel 版本缓存在 `~/.cache/viola/grouped/`。再次遇到同一配置直接使用缓存。

可选 `rotation_size=64` 在每 64 个输入通道内做固定 Regular Hadamard 变换，再量化为
FP8；默认 `0` 保持原路径。它要求输入通道数是 64 的倍数，适合对离群值敏感的层做消融。
主权重和 BF16/FP16 参数仍在原坐标系中，生成计算用 FP8 权重时才做旋转。反向使用旋转后的 FP8 张量，并将
dX/dW 原地逆旋转回原坐标；不额外保留完整的浮点旋转副本。此选项不改变 FP8 格式、
per-tensor scale 或累加策略，也不保证所有分布的误差都会降低。
`layer.dequantize_weight()` 返回原坐标下的解量化权重。恢复 checkpoint 时使用相同的
`rotation_size` 配置。

也可以在 FP16 中计算每段局部和，再汇总到 FP32：

```python
layer = viola.nn.Linear(
    4096, 16384, device="cuda", dtype=torch.bfloat16,
    accum_dtype="fp16", accum_block_size=512,
    accum_reduce_dtype="fp32", autotune=True,
)
```

这里的 `512` 是沿矩阵乘法的 K 维每 512 项汇总一次，forward、dX、dW 都应用这一策略；
dW 的 K 维对应展开后的 token 数。局部和与分块总和保留在寄存器中，量化 scale 根据
分块长度为 FP16 累加预留数值范围。`accum_reduce_dtype` 可选 `fp32` 或 `bf16`。
最终梯度的存储精度仍由 `viola.set_gradient_dtype()` 独立设置。

用 `autotune=True` 为实际矩阵形状搜索 GEMM 配置：

```python
layer = viola.nn.Linear(4096, 16384, mode="memory", autotune=True,
                        device="cuda", dtype=torch.bfloat16)
```

前向、dX、dW 根据各自的矩阵形状和 FP8 类型分别搜索分块、线程数、流水线深度，
搜索会根据当前 GPU 的共享内存容量和矩阵形状筛选分块，再测量 warp 布局、块的遍历
顺序和数据搬运方式，并对最终候选进行正反顺序复测。分块模式同时计入局部和、总和的
寄存器占用，前向还比较原生 GEMM 与显式 MMA 布局。候选配置通过数值核对后才参与选择。
配置按 GPU 型号、计算能力、形状、类型、累加方式、工具链和源码缓存到
`~/.cache/viola/linear/`，后续运行直接加载。
默认使用 FP32 累加；autotune 本身不改变计算精度。
默认不开启搜索；首次使用新形状时会编译并计时，使用 CUDA Graph 前先完成普通前后向预热。

训练前可单独预调优 4 倍扩展 MLP 所需的全部 GEMM：

```bash
TILELANG_AUTO_TUNING_CPU_COUNTS=4 PYTHONPATH=. python checks/autotune_linear.py \
  --tokens 16384 --hidden 4096 --expansion 4

PYTHONPATH=. python checks/benchmark_mlp.py --tokens 16384 --autotune
```

两条命令都可追加 `--accum-dtype fp16 --accum-block-size 512 --accum-reduce-dtype fp32`
来调优和测量分块版本。

## 精度策略与主权重

```python
viola.precision.set_master_weight_dtype("fp32")
viola.precision.set_gradient_dtype("bf16")

# 顶层提供相同函数的快捷导出。
viola.set_master_weight_dtype("fp32")
viola.set_gradient_dtype("bf16")

model = viola.nn.Linear(1024, 4096, device="cuda", dtype=torch.bfloat16)
optimizer = viola.optim.FP8Muon(model.parameters(), lr=1e-3)
viola.optim.bind_optimizer(model, optimizer)
```

先设置精度并把模型移到训练设备，再创建、绑定优化器。主权重与参数梯度分别支持
`fp32` / `bf16` / `fp16`，不跟 Linear 的计算格式、Muon 的 FP8 状态或辅助 Adam
的 `adam_state_dtype` 绑定。主权重随机舍入可用
`set_master_weight_dtype("bf16", rounding="stochastic", seed=42)` 单独启用。

`bind_optimizer` 把优化器绑定到真实数值的主权重；`step()` 前直接引用累积梯度，
更新后从主权重重新量化 FP8 模型权重，或写回 BF16/FP16/普通模型参数。
模型存储与优化器主权重独立：默认主权重仍为 FP32；选择与 BF16/FP16 参数相同精度的
主权重时可直接复用参数，不再额外分配主权重副本。
不保存第二份高精度梯度。它也支持普通 AdamW：主权重和梯度 dtype 不同时使用
Viola 的 TileLang AdamW 更新，不做整份梯度升精度复制。

不绑定主权重时，FP8Muon 直接更新传入的普通参数。`set_master_weight_dtype` 本身不创建
主权重，主权重由绑定器管理；参数梯度设置在创建 FP8Muon 或绑定时应用。
这些策略不修改算子内部计算、激活梯度或第三方框架的通信缓冲，亦不追溯修改已有实例。

保存和恢复训练使用常规 `model.state_dict()` 与 `optimizer.state_dict()`。
绑定器将主权重一起放入 optimizer checkpoint。FP8 动量按原始字节保存，scale、Adam
状态保留自己的 dtype，加载时不会先膨胀为完整 FP32 动量。
保持参数顺序和分组一致；Linear 的前后向格式也按原配置构造。

## 运行

需要 PyTorch ≥ 2.11、TileLang 0.1.15、NVIDIA GPU 和可用 CUDA 编译工具链。
在仓库根目录设置 `PYTHONPATH=.` 即可使用，无需安装 Viola 本身。

```bash
pip install -r requirements.txt
PYTHONPATH=. python examples/train.py
PYTHONPATH=. python examples/linear.py
PYTHONPATH=. python examples/linear.py --weight-dtype fp16 --scaling grouped

# 普通 DDP 示例：各 rank 持有自己的完整优化器状态。
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 examples/train.py
```

验证入口：

```bash
PYTHONPATH=. python checks/optimizer.py
PYTHONPATH=. python checks/linear.py
PYTHONPATH=. python checks/quantization.py
PYTHONPATH=. python checks/precision_kernels.py
```

真实 ImageNet latent 的 DiT 对照脚本见 [experiments](experiments/README.md)。
新版本使用普通 checkpoint，不提供旧插件 checkpoint 的转换或恢复。

算法参考：[Muon](https://github.com/KellerJordan/Muon)、
[Gram Newton–Schulz](https://tridao.me/blog/2026/gram-newton-schulz/)。
