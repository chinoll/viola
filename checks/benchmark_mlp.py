"""Large 4x-expansion MLP, including SiLU and all input/parameter gradients."""

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch import nn
import tilelang

import viola
from benchmark_linear import measure, save


def run_case(tokens, width, expansion, mode, iterations, autotune=False, linear_options=None,
              check_finite=True):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(20261029)
    model = nn.Sequential(nn.Linear(width, expansion * width, bias=False, device="cuda", dtype=torch.bfloat16),
                          nn.SiLU(),
                          nn.Linear(expansion * width, width, bias=False, device="cuda", dtype=torch.bfloat16))
    if mode != "bf16":
        for index in (0, 2):
            model[index] = viola.nn.Linear.from_float(model[index], mode=mode, autotune=autotune,
                                                      **(linear_options or {}))
    optimizer = viola.optim.bind_optimizer(model, torch.optim.SGD(model.parameters(), lr=.001, foreach=False))
    torch.manual_seed(123)
    x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)

    def clear():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None

    def forward():
        return model(x)

    def training():
        clear()
        model(x).backward(dy)

    @torch.no_grad()
    def refresh():
        for param, master, layer in optimizer._viola_linear_masters.pairs:
            if layer is None:
                param.copy_(master)
            else:
                layer.refresh_weight(master)

    def training_refresh():
        training()
        refresh()

    for _ in range(3):
        training()
    torch.cuda.synchronize()
    assert all(p.grad.dtype == torch.bfloat16 for p in model.parameters())
    if check_finite:
        assert all(bool(p.grad.isfinite().all()) for p in model.parameters())
        assert bool(x.grad.isfinite().all())
    clear()
    tensors = [x, dy, *model.parameters(), *model.buffers(),
               *(p for group in optimizer.param_groups for p in group["params"])]
    storage_bytes = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    persistent = sum(storage_bytes.values())
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    training()
    torch.cuda.synchronize()
    peak = persistent + torch.cuda.max_memory_allocated() - allocated_before
    clear()
    row = dict(tokens=tokens, width=width, expansion=expansion, mode=mode,
               linear_options=linear_options or {},
               finite_check=check_finite,
               parameters=sum(p.numel() for p in model.parameters()),
               input_mib=x.nbytes / 2**20,
               model_weight_mib=sum(p.nbytes for p in model.parameters()) / 2**20,
               transpose_mib=sum(layer.weight_transpose.nbytes for layer in model.modules()
                                 if isinstance(layer, viola.nn.Linear)) / 2**20,
               master_mib=sum(p.nbytes for g in optimizer.param_groups for p in g["params"]) / 2**20,
               persistent_with_inputs_mib=persistent / 2**20, train_peak_mib=peak / 2**20)
    for name, fn in (("forward", forward), ("train", training)):
        for execution in ("graph", "eager"):
            row[name + "_" + execution] = measure(fn, graph=execution == "graph", iterations=iterations)
            clear()
    row["train_refresh_graph"] = measure(training_refresh, graph=True, iterations=iterations)
    clear()
    with torch.no_grad():
        row["weight_refresh_graph"] = measure(refresh, graph=True, iterations=iterations)
    row["train_tflops"] = 12*tokens*width*width*expansion / (row["train_graph"]["median_ms"] * 1e9)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="16384,4096,32768")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--modes", default="bf16,memory,speed")
    parser.add_argument("--accum-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--accum-block-size", type=int, default=0)
    parser.add_argument("--accum-reduce-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--output", default="results/mlp-performance.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  tilelang=tilelang.__version__, architecture="Linear -> SiLU -> Linear, expansion=4, no bias",
                  precision="BF16 I/O/gradients; FP32 master; FP8 E4M3 forward/E5M2 backward",
                  graph_unroll=8, optimizer_math_included=False, scaling="current",
                  autotune=args.autotune,
                  memory_scope="full MLP with retained root X/dY, master+grads+activations+temporaries; no optimizer state",
                  results=[])
    begin = time.perf_counter()
    modes = args.modes.split(",")
    linear_options = dict(accum_dtype=args.accum_dtype, accum_block_size=args.accum_block_size,
                          accum_reduce_dtype=args.accum_reduce_dtype)
    for index, tokens in enumerate(map(int, args.tokens.split(","))):
        for mode in modes[index % len(modes):] + modes[:index % len(modes)]:
            print(f"start tokens={tokens} width={args.width} expansion={args.expansion} {mode}", flush=True)
            row = run_case(tokens, args.width, args.expansion, mode, args.iterations, args.autotune, linear_options)
            report["results"].append(row)
            report["elapsed_s"] = time.perf_counter() - begin
            save(report, Path(args.output))
            print(f"done {mode} fwd={row['forward_graph']['median_ms']:.4f} ms "
                  f"train={row['train_graph']['median_ms']:.4f} ms "
                  f"eager={row['train_eager']['median_ms']:.4f} ms "
                  f"peak={row['train_peak_mib']:.1f} MiB", flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
