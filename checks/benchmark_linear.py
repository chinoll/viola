"""Shape sweep: current FP8 Linear vs native BF16, without optimizer math.

M is the flattened token count; X=[M,K], W=[N,K]. Both paths use BF16
inputs, outputs and parameter gradients, with an FP32 master. Training includes
dX and dW. FP8 quantization, scale reductions and transposes are timed.
CUDA Graph timings isolate device work; eager timings include Python dispatch.
Peak memory is an isolated layer with externally retained X/dY, including its
master, gradients and temporaries. It is not a full-model activation-memory test.
"""

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import tilelang

import viola
from viola.nn import _linear_kernels as kernels


def measure(fn, *, graph, repeats=3, iterations=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    unroll = 8 if graph else 1
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            for _ in range(unroll):
                fn()
        runner = captured.replay
        for _ in range(3):
            runner()
        torch.cuda.synchronize()
        # Shape compilation may leave the GPU at idle clocks. Warm for about
        # 30 ms of device work before collecting samples, without clock locks.
        probe_start = torch.cuda.Event(enable_timing=True)
        probe_end = torch.cuda.Event(enable_timing=True)
        probe_start.record()
        runner()
        probe_end.record()
        probe_end.synchronize()
        for _ in range(min(4096, math.ceil(30 / probe_start.elapsed_time(probe_end)))):
            runner()
        torch.cuda.synchronize()
    else:
        runner = fn

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        if graph:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            begin = time.perf_counter()
        for _ in range(iterations):
            runner()
        if graph:
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000
        samples.append(elapsed / (iterations * unroll))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def run_case(m, k, n, mode, repeats, iterations, autotune=False):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(20261029)
    model = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
    if mode != "bf16":
        model = viola.nn.Linear.from_float(model, mode=mode,
                                          forward_dtype="e4m3", backward_dtype="e5m2", autotune=autotune)
    optimizer = viola.optim.bind_optimizer(
        model, torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False))
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    def clear():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None

    def forward():
        return model(x)

    def training():
        clear()
        model(x).backward(dy)

    # Compile every path and initialize cuBLAS workspaces before measuring.
    for _ in range(3):
        training()
    torch.cuda.synchronize()
    assert model.weight.grad.dtype == torch.bfloat16
    assert bool(model.weight.grad.isfinite().all()) and bool(x.grad.isfinite().all())
    clear()
    torch.cuda.synchronize()
    # Count owned tensors explicitly: a cuBLAS workspace first initialized by
    # one case must not appear as that layer's persistent memory advantage/cost.
    tensors = [x, dy, *model.parameters(), *model.buffers(),
               *(p for group in optimizer.param_groups for p in group["params"])]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    persistent = sum(storages.values())
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    training()
    torch.cuda.synchronize()
    peak = persistent + torch.cuda.max_memory_allocated() - allocated_before
    clear()

    row = dict(m=m, k=k, n=n, mode=mode,
               input_mib=x.nbytes / 2**20, weight_elements=k*n,
               persistent_with_inputs_mib=persistent / 2**20,
               train_peak_mib=peak / 2**20,
               model_weight_mib=model.weight.nbytes / 2**20,
               transpose_mib=(model.weight_transpose.nbytes / 2**20 if mode != "bf16" else 0),
               master_mib=sum(p.nbytes for g in optimizer.param_groups for p in g["params"]) / 2**20)
    for name, fn in (("forward", forward), ("train", training)):
        row[name + "_graph"] = measure(fn, graph=True, repeats=repeats, iterations=iterations)
        clear()
        row[name + "_eager"] = measure(fn, graph=False, repeats=repeats, iterations=iterations)
        clear()

    # Prequantized GEMMs exclude quantization and layout conversion, exposing
    # whether the bottleneck is the GEMM itself or the surrounding operators.
    with torch.no_grad():
        master = optimizer.param_groups[0]["params"][0]
        if mode == "bf16":
            gemms = {
                "gemm_forward": lambda: F.linear(x, model.weight),
                "gemm_dx": lambda: dy @ model.weight,
                "gemm_dw": lambda: dy.T @ x,
                "weight_refresh": lambda: model.weight.copy_(master),
            }
        else:
            sx = kernels.tensor_scale(x)
            sg = kernels.tensor_scale(dy, fp8_dtype=torch.float8_e5m2)
            qx, qxt = kernels.quantize(x, sx, transpose=True)
            qg, qgt = kernels.quantize(dy, sg, transpose=True, fp8_dtype=torch.float8_e5m2)
            wt = model.weight_transpose if model.weight_transpose.numel() else kernels.transpose(model.weight)
            gemms = {
                "gemm_forward": lambda: kernels.gemm(qx, model.weight, sx, model.weight_scale,
                                                     out_dtype=torch.bfloat16, autotune=autotune),
                "gemm_dx": lambda: kernels.gemm(qg, wt, sg, model.weight_scale,
                                                out_dtype=torch.bfloat16, autotune=autotune),
                "gemm_dw": lambda: kernels.gemm(qgt, qxt, sg, sx, out_dtype=torch.bfloat16, autotune=autotune),
                "weight_refresh": lambda: model.refresh_weight(master),
            }
        for name, fn in gemms.items():
            row[name + "_graph"] = measure(fn, graph=True, repeats=repeats, iterations=iterations)

    row["forward_tflops"] = 2*m*k*n / (row["forward_graph"]["median_ms"] * 1e9)
    row["train_tflops"] = 6*m*k*n / (row["train_graph"]["median_ms"] * 1e9)
    # Include weight requantization once per update, without conflating the
    # Linear measurement with a particular optimizer's arithmetic/state cost.
    row["train_plus_refresh_graph_ms"] = (row["train_graph"]["median_ms"]
                                           + row["weight_refresh_graph"]["median_ms"])
    return row


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for row in report["results"]:
        flat = {key: (value["median_ms"] if isinstance(value, dict) else value)
                for key, value in row.items()}
        rows.append(flat)
    if rows:
        with output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="64,256,1024,4096,16384")
    parser.add_argument("--dimensions", default="768:768,768:3072,3072:768,2048:2048,2048:8192,8192:2048")
    parser.add_argument("--modes", default="bf16,memory,speed")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--output", default="results/linear-performance.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    report = dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  torch=torch.__version__, cuda=torch.version.cuda, tilelang=tilelang.__version__,
                  bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                  precision="BF16 I/O/gradients; FP32 master; FP8 E4M3 forward/E5M2 backward",
                  bias=False, optimizer_math_included=False, scaling="current",
                  autotune=args.autotune,
                  graph_unroll=8, repeats=args.repeats, iterations=args.iterations,
                  memory_scope="single layer, retained X/dY, master+grads+temporaries, no optimizer state",
                  results=[])
    modes = args.modes.split(",")
    shapes = [(m, k, n) for k, n in (map(int, pair.split(":")) for pair in args.dimensions.split(","))
              for m in map(int, args.tokens.split(","))]
    begin = time.perf_counter()
    for index, (m, k, n) in enumerate(shapes):
        # Rotate measurement order to reduce systematic clock/order bias.
        order = modes[index % len(modes):] + modes[:index % len(modes)]
        for mode in order:
            print(f"start {index+1}/{len(shapes)} M={m} K={k} N={n} {mode}", flush=True)
            row = run_case(m, k, n, mode, args.repeats, args.iterations, args.autotune)
            report["results"].append(row)
            report["elapsed_s"] = time.perf_counter() - begin
            save(report, Path(args.output))
            print(f"done {mode} fwd={row['forward_graph']['median_ms']:.4f} ms "
                  f"train={row['train_graph']['median_ms']:.4f} ms "
                  f"eager_train={row['train_eager']['median_ms']:.4f} ms "
                  f"peak={row['train_peak_mib']:.1f} MiB", flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
