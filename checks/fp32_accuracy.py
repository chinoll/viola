"""Separate FP8 quantization error from FP32 accumulation/output rounding."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from benchmark_linear import measure
from viola.nn import _linear_kernels as kernels


SHAPES = {
    "forward": (16384, 16384, 4096, "float8_e4m3", "float8_e4m3"),
    "dx": (16384, 4096, 16384, "float8_e5m2", "float8_e4m3"),
    "dw": (4096, 16384, 16384, "float8_e5m2", "float8_e4m3"),
}


def errors(actual, expected):
    # Chunk metrics so this measurement does not require several full FP32 copies.
    difference2 = torch.zeros((), device=actual.device, dtype=torch.float64)
    reference2 = torch.zeros_like(difference2)
    maximum = torch.zeros_like(difference2)
    finite = torch.ones((), device=actual.device, dtype=torch.bool)
    for start in range(0, actual.shape[0], 128):
        a, b = actual[start:start + 128].double(), expected[start:start + 128].double()
        delta = a - b
        difference2 += delta.square().sum()
        reference2 += b.square().sum()
        maximum = torch.maximum(maximum, delta.abs().max())
        finite &= a.isfinite().all()
    return dict(relative_l2=(difference2 / reference2).sqrt().item(),
                max_abs=maximum.item(), rms_abs=(difference2 / actual.numel()).sqrt().item(),
                reference_rms=(reference2 / actual.numel()).sqrt().item(), finite=finite.item())


def run(role, config, previous=None, accum_block_size=0):
    torch.manual_seed(20261029)
    m, n, k, at, bt = SHAPES[role]
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    if role != "dw":
        b.uniform_(-1 / (k if role == "forward" else n) ** .5,
                   1 / (k if role == "forward" else n) ** .5)
    qa_dtype = torch.float8_e4m3fn if at == "float8_e4m3" else torch.float8_e5m2
    limit = 2.0 ** math.floor(math.log2(math.sqrt(32768 / accum_block_size))) if accum_block_size else None
    sa = kernels.tensor_scale(a, fp8_dtype=qa_dtype, quantization_limit=limit)
    sb = kernels.tensor_scale(b, quantization_limit=limit)
    qa, _ = kernels.quantize(a, sa, fp8_dtype=qa_dtype, quantization_limit=limit)
    qb, _ = kernels.quantize(b, sb, quantization_limit=limit)
    options = dict(config)
    copy_mode = options.pop("copy_mode", "default")
    explicit = options.pop("explicit_mma", at != bt)
    factory = kernels._mixed_gemm_kernel if explicit or at != bt else kernels._gemm_kernel
    if factory is kernels._gemm_kernel:
        options.pop("row_warps", None)
    options["accum_dtype"] = "float16" if accum_block_size else "float32"
    if accum_block_size:
        options.update(accum_block_size=accum_block_size, accum_reduce_dtype="float32")
    bias = torch.empty(0, device="cuda", dtype=torch.bfloat16)

    def compile_kernel(dtype):
        args = (m, n, k, at, bt, dtype, "bfloat16", False)
        return kernels._configured_kernel(factory is kernels._mixed_gemm_kernel, args,
                                          tuple(sorted(options.items())), copy_mode, 0)

    output = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    compiled = compile_kernel("bfloat16")
    compiled(qa, qb, sa, sb, bias, output)
    reference = a @ b.T
    report = dict(role=role, shape_mnk=[m, n, k], config=config,
                  accum_block_size=accum_block_size, quantization_limit=limit,
                  versus_bf16=errors(output, reference))
    del reference, a, b
    if previous is not None:
        old_factory = kernels._mixed_gemm_kernel if at != bt or accum_block_size else kernels._gemm_kernel
        old_options = dict(previous)
        if accum_block_size:
            old_options.update(accum_dtype="float16", accum_block_size=accum_block_size,
                               accum_reduce_dtype="float32")
        old = old_factory(m, n, k, at, bt, "bfloat16", "bfloat16", False, **old_options)
        old_output = torch.full_like(output, float("nan"))
        old(qa, qb, sa, sb, bias, old_output)
        report["versus_previous_kernel"] = errors(output, old_output)
        runners = dict(previous=lambda: old(qa, qb, sa, sb, bias, old_output),
                       tuned=lambda: compiled(qa, qb, sa, sb, bias, output))
        report["timing_ms"] = dict(previous=[], tuned=[])
        for name in ("previous", "tuned", "tuned", "previous"):
            timing = measure(runners[name], graph=True, iterations=8)
            report["timing_ms"][name].append(timing["median_ms"])
        del old_output

    # FP64 reference on 64 x 64 evenly spaced output entries, using the same
    # FP8 operands. This removes FP8 input error from the accumulator comparison.
    rows = torch.linspace(0, m - 1, 64, device="cuda").long()
    cols = torch.linspace(0, n - 1, 64, device="cuda").long()
    aa = qa.view(torch.uint8)[rows].view(qa.dtype).double()
    bb = qb.view(torch.uint8)[cols].view(qb.dtype).double()
    exact = (aa @ bb.T) * sa.double() * sb.double()
    report["same_fp8_bf16_output"] = errors(output[rows[:, None], cols], exact)
    del output, aa, bb
    output32 = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float32)
    compile_kernel("float32")(qa, qb, sa, sb, bias, output32)
    report["same_fp8_fp32_output"] = errors(output32[rows[:, None], cols], exact)
    report["all_fp32_outputs_finite"] = output32.isfinite().all().item()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", default="forward,dx,dw")
    parser.add_argument("--configs", required=True, help="JSON mapping role to GEMM configuration")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--accum-block-size", type=int, default=0)
    parser.add_argument("--output", default="results/fp32-accuracy.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    configs = json.loads(Path(args.configs).read_text())
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  reference="BF16 operands, FP32 accumulation, BF16 output; identical input values",
                  accumulator_reference="4096 entries per GEMM, FP64 dot products of identical FP8 operands",
                  results=[])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for role in args.roles.split(","):
        config = configs[role]
        if args.autotune:
            from viola.nn._linear_tuning import get_config
            m, n, k, at, bt = SHAPES[role]
            config = get_config(m, n, k, at, bt, "bfloat16", "bfloat16", False, 0,
                                "float16" if args.accum_block_size else "float32",
                                args.accum_block_size, "float32")
        row = run(role, config, previous=configs[role] if args.autotune else None,
                  accum_block_size=args.accum_block_size)
        report["results"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
