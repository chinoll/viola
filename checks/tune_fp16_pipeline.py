"""FP16-only speed search over copy pipelines and larger compute warps."""

import argparse
import json
from pathlib import Path

import torch
from tilelang.autotuner import AutoTuner

from viola.nn import _linear_kernels as kernels
from tune_fp16_speed import SHAPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="dw")
    args = parser.parse_args()
    torch.set_num_threads(4)
    m, n, k, at, bt = SHAPES[args.role]
    panel = 16 if args.role == "dw" else 8
    candidates = [
        (256, 256, 64, 2, 256, 2), (256, 256, 64, 3, 256, 2),
        (256, 256, 64, 2, 512, 4), (256, 256, 64, 3, 512, 4),
        (256, 256, 64, 2, 512, 8), (256, 256, 32, 3, 512, 4),
        (256, 256, 32, 4, 512, 4), (128, 256, 128, 2, 256, 2),
        (128, 256, 64, 2, 256, 2), (128, 256, 64, 3, 256, 2),
        (256, 128, 128, 2, 256, 4), (256, 128, 64, 2, 256, 4),
        (128, 128, 128, 2, 256, 2), (128, 128, 64, 2, 256, 2),
        (256, 256, 32, 4, 256, 2), (256, 256, 32, 3, 256, 2),
    ]
    configs = []
    for values in candidates:
        cfg = dict(zip(("bm", "bn", "bk", "stages", "threads", "row_warps"), values))
        for explicit in ([False, True] if at == bt else [True]):
            configs.append(dict(cfg, explicit_mma=explicit))

    def kernel(bm=256, bn=256, bk=64, stages=2, threads=256, row_warps=2, explicit_mma=True):
        factory = kernels._mixed_gemm_kernel if explicit_mma else kernels._gemm_kernel
        kwargs = dict(bm=bm, bn=bn, bk=bk, stages=stages, threads=threads,
                      accum_dtype="float16", swizzle=panel)
        if explicit_mma:
            kwargs["row_warps"] = row_warps
        return factory.get_tir(m, n, k, at, bt, "bfloat16", "bfloat16", False, **kwargs)

    inputs = []
    for rows, fmt in ((m, at), (n, bt)):
        dtype = torch.float8_e4m3fn if fmt == "float8_e4m3" else torch.float8_e5m2
        value = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        scale = kernels.tensor_scale(value, fp8_dtype=dtype)
        q, _ = kernels.quantize(value, scale, fp8_dtype=dtype)
        inputs.append((q, scale))
        del value
    a, sa = inputs[0]; b, sb = inputs[1]
    bias = torch.empty(0, device="cuda", dtype=torch.bfloat16)

    def supply(device):
        return [a, b, sa, sb, bias]

    report = dict(role=args.role, shape=[m, n, k], results=[])
    path = Path(f"results/fp16-pipeline-{args.role}.json")
    backends = [
        ("default", {}),
        ("no_warp_specialization", {"tl.disable_warp_specialized": True}),
        ("cp_async", {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True,
                      "tl.enable_async_copy": True}),
    ]
    for name, passes in backends:
        print("START " + name, flush=True)
        tuner = (AutoTuner.from_kernel(kernel, configs=configs)
                 .set_compile_args(out_idx=[-1], target="cuda", pass_configs=passes)
                 .set_profile_args(supply_prog=supply, skip_check=True,
                                   cache_input_tensors=False, backend="cudagraph"))
        result = tuner.run(warmup=5, rep=15, timeout=120)
        row = dict(backend=name, passes=passes, config=result.config, ms=result.latency,
                   tflops=2*m*n*k/(result.latency*1e9))
        report["results"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
