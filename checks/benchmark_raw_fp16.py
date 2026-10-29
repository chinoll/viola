"""Speed ceiling with unchanged FP8 ranges; FP16 overflow is allowed here."""

import gc
import json
from pathlib import Path

import torch
import viola
from viola.nn import _linear_kernels as kernels, _linear_tuning

from benchmark_linear import measure
from benchmark_mlp import run_case


def main():
    torch.set_num_threads(4)
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    tile = dict(bm=128, bn=128, bk=128, threads=256, stages=3)
    _linear_tuning.get_config = lambda *args: tile.copy()
    report = dict(gpu=torch.cuda.get_device_name(), tile=tile,
                  scope="timing only; original FP8 range; overflow allowed; no optimizer math",
                  mlp=[], gemm=[])
    path = Path("results/raw-fp16-speed.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    for dtype in ("fp32", "fp16"):
        print("MLP " + dtype, flush=True)
        # An explicit large override restores the format's full code range.
        options = dict(accum_dtype=dtype, quantization_limit=1e9)
        row = run_case(16384, 4096, 4, "memory", 3, True, options, check_finite=False)
        row["accum_dtype"] = dtype
        report["mlp"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("RESULT " + json.dumps(row), flush=True)
    for role, m, n, k, fmt in [("forward_up", 16384, 16384, 4096, torch.float8_e4m3fn),
                               ("backward_up", 16384, 4096, 16384, torch.float8_e5m2),
                               ("backward_down_dw", 4096, 16384, 16384, torch.float8_e5m2)]:
        gc.collect(); torch.cuda.empty_cache()
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        sa = kernels.tensor_scale(a, fp8_dtype=fmt); sb = kernels.tensor_scale(b)
        qa, _ = kernels.quantize(a, sa, fp8_dtype=fmt)
        qb, _ = kernels.quantize(b, sb)
        del a, b
        for dtype in ("fp32", "fp16"):
            def run():
                return kernels.gemm(qa, qb, sa, sb, out_dtype=torch.bfloat16,
                                    accum_dtype=dtype, autotune=True)
            output = run()
            finite_fraction = output.isfinite().float().mean().item()
            del output
            timing = measure(run, graph=True, iterations=10)
            row = dict(role=role, shape=[m, n, k], accum_dtype=dtype,
                       finite_fraction=finite_fraction, timing=timing,
                       tflops=2*m*n*k/(timing["median_ms"]*1e9))
            report["gemm"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print("GEMM " + json.dumps(row), flush=True)
        del qa, qb, sa, sb


if __name__ == "__main__":
    main()
