"""FP16 throughput search. No numerical filtering; original FP8 code ranges."""

import argparse
import json
from pathlib import Path

import torch
from tilelang.autotuner import AutoTuner

from viola.nn import _linear_kernels as kernels


SHAPES = {
    "dw": (4096, 16384, 16384, "float8_e5m2", "float8_e4m3"),
    "dx": (16384, 4096, 16384, "float8_e5m2", "float8_e4m3"),
    "forward": (16384, 16384, 4096, "float8_e4m3", "float8_e4m3"),
    "forward_down": (16384, 4096, 16384, "float8_e4m3", "float8_e4m3"),
    "dx_down": (16384, 16384, 4096, "float8_e5m2", "float8_e4m3"),
    "dw_swapped": (16384, 4096, 16384, "float8_e4m3", "float8_e5m2"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", default="dw")
    args = parser.parse_args()
    torch.set_num_threads(4)
    for role in args.roles.split(","):
        m, n, k, at, bt = SHAPES[role]
        configs = []
        for bm, bn in ((64, 128), (128, 64), (128, 128), (128, 256), (256, 128), (256, 256)):
            for bk, stages in ((64, 2), (64, 3), (128, 2), (128, 3), (256, 2)):
                if (bm + bn) * bk * stages > 98304:
                    continue
                for threads in (128, 256):
                    if bm * bn // threads > 256:
                        continue
                    warps = threads // 32
                    row = min((2, 4), key=lambda r: abs(bm / r - bn / (warps // r)))
                    configs.append(dict(bm=bm, bn=bn, bk=bk, stages=stages, threads=threads,
                                        row_warps=row, swizzle=8, swizzle_order="row", explicit_mma=True))

        def kernel(bm=128, bn=128, bk=128, threads=256, stages=3,
                   row_warps=4, swizzle=0, swizzle_order="row", explicit_mma=True):
            factory = kernels._mixed_gemm_kernel if explicit_mma or at != bt else kernels._gemm_kernel
            options = dict(bm=bm, bn=bn, bk=bk, threads=threads, stages=stages,
                           accum_dtype="float16", swizzle=swizzle, swizzle_order=swizzle_order)
            if factory is kernels._mixed_gemm_kernel:
                options["row_warps"] = row_warps
            return factory.get_tir(m, n, k, at, bt, "bfloat16", "bfloat16", False, **options)

        values = []
        for rows, fmt in ((m, at), (n, bt)):
            dtype = torch.float8_e4m3fn if fmt == "float8_e4m3" else torch.float8_e5m2
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            scale = kernels.tensor_scale(x, fp8_dtype=dtype)
            q, _ = kernels.quantize(x, scale, fp8_dtype=dtype)
            values.append((q, scale))
            del x
        a, sa = values[0]; b, sb = values[1]
        bias = torch.empty(0, device="cuda", dtype=torch.bfloat16)

        def supply(device):
            return [a, b, sa, sb, bias]

        def search(candidates):
            tuner = (AutoTuner.from_kernel(kernel, configs=candidates)
                     .set_compile_args(out_idx=[-1], target="cuda")
                     .set_profile_args(supply_prog=supply, skip_check=True,
                                       cache_input_tensors=False, backend="cudagraph"))
            return tuner.run(warmup=5, rep=15, timeout=120)

        print(f"START {role}: {len(configs)} tiles", flush=True)
        first = search(configs)
        report = dict(role=role, shape=[m, n, k], types=[at, bt],
                      first=dict(config=first.config, ms=first.latency))
        path = Path(f"results/fp16-search-{role}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        candidates = []
        for row in (1, 2, 4, 8):
            if row > first.config["threads"] // 32:
                continue
            if first.config["bm"] // row < 16:
                continue
            for swizzle, order in ((0, "row"), (4, "row"), (8, "row"), (16, "row"),
                                   (4, "column"), (8, "column"), (16, "column")):
                candidates.append(dict(first.config, row_warps=row, swizzle=swizzle, swizzle_order=order))
        if at == bt:
            candidates.append(dict(first.config, explicit_mma=False))
        print(f"REFINE {role}: {len(candidates)} layouts", flush=True)
        second = search(candidates)
        report["best"] = dict(config=second.config, ms=second.latency, tflops=2*m*n*k/(second.latency*1e9))
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
