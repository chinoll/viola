"""Compare native and explicit MMA layouts for blocked FP16 forward GEMMs."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import tilelang
import tilelang.language as T

from benchmark_linear import measure
from viola.nn import _linear_kernels as kernels


@tilelang.jit
def native_block_kernel(m, n, k, block, out_dtype="bfloat16", bm=128, bn=128,
                        bk=128, threads=256, stages=2, swizzle=0):
    @T.prim_func
    def kernel(a: T.Tensor((m, k), "float8_e4m3"), b: T.Tensor((n, k), "float8_e4m3"),
               sa: T.Tensor((1,), "float32"), sb: T.Tensor((1,), "float32"),
               out: T.Tensor((m, n), out_dtype)):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            if swizzle:
                T.use_swizzle(swizzle, order="row")
            ash = T.alloc_shared((bm, bk), "float8_e4m3")
            bsh = T.alloc_shared((bn, bk), "float8_e4m3")
            partial = T.alloc_fragment((bm, bn), "float16")
            total = T.alloc_fragment((bm, bn), "float32")
            result = T.alloc_fragment((bm, bn), out_dtype)
            T.clear(partial)
            T.clear(total)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, ko * bk], ash)
                T.copy(b[bx * bn, ko * bk], bsh)
                T.gemm(ash, bsh, partial, transpose_B=True)
                if (ko + 1) % (block // bk) == 0 or ko == T.ceildiv(k, bk) - 1:
                    for i, j in T.Parallel(bm, bn):
                        total[i, j] += T.cast(partial[i, j], "float32")
                    T.clear(partial)
            for i, j in T.Parallel(bm, bn):
                result[i, j] = total[i, j] * sa[0] * sb[0]
            T.copy(result, out[by * bm, bx * bn])
    return kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="128,512")
    parser.add_argument("--roles", default="up,down")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = dict(gpu=torch.cuda.get_device_name(), results=[])
    path = Path("results/native-block-forward.json")
    for block in map(int, args.blocks.split(",")):
        for role in args.roles.split(","):
            m, n, k = (16384, 16384, 4096) if role == "up" else (16384, 4096, 16384)
            torch.manual_seed(41)
            limit = 2.0 ** math.floor(math.log2(math.sqrt(32768 / block)))
            a0 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b0 = torch.empty(n, k, device="cuda", dtype=torch.bfloat16).uniform_(-1 / math.sqrt(k), 1 / math.sqrt(k))
            sa = kernels.tensor_scale(a0, quantization_limit=limit)
            sb = kernels.tensor_scale(b0, quantization_limit=limit)
            a, _ = kernels.quantize(a0, sa, quantization_limit=limit)
            b, _ = kernels.quantize(b0, sb, quantization_limit=limit)
            del a0, b0
            reference = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            empty_bias = reference.new_empty(0)
            baseline_config = dict(bm=128, bn=128, bk=128, threads=256, stages=3)
            baseline = kernels._mixed_gemm_kernel(m, n, k, "float8_e4m3", "float8_e4m3", "bfloat16", "bfloat16",
                                                   False, accum_dtype="float16", accum_block_size=block,
                                                   accum_reduce_dtype="float32", **baseline_config)
            baseline(a, b, sa, sb, empty_bias, reference)
            candidates = []
            for bm, bn, bk, threads, stages, copy_mode in (
                (128, 128, 128, 256, 2, "default"), (128, 128, 128, 256, 3, "default"),
                (128, 128, 64, 256, 3, "default"), (64, 256, 128, 256, 2, "default"),
                (128, 256, 128, 256, 2, "no_warp_specialization"),
                (256, 128, 128, 256, 2, "no_warp_specialization"),
                (128, 256, 128, 256, 2, "cp_async"), (256, 128, 128, 256, 2, "cp_async"),
                (128, 256, 64, 512, 2, "no_warp_specialization"),
                (256, 128, 64, 512, 2, "no_warp_specialization"),
            ):
                config = dict(bm=bm, bn=bn, bk=bk, threads=threads, stages=stages, swizzle=8)
                compiled = tilelang.compile(native_block_kernel.get_tir(m, n, k, block, **config),
                                           target="cuda", pass_configs=kernels.COPY_PASSES[copy_mode])
                out = torch.full_like(reference, float("nan"))
                compiled(a, b, sa, sb, out)
                finite = bool(out.isfinite().all())
                delta = (out.float() - reference.float())
                error = (delta.norm() / reference.float().norm()).item()
                row = dict(block=block, role=role, config=dict(config, copy_mode=copy_mode),
                           finite=finite, versus_explicit_relative_l2=error, max_abs=delta.abs().max().item())
                assert finite and error < .005, row
                del delta
                row["timing"] = measure(lambda: compiled(a, b, sa, sb, out), graph=True, iterations=5)
                candidates.append((row, compiled))
                report["results"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                del out
            best, compiled = min(candidates, key=lambda pair: pair[0]["timing"]["median_ms"])
            out = torch.empty_like(reference)
            timings = dict(baseline=[], native=[])
            for name in ("baseline", "native", "native", "baseline"):
                run = (lambda: baseline(a, b, sa, sb, empty_bias, reference)) if name == "baseline" else (lambda: compiled(a, b, sa, sb, out))
                timings[name].append(measure(run, graph=True, iterations=8)["median_ms"])
            comparison = dict(block=block, role=role, best=best["config"], timings=timings)
            report.setdefault("comparisons", []).append(comparison)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print("COMPARISON " + json.dumps(comparison), flush=True)
            del a, b, reference, out, candidates, compiled, baseline
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
