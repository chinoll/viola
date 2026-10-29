"""TileLang GEMM search, with validated per-device/per-shape configurations."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch
import tilelang
from tilelang.autotuner import AutoTuner

from ._linear_kernels import _gemm_kernel, _mixed_gemm_kernel, COPY_PASSES


def _torch_dtype(name):
    return {"float8_e4m3": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}.get(name, getattr(torch, name, None))


def candidate_configs(properties, accum_dtype, accum_block_size, mixed,
                      accum_reduce_dtype="float32", copy_mode="default"):
    """Prune shared-memory and register-heavy tiles using the current GPU."""
    configs = []
    for bm, bn in ((64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128), (256, 256)):
        for bk, stages in ((32, 3), (64, 2), (64, 3), (128, 2), (128, 3), (256, 2)):
            if accum_block_size and accum_block_size % bk:
                continue
            if (bm + bn) * bk * stages + 2048 > properties.shared_memory_per_block_optin:
                continue
            for threads in (128, 256, 512):
                accumulator_words = .5 if accum_dtype == "float16" else 1
                if accum_block_size:
                    accumulator_words += .5 if accum_reduce_dtype == "bfloat16" else 1
                registers = bm * bn // threads * accumulator_words + 32
                producer_threads = 128 if properties.major >= 9 and copy_mode == "default" else 0
                if registers * (threads + producer_threads) > 65536:
                    continue
                warps = threads // 32
                row = min((2, 4), key=lambda r: abs(bm / r - bn / (warps // r)))
                configs.append(dict(bm=bm, bn=bn, bk=bk, threads=threads, stages=stages,
                                    row_warps=row if mixed else 0, explicit_mma=mixed,
                                    swizzle=8, swizzle_order="row"))
    return configs


def _remeasure(kernel, candidates, inputs, expected, half):
    """Compare finalists in both orders after compilation, at steady clocks."""
    configs = list({json.dumps(c, sort_keys=True): c for c in candidates}.values())
    output = torch.empty_like(expected)
    graphs, compiled_kernels = [], []
    for config in configs:
        options = dict(config)
        copy_mode = options.pop("copy_mode")
        compiled = tilelang.compile(kernel(**options), target="cuda", pass_configs=COPY_PASSES[copy_mode])
        compiled_kernels.append(compiled)
        output.fill_(float("nan"))
        compiled(*inputs, output)
        if half:
            assert bool(output.isfinite().all()), "Non-finite GEMM output"
            assert ((output.float() - expected.float()).norm() / expected.float().norm()).item() <= .05
        else:
            torch.testing.assert_close(output, expected, rtol=.01, atol=.01)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(8):
                compiled(*inputs, output)
        graphs.append(graph)
    samples = [[] for _ in graphs]
    indices = list(range(len(graphs)))
    for index in indices + indices[::-1]:
        graph = graphs[index]
        for _ in range(4):
            graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):
            graph.replay()
        end.record()
        end.synchronize()
        samples[index].append(start.elapsed_time(end) / 64)
    winner = min(indices, key=lambda i: median(samples[i]))
    return configs[winner], median(samples[winner])


@lru_cache(maxsize=256)
def get_config(m, n, k, a_dtype, b_dtype, out_dtype, bias_dtype, has_bias, device_index,
               accum_dtype="float32", accum_block_size=0, accum_reduce_dtype="float32"):
    properties = torch.cuda.get_device_properties(device_index)
    gpu = properties.name
    source = Path(__file__).with_name("_linear_kernels.py").read_bytes() + Path(__file__).read_bytes()
    mixed = a_dtype != b_dtype
    configs = candidate_configs(properties, accum_dtype, accum_block_size, mixed, accum_reduce_dtype)
    identity = dict(gpu=gpu, capability=torch.cuda.get_device_capability(device_index),
                    multiprocessors=properties.multi_processor_count,
                    shared_memory_limit=properties.shared_memory_per_block_optin,
                    tilelang=tilelang.__version__, torch=torch.__version__,
                    kernel=hashlib.sha256(source).hexdigest(), configs=configs,
                    shape=[m, n, k], types=[a_dtype, b_dtype, out_dtype, bias_dtype], bias=has_bias,
                    accumulator=accum_dtype, accumulation_block=accum_block_size,
                    accumulation_reducer=accum_reduce_dtype)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path.home() / ".cache/viola/linear" / (digest + ".json")
    if path.exists():
        return json.loads(path.read_text())["config"]

    # The closure contains only shape/type scalars, as required by AutoTuner.
    def kernel(bm=64, bn=64, bk=64, threads=128, stages=3, row_warps=0,
               explicit_mma=False, swizzle=0, swizzle_order="row"):
        factory = _mixed_gemm_kernel if mixed or explicit_mma else _gemm_kernel
        options = dict(accum_block_size=accum_block_size, accum_reduce_dtype=accum_reduce_dtype)
        if factory is _mixed_gemm_kernel:
            options["row_warps"] = row_warps
        return factory.get_tir(m, n, k, a_dtype, b_dtype, out_dtype, bias_dtype, has_bias,
                               bm=bm, bn=bn, bk=bk, threads=threads, stages=stages,
                               accum_dtype=accum_dtype, swizzle=swizzle, swizzle_order=swizzle_order, **options)

    with torch.cuda.device(device_index), torch.no_grad(), torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(41)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(_torch_dtype(a_dtype))
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(_torch_dtype(b_dtype))
        sa = torch.ones(1, device="cuda")
        sb = torch.ones(1, device="cuda")
        bias = torch.zeros(n if has_bias else 0, device="cuda", dtype=_torch_dtype(bias_dtype))

        def supply(device):
            return [a, b, sa, sb, bias]

        def reference(a, b, sa, sb, bias):
            result = (a.float() @ b.float().T) * sa * sb
            if has_bias:
                result += bias.float()
            return result.to(_torch_dtype(out_dtype))

        def check_accumulation(actual, expected):
            # Long reduced-precision sums need a norm-based error budget;
            # elementwise relative error is ill-conditioned near cancellation.
            for output, target in zip(actual, expected):
                assert bool(output.isfinite().all()), "Non-finite GEMM output"
                error = (output.float() - target.float()).norm() / target.float().norm().clamp_min(1e-20)
                assert error.item() <= .05, f"Accumulator relative L2 error: {error.item():.6f}"

        def search(candidates, copy_mode="default"):
            tuner = (AutoTuner.from_kernel(kernel, configs=candidates)
                     .set_compile_args(out_idx=[-1], target="cuda", pass_configs=COPY_PASSES[copy_mode])
                     .set_profile_args(supply_prog=supply, ref_prog=reference,
                                       rtol=.01, atol=.01, max_mismatched_ratio=.001,
                                       manual_check_prog=check_accumulation if accum_dtype == "float16" else None,
                                       cache_input_tensors=False, backend="cudagraph"))
            return tuner.run(warmup=5, rep=20, timeout=120)

        result = search(configs)
        best_config, best_latency = dict(result.config), result.latency
        finalists = [dict(best_config, copy_mode="default")]
        layouts = []
        implementations = [True] if mixed else [False, True]
        for explicit in implementations:
            rows = (1, 2, 4, 8) if explicit else (0,)
            for row in rows:
                if explicit and (row > best_config["threads"] // 32 or best_config["bm"] // row < 16):
                    continue
                for panel, order in ((0, "row"), (4, "row"), (8, "row"), (16, "row"),
                                      (4, "column"), (8, "column"), (16, "column")):
                    layouts.append(dict(best_config, explicit_mma=explicit, row_warps=row,
                                        swizzle=panel, swizzle_order=order))
        result = search(layouts)
        finalists.append(dict(result.config, copy_mode="default"))
        if result.latency < best_latency:
            best_config, best_latency = dict(result.config), result.latency
        best_copy_mode = "default"
        for copy_mode in ("no_warp_specialization", "cp_async"):
            # Disabling the producer warps can make a larger tile fit when both
            # FP16 partial sums and FP32 block totals occupy registers.
            copy_configs = candidate_configs(properties, accum_dtype, accum_block_size, mixed,
                                             accum_reduce_dtype, copy_mode)
            copy_candidates = [best_config] + [c for c in copy_configs if c["threads"] == 256 and
                                               c["stages"] == 2 and c["bk"] in (64, 128) and
                                               (c["bm"], c["bn"]) in ((128, 128), (128, 256), (256, 128))]
            result = search(copy_candidates, copy_mode)
            finalists.append(dict(result.config, copy_mode=copy_mode))
            if result.latency < best_latency:
                best_config, best_latency, best_copy_mode = dict(result.config), result.latency, copy_mode
        best_config["copy_mode"] = best_copy_mode
        finalists.extend([best_config, dict(best_config, swizzle=0, swizzle_order="row")])
        best_config, best_latency = _remeasure(kernel, finalists, supply(device_index),
                                               reference(a, b, sa, sb, bias), accum_dtype == "float16")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(**identity, config=best_config, latency_ms=best_latency), indent=2) + "\n")
    temporary.replace(path)
    return best_config
