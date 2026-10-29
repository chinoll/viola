"""TileLang GEMM search, with validated per-device/per-shape configurations."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch
import tilelang
from tilelang.autotuner import AutoTuner

from ._linear_kernels import _gemm_kernel, _mixed_gemm_kernel


CONFIGS = [dict(zip(("bm", "bn", "bk", "threads", "stages"), values)) for values in (
    (64, 64, 64, 128, 3),
    (64, 128, 64, 128, 3),
    (128, 64, 64, 128, 3),
    (128, 128, 64, 128, 3),
    (128, 128, 64, 256, 3),
    (128, 128, 128, 256, 3),
    (128, 128, 128, 128, 2),
    (128, 256, 64, 256, 3),
    (256, 128, 64, 256, 3),
    (64, 128, 128, 128, 3),
    (128, 64, 128, 128, 3),
    (128, 256, 128, 256, 2),
    (256, 128, 128, 256, 2),
    (64, 64, 128, 128, 3),
    (128, 128, 32, 256, 3),
    (128, 128, 64, 256, 4),
)]


def _torch_dtype(name):
    return {"float8_e4m3": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}.get(name, getattr(torch, name, None))


@lru_cache(maxsize=256)
def get_config(m, n, k, a_dtype, b_dtype, out_dtype, bias_dtype, has_bias, device_index):
    gpu = torch.cuda.get_device_name(device_index)
    source = Path(__file__).with_name("_linear_kernels.py").read_bytes()
    identity = dict(gpu=gpu, capability=torch.cuda.get_device_capability(device_index),
                    tilelang=tilelang.__version__, torch=torch.__version__,
                    kernel=hashlib.sha256(source).hexdigest(), configs=CONFIGS,
                    shape=[m, n, k], types=[a_dtype, b_dtype, out_dtype, bias_dtype], bias=has_bias)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path.home() / ".cache/viola/linear" / (digest + ".json")
    if path.exists():
        return json.loads(path.read_text())["config"]

    # The closure contains only shape/type scalars, as required by AutoTuner.
    def kernel(bm=64, bn=64, bk=64, threads=128, stages=3):
        factory = _gemm_kernel if a_dtype == b_dtype else _mixed_gemm_kernel
        return factory.get_tir(m, n, k, a_dtype, b_dtype, out_dtype, bias_dtype, has_bias,
                               bm=bm, bn=bn, bk=bk, threads=threads, stages=stages)

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

        tuner = (AutoTuner.from_kernel(kernel, configs=CONFIGS)
                 .set_compile_args(out_idx=[-1], target="cuda")
                 .set_profile_args(supply_prog=supply, ref_prog=reference,
                                   rtol=.01, atol=.01, max_mismatched_ratio=.001,
                                   cache_input_tensors=False, backend="cudagraph"))
        result = tuner.run(warmup=5, rep=20, timeout=120)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(**identity, config=result.config, latency_ms=result.latency), indent=2) + "\n")
    temporary.replace(path)
    return result.config
