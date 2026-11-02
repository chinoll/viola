"""Small per-device/shape search for grouped and MXFP8 GEMMs."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch
import tilelang

from ._grouped_kernels import compiled_gemm, scale_shape, decode_scale


@lru_cache(maxsize=256)
def get_config(args, device):
    m, n, k, ad, bd, od, bias_dtype, has_bias, ar, ak, br, bk_scale, native, accum, partial, reducer = args
    properties = torch.cuda.get_device_properties(device)
    identity = dict(gpu=properties.name, capability=[properties.major, properties.minor],
                    torch=torch.__version__, tilelang=tilelang.__version__, args=args,
                    code=hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('_grouped_kernels.py').read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path.home() / '.cache/viola/grouped' / (digest + '.json')
    if path.exists():
        return json.loads(path.read_text())['config']
    formats = {'float8_e4m3': torch.float8_e4m3fn, 'float8_e5m2': torch.float8_e5m2}
    with torch.cuda.device(device), torch.no_grad(), torch.random.fork_rng(devices=[device]):
        torch.manual_seed(41)
        a = torch.randn(m, k, device='cuda').clamp(-3, 3).to(formats[ad])
        b = torch.randn(n, k, device='cuda').clamp(-3, 3).to(formats[bd])
        sf_dtype = torch.uint8 if native else torch.float32
        sa = torch.randint(125, 130, scale_shape((m, k), (ar, ak)), device='cuda').to(sf_dtype)
        sb = torch.randint(125, 130, scale_shape((n, k), (br, bk_scale)), device='cuda').to(sf_dtype)
        if not native:
            sa = (sa - 127).exp2()
            sb = (sb - 127).exp2()
        bias = torch.zeros(n if has_bias else 0, device='cuda', dtype=getattr(torch, bias_dtype))
        output = torch.empty(m, n, device='cuda', dtype=getattr(torch, od))
        # An independent FP64 witness catches scale layout mistakes.
        ri = torch.linspace(0, m - 1, min(m, 32), device='cuda').long()
        ci = torch.linspace(0, n - 1, min(n, 32), device='cuda').long()
        kk = torch.arange(k, device='cuda')
        da = a.view(torch.uint8)[ri].view(a.dtype).float() * decode_scale(sa)[ri[:, None] // ar, kk[None, :] // ak]
        db = b.view(torch.uint8)[ci].view(b.dtype).float() * decode_scale(sb)[ci[:, None] // br, kk[None, :] // bk_scale]
        expected = (da.double() @ db.double().T).float()
        candidates = [(64, 64, 32, 128), (64, 64, 64, 128), (64, 128, 64, 128),
                      (128, 64, 64, 256), (128, 128, 64, 256), (128, 128, 128, 256)]
        rows, seen = [], set()
        def profile(config):
            key = tuple(sorted(config.items()))
            if key in seen:
                return
            seen.add(key)
            kernel = compiled_gemm(args, key, device)
            kernel(a, b, sa, sb, bias, output)
            actual = output[ri[:, None], ci[None, :]].float()
            error = (actual - expected).norm() / expected.norm()
            assert error.item() < (.04 if accum == 'float16' or reducer == 'bfloat16' else .01), (config, error.item())
            for _ in range(2):
                kernel(a, b, sa, sb, bias, output)
                assert torch.equal(output[ri[:, None], ci[None, :]].float(), actual), config
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    kernel(a, b, sa, sb, bias, output)
            for _ in range(5):
                graph.replay()
            samples = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    graph.replay()
                end.record(); end.synchronize()
                samples.append(start.elapsed_time(end) / 64)
            rows.append((median(samples), config))
        for bm, bn, bk, threads in candidates:
            profile(dict(bm=bm, bn=bn, bk=bk, threads=threads, stages=2, cache_scales=True, copy_mode='default'))
        base = min(rows, key=lambda row: row[0])[1]
        for stages in (1, 2):
            for cached in (True, False):
                for copy_mode in ('default', 'cp_async'):
                    profile(dict(base, stages=stages, cache_scales=cached, copy_mode=copy_mode))
        latency, config = min(rows, key=lambda row: row[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(identity=identity, config=config, latency_ms=latency,
                                        candidates=[dict(latency_ms=t, config=c) for t, c in rows]), indent=2) + '\n')
    temporary.replace(path)
    return config
