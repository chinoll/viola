"""Cache fused GEMM plans per GPU, token count, widths and precision recipe."""

import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch
import tilelang

from ._mlp_kernels import compiled
from ._grouped_kernels import dequantize


_PLANS = {}
_SOURCE = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('_mlp_kernels.py').read_bytes()).hexdigest()


def get_config(args, operands, device):
    memory_key = (device, args)
    if memory_key in _PLANS:
        return _PLANS[memory_key]
    prop = torch.cuda.get_device_properties(device)
    identity = dict(gpu=prop.name, capability=[prop.major, prop.minor], args=args,
                    torch=torch.__version__, tilelang=tilelang.__version__, code=_SOURCE)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path.home() / '.cache/viola/mlp' / (key + '.json')
    if path.exists():
        config = json.loads(path.read_text())['config']
    else:
        # Compile a small hardware-pruned set on the first real invocation.
        backward = args[17]
        encoded_output = args[19]
        shapes = ((32, 64, 64, 128), (64, 64, 64, 128), (64, 128, 64, 256),
                  (128, 64, 64, 256), (128, 128, 64, 256), (64, 64, 128, 128),
                  (128, 128, 128, 256), (64, 64, 256, 128))
        if backward:
            shapes += ((128, 256, 64, 256), (256, 128, 64, 256))
        scores, seen = [], set()
        expected = None
        expected_t = None

        def profile(config):
            nonlocal expected, expected_t
            bm, bn, bk, stages = (config[k] for k in ('bm', 'bn', 'bk', 'stages'))
            buffers = 1 if backward else 2
            scale_bytes = 1 if args[12] == 'mxfp8' else 4
            smem = (bm + buffers * bn) * bk * stages + (bm + buffers * bn) * (bk // 32) * scale_bytes
            if encoded_output:
                smem += bm * bn * (4 if args[5] == 'float32' else 2)
            key = tuple(sorted(config.items()))
            if smem > prop.shared_memory_per_block_optin or key in seen:
                return
            seen.add(key)
            fn = compiled(args, tuple(sorted(config.items())), device)
            fn(*operands)
            actual = operands[-1]
            decoded = dequantize(actual, operands[-4], (1, 32)) if encoded_output else actual.float()
            if expected is None:
                expected = decoded.clone()
            error = (decoded - expected).norm() / expected.norm().clamp_min(1e-30)
            assert float(error) < .005, (config, float(error))
            del decoded
            if args[21]:
                decoded_t = dequantize(operands[-3], operands[-2], (1, 32))
                if expected_t is None:
                    expected_t = decoded_t.clone()
                error_t = (decoded_t - expected_t).norm() / expected_t.norm().clamp_min(1e-30)
                assert float(error_t) < .005, (config, 'transpose', float(error_t))
                del decoded_t
            previous = actual.clone()
            fn(*operands)
            assert torch.equal(previous.view(torch.uint8), actual.view(torch.uint8)), (config, 'nondeterministic')
            del previous
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    fn(*operands)
            for _ in range(3):
                graph.replay()
            times = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(4):
                    graph.replay()
                end.record(); end.synchronize()
                times.append(start.elapsed_time(end) / 32)
            scores.append((median(times), config))
            del graph
        for bm, bn, bk, threads in shapes:
            profile(dict(bm=bm, bn=bn, bk=bk, threads=threads, stages=2, copy_mode='default'))
        for _, base in sorted(scores, key=lambda row: row[0])[:2]:
            for stages in (1, 2, 3):
                for copy_mode in ('default', 'cp_async'):
                    profile(dict(base, stages=stages, copy_mode=copy_mode))
        _, config = min(scores, key=lambda row: row[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(identity=identity, config=config, candidates=scores), indent=2) + '\n')
        temporary.replace(path)
    _PLANS[memory_key] = config
    return config
