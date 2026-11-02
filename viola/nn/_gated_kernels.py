"""Fused gated activations and their backward; no activation-sized scratch."""

import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch
import tilelang
import tilelang.language as T

from ._linear_kernels import _dtype


@tilelang.jit
def forward_kernel(hidden, dtype, activation, block=1024, threads=128):
    rows = T.symbolic('rows')

    @T.prim_func
    def kernel(z: T.Tensor((rows, 2 * hidden), dtype), out: T.Tensor((rows, hidden), dtype)):
        with T.Kernel(T.ceildiv(rows * hidden, block), threads=threads) as bid:
            for i in T.Parallel(block):
                index = bid * block + i
                if index < rows * hidden:
                    r, c = index // hidden, index % hidden
                    g = T.cast(z[r, c], 'float32')
                    u = T.cast(z[r, c + hidden], 'float32')
                    if activation == 'silu':
                        a = g / (1.0 + T.exp(-g))
                    elif activation == 'gelu':
                        a = 0.5 * g * (1.0 + T.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
                    else:
                        a = T.max(g, 0.0)
                    out[r, c] = a * u
    return kernel


@tilelang.jit
def backward_kernel(hidden, dtype, activation, block=1024, threads=128):
    rows = T.symbolic('rows')

    @T.prim_func
    def kernel(z: T.Tensor((rows, 2 * hidden), dtype), dy: T.Tensor((rows, hidden), dtype),
               dz: T.Tensor((rows, 2 * hidden), dtype)):
        with T.Kernel(T.ceildiv(rows * hidden, block), threads=threads) as bid:
            for i in T.Parallel(block):
                index = bid * block + i
                if index < rows * hidden:
                    r, c = index // hidden, index % hidden
                    g = T.cast(z[r, c], 'float32')
                    u = T.cast(z[r, c + hidden], 'float32')
                    d = T.cast(dy[r, c], 'float32')
                    if activation == 'silu':
                        s = 1.0 / (1.0 + T.exp(-g))
                        a = g * s
                        derivative = s * (1.0 + g * (1.0 - s))
                    elif activation == 'gelu':
                        t = T.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))
                        a = 0.5 * g * (1.0 + t)
                        derivative = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * g * g)
                    else:
                        a = T.max(g, 0.0)
                        derivative = T.if_then_else(g > 0.0, 1.0, 0.0)
                    dz[r, c] = d * u * derivative
                    dz[r, c + hidden] = d * a
    return kernel


_CONFIGS = ((256, 128), (1024, 128), (2048, 256))
_PLANS = {}
_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def get_config(z, activation, backward=False, dy=None):
    hidden = z.shape[-1] // 2
    device = torch.cuda.get_device_properties(z.device)
    identity = dict(gpu=device.name, capability=[device.major, device.minor],
                    shape=list(z.shape), dtype=_dtype(z), activation=activation, backward=backward,
                    torch=torch.__version__, tilelang=tilelang.__version__, code=_SOURCE_HASH)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if key in _PLANS:
        return _PLANS[key]
    path = Path.home() / '.cache/viola/gated' / (key + '.json')
    if path.exists():
        config = tuple(json.loads(path.read_text())['config'])
    else:
        output = torch.empty_like(z) if backward else torch.empty((z.shape[0], hidden), device=z.device, dtype=z.dtype)
        args = (z, dy, output) if backward else (z, output)
        factory = backward_kernel if backward else forward_kernel
        scores = []
        with torch.cuda.device(z.device):
            for config in _CONFIGS:
                kernel = factory(hidden, _dtype(z), activation, *config)
                kernel(*args)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):
                        kernel(*args)
                for _ in range(3):
                    graph.replay()
                samples = []
                for _ in range(3):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(4):
                        graph.replay()
                    end.record(); end.synchronize()
                    samples.append(start.elapsed_time(end) / 32)
                scores.append((median(samples), config))
                del graph
        _, config = min(scores)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(identity=identity, config=config, candidates=scores), indent=2) + '\n')
        temporary.replace(path)
    _PLANS[key] = config
    return config


class GatedActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, activation, autotune):
        hidden = z.shape[-1] // 2
        flat = z.reshape(-1, 2 * hidden).contiguous()
        config = get_config(flat, activation) if autotune else (1024, 128)
        out = torch.empty((flat.shape[0], hidden), device=z.device, dtype=z.dtype)
        forward_kernel(hidden, _dtype(flat), activation, *config)(flat, out)
        ctx.save_for_backward(flat)
        ctx.shape, ctx.activation, ctx.autotune = z.shape, activation, autotune
        return out.reshape(*z.shape[:-1], hidden)

    @staticmethod
    def backward(ctx, dy):
        z, = ctx.saved_tensors
        hidden = z.shape[-1] // 2
        dy = dy.reshape(-1, hidden).contiguous()
        config = get_config(z, ctx.activation, backward=True, dy=dy) if ctx.autotune else (1024, 128)
        dz = torch.empty_like(z)
        backward_kernel(hidden, _dtype(z), ctx.activation, *config)(z, dy, dz)
        return dz.reshape(ctx.shape), None, None
