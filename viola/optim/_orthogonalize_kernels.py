"""TileLang Newton–Schulz GEMMs with polynomial/update epilogues."""

import hashlib
import json
import os
from pathlib import Path
from statistics import median
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def matmul_kernel(dtype, transpose_b, epilogue, batch, m, n, k,
                  bm=64, bn=64, bk=64, threads=128, stages=2):
    bshape = (batch, n, k) if transpose_b else (batch, k, n)

    @T.prim_func
    def main(a: T.Tensor((batch, m, k), dtype), b: T.Tensor(bshape, dtype),
             residual: T.Tensor((batch, m, n) if epilogue else (0, 0, 0), dtype),
             out: T.Tensor((batch, m, n), dtype)):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), batch, threads=threads) as (bx, by, bz):
            ash = T.alloc_shared((bm, bk), dtype)
            bsh = T.alloc_shared((bn, bk) if transpose_b else (bk, bn), dtype)
            acc = T.alloc_fragment((bm, bn), 'float32')
            result = T.alloc_fragment((bm, bn), dtype)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[bz, by * bm, ko * bk], ash)
                if transpose_b:
                    T.copy(b[bz, bx * bn, ko * bk], bsh)
                else:
                    T.copy(b[bz, ko * bk, bx * bn], bsh)
                T.gemm(ash, bsh, acc, transpose_B=transpose_b)
            for i, j in T.Parallel(bm, bn):
                # The reference rounds GEMM outputs before scalar arithmetic.
                value = T.cast(T.cast(acc[i, j], dtype), 'float32')
                if epilogue:
                    r = T.if_then_else(by * bm + i < m and bx * bn + j < n,
                                       T.cast(residual[bz, by * bm + i, bx * bn + j], 'float32'), 0.)
                    if epilogue == 1:
                        value = -4.7750 * r + 2.0315 * value
                    else:
                        value += 3.4445 * r
                result[i, j] = value
            T.copy(result, out[bz, by * bm, bx * bn])
    return main


@lru_cache(maxsize=512)
def compiled(args, options, device):
    options = dict(options)
    copy_mode = options.pop('copy_mode', 'default')
    passes = {} if copy_mode == 'default' else {
        'tl.disable_warp_specialized': True, 'tl.disable_tma_lower': True,
        'tl.enable_async_copy': True}
    with torch.cuda.device(device):
        return tilelang.compile(matmul_kernel.get_tir(*args, **options), target='cuda', pass_configs=passes)


def _matmul(a, b, config, *, transpose_b=False, residual=None, epilogue=0):
    batch, m, k = a.shape
    n = b.shape[-2] if transpose_b else b.shape[-1]
    out = a.new_empty((batch, m, n))
    dtype = 'bfloat16' if a.dtype == torch.bfloat16 else 'float16'
    args = (dtype, transpose_b, epilogue, batch, m, n, k)
    operands = (a, b, residual if residual is not None else a.new_empty(0, 0, 0), out)
    if config is None:
        config = plan(args, operands, a.device.index)
    compiled(args, tuple(sorted(config.items())), a.device.index)(*operands)
    return out


def run(matrix, steps, method, config):
    shape = matrix.shape
    tall = shape[-2] > shape[-1]
    dtype = torch.bfloat16 if method == 'standard' else torch.float16
    x = matrix.to(dtype)
    if tall:
        x = x.mT
    x = (x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)).contiguous()
    x = x.reshape(-1, x.shape[-2], x.shape[-1])
    if method == 'standard' or x.shape[-2] == x.shape[-1]:
        for _ in range(steps):
            r = _matmul(x, x, config, transpose_b=True)
            z = _matmul(r, r, config, residual=r, epilogue=1)
            x = _matmul(z, x, config, residual=x, epilogue=2)
    else:
        for length in (min(steps, 2), max(steps - 2, 0)):
            if length:
                r = _matmul(x, x, config, transpose_b=True)
                for i in range(length):
                    z = _matmul(r, r, config, residual=r, epilogue=1)
                    if i == 0:
                        q = z.clone()
                        q.diagonal(dim1=-2, dim2=-1).add_(3.4445)
                    else:
                        q = _matmul(z, q, config, residual=q, epilogue=2)
                    if i + 1 < length:
                        rz = _matmul(z, r, config, residual=r, epilogue=2)
                        r = _matmul(z, rz, config, residual=rz, epilogue=2)
                x = _matmul(q, x, config)
    if tall:
        x = x.mT
    return x.reshape(shape)


_PLANS = {}
_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def plan(args, operands, device):
    memory_key = (device, args)
    if memory_key in _PLANS:
        return _PLANS[memory_key]
    prop = torch.cuda.get_device_properties(device)
    identity = dict(gpu=prop.name, capability=[prop.major, prop.minor], args=args,
                    torch=torch.__version__, tilelang=tilelang.__version__, source=_SOURCE)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path.home() / '.cache/viola/orthogonalize' / (key + '.json')
    if path.exists():
        config = json.loads(path.read_text())['config']
    else:
        dtype, transpose_b, epilogue, batch, m, n, k = args
        a, b, residual, output = operands
        # Validate each schedule against FP32 arithmetic on the same operands.
        expected = a.float() @ (b.float().mT if transpose_b else b.float())
        expected = expected.to(a.dtype).float()
        if epilogue == 1:
            expected = -4.7750 * residual.float() + 2.0315 * expected
        elif epilogue == 2:
            expected = 3.4445 * residual.float() + expected
        expected = expected.to(a.dtype)
        norm = expected.float().norm().clamp_min(1e-30)
        candidates = []
        for bm, bn in ((32, 64), (64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)):
            for bk in (32, 64, 128):
                threads = 128 if bm * bn <= 8192 else 256
                if (bm + bn) * bk * 2 * 2 > prop.shared_memory_per_block_optin:
                    continue
                candidates.append(dict(bm=bm, bn=bn, bk=bk, threads=threads, stages=2, copy_mode='default'))
        scores, seen = [], set()

        def build(config):
            return config, compiled(args, tuple(sorted(config.items())), device)

        def profile(config, fn):
            seen.add(tuple(sorted(config.items())))
            fn(*operands)
            error = float((output.float() - expected.float()).norm() / norm)
            assert error < .02 and torch.isfinite(output).all(), (args, config, error)
            previous = output.clone()
            fn(*operands)
            assert torch.equal(previous, output), (config, 'nondeterministic')
            del previous
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    fn(*operands)
            for _ in range(5):
                graph.replay()
            times = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):
                    graph.replay()
                end.record(); end.synchronize()
                times.append(start.elapsed_time(end) / 40)
            scores.append((median(times), config))
            del graph

        # Limit host compilation concurrency; only one candidate runs on GPU.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for config, fn in pool.map(build, candidates):
                profile(config, fn)
            finalists = [c for _, c in sorted(scores, key=lambda row: row[0])[:2]]
            refinements = []
            for base in finalists:
                for stages in (1, 2, 3):
                    for copy_mode in ('default', 'cp_async'):
                        config = dict(base, stages=stages, copy_mode=copy_mode)
                        smem = (config['bm'] + config['bn']) * config['bk'] * 2 * stages
                        if smem <= prop.shared_memory_per_block_optin and tuple(sorted(config.items())) not in seen:
                            refinements.append(config)
            for config, fn in pool.map(build, refinements):
                profile(config, fn)
        _, config = min(scores, key=lambda row: row[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(identity=identity, config=config, candidates=scores), indent=2) + '\n')
        temporary.replace(path)
    _PLANS[memory_key] = config
    return config


def orthogonalize(matrix, steps=5, method='standard', autotune=True):
    config = None if autotune else dict(bm=64, bn=64, bk=64, threads=128, stages=2, copy_mode='default')
    return run(matrix, steps, method, config)
