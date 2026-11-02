"""End-to-end NS timing and update error against the existing compiled recipe."""

import argparse
import json
from pathlib import Path
import time

import torch
from viola.optim._orthogonalize import METHODS
from viola.optim._orthogonalize_kernels import orthogonalize
from benchmark_linear import measure


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', default='256x1024,1024x4096,4096x4096')
    p.add_argument('--methods', default='standard,gram')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(31)
    results = []
    for method in args.methods.split(','):
        for shape in args.shapes.split(','):
            shape = tuple(map(int, shape.split('x')))
            x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
            original = lambda: METHODS[method](x, steps=5)
            candidate = lambda: orthogonalize(x, steps=5, method=method, autotune=True)
            yref = original(); torch.cuda.synchronize()
            start = time.monotonic()
            y = candidate(); torch.cuda.synchronize()
            tune_seconds = time.monotonic() - start
            error = float((y.float() - yref.float()).norm() / yref.float().norm())
            cosine = float(torch.nn.functional.cosine_similarity(y.float().flatten(), yref.float().flatten(), dim=0))
            assert torch.isfinite(y).all()
            row = dict(method=method, shape=shape, relative_l2=error, cosine=cosine,
                       tune_seconds=tune_seconds, torch=measure(original, graph=True, repeats=3, iterations=10),
                       tilelang=measure(candidate, graph=True, repeats=3, iterations=10))
            results.append(row); print(json.dumps(row), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(results=results, complete=False), indent=2) + '\n')
    args.output.write_text(json.dumps(dict(results=results, complete=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
