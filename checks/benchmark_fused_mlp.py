"""Whole gated MLP: fused GEMM vs unfused FP8, plus a BF16 reference."""

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch import nn
import viola
from viola.nn._gated_kernels import GatedActivation
from benchmark_linear import measure


class Unfused(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gate_up, self.down = model.gate_up, model.down

    def forward(self, x):
        return self.down(GatedActivation.apply(self.gate_up(x), 'silu', False))


def run(shape, backend, storage, scaling):
    gc.collect(); torch.cuda.empty_cache()
    m, d, h = shape
    torch.manual_seed(131)
    if backend == 'bf16':
        model = nn.Module()
        model.gate_up = nn.Linear(d, 2 * h, bias=False, device='cuda', dtype=torch.bfloat16)
        model.down = nn.Linear(h, d, bias=False, device='cuda', dtype=torch.bfloat16)
    else:
        model = viola.nn.MLP(d, h, bias=False, weight_dtype=storage, scaling=scaling,
                             autotune=True, device='cuda', dtype=torch.bfloat16)
    if backend != 'fused':
        model = Unfused(model)
    optimizer = viola.optim.bind_optimizer(model, torch.optim.SGD(model.parameters(), lr=.001))
    torch.manual_seed(231)
    x = torch.randn(m, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)

    def clear():
        optimizer.zero_grad(set_to_none=True); x.grad = None

    def forward():
        return model(x)

    def training():
        clear(); model(x).backward(dy)

    start = time.monotonic()
    for _ in range(2):
        training()
    torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    row = dict(backend=backend, storage=storage, scaling=scaling, shape=shape,
               warmup_seconds=time.monotonic() - start)
    print(json.dumps(dict(event='warmed', **row)), flush=True)
    clear()
    tensors = [*model.parameters(), *model.buffers(), *(p for g in optimizer.param_groups for p in g['params'])]
    owned = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors if t.numel()}
    row['model_and_master_mib'] = sum(owned.values()) / 2**20
    del tensors, owned
    gc.collect(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    training(); torch.cuda.synchronize()
    row['extra_peak_mib'] = (torch.cuda.max_memory_allocated() - base) / 2**20
    row['owned_peak_mib'] = row['model_and_master_mib'] + (x.nbytes + dy.nbytes) / 2**20 + row['extra_peak_mib']
    clear()
    for name, fn in (('forward', forward), ('fwd_bwd', training)):
        row[name] = measure(fn, graph=True, repeats=5, iterations=10)
        clear()
    row['eager'] = measure(training, graph=False, repeats=3, iterations=10)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', default='2048,4096,16384')
    p.add_argument('--dim', type=int, default=4096)
    p.add_argument('--hidden', type=int, default=16384)
    p.add_argument('--scaling', default='mxfp8')
    p.add_argument('--storage', default='bf16')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    viola.set_master_weight_dtype('fp32'); viola.set_gradient_dtype('bf16')
    results = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   master_dtype='fp32', gradient_dtype='bf16',
                   activation_dtype='bf16', mode='memory',
                   fp8_forward='e4m3', fp8_backward='e5m2', accumulation='fp32',
                   complete=False, results=[])
    for tokens in map(int, args.tokens.split(',')):
        shape = (tokens, args.dim, args.hidden)
        for backend in ('bf16', 'unfused', 'fused', 'unfused'):
            row = run(shape, backend, args.storage, args.scaling)
            results['results'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    results['complete'] = True
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
