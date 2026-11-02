"""Warmed FP8 compute: native BF16, FP8 storage, and on-demand BF16 storage."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
import tilelang

import viola
from benchmark_linear import measure


def run_case(shape, kind, recipe, repeats, iterations):
    m, k, n = shape
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(177)
    if kind == 'linear':
        model = nn.Linear(k, n, bias=False, device='cuda', dtype=torch.bfloat16)
    else:
        model = nn.Sequential(nn.Linear(k, n, bias=False, device='cuda', dtype=torch.bfloat16),
                              nn.SiLU(), nn.Linear(n, k, bias=False, device='cuda', dtype=torch.bfloat16))
    if recipe['backend'] == 'viola':
        options = dict(weight_dtype=recipe['storage'], mode=recipe['mode'], scaling=recipe['scaling'],
                       autotune=True, accum_dtype='fp32', forward_dtype='e4m3', backward_dtype='e5m2')
        if kind == 'linear':
            model = viola.nn.Linear.from_float(model, **options)
        else:
            for i in (0, 2):
                model[i] = viola.nn.Linear.from_float(model[i], **options)
    optimizer = viola.optim.bind_optimizer(model, torch.optim.SGD(model.parameters(), lr=.001))
    # from_float consumes initializer randomness. Give every recipe identical operands.
    torch.manual_seed(277)
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n if kind == 'linear' else k, device='cuda', dtype=torch.bfloat16)

    def clear():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None

    def forward():
        return model(x)

    def training():
        clear()
        model(x).backward(dy)

    @torch.no_grad()
    def refresh():
        for param, master, layer in optimizer._viola_linear_masters.pairs:
            if layer is None:
                param.copy_(master)
            else:
                layer.refresh_weight(master)

    def training_refresh():
        training()
        refresh()

    start = time.monotonic()
    for _ in range(3):
        training()
    torch.cuda.synchronize()
    warmup = time.monotonic() - start
    assert torch.isfinite(x.grad).all()
    assert all(p.grad.dtype == torch.bfloat16 and torch.isfinite(p.grad).all() for p in model.parameters())
    clear()
    row = dict(**recipe, kind=kind, shape=shape, warmup_seconds=warmup)
    print(json.dumps(dict(event='warmed', **row)), flush=True)
    for name, fn in (('forward', forward), ('fwd_bwd', training), ('fwd_bwd_writeback', training_refresh)):
        row[name] = measure(fn, graph=True, repeats=repeats, iterations=iterations)
        clear()
    row['fwd_bwd_eager'] = measure(training, graph=False, repeats=repeats, iterations=iterations)
    clear()
    gc.collect()
    torch.cuda.empty_cache()
    tensors = [*model.parameters(), *model.buffers(),
               *(p for group in optimizer.param_groups for p in group['params'])]
    owned = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors if t.numel()}
    row['persistent_model_and_master_mib'] = sum(owned.values()) / 2**20
    row['retained_inputs_mib'] = (x.nbytes + dy.nbytes) / 2**20
    del tensors, owned
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    training()
    torch.cuda.synchronize()
    row['fwd_bwd_extra_peak_mib'] = (torch.cuda.max_memory_allocated() - before) / 2**20
    row['peak_owned_mib'] = (row['persistent_model_and_master_mib'] + row['retained_inputs_mib']
                             + row['fwd_bwd_extra_peak_mib'])
    clear()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shape', default='4096,4096,4096', help='tokens,input,output; MLP is input -> output -> input')
    p.add_argument('--kind', choices=('linear', 'mlp'), default='linear')
    p.add_argument('--scalings', default='current,grouped,mxfp8')
    p.add_argument('--storages', default='fp8,bf16')
    p.add_argument('--modes', default='memory,fast')
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    viola.set_master_weight_dtype('fp32')
    viola.set_gradient_dtype('bf16')
    shape = list(map(int, args.shape.split(',')))
    baseline = dict(backend='pytorch', storage='bf16', scaling='none', mode='none')
    recipes = [dict(backend='viola', storage=w, scaling=s, mode=mode)
               for s in args.scalings.split(',') for w in args.storages.split(',') for mode in args.modes.split(',')]
    if args.reverse:
        recipes.reverse()
    # Recheck the native baseline at the end to expose thermal/order drift.
    recipes = [baseline, *recipes, baseline]
    root = Path(viola.__file__).parent
    report = dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  torch=torch.__version__, tilelang=tilelang.__version__, kind=args.kind, shape=shape,
                  precision='BF16 inputs/outputs/gradients, FP32 master; E4M3 W/X, E5M2 dY, FP32 accumulator',
                  bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                  autotune=True, optimizer_math_included=False, graph_unroll=8,
                  quantization_and_transposes_included=True, forward_includes_training_activation_saves=True,
                  repeats=args.repeats, iterations=args.iterations, reverse=args.reverse,
                  sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')},
                  results=[], complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, recipe in enumerate(recipes):
        print(json.dumps(dict(event='start', index=index, **recipe, shape=shape)), flush=True)
        row = run_case(shape, args.kind, recipe, args.repeats, args.iterations)
        report['results'].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(event='result', **row)), flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
