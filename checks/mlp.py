"""Compare fused MLP values/gradients with the same unfused FP8 recipe."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import viola


def reference(model, x):
    g, u = model.gate_up(x).chunk(2, -1)
    act = {'silu': F.silu, 'relu': F.relu,
           'gelu': lambda a: F.gelu(a, approximate='tanh')}[model.activation]
    return model.down((act(g.float()) * u.float()).to(x.dtype))


def check(options, *, dtype=torch.bfloat16, freeze=False):
    torch.manual_seed(131)
    hidden = 192 if options.get('rotation_size') else 160
    model = viola.nn.MLP(128, hidden, 96, device='cuda', dtype=dtype, **options)
    if freeze:
        model.gate_up.requires_grad_(False)
    opt = viola.optim.bind_optimizer(model, torch.optim.SGD(model.parameters(), lr=.001))
    x = torch.randn(2, 33, 128, device='cuda', dtype=dtype, requires_grad=not freeze)
    dy = torch.randn(2, 33, 96, device='cuda', dtype=dtype)
    before = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
    y = model(x)
    y.backward(dy)
    actual = [y.detach(), x.grad] + [p.grad for p in model.parameters()]
    actual = [v.detach().clone() if v is not None else None for v in actual]
    model.load_state_dict(before)
    opt.zero_grad(set_to_none=True); x.grad = None
    target = reference(model, x)
    target.backward(dy)
    expected = [target.detach(), x.grad] + [p.grad for p in model.parameters()]
    names = ['y', 'dx', *dict(model.named_parameters())]
    errors = {}
    for name, a, b in zip(names, actual, expected):
        assert (a is None) == (b is None), name
        if a is not None:
            errors[name] = float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
    assert max(errors.values()) < .025, (options, errors)
    opt.step()
    with torch.no_grad():
        assert torch.isfinite(model(x)).all()
    # Multiple pending forwards and gradient accumulation use distinct contexts.
    opt.zero_grad(set_to_none=True)
    (model(x).float().square().mean() + model(x * .7).float().square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    return dict(options=options, dtype=str(dtype), frozen_up=freeze, relative_l2=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    viola.set_master_weight_dtype('fp32')
    viola.set_gradient_dtype('bf16')
    recipes = [dict(weight_dtype='bf16', scaling=s) for s in ('current', 'delayed', 'grouped', 'mxfp8')]
    recipes += [dict(weight_dtype='fp8', scaling=s, mode='fast', activation='gelu') for s in ('current', 'grouped', 'mxfp8')]
    recipes += [dict(weight_dtype='fp16', scaling='current', forward_dtype='e5m2', backward_dtype='e4m3', activation='relu'),
                dict(weight_dtype='bf16', scaling='grouped', accum_dtype='fp16', accum_block_size=32),
                dict(weight_dtype='bf16', scaling='current', accum_dtype='fp16', accum_block_size=32),
                dict(weight_dtype='bf16', scaling='current', rotation_size=64, saved_input_dtype='bf16'),
                dict(weight_dtype='bf16', scaling='mxfp8', bias=False, saved_input_dtype='bf16')]
    results = []
    for options in recipes:
        row = check(options, dtype=torch.float16 if options['weight_dtype'] == 'fp16' else torch.bfloat16)
        results.append(row); print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(results=results, complete=False), indent=2) + '\n')
    results.append(check(dict(weight_dtype='bf16', scaling='current'), freeze=True))
    args.output.write_text(json.dumps(dict(results=results, complete=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
