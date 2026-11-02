"""BF16/FP16 weight storage: FP8 math, cache lifetime, accumulation and resume."""

import argparse
import copy
import json
from pathlib import Path

import torch

import viola
from viola.nn import _grouped_kernels as grouped
from viola.nn import _rotation_kernels as rotation


def error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)).item()


def tensor_reference(x, dtype, scale, limit=None):
    limit = torch.finfo(dtype).max if limit is None else limit
    value, scale = x.detach().float().cpu(), scale.detach().float().cpu()
    return ((value / scale).clamp(-limit, limit).to(dtype).float() * scale).to(x.device)


def group_reference(x, dtype, group, native, limit=None, transposed=False):
    q, s = grouped.quantize(x.detach().cpu(), dtype, group, native=native,
                            limit=limit, transposed=transposed)
    return grouped.dequantize(q, s, group).to(x.device)


def check_math(scaling, mode, *, saved='fp8', rotate=0, formats=('e4m3', 'e5m2'),
               gw=(32, 32), half=False, need_x=True, need_w=True, storage='bf16', activations='bf16'):
    torch.manual_seed(121)
    viola.set_master_weight_dtype('fp32')
    viola.set_gradient_dtype('bf16')
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[activations]
    source = torch.nn.Linear(128, 96, device='cuda', dtype=dtype)
    layer = viola.nn.Linear.from_float(source, weight_dtype=storage, mode=mode, scaling=scaling,
        saved_input_dtype=saved, rotation_size=rotate, weight_group_shape=gw,
        forward_dtype=formats[0], backward_dtype=formats[1],
        accum_dtype='fp16' if half else 'fp32', accum_block_size=32 if half else 0)
    layer.weight.requires_grad_(need_w)
    x = torch.randn(65, 128, device='cuda', dtype=dtype, requires_grad=need_x)
    dy = torch.randn(65, 96, device='cuda', dtype=dtype)
    if scaling == 'delayed':
        layer.weight_quantizer.history.fill_(layer.weight.float().abs().max() * 2)
        layer.weight_quantizer.index = 1
    captured = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (captured.append(t), t)[1], lambda t: t):
        y = layer(x)
    # Floating storage never publishes FP8 W/W.T as module buffers or checkpoint payloads.
    assert layer.weight.dtype == {'bf16':torch.bfloat16, 'fp16':torch.float16}[storage]
    assert layer._initial_weight is None
    assert layer.weight_transpose.numel() == layer.weight_scale.numel() == 0
    assert captured[2] is layer.weight
    cached = captured[4]
    assert bool(cached.numel()) == (mode == 'fast' and need_x)
    assert not any(isinstance(t, torch.Tensor) and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                   and t.numel() for t in layer.state_dict().values())

    if layer._grouped_scaling:
        native = layer._native_scaling
        gx, gg = layer.input_group_shape, layer.grad_group_shape
        lw = layer._group_limit(gw, layer.fp8_dtype, weight=True)
        lx, lg = layer._group_limit(gx, layer.fp8_dtype), layer._group_limit(gg, layer.backward_fp8_dtype)
        qx = group_reference(x, layer.fp8_dtype, gx, native, lx)
        qw = group_reference(layer.weight, layer.fp8_dtype, gw, native, lw)
        qg = group_reference(dy, layer.backward_fp8_dtype, gg, native, lg)
        wt = group_reference(layer.weight, layer.fp8_dtype, grouped.transpose_group(gw), native, lw, True)
        xt = group_reference(x, layer.fp8_dtype, gx, native, lx, True)
        gt = group_reference(dy, layer.backward_fp8_dtype, gg, native, lg, True)
        expected = dict(y=qx @ qw.T + layer.bias.float(), dx=qg @ wt.T, dw=gt @ xt.T)
    else:
        rx = rotation.rotate_cpu(x.detach().cpu()).cuda() if rotate else x
        rw = rotation.rotate_cpu(layer.weight.detach().cpu()).cuda() if rotate else layer.weight
        qx = tensor_reference(rx, layer.fp8_dtype, captured[1], layer._quantization_limit(max(x.shape), layer.fp8_dtype))
        qw = tensor_reference(rw, layer.fp8_dtype, captured[3], layer._quantization_limit(max(layer.weight.shape), layer.fp8_dtype))
        limit = layer._quantization_limit(max(dy.shape), layer.backward_fp8_dtype)
        sg = dy.float().abs().max().clamp_min(1e-30) / (limit or torch.finfo(layer.backward_fp8_dtype).max)
        qg = tensor_reference(dy, layer.backward_fp8_dtype, sg, limit)
        expected = dict(y=qx @ qw.T + layer.bias.float(), dx=qg @ qw, dw=qg.T @ qx)
        if rotate:
            # Inverse rotation reads the GEMM result in its requested output dtype.
            for name in ('dx', 'dw'):
                out_dtype = x.dtype if name == 'dx' else layer.weight.grad_dtype
                expected[name] = rotation.rotate_cpu(expected[name].to(out_dtype).cpu()).cuda()
    y.backward(dy)
    actual = dict(y=y, dx=x.grad, dw=layer.weight.grad)
    errors = {name: error(value, expected[name].to(value.dtype)) for name, value in actual.items() if value is not None}
    assert max(errors.values()) < (.015 if half else .004), errors
    assert (x.grad is not None) == need_x and (layer.weight.grad is not None) == need_w
    torch.testing.assert_close(layer.bias.grad, dy.sum(0, dtype=layer.bias.grad_dtype), rtol=0, atol=0)
    with torch.no_grad():
        assert torch.isfinite(layer(x.detach())).all()
    return dict(weight_dtype=storage, activations=activations, scaling=scaling, mode=mode,
                saved_input=saved, rotation=rotate, formats=formats,
                weight_group=gw, fp16_accumulation=half, need_x=need_x, need_w=need_w, arithmetic_relative_l2=errors)


def check_resume(master_dtype, grad_dtype, optimizer_name, storage='bf16'):
    torch.manual_seed(231)
    viola.set_master_weight_dtype(master_dtype)
    viola.set_gradient_dtype(grad_dtype)
    options = dict(weight_dtype=storage, mode='fast', scaling='delayed', saved_input_dtype='bf16',
                   device='cuda', dtype=torch.bfloat16)

    def make():
        model = viola.nn.Linear(128, 96, **options)
        optimizer = (viola.optim.FP8Muon(model.parameters(), lr=.003) if optimizer_name == 'muon'
                     else torch.optim.AdamW(model.parameters(), lr=.003))
        return model, viola.optim.bind_optimizer(model, optimizer)

    model, optimizer = make()
    assert (model._master_ref() is model.weight) == (master_dtype == storage)
    batches = [torch.randn(32, 128, device='cuda', dtype=torch.bfloat16) * scale for scale in (1., 3.)]

    def step(model, optimizer):
        optimizer.zero_grad()
        # Keep two forwards live: backward must use each call's saved scale/cache.
        losses = [model(x).float().square().mean() / len(batches) for x in batches]
        for loss in reversed(losses):
            loss.backward()
        assert model.weight.grad.dtype == getattr(torch, {'bf16':'bfloat16', 'fp32':'float32'}[grad_dtype])
        previous = model.weight.detach().clone()
        index = model.weight_quantizer.index
        optimizer.step()
        assert model.weight_quantizer.index == index
        assert not torch.equal(model.weight, previous)
        assert model.weight_transpose.numel() == 0
        return sum(loss.detach().item() for loss in losses)

    initial = step(model, optimizer)
    state = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    restored, opt2 = make()
    restored.load_state_dict(state['model'])
    opt2.load_state_dict(state['optimizer'])
    a, b = step(model, optimizer), step(restored, opt2)
    assert a == b
    torch.testing.assert_close(model.weight, restored.weight, rtol=0, atol=0)
    torch.testing.assert_close(model._master_ref(), restored._master_ref(), rtol=0, atol=0)
    torch.testing.assert_close(model.bias, restored.bias, rtol=0, atol=0)
    return dict(weight_dtype=storage, master=master_dtype, gradient=grad_dtype, optimizer=optimizer_name,
                initial_loss=initial, next_loss=a, exact_resume=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/floating-weight-storage.json'))
    parser.add_argument('--weight-dtypes', default='bf16,fp16')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = dict(gpu=torch.cuda.get_device_name(), numerics=[], resume=[], complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(kind, row):
        report[kind].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(kind=kind, **row)), flush=True)

    for storage in args.weight_dtypes.split(','):
        for scaling in ('current', 'delayed', 'grouped', 'mxfp8'):
            for mode in ('memory', 'fast'):
                save('numerics', check_math(scaling, mode, storage=storage))
        save('numerics', check_math('current', 'memory', storage=storage, saved='bf16', rotate=64))
        save('numerics', check_math('current', 'fast', storage=storage, formats=('e5m2', 'e4m3')))
        save('numerics', check_math('grouped', 'memory', storage=storage, gw=(1, 32), saved='bf16'))
        save('numerics', check_math('mxfp8', 'fast', storage=storage, gw=(1, 32), saved='bf16'))
        save('numerics', check_math('current', 'memory', storage=storage, half=True))
        save('numerics', check_math('grouped', 'fast', storage=storage, half=True))
        save('numerics', check_math('current', 'fast', storage=storage, need_x=False))
        save('numerics', check_math('grouped', 'memory', storage=storage, need_w=False))
        save('numerics', check_math('current', 'memory', storage=storage, activations='fp16', saved='fp16'))
        for master, grad, opt in (('fp32','bf16','adamw'), (storage,'fp32','adamw'), (storage,'bf16','muon')):
            save('resume', check_resume(master, grad, opt, storage))
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
