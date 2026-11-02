"""Check grouped/MXFP8 encoding, GEMMs, autograd and optimizer writeback."""

import argparse
import gc
import json
from pathlib import Path
import time

import torch

import viola
from viola.nn import _grouped_kernels as g


def error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def reference(x, dtype, group, native, limit=None, transposed=False):
    q, s = g.quantize(x.detach().cpu(), dtype, group, native=native, limit=limit, transposed=transposed)
    return g.dequantize(q, s, group).cuda()


def check_kernel(native, ga, gb, formats, shape, half=False):
    m, k, n = shape
    dtype_a, dtype_b = formats
    a = (torch.randn(m, k, device='cuda') * torch.exp2(torch.randint(-5, 6, (m, 1), device='cuda'))).bfloat16()
    b = (torch.randn(n, k, device='cuda') * torch.exp2(torch.randint(-5, 6, (n, 1), device='cuda'))).bfloat16()
    limit = 8. if half else None
    qa, sa = g.quantize(a, dtype_a, ga, native=native, limit=limit)
    qb, sb = g.quantize(b, dtype_b, gb, native=native, limit=limit)
    da, db = g.dequantize(qa, sa, ga), g.dequantize(qb, sb, gb)
    encoding = [error(da, reference(a, dtype_a, ga, native, limit)),
                error(db, reference(b, dtype_b, gb, native, limit))]
    assert max(encoding) < 1e-5, encoding
    bias = torch.randn(n, device='cuda')
    actual = g.gemm(qa, qb, sa, sb, ga, gb, out_dtype=torch.float32, native=native,
                    accum_dtype='fp16' if half else 'fp32', bias=bias)
    expected = da @ db.T + bias
    arithmetic = error(actual, expected)
    assert torch.isfinite(actual).all()
    assert arithmetic < (.01 if half else 2e-5), arithmetic
    for _ in range(2):
        repeated = g.gemm(qa, qb, sa, sb, ga, gb, out_dtype=torch.float32, native=native,
                          accum_dtype='fp16' if half else 'fp32', bias=bias)
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    return dict(native=native, group_a=ga, group_b=gb, formats=[str(d) for d in formats],
                shape=shape, fp16_accumulation=half, encoding_relative_l2=encoding,
                arithmetic_relative_l2=arithmetic, versus_bf16=error(actual, (a @ b.T).float() + bias))


def check_layer(scaling, mode, gw, gx, gg, saved='fp8', half=False):
    source = torch.nn.Linear(128, 96, device='cuda', dtype=torch.bfloat16)
    source.weight.data[:, ::17] *= 16
    layer = viola.nn.Linear.from_float(source, scaling=scaling, mode=mode,
        weight_group_shape=gw, input_group_shape=gx, grad_group_shape=gg,
        saved_input_dtype=saved, accum_dtype='fp16' if half else 'fp32',
        accum_block_size=32 if half else 0)
    x = torch.randn(65, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(65, 96, device='cuda', dtype=torch.bfloat16)
    native = scaling == 'mxfp8'
    y = layer(x)
    y.backward(dy)
    ref_x = reference(x, layer.fp8_dtype, gx, native, layer._group_limit(gx, layer.fp8_dtype))
    ref_w = layer.dequantize_weight()
    ref_g = reference(dy, layer.backward_fp8_dtype, gg, native, layer._group_limit(gg, layer.backward_fp8_dtype))
    if gw[0] >= 32:
        wt = ref_w.T
    else:
        wt = reference(ref_w, layer.fp8_dtype, gw, native, layer._group_limit(gw, layer.fp8_dtype, weight=True), True)
    xt = reference(x, layer.fp8_dtype, gx, native, layer._group_limit(gx, layer.fp8_dtype), True)
    gt = reference(dy, layer.backward_fp8_dtype, gg, native, layer._group_limit(gg, layer.backward_fp8_dtype), True)
    expected = dict(y=(ref_x @ ref_w.T + source.bias.float()).bfloat16(),
                    dx=(ref_g @ wt.T).bfloat16(), dw=(gt @ xt.T).bfloat16(), db=dy.sum(0))
    actual = dict(y=y, dx=x.grad, dw=layer.weight.grad, db=layer.bias.grad)
    arithmetic = {key: error(actual[key], expected[key]) for key in actual}
    assert max(arithmetic.values()) < (.012 if half else .003), arithmetic
    baseline = dict(y=source(x), dx=dy @ source.weight, dw=dy.T @ x, db=dy.sum(0))
    numerical = {key: error(actual[key], baseline[key]) for key in actual}
    layer.zero_grad(); x.grad = None
    # Load reconstructs the same transposed FP8 cache and preserves scale dtype.
    restored = viola.nn.Linear(128, 96, device='cuda', dtype=torch.bfloat16, scaling=scaling, mode=mode,
        weight_group_shape=gw, input_group_shape=gx, grad_group_shape=gg,
        saved_input_dtype=saved, accum_dtype='fp16' if half else 'fp32', accum_block_size=32 if half else 0)
    restored.load_state_dict(layer.state_dict())
    restored.bfloat16()
    torch.testing.assert_close(restored(x), y, rtol=0, atol=0)
    optimizer = viola.optim.bind_optimizer(layer, torch.optim.SGD(layer.parameters(), lr=.001))
    before = layer.weight.detach().clone()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(); x.grad = None
        loss = layer(x).float().square().mean()
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    assert torch.isfinite(layer.dequantize_weight()).all()
    assert not torch.equal(before.view(torch.uint8), layer.weight.view(torch.uint8))
    row = dict(scaling=scaling, mode=mode, weight_group=gw, input_group=gx, grad_group=gg,
               saved_input=saved, fp16_accumulation=half, arithmetic_relative_l2=arithmetic,
               versus_bf16=numerical, optimizer_losses=losses)
    del layer, restored, optimizer, source, x, dy, actual, baseline, y
    gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2041)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    viola.set_master_weight_dtype('fp32'); viola.set_gradient_dtype('bf16')
    report = dict(gpu=torch.cuda.get_device_name(), kernels=[], layers=[], complete=False)
    started = time.monotonic()
    def save(kind, row):
        report[kind].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(kind=kind, **row)), flush=True)
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    for native in (False, True):
        for ga, gb, formats, shape in [((1,32),(32,32),(e4,e4),(65,96,128)),
                                       ((1,64),(32,64),(e5,e4),(96,65,64)),
                                       ((32,64),(64,32),(e4,e5),(64,160,96)),
                                       ((1,32),(32,32),(e5,e4),(65,4096,128))]:
            save('kernels', check_kernel(native,ga,gb,formats,shape))
    save('kernels',check_kernel(False,(1,32),(32,32),(e5,e4),(65,96,128),True))
    for scaling in ('grouped','mxfp8'):
        save('layers', check_layer(scaling,'memory',(32,32),(1,32),(1,32)))
        save('layers', check_layer(scaling,'fast',(32,64),(32,32),(1,64),saved='bf16'))
        save('layers', check_layer(scaling,'fast',(1,32),(1,32),(1,32)))
    save('layers',check_layer('grouped','memory',(32,32),(1,32),(1,32),half=True))
    report.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
