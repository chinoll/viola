"""Numerical, accumulation, checkpoint and storage checks for TileLang Linear."""

import copy
import gc
import json
from pathlib import Path

import torch
import viola
from viola.nn._linear_kernels import _gemm_kernel, _mixed_gemm_kernel


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)).item()


def quantized(x, dtype, scale=None):
    limit = torch.finfo(dtype).max
    scale = x.float().abs().max().clamp_min(1e-30) / limit if scale is None else scale
    return (x.float() / scale).clamp(-limit, limit).to(dtype).float() * scale


def numerical(fmt, backward, mode, shape, gradient):
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype(gradient)
    m, k, n = shape
    torch.manual_seed(127)
    layer = viola.nn.Linear(k, n, mode=mode, forward_dtype=fmt, backward_dtype=backward, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    y = layer(x)
    y.backward(dy)
    qx, qw, qdy = quantized(x, layer.fp8_dtype), layer.weight.float() * layer.weight_scale, quantized(dy, layer.backward_fp8_dtype)
    refs = {"y": qx @ qw.T + layer.bias.float(), "dx": qdy @ qw, "dw": qdy.T @ qx}
    actual = {"y": y, "dx": x.grad, "dw": layer.weight.grad}
    errors = {key: relative_error(actual[key], refs[key]) for key in actual}
    for key in actual:
        assert errors[key] < (0.003 if actual[key].dtype == torch.bfloat16 else 2e-5), errors
    torch.testing.assert_close(layer.bias.grad, dy.sum(0, dtype=layer.bias.grad_dtype))
    reference_fp32 = x.float() @ layer._initial_weight.T + layer.bias.float()
    errors["y_vs_unquantized_fp32"] = relative_error(y, reference_fp32)
    errors["dx_vs_unquantized_fp32"] = relative_error(x.grad, dy.float() @ layer._initial_weight)
    errors["dw_vs_unquantized_fp32"] = relative_error(layer.weight.grad, dy.float().T @ x.float())
    return dict(forward=fmt, backward=backward, mode=mode, shape=shape, gradient=gradient, errors=errors)


def update_and_resume(fmt, mode, scaling, gradient, master_dtype="fp32"):
    viola.set_master_weight_dtype(master_dtype)
    viola.set_gradient_dtype(gradient)
    layer = viola.nn.Linear(256, 192, mode=mode, forward_dtype=fmt, scaling=scaling,
                         device="cuda", dtype=torch.bfloat16)
    optimizer = viola.optim.bind_optimizer(layer, torch.optim.AdamW(layer.parameters(), lr=.002))
    assert layer._initial_weight is None
    x = [torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    def step(model, opt):
        opt.zero_grad()
        for batch in x:
            model(batch).float().square().mean().div(len(x)).backward()
        before = model.weight.detach().view(torch.uint8).clone()
        opt.step()
        assert not torch.equal(before, model.weight.view(torch.uint8))
        torch.testing.assert_close(model.weight.float() * model.weight_scale,
                                   quantized(model._master_ref(), model.fp8_dtype, model.weight_scale),
                                   rtol=0, atol=0)
        assert model.weight.grad.dtype == {"fp32": torch.float32, "bf16": torch.bfloat16}[gradient]
        if model.cache_weight_transpose:
            assert torch.equal(model.weight.T.contiguous().view(torch.uint8), model.weight_transpose.view(torch.uint8))
        else:
            assert model.weight_transpose.numel() == 0

    step(layer, optimizer)
    state = copy.deepcopy({"model": layer.state_dict(), "optimizer": optimizer.state_dict()})
    restored = viola.nn.Linear(256, 192, mode=mode, forward_dtype=fmt, scaling=scaling,
                            device="cuda", dtype=torch.bfloat16)
    restored.load_state_dict(state["model"])
    opt2 = viola.optim.bind_optimizer(restored, torch.optim.AdamW(restored.parameters(), lr=.002))
    opt2.load_state_dict(state["optimizer"])
    step(layer, optimizer)
    step(restored, opt2)
    assert torch.equal(layer.weight.view(torch.uint8), restored.weight.view(torch.uint8))
    torch.testing.assert_close(layer.bias, restored.bias, rtol=0, atol=0)
    for p, q in zip(optimizer.param_groups[0]["params"], opt2.param_groups[0]["params"]):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=False)
    assert layer.weight.grad.count_nonzero() == 0
    optimizer.zero_grad()
    assert layer.weight.grad is None
    return dict(format=fmt, mode=mode, scaling=scaling, gradient=gradient, master=master_dtype,
                three_microbatches=True, exact_resume=True)


def presets_and_storage():
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    source = torch.nn.Linear(512, 512, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    outputs, grads = [], []
    rows = []
    for mode in ("memory", "speed"):
        layer = viola.nn.Linear.from_float(source, mode=mode)
        opt = viola.optim.bind_optimizer(layer, torch.optim.AdamW(layer.parameters(), lr=.001))
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t), t)[1], lambda t: t):
            y = layer(x)
        outputs.append(y.detach().clone())
        y.float().square().mean().backward()
        grads.append(layer.weight.grad.clone())
        rows.append(dict(mode=mode,
                         weight_bytes=layer.weight.nbytes,
                         transpose_bytes=layer.weight_transpose.nbytes,
                         saved_input_bytes=saved[0].nbytes,
                         gradient_bytes=layer.weight.grad.nbytes,
                         master_bytes=layer._master_ref().nbytes))
        assert saved[0].dtype == torch.float8_e4m3fn
        assert layer.weight.grad.nbytes == 2 * layer.weight.numel()
        del saved, y, layer, opt
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)
    # CPU construction, dtype/device moves, frozen weights, saved-input override.
    layer = viola.nn.Linear(512, 512, saved_input_dtype="bf16", cache_weight_transpose=True).bfloat16().cuda()
    assert layer.weight.dtype == torch.float8_e4m3fn and layer.weight_scale.dtype == torch.float32
    layer(x).sum().backward()
    layer.weight.requires_grad_(False)
    layer(x).sum().backward()
    return rows


def training(fmt, steps=100):
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    torch.manual_seed(777)
    ref = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    # Match inputs/targets as well as initialization across precision variants.
    data = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * .3
    model = viola.nn.Linear.from_float(ref, forward_dtype=fmt) if fmt else ref
    opt = viola.optim.bind_optimizer(model, torch.optim.AdamW(model.parameters(), lr=.005))
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = model(data).float().sub(target.float()).square().mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * .15, losses[-1]
    return dict(format=fmt or "bf16", steps=steps, initial_loss=losses[0], final_loss=losses[-1], loss=losses)


def peak_memory(mode):
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    model = (torch.nn.Linear(1024, 4096, device="cuda", dtype=torch.bfloat16) if mode == "bf16"
             else viola.nn.Linear(1024, 4096, mode=mode, device="cuda", dtype=torch.bfloat16))
    optimizer = viola.optim.bind_optimizer(model, torch.optim.SGD(model.parameters(), lr=.001, foreach=False))
    x = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Compile/warm first; the second pass measures tensor allocation, not compilation.
    for i in range(2):
        optimizer.zero_grad()
        x.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        y = model(x)
        y.backward(torch.ones_like(y))
        optimizer.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        del y
    return dict(mode=mode, shape=[512, 1024, 4096], peak_allocated_mib=peak / 2**20)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(47)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    report["numerics"] = [numerical(fmt, backward, mode, (37, 80, 48) if mode == "memory" else (128, 256, 192), grad)
                           for fmt in ("e4m3", "e5m2") for backward in ("e4m3", "e5m2")
                           for mode, grad in (("memory", "fp32"), ("speed", "bf16"))]
    print("numerics passed", flush=True)
    report["update_resume"] = [update_and_resume(fmt, mode, scaling, grad)
                                for fmt in ("e4m3", "e5m2") for mode, scaling, grad in
                                (("memory", "current", "bf16"), ("speed", "delayed", "fp32"))]
    report["update_resume"].append(update_and_resume("e4m3", "speed", "current", "fp32", "bf16"))
    report["storage"] = presets_and_storage()
    print("update, checkpoint and storage checks passed", flush=True)
    report["training"] = [training(fmt) for fmt in (None, "e4m3", "e5m2")]
    report["peak_memory"] = [peak_memory(mode) for mode in ("bf16", "memory", "speed")]
    report["ptx"] = {}
    for fmt in ("e4m3", "e5m2"):
        kernel = _gemm_kernel(128, 192, 256, "float8_" + fmt, "float8_" + fmt, "bfloat16", "bfloat16", True)
        instructions = sorted(set(line.strip().split("{")[0].strip() for line in kernel._get_ptx().splitlines() if "mma.sync" in line))
        assert any(fmt in line for line in instructions)
        report["ptx"][fmt] = instructions
    kernel = _mixed_gemm_kernel(128, 192, 256, "float8_e5m2", "float8_e4m3", "float32", "float32", False)
    report["ptx"]["mixed_backward"] = sorted(set(line.strip().split("{")[0].strip() for line in kernel._get_ptx().splitlines() if "mma.sync" in line))
    assert any("e5m2.e4m3" in line for line in report["ptx"]["mixed_backward"])
    path = Path("results/linear-5060ti.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training"}, indent=2))
    print("training:", [{k: v for k, v in row.items() if k != "loss"} for row in report["training"]])
    print(path)


if __name__ == "__main__":
    main()
