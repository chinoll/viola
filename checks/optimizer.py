"""Standalone Muon arithmetic, state storage, checkpoint and FP8 Linear training."""
import copy
import json
from pathlib import Path
import sys

import torch
import viola
from viola.optim._kernels import dequantize, state_sizes
from viola.optim._orthogonalize import METHODS


class UpdateProbe:
    def __init__(self):
        self.values = {}

    def momentum(self, p, grad, old_q, old_scale, q, scale, matrix, group, loss_scale, quant):
        old = dequantize(old_q, old_scale, tuple(matrix.shape), **quant)
        fresh = group["momentum"] * old + (1 - group["momentum"]) * grad.flatten().float()
        reference = ((1 - group["momentum"]) * grad.flatten().float() + group["momentum"] * fresh
                     if group["nesterov"] else fresh)
        torch.testing.assert_close(matrix.flatten().float(), reference.bfloat16().float(), atol=2e-6, rtol=.01)
        error = (dequantize(q, scale, tuple(matrix.shape), **quant) - fresh).norm() / fresh.norm()
        assert error < .06, error

    def update(self, p, update, group, ns):
        self.values[id(p)] = update.float().reshape_as(p).clone()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def state_and_updates(granularity, rounding, method):
    params = [torch.nn.Parameter(torch.randn(*shape, device="cuda")) for shape in
              ((32, 64), (4, 4, 3, 3), (2, 16, 32), (32,))]
    groups = [dict(params=params[:1], use_muon=True), dict(params=params[1:2], use_muon=True, nesterov=False),
              dict(params=params[2:3], batched=True),
              dict(params=params[3:], use_muon=False)]
    opt = viola.optim.FP8Muon(groups, lr=.003, weight_decay=.07, tile_size=64,
                              granularity=granularity, rounding=rounding, rounding_seed=17,
                              ns_method=method, adam_state_dtype="bf16")
    probe = UpdateProbe()
    opt.observer = probe
    for step in range(3):
        before = [p.detach().clone() for p in params]
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
        for p, previous in zip(params[:3], before):
            shape = p.shape if p is params[2] else (p.shape[0], p.numel() // p.shape[0])
            scale = .2 * max(shape[-2:])**.5
            expected = previous * (1 - .003 * .07) - .003 * scale * probe.values[id(p)]
            torch.testing.assert_close(p, expected, rtol=2e-6, atol=5e-7)
            state = opt.state[p]
            qn, sn = state_sizes(shape, granularity=granularity, tile_size=64, block_shape=(16, 16))
            assert state["momentum_q"].dtype == torch.uint8 and state["momentum_q"].numel() == qn
            assert state["momentum_scale"].dtype == torch.float32 and state["momentum_scale"].numel() == sn
            assert state["step"] == step + 1
            assert set(state) == {"step", "momentum_q", "momentum_scale"}
    assert opt.state[params[3]]["exp_avg"].dtype == torch.bfloat16
    saved_weights, saved = [p.detach().clone() for p in params], copy.deepcopy(opt.state_dict())
    opt.observer = None
    opt.step()
    expected_weights, expected_state = [p.detach().clone() for p in params], copy.deepcopy(opt.state_dict())
    with torch.no_grad():
        for p, value in zip(params, saved_weights):
            p.copy_(value)
    opt.load_state_dict(saved)
    equal(opt.state_dict(), saved)
    opt.step()
    equal([p.detach() for p in params], expected_weights)
    equal(opt.state_dict(), expected_state)
    # Missing gradients do not apply decay or advance a parameter's state.
    before = params[0].detach().clone()
    step = opt.state[params[0]]["step"]
    params[0].grad = None
    opt.step()
    equal(params[0].detach(), before)
    assert opt.state[params[0]]["step"] == step
    return dict(granularity=granularity, rounding=rounding, ns_method=method, exact_resume=True)


def standalone_training():
    torch.manual_seed(9)
    model = torch.nn.Linear(64, 32, device="cuda")
    opt = viola.optim.FP8Muon(model.parameters(), lr=.01, tile_size=64)
    x = torch.randn(128, 64, device="cuda")
    losses = []
    for _ in range(100):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(x).float() - x[:, :32]).square().mean()
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * .1
    return dict(steps=100, initial_loss=losses[0], final_loss=losses[-1])


def linear_integration(master="fp32", gradient="bf16"):
    viola.set_master_weight_dtype(master, rounding="nearest" if master == "fp32" else "stochastic", seed=37)
    viola.set_gradient_dtype(gradient)
    model = viola.nn.Linear(64, 32, device="cuda", dtype=torch.bfloat16)
    opt = viola.optim.bind_optimizer(model, viola.optim.FP8Muon(model.parameters(), lr=.01, tile_size=64))
    x = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    def update():
        opt.zero_grad()
        for micro in x.chunk(2):
            (model(micro).float().square().mean() / 2).backward()
        opt.step()
    for _ in range(3):
        update()
    saved = copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict()))
    update()
    expected = copy.deepcopy(dict(model=model.state_dict(), optimizer=opt.state_dict()))
    model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    update()
    equal(model.state_dict(), expected["model"])
    equal(opt.state_dict(), expected["optimizer"])
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    assert model.weight.grad.dtype == dtypes[gradient]
    assert model._master_ref().dtype == dtypes[master]
    return dict(microbatches=2, exact_resume=True, master=master, gradient=gradient)


def nonfinite_policy():
    model = viola.nn.Linear(64, 32, device="cuda", scaling="delayed")
    opt = viola.optim.bind_optimizer(model, viola.optim.FP8Muon(model.parameters(), tile_size=64, skip_nonfinite=True))
    model.weight.grad = torch.full(model.weight.shape, float("nan"), device="cuda", dtype=model.weight.grad_dtype)
    before = copy.deepcopy(model.state_dict())
    opt.step()
    equal(model.state_dict(), before)
    assert opt.step_was_skipped and not opt.state
    opt.skip_nonfinite = False
    opt.step()
    assert not opt.step_was_skipped and torch.isnan(model._master_ref()).any()
    return dict(default_propagates_nan=True, explicit_skip_preserves_quantizer=True)


def main():
    torch.manual_seed(31)
    rows = [state_and_updates(quant, rounding, method) for quant, rounding, method in
            (("tile", "nearest", "standard"), ("block", "stochastic", "standard"),
             ("tile", "stochastic", "gram"))]
    result = dict(gpu=torch.cuda.get_device_name(), cases=rows,
                  training=standalone_training(),
                  linear_integration=[linear_integration(m, g) for m, g in
                                      (("fp32", "bf16"), ("bf16", "fp32"), ("fp16", "bf16"))],
                  nonfinite=nonfinite_policy())
    assert "deepspeed" not in sys.modules
    result["deepspeed_imported"] = False
    path = Path("results/standalone-optimizer.json"); path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
