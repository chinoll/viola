"""Hadamard-64: training cost first, then BF16 references with outliers."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from torch import nn

import viola
from viola.nn import _rotation_kernels as rotation
from benchmark_linear import measure


def errors(actual, reference):
    actual, reference = actual.float(), reference.float()
    difference = actual - reference
    per_row = difference.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-30)
    quantiles = torch.quantile(per_row, torch.tensor([.5, .95, .99], device=per_row.device))
    return dict(relative_l2=(difference.norm() / reference.norm().clamp_min(1e-30)).item(),
                max_abs=difference.abs().max().item(),
                max_abs_over_ref_max=(difference.abs().max() / reference.abs().max().clamp_min(1e-30)).item(),
                rms_abs=difference.square().mean().sqrt().item(),
                cosine=torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item(),
                row_relative_l2_p50=quantiles[0].item(), row_relative_l2_p95=quantiles[1].item(),
                row_relative_l2_p99=quantiles[2].item(), row_relative_l2_max=per_row.max().item(),
                nonfinite=(~actual.isfinite()).sum().item())


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")


def make_layer(source, rotated, block=1024, autotune=False):
    layer = viola.nn.Linear.from_float(
        source, mode="memory", rotation_size=64 if rotated else 0,
        autotune=autotune, accum_dtype="fp16" if block else "fp32",
        accum_block_size=block, accum_reduce_dtype="fp32")
    optimizer = viola.optim.bind_optimizer(layer, torch.optim.SGD(layer.parameters(), lr=1e-4))
    return layer, optimizer


def speed_case(rotated, tokens, width, expansion, repeats):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(41)
    first = nn.Linear(width, width * expansion, bias=False, device="cuda", dtype=torch.bfloat16)
    second = nn.Linear(width * expansion, width, bias=False, device="cuda", dtype=torch.bfloat16)
    a, oa = make_layer(first, rotated, autotune=True)
    b, ob = make_layer(second, rotated, autotune=True)
    del first, second
    x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)

    def clear():
        oa.zero_grad(set_to_none=True)
        ob.zero_grad(set_to_none=True)
        x.grad = None

    def train():
        clear()
        b(torch.nn.functional.silu(a(x))).backward(dy)

    def refresh():
        a.refresh_weight(a._master_ref())
        b.refresh_weight(b._master_ref())

    def with_refresh():
        train()
        refresh()

    # All compilation and the existing per-shape GEMM autotune precede timing.
    for _ in range(3):
        train()
    clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    clear()
    row = dict(rotation=64 if rotated else 0, tokens=tokens, width=width,
               expansion=expansion, peak_mib=peak)
    for name, function in (("fwd_bwd", train), ("weight_refresh", refresh),
                           ("fwd_bwd_refresh", with_refresh)):
        row[name] = measure(function, graph=True, iterations=repeats)
        clear()
    return row


def weight_pattern(base, kind, amplitude):
    w = base.clone()
    if kind == "spikes":
        # Fixed positions; 0.1% of weights have enlarged magnitude.
        w.flatten()[::1000] *= amplitude
    elif kind == "channels":
        w[:, ::256] *= amplitude
    elif kind == "rows":
        w[::256] *= amplitude
    elif kind == "patch":
        w[:64, :64] *= amplitude
    elif kind == "row_offset":
        w[::256] += amplitude / math.sqrt(w.shape[1])
    return w


def input_pattern(base, kind):
    x = base.clone()
    if kind.startswith("global_"):
        x *= float(kind.removeprefix("global_"))
    elif kind == "token_scales":
        x *= torch.logspace(-2, 2, x.shape[0], device=x.device).to(x.dtype)[:, None]
    elif kind == "channel_spikes":
        x[:, ::256] *= 64
    elif kind == "mixed":
        x *= torch.logspace(-2, 2, x.shape[0], device=x.device).to(x.dtype)[:, None]
        x[:, ::256] *= 64
    return x.requires_grad_(True)


def numerical_case(weight, x, dy, rotated, block, expected):
    source = nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                       device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(weight)
    layer, optimizer = make_layer(source, rotated, block=block)
    del source
    x.grad = None
    actual = layer(x)
    actual.backward(dy)
    with torch.no_grad():
        return dict(rotation=64 if rotated else 0, accum_block=block,
                    y=errors(actual, expected[0]), dx=errors(x.grad, expected[1]),
                    dw=errors(layer.weight.grad, expected[2]),
                    weight=errors(layer.dequantize_weight(), weight))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("speed", "numerics"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--blocks", default="1024,0")
    parser.add_argument("--configs", type=Path)
    args = parser.parse_args()
    configure()
    if args.configs:
        # Reuse the same previously autotuned GEMM tiles on both sides. The
        # GEMM kernels themselves are unchanged by this rotation experiment.
        from viola.nn import _linear_tuning as tuning
        configs = {(r["m"], r["n"], r["k"], r["a"], r["b"]): r["config"]
                   for r in json.loads(args.configs.read_text())["blocked"]}
        tuning.get_config = lambda m, n, k, at, bt, *a, **kw: dict(configs[(m, n, k, at, bt)])
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, phase=args.phase,
                  reference="BF16 torch.nn.Linear, BF16 input/output/dY/dX/dW; reduced-precision reduction disabled",
                  precision="E4M3 forward, E5M2 backward, FP32 master, BF16 gradients",
                  rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(row):
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    # Small algebra check before performance measurement: same transform and
    # inverse, independent of FP8 rounding and the BF16 numerical reference.
    torch.manual_seed(41)
    v = torch.randn(35, 192, device="cuda")
    h4 = (torch.ones(4, 4, device="cuda") - 2 * torch.eye(4, device="cuda").flip(1)) / 2
    h64 = torch.kron(torch.kron(h4, h4), h4)
    expected = (v.reshape(-1, 64) @ h64).reshape(v.shape)
    actual = rotation.inverse_(v.clone())
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(rotation.inverse_(actual), v, atol=2e-6, rtol=2e-6)

    if args.phase == "speed":
        for rotated in (False, True, True, False):
            save(speed_case(rotated, args.tokens, args.width, args.expansion, args.repeats))
    else:
        torch.manual_seed(42)
        m, k, n = 2048, 2048, 4096
        base_weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
        base_x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        patterns = [("normal", 1)] + [(kind, amplitude) for kind in ("spikes", "channels")
                                     for amplitude in (16, 64, 256)]
        patterns += [("rows", 64), ("patch", 64), ("row_offset", 64)]
        for kind, amplitude in patterns:
            weight = weight_pattern(base_weight, kind, amplitude)
            for input_kind in ("global_0.01", "global_1", "global_100", "token_scales", "channel_spikes", "mixed"):
                x = input_pattern(base_x, input_kind)
                ref_weight = weight.detach().requires_grad_(True)
                y = torch.nn.functional.linear(x, ref_weight)
                y.backward(dy)
                expected = (y.detach(), x.grad.detach(), ref_weight.grad.detach())
                for block in map(int, args.blocks.split(",")):
                    for rotated in (False, True):
                        row = numerical_case(weight, x, dy, rotated, block, expected)
                        row.update(weight_pattern=kind, outlier_multiplier=amplitude, input_pattern=input_kind,
                                   weight_absmax=weight.abs().max().item(), input_absmax=x.abs().max().item())
                        save(row)
                del x, y, ref_weight, expected
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
