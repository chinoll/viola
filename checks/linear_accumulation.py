"""FP16 MMA accumulation: isolated rounding error and matched MLP training."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from torch import nn

import viola
from viola.nn import _linear_kernels as kernels


OPTIONS = {
    "fp32": {},
    "fp32_limit16": {"quantization_limit": 16.0},
    "fp16": {"accum_dtype": "fp16"},
    "block128_fp32": {"accum_dtype": "fp16", "accum_block_size": 128, "accum_reduce_dtype": "fp32"},
    "block128_bf16": {"accum_dtype": "fp16", "accum_block_size": 128, "accum_reduce_dtype": "bf16"},
    "block512_fp32": {"accum_dtype": "fp16", "accum_block_size": 512, "accum_reduce_dtype": "fp32"},
    "block512_bf16": {"accum_dtype": "fp16", "accum_block_size": 512, "accum_reduce_dtype": "bf16"},
}


def metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    finite = actual.isfinite()
    result = dict(finite_fraction=finite.float().mean().item(),
                  nan_count=actual.isnan().sum().item(), inf_count=actual.isinf().sum().item())
    if bool(finite.all()):
        difference = actual - reference
        result.update(relative_l2=(difference.norm() / reference.norm().clamp_min(1e-20)).item(),
                      max_abs=difference.abs().max().item(),
                      rms_abs=difference.square().mean().sqrt().item())
    return result


def gemm_errors(names):
    rows = []
    for reduction in (256, 4096, 16384):
        for distribution in ("normal", "positive"):
            torch.manual_seed(41)
            a = torch.randn(128, reduction, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(128, reduction, device="cuda", dtype=torch.bfloat16)
            if distribution == "positive":
                a, b = a.abs() + .25, b.abs() + .25
            unquantized = a.float() @ b.float().T
            for left_format in (torch.float8_e4m3fn, torch.float8_e5m2):
                for name in names:
                    options = OPTIONS[name]
                    block = options.get("accum_block_size", 0)
                    limit = options.get("quantization_limit")
                    if options.get("accum_dtype") == "fp16":
                        limit = 2.0 ** math.floor(math.log2(math.sqrt(32768 / (block or reduction))))
                    sa = kernels.tensor_scale(a, fp8_dtype=left_format, quantization_limit=limit)
                    sb = kernels.tensor_scale(b, quantization_limit=limit)
                    qa, _ = kernels.quantize(a, sa, fp8_dtype=left_format, quantization_limit=limit)
                    qb, _ = kernels.quantize(b, sb, quantization_limit=limit)
                    gemm_options = {k: v for k, v in options.items() if k != "quantization_limit"}
                    output = kernels.gemm(qa, qb, sa, sb, out_dtype=torch.bfloat16, **gemm_options)
                    reference = kernels.gemm(qa, qb, sa, sb, out_dtype=torch.bfloat16)
                    row = dict(name=name, reduction=reduction, distribution=distribution,
                               left_format=str(left_format), quantization_limit=limit,
                               accumulation_error=metrics(output, reference),
                               total_error=metrics(output, unquantized))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows


def linear_errors(names):
    rows = []
    for tokens in (256, 4096, 16384):
        torch.manual_seed(27)
        source = nn.Linear(512, 512, bias=False, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        reference = dict(y=x.float() @ source.weight.float().T,
                         dx=dy.float() @ source.weight.float(), dw=dy.float().T @ x.float())
        for name in names:
            layer = viola.nn.Linear.from_float(source, **OPTIONS[name])
            batch = x.detach().requires_grad_(True)
            output = layer(batch)
            output.backward(dy)
            actual = dict(y=output, dx=batch.grad, dw=layer.weight.grad)
            row = dict(name=name, tokens=tokens, errors={key: metrics(actual[key], reference[key]) for key in actual})
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def train(names, steps):
    # A finite synthetic regression task, not a generative-quality evaluation.
    torch.manual_seed(812)
    teacher = nn.Sequential(nn.Linear(256, 1024, bias=False, device="cuda", dtype=torch.bfloat16),
                            nn.SiLU(), nn.Linear(1024, 256, bias=False, device="cuda", dtype=torch.bfloat16))
    data = torch.randn(8192, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        targets = teacher(data)
    del teacher
    rows, baseline_weights = [], None
    for name in names:
        torch.manual_seed(777)
        model = nn.Sequential(nn.Linear(256, 1024, bias=False, device="cuda", dtype=torch.bfloat16),
                              nn.SiLU(), nn.Linear(1024, 256, bias=False, device="cuda", dtype=torch.bfloat16))
        if name != "bf16":
            for index in (0, 2):
                model[index] = viola.nn.Linear.from_float(model[index], **OPTIONS[name])
        optimizer = viola.optim.bind_optimizer(model, torch.optim.AdamW(model.parameters(), lr=1e-3))
        generator = torch.Generator(device="cuda").manual_seed(20261029)
        losses, norms, failed_step = [], [], None
        for step in range(steps):
            tokens = (256, 1024, 4096)[(step // 32) % 3]
            indices = torch.randint(data.shape[0], (tokens,), device="cuda", generator=generator)
            optimizer.zero_grad()
            prediction = model(data[indices])
            loss = (prediction.float() - targets[indices].float()).square().mean()
            loss.backward()
            norm = sum(p.grad.float().square().sum() for p in model.parameters()).sqrt()
            losses.append(loss.item()); norms.append(norm.item())
            if not math.isfinite(losses[-1]) or not math.isfinite(norms[-1]):
                failed_step = step + 1
                break
            optimizer.step()
            if (step + 1) % 100 == 0:
                print(json.dumps(dict(name=name, step=step + 1, loss=losses[-1], grad_norm=norms[-1])), flush=True)
        with torch.no_grad():
            validation = (model(data[:1024]).float() - targets[:1024].float()).square().mean().item()
        weights = torch.cat([p.detach().float().flatten() for group in optimizer.param_groups for p in group["params"]])
        if name == "fp32":
            baseline_weights = weights.clone()
        row = dict(name=name, requested_steps=steps, completed_steps=len(losses), nonfinite_step=failed_step,
                   initial_loss=losses[0], final_loss=losses[-1], last100_mean=sum(losses[-100:])/len(losses[-100:]),
                   evaluation_loss=validation, grad_norm_max=max(norms), losses=losses, grad_norms=norms)
        if baseline_weights is not None:
            row["weights_vs_fp32_accum"] = metrics(weights, baseline_weights)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("losses", "grad_norms")}), flush=True)
        del model, optimizer, weights
        gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("numerics", "training"))
    parser.add_argument("--variants", default="fp32,fp32_limit16,fp16,block128_fp32,block128_bf16")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    names = args.variants.split(",")
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, task=args.task,
                  options={name: OPTIONS.get(name, {}) for name in names},
                  precision="FP32 master; BF16 inputs, outputs and parameter gradients; only accumulator/scaling vary")
    if args.task == "numerics":
        report["gemm"] = gemm_errors(names)
        report["linear"] = linear_errors(names)
    else:
        report["training"] = train(names, args.steps)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
