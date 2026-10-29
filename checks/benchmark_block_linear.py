"""Blocked FP16 MLP timing and actual BF16 Linear forward/backward references."""

import argparse
import gc
import json
from pathlib import Path

import torch
from torch import nn

import viola
from viola.nn import _linear_tuning as tuning
from benchmark_mlp import run_case
from fp32_accuracy import errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/block-linear.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    native_rows = json.loads((args.tuning_results / "native-block-forward.json").read_text())["comparisons"]
    configs = {}
    for block in (128, 512):
        rows = json.loads((args.tuning_results / f"block{block}-autotune.json").read_text())["results"]
        assert len(rows) == 5, "Wait for the five MLP GEMM shapes to finish tuning"
        explicit = {(r["m"], r["n"], r["k"], r["a"], r["b"]): r["config"] for r in rows}
        native = dict(explicit)
        for row in native_rows:
            if row["block"] != block:
                continue
            m, n, k = (16384, 16384, 4096) if row["role"] == "up" else (16384, 4096, 16384)
            native[m, n, k, "float8_e4m3", "float8_e4m3"] = dict(row["best"], explicit_mma=False)
        configs[block] = dict(explicit=explicit, native=native)

    report = dict(gpu=torch.cuda.get_device_name(), timing=[], linear_accuracy=[])
    args.output.parent.mkdir(exist_ok=True, parents=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def use_configs(block, variant):
        def get_config(m, n, k, at, bt, *args, **kwargs):
            if variant == "fixed":
                return dict(bm=128, bn=128, bk=128, threads=256, stages=3, explicit_mma=True)
            return dict(configs[block][variant][m, n, k, at, bt])
        tuning.get_config = get_config

    for variant, block in (("fixed", 128), ("fixed", 512), ("explicit", 128),
                            ("native", 128), ("explicit", 512), ("native", 512)):
        use_configs(block, variant)
        row = run_case(16384, 4096, 4, "memory", 3, autotune=True,
                       linear_options=dict(accum_dtype="fp16", accum_block_size=block, accum_reduce_dtype="fp32"))
        row.update(variant=variant, block=block)
        report["timing"].append(row)
        save()
        print("MLP " + json.dumps(dict(variant=variant, block=block,
                                       forward_ms=row["forward_graph"]["median_ms"],
                                       fwd_bwd_ms=row["train_graph"]["median_ms"],
                                       peak_mib=row["train_peak_mib"])), flush=True)

    # Use nn.Linear itself, with the same weights, input and upstream gradient.
    # The optimized implementation is selected by the MLP timings above.
    for in_features, out_features in ((4096, 16384), (16384, 4096)):
        torch.manual_seed(41)
        source = nn.Linear(in_features, out_features, bias=False, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(16384, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(16384, out_features, device="cuda", dtype=torch.bfloat16)
        y = source(x)
        y.backward(dy)
        reference = dict(y=y.detach(), dx=x.grad.detach(), dw=source.weight.grad.detach())
        for block in (128, 512):
            rows = [r for r in report["timing"] if r["block"] == block and r["variant"] != "fixed"]
            variant = min(rows, key=lambda r: r["train_graph"]["median_ms"])["variant"]
            use_configs(block, variant)
            layer = viola.nn.Linear.from_float(source, mode="memory", autotune=True,
                                                accum_dtype="fp16", accum_block_size=block, accum_reduce_dtype="fp32")
            batch = x.detach().requires_grad_(True)
            output = layer(batch)
            output.backward(dy)
            with torch.no_grad():
                actual = dict(y=output, dx=batch.grad, dw=layer.weight.grad)
                row = dict(block=block, variant=variant, tokens=16384,
                           in_features=in_features, out_features=out_features,
                           versus_bf16_linear={key: errors(actual[key], reference[key]) for key in actual})
            report["linear_accuracy"].append(row)
            save()
            print("ACCURACY " + json.dumps(row), flush=True)
            del actual, output, layer, batch
        del source, x, dy, y, reference
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
