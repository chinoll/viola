"""Compare matched independent runs: loss curves and final FP32 master weights."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch


def read_json(path):
    return json.loads(path.read_text())


def losses(root):
    return {r["step"]: r for line in (root / "steps-rank0.jsonl").open() if (r := json.loads(line))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp8", type=Path, required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    left, right = read_json(args.fp8 / "settings.json"), read_json(args.bf16 / "settings.json")
    matched = ("seed", "batch", "micro_batch", "world_size", "epochs", "lr", "model", "dataset")
    for key in matched:
        assert left[key] == right[key], f"Training settings differ: {key}"
    assert left["source_sha256"]["experiments/dit.py"] == right["source_sha256"]["experiments/dit.py"]
    steps = read_json(args.fp8 / "complete.json")["steps"]
    assert steps == read_json(args.bf16 / "complete.json")["steps"]
    args.output.mkdir(parents=True, exist_ok=True)
    a, b = losses(args.fp8), losses(args.bf16)
    assert a.keys() == b.keys()
    fields = ("step", "fp8_loss", "bf16_loss", "loss_difference", "fp8_probe", "bf16_probe")
    with (args.output / "loss_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in sorted(a):
            writer.writerow(dict(step=step, fp8_loss=a[step]["loss"], bf16_loss=b[step]["loss"],
                loss_difference=a[step]["loss"] - b[step]["loss"],
                fp8_probe=a[step].get("fixed_probe_loss", ""), bf16_probe=b[step].get("fixed_probe_loss", "")))

    checkpoints = [torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
                   for root in (args.fp8, args.bf16)]
    weights = [dict(zip(c["master_names"], c["optimizer"]["viola_masters"]["weights"])) for c in checkpoints]
    names = checkpoints[0]["master_names"]
    assert names == checkpoints[1]["master_names"]
    totals = dict(error_squared_sum=0., reference_squared_sum=0., actual_squared_sum=0.,
                  dot=0., absolute_sum=0., max_abs=0., elements=0)
    rows = []
    for name in names:
        # Compare the actual saved masters, independently of model storage dtype.
        x, y = [state[name].contiguous().double() for state in weights]
        delta = x - y
        es, rs, xs = delta.square().sum().item(), y.square().sum().item(), x.square().sum().item()
        dot, absolute, maximum = (x * y).sum().item(), delta.abs().sum().item(), delta.abs().max().item()
        rows.append(dict(name=name, shape=list(x.shape), relative_l2=math.sqrt(es / rs) if rs else None,
                         cosine=dot / math.sqrt(xs * rs) if xs and rs else None,
                         max_abs=maximum, mean_abs=absolute / x.numel(), rmse=math.sqrt(es / x.numel())))
        for key, value in (("error_squared_sum", es), ("reference_squared_sum", rs),
                           ("actual_squared_sum", xs), ("dot", dot), ("absolute_sum", absolute)):
            totals[key] += value
        totals["max_abs"] = max(totals["max_abs"], maximum)
        totals["elements"] += x.numel()
        del x, y, delta
    summary = dict(steps=steps, reference="BF16 momentum training; FP32 master weights",
        note="Independent training trajectories: final weight difference includes accumulated trajectory divergence.",
        parameters=totals["elements"], relative_l2=math.sqrt(totals["error_squared_sum"] / totals["reference_squared_sum"]),
        cosine=totals["dot"] / math.sqrt(totals["actual_squared_sum"] * totals["reference_squared_sum"]),
        max_abs=totals["max_abs"], mean_abs=totals["absolute_sum"] / totals["elements"],
        rmse=math.sqrt(totals["error_squared_sum"] / totals["elements"]),
        final_100_loss={key:sum(r["loss"] for r in list(data.values())[-100:])/min(100,len(data))
                        for key,data in (("fp8",a),("bf16",b))})
    (args.output / "weight_errors.json").write_text(json.dumps(rows, indent=2))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
