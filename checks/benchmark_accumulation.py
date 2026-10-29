"""Compare accumulator strategies with identical GEMM tiles, before tuning."""

import argparse
import gc
import json
from pathlib import Path

import torch
import viola
from viola.nn import _linear_tuning

from benchmark_mlp import run_case
from linear_accumulation import OPTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--variants", default="fp32,block128_bf16,block128_fp32,block512_bf16,block512_fp32,fp16,bf16")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--output", default="results/accum-speed.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    tile = dict(bm=128, bn=128, bk=128, threads=256, stages=3)
    # A controlled timing experiment: no search, identical tiles for every
    # accumulator. This replacement is local to this benchmark process.
    if not args.autotune:
        _linear_tuning.get_config = lambda *args: tile.copy()
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  scope="16384-token MLP, BF16 IO/gradients, FP32 master, no optimizer arithmetic",
                  selection="autotuned" if args.autotune else "fixed matched tiles; not autotuned",
                  tile=None if args.autotune else tile, results=[])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in args.variants.split(","):
        print("START " + name, flush=True)
        row = run_case(args.tokens, args.width, 4, "bf16" if name == "bf16" else "memory",
                       3, True, OPTIONS.get(name, {}))
        row["name"] = name
        report["results"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("RESULT " + json.dumps(row), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
