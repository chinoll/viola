"""Pre-tune all FP8 forward/dX/dW GEMMs for a 4x-expansion MLP."""

import argparse
import json
from pathlib import Path
import time

import torch

from viola.nn._linear_tuning import get_config, candidate_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--accum-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--accum-block-size", type=int, default=0)
    parser.add_argument("--accum-reduce-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--output", default="results/linear-autotune.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    h, expanded, tokens = args.hidden, args.hidden * args.expansion, args.tokens
    shapes = []
    for k, n in ((h, expanded), (expanded, h)):
        shapes.extend([(tokens, n, k, "float8_e4m3", "float8_e4m3"),
                       (tokens, k, n, "float8_e5m2", "float8_e4m3"),
                       (n, k, tokens, "float8_e5m2", "float8_e4m3")])
    shapes = list(dict.fromkeys(shapes))
    shapes.sort(key=lambda shape: (shape[3] == shape[4], shape[0] >= shape[1]))
    report = dict(gpu=torch.cuda.get_device_name(), results=[])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, (m, n, k, a, b) in enumerate(shapes):
        print(f"tune {index+1}/{len(shapes)} M={m} N={n} K={k} {a} x {b}", flush=True)
        begin = time.perf_counter()
        config = get_config(m, n, k, a, b, "bfloat16", "bfloat16", False, torch.cuda.current_device(),
                            {"fp32": "float32", "fp16": "float16"}[args.accum_dtype], args.accum_block_size,
                            {"fp32": "float32", "bf16": "bfloat16"}[args.accum_reduce_dtype])
        candidates = candidate_configs(torch.cuda.get_device_properties(0),
                                       {"fp32": "float32", "fp16": "float16"}[args.accum_dtype],
                                       args.accum_block_size, a != b,
                                       {"fp32": "float32", "bf16": "bfloat16"}[args.accum_reduce_dtype])
        row = dict(m=m, n=n, k=k, a=a, b=b, candidates=len(candidates), config=config,
                   elapsed_s=time.perf_counter()-begin)
        report["results"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
