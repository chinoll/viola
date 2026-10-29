"""Compare standalone FP8/BF16/FP16 Muon compute with identical NS and gradients."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from momentum16 import Momentum16
from viola.optim import FP8Muon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/standalone-muon-benchmark.json")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(17)
    base = torch.randn(1024, 1024, device="cuda")
    gradient = torch.randn_like(base, dtype=torch.bfloat16)
    rows = []
    for dtype in ("fp8", "bf16", "fp16"):
        param = torch.nn.Parameter(base.clone())
        param.grad_dtype = torch.bfloat16
        param.grad = gradient
        optimizer = (FP8Muon([param]) if dtype == "fp8" else
                     Momentum16([param], momentum_dtype=getattr(torch, {"bf16": "bfloat16", "fp16": "float16"}[dtype])))
        for _ in range(3):
            optimizer.step()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.steps):
            optimizer.step()
        end.record(); end.synchronize()
        state_bytes = sum(t.nbytes for s in optimizer.state.values() for t in s.values() if isinstance(t, torch.Tensor))
        rows.append(dict(momentum=dtype, ms=begin.elapsed_time(end) / args.steps, momentum_state_bytes=state_bytes))
    result = dict(gpu=torch.cuda.get_device_name(), scope="optimizer only, no model forward/backward", results=rows)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
