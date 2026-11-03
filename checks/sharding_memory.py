"""Measure independent sharding memory on the same eight-layer model."""

import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn

import viola


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2048, 2048, bias=False) for _ in range(8))

    def forward(self, x):
        for layer in self.layers:
            x = x + .1 * torch.nn.functional.silu(layer(x))
        return x


def main():
    parser = argparse.ArgumentParser()
    for flag in ("parameters", "gradients", "optimizer", "ema"):
        parser.add_argument("--shard-" + flag, action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--communication", choices=("fp32", "bf16", "fp8"), default="fp32")
    parser.add_argument("--fp8-format", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument("--bucket-mb", type=float, default=4)
    args = parser.parse_args()
    engine = viola.violaEngine(dict(sharding=dict(parameters=args.shard_parameters, gradients=args.shard_gradients,
        optimizer=args.shard_optimizer, ema=args.shard_ema),
        gradient_accumulation_steps=2, precision=dict(master_weight_dtype="fp32", gradient_dtype="fp32"),
        communication=dict(bucket_mb=args.bucket_mb, dtype=args.communication, fp8_format=args.fp8_format)))
    torch.manual_seed(42)
    model = engine.prepare_model(Model().bfloat16())
    # SGD isolates the gradient/storage change from Muon's temporary NS workspace.
    engine.prepare_optimizer(lambda params: torch.optim.SGD(params, lr=.001, momentum=.9, foreach=False))
    torch.manual_seed(100 + engine.rank)
    x = torch.randn(64, 2048, device=engine.device, dtype=torch.bfloat16)
    rows = []
    for step in range(7):
        dist.barrier()
        torch.cuda.synchronize()
        if step == 2:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for micro in range(2):
            with engine.accumulate():
                loss = (model(x).float() - x.float()).square().mean()
                engine.backward(loss)
                if step >= 2 and micro == 1:
                    after_backward = torch.cuda.memory_allocated()
                    gradients = engine.state.gradients.accumulator.numel() * 4
                engine.step()
        torch.cuda.synchronize()
        if step >= 2:
            rows.append(dict(seconds=time.perf_counter() - start, loss=loss.item(),
                             after_backward_bytes=after_backward, gradient_bytes=gradients))
    result = dict(rank=engine.rank, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), steps=rows)
    results = [None] * engine.world_size
    dist.all_gather_object(results, result)
    if engine.is_main_process:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = dict(sharding=engine.state.sharding, communication=args.communication, fp8_format=args.fp8_format,
                      bucket_mb=args.bucket_mb, ranks=results)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
