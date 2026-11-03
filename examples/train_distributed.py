"""PYTHONPATH=. python -m viola launch --config examples/distributed.yaml examples/train_distributed.py"""

import argparse

import torch
import torch.distributed as dist
from torch import nn

import viola


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20, help="Total optimizer updates")
    parser.add_argument("--batch-size", type=int, default=32, help="Local microbatch size")
    parser.add_argument("--fp8", action="store_true", help="Use Viola's FP8 MLP")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()

    engine = viola.violaEngine()
    torch.manual_seed(42)
    if args.fp8:
        model = viola.nn.MLP(128, 256, device=engine.device, dtype=torch.bfloat16)
    else:
        model = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128)).bfloat16()
    model = engine.prepare_model(model)
    optimizer = engine.prepare_optimizer(lambda params: viola.optim.FP8Muon(params, lr=1e-3))
    if args.resume:
        engine.load_checkpoint(args.resume)

    torch.manual_seed(100 + engine.rank)
    x = torch.randn(args.batch_size, 128, device=engine.device, dtype=torch.bfloat16)
    while engine.global_step < args.steps:
        with engine.accumulate():
            with engine.autocast():
                loss = (model(x).float() - x.float()).square().mean()
            engine.backward(loss)
            updated = engine.step(optimizer)
        if updated and (engine.global_step == 1 or engine.global_step % 5 == 0):
            value = engine.reduce(loss).item()
            if engine.is_main_process:
                print(f"step={engine.global_step} loss={value:.6f} ema_updates={engine.ema.updates}", flush=True)
    if args.checkpoint:
        engine.save_checkpoint(args.checkpoint)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
