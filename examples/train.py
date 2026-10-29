"""Ordinary PyTorch training: python examples/train.py or torchrun for DDP."""
import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import viola


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--mode", choices=("fast", "balanced", "stable"), default="balanced")
    parser.add_argument("--rounding", choices=("nearest", "stochastic"), default="nearest")
    args = parser.parse_args()
    rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.SiLU(), torch.nn.Linear(256, 64)).cuda()
    optimizer = viola.optim.FP8Muon(model.parameters(), lr=.003, mode=args.mode, rounding=args.rounding)
    if world > 1:
        model = DDP(model, device_ids=[rank])
    x = torch.randn(128, 128, device="cuda")
    for step in range(args.steps):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(x).float() - x[:, :64]).square().mean()
        loss.backward()
        optimizer.step()
        if rank == 0 and (step == 0 or (step + 1) % 20 == 0):
            print(f"step={step + 1} loss={loss.item():.6f}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
