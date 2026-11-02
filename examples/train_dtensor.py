"""DTensor data parallelism + owner-sharded FP8 Muon, without FSDP.

PYTHONPATH=. torchrun --standalone --nproc-per-node=2 examples/train_dtensor.py
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

import viola
from viola.distributed import ShardedEMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="Local batch size")
    parser.add_argument("--ema-decay", type=float, default=0, help="0 disables EMA")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # The model keeps its BF16 weights; FP8Muon keeps one FP8 momentum per owner.
    viola.set_master_weight_dtype(None)
    viola.set_gradient_dtype("fp32")
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128)).cuda().bfloat16()
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            module.register_parameter(name, nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Replicate()])))
    optimizer = viola.optim.FP8Muon(model.parameters(), lr=1e-3)
    viola.optim.bind_optimizer(model, optimizer)
    ema = ShardedEMA(model, optimizer, decay=args.ema_decay)

    # Each rank contributes a different data shard. DTensor's global mean
    # produces Partial gradients with the correct global loss normalization.
    torch.manual_seed(100 + dist.get_rank())
    local_x = torch.randn(args.batch_size, 128, device="cuda", dtype=torch.bfloat16)
    x = DTensor.from_local(local_x, mesh, [Shard(0)])
    for step in range(args.steps):
        optimizer.zero_grad()
        loss = (model(x).float() - x.float()).square().mean()
        loss.backward()
        optimizer.step()  # Includes owner reduction, update, sync and EMA hook.
        if step == 0 or (step + 1) % 5 == 0:
            value = loss.detach().full_tensor().item()  # Collective on every rank.
            if dist.get_rank() == 0:
                print(f"step={step + 1} loss={value:.6f} ema_updates={ema.updates}", flush=True)

    ema.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
