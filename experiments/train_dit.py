"""Real ImageNet DiT training with per-update, per-matrix momentum diagnostics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import viola
from dit import DiT
from audit import MomentumAudit


class LatentBatches(Dataset):
    def __init__(self, root, batch, rank, world, epoch, seed):
        root = Path(root)
        meta = json.loads((root / "metadata.json").read_text())
        n = meta["samples"]
        self.data = np.memmap(root / "latents.f32", mode="r", dtype="float32", shape=(n, 2, 4, 32, 32))
        self.labels = np.memmap(root / "labels.i64", mode="r", dtype="int64", shape=(n,))
        rng = np.random.default_rng(seed + epoch)
        self.order = rng.permutation(n)
        self.flip = rng.integers(0, 2, n, dtype=np.int8)
        self.batch, self.local, self.rank = batch, batch // world, rank

    def __len__(self):
        return len(self.order) // self.batch

    def __getitem__(self, i):
        start = i * self.batch + self.rank * self.local
        idx = self.order[start:start + self.local]
        return torch.from_numpy(self.data[idx, self.flip[idx]]), torch.from_numpy(self.labels[idx].copy())


def make_config(args, world):
    local = args.batch // world
    micro = min(args.micro_batch, local)
    assert args.batch % world == 0 and local % micro == 0
    return dict(train_micro_batch_size_per_gpu=micro, gradient_accumulation_steps=local // micro,
        bf16=dict(enabled=True, check_grad_overflow=False), data_types=dict(grad_accum_dtype="fp32"),
        gradient_clipping=0., zero_optimization=dict(stage=args.stage, overlap_comm=False,
        reduce_bucket_size=8000000, allgather_bucket_size=8000000, parameter_alignment=True),
        zero_allow_untested_optimizer=True, steps_per_print=10000000,
        wall_clock_breakdown=False)


@torch.no_grad()
def probe(engine, data, rank, world):
    """Fixed training-distribution probes; not held-out or generative FID evaluation."""
    engine.eval()
    total = torch.zeros(2, device=engine.device)
    indices = np.random.default_rng(1701).choice(len(data.order), 1024, replace=False)[rank::world]
    rng = torch.Generator(device=engine.device).manual_seed(9999)
    # Draw globally and select rows so probes do not change with DP count.
    noise = torch.randn(1024, 4, 32, 32, generator=rng, device=engine.device)[rank::world]
    times = torch.randn(1024, generator=rng, device=engine.device).sigmoid()[rank::world]
    for start in range(0, len(indices), 32):
        idx = indices[start:start + 32]
        x = torch.from_numpy(data.data[idx, 0]).to(engine.device).float() * .18215
        y = torch.from_numpy(data.labels[idx].copy()).to(engine.device)
        t = times[start:start + 32]
        eps = noise[start:start + 32]
        z = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * eps
        prediction = engine(z.bfloat16(), t, y).float()
        errors = (prediction - (eps - x)).square().flatten(1).mean(1)
        total[0] += errors.sum(); total[1] += len(idx)
    dist.all_reduce(total)
    engine.train()
    return (total[0] / total[1]).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--stage", type=int, default=2, choices=(1, 2))
    p.add_argument("--mode", default="balanced", choices=("fast", "balanced", "stable"))
    p.add_argument("--momentum-dtype", default="fp8", choices=("fp8", "bf16"))
    p.add_argument("--master-weight-dtype", choices=("fp32", "bf16", "fp16"))
    p.add_argument("--gradient-dtype", choices=("fp32", "bf16", "fp16"))
    p.add_argument("--master-rounding", choices=("nearest", "stochastic"), default="nearest")
    p.add_argument("--adam-state-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    p.add_argument("--rounding", default="nearest", choices=("nearest", "stochastic"))
    p.add_argument("--rounding-seed", type=int, default=0)
    p.add_argument("--granularity", default="tile", choices=("tile", "block"))
    p.add_argument("--tile-size", type=int)
    p.add_argument("--block-shape", type=int, nargs=2)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ns-interval", type=int, default=100)
    p.add_argument("--probe-interval", type=int, default=500)
    args = p.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    import deepspeed
    from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.master_weight_dtype is not None:
        viola.set_master_weight_dtype(args.master_weight_dtype, rounding=args.master_rounding)
    if args.gradient_dtype is not None:
        viola.set_gradient_dtype(args.gradient_dtype)
    viola.enable_fp8_muon(mode=args.mode, ns_method="standard", rounding=args.rounding,
        rounding_seed=args.rounding_seed, granularity=args.granularity,
        tile_size=args.tile_size, block_shape=args.block_shape, adam_state_dtype=args.adam_state_dtype)
    if args.momentum_dtype == "bf16":
        from momentum16 import reference_partition
        import viola.plugin as plugin
        plugin._partition = reference_partition(torch.bfloat16)
        if rank == 0:
            print("BF16 reference training: ZeRO hooks shared; momentum storage/update is BF16, not FP8.", flush=True)
    model = DiT()
    muon, adam = [], []
    for name, param in model.named_parameters():
        (muon if name.startswith("blocks.") and param.ndim == 2 else adam).append(param)
    optimizer = MuonWithAuxAdam([
        dict(params=muon, use_muon=True, lr=args.lr, momentum=.95, weight_decay=.01,
             ns_method="standard", ns_steps=5, nesterov=True, adjust_lr_fn="match_rms_adamw"),
        dict(params=adam, use_muon=False, lr=args.lr, betas=(.9, .95), eps=1e-8, weight_decay=.01),
    ], adam_optimizer=torch.optim.AdamW)
    cfg = make_config(args, world)
    engine = deepspeed.initialize(model=model, optimizer=optimizer, config=cfg)[0]
    audit = None
    if args.momentum_dtype == "fp8":
        audit = MomentumAudit(engine.module, Path(args.output) / "metrics", rank, args.ns_interval)
        engine.optimizer._viola_observer = audit
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    log = (output / f"steps-rank{rank}.jsonl").open("a", buffering=1)
    metadata = json.loads((Path(args.data) / "metadata.json").read_text())
    steps_per_epoch = metadata["samples"] // args.batch
    if rank == 0:
        manifest = dict(vars(args), world_size=world, tokens_per_image=64, token_budget=args.batch * 64,
            model="DiT-B/4", pixel_patch=32, width=768, depth=12, heads=12,
            parameters=sum(p.numel() for p in model.parameters()),
            muon_parameters=sum(p.numel() for p in muon), steps_per_epoch=steps_per_epoch,
            total_steps=args.epochs * steps_per_epoch, torch=torch.__version__, deepspeed=deepspeed.__version__,
            deepspeed_config=cfg, dataset=metadata, objective="FM v-prediction, logit-normal t",
            data_remainder_policy="drop incomplete global batch each epoch",
            reference_precision="bf16",
            comparison=("same-gradient shadow BF16 momentum and NS update; not separate convergence runs"
                        if audit else "independent training with single BF16 momentum storage"),
            source_sha256={str(f.relative_to(Path(__file__).parent.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                for folder in (Path(__file__).parent, Path(viola.__file__).parent) for f in folder.glob('*.py')})
        (output / "settings.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest), flush=True)
    micro = cfg["train_micro_batch_size_per_gpu"]
    local = args.batch // world
    step = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        data = LatentBatches(args.data, args.batch, rank, world, epoch, args.seed)
        loader = DataLoader(data, batch_size=None, num_workers=args.workers, pin_memory=True,
                            **(dict(prefetch_factor=2) if args.workers else {}))
        for x, y in loader:
            begin = time.monotonic()
            if audit:
                audit.step = step + 1
            lr = args.lr * min((step + 1) / 1000, 1.)
            for group in engine.optimizer.optimizer.param_groups:
                group["lr"] = lr
            x = x.to(engine.device, non_blocking=True).float() * .18215
            y = y.to(engine.device, non_blocking=True)
            generator = torch.Generator(device=engine.device).manual_seed(args.seed + 100000 + step)
            noise = torch.randn(args.batch, 4, 32, 32, generator=generator, device=engine.device)
            t = torch.randn(args.batch, generator=generator, device=engine.device).sigmoid()
            drop = torch.rand(args.batch, generator=generator, device=engine.device) < .1
            section = slice(rank * local, (rank + 1) * local)
            noise, t, drop = noise[section], t[section], drop[section]
            y = y.masked_fill(drop, 1000)
            loss_sum = torch.zeros((), device=engine.device)
            for start in range(0, local, micro):
                clean, eps, times = x[start:start + micro], noise[start:start + micro], t[start:start + micro]
                z = (1 - times[:, None, None, None]) * clean + times[:, None, None, None] * eps
                prediction = engine(z.bfloat16(), times, y[start:start + micro])
                loss = (prediction.float() - (eps - clean)).square().mean()
                engine.backward(loss)
                engine.step()
                loss_sum += loss.detach() * (micro / local)
            dist.all_reduce(loss_sum); loss_sum /= world
            if audit:
                diagnostics = audit.finish(engine.optimizer.overflow)
            else:
                states = [s["single_16bit_momentum"] for s in engine.optimizer.optimizer.state.values()
                          if "single_16bit_momentum" in s]
                assert all(s.dtype == torch.bfloat16 for s in states)
                diagnostics = dict(momentum_dtype="bf16", momentum_state_bytes=sum(s.nbytes for s in states))
            torch.cuda.synchronize()
            step += 1
            record = dict(step=step, epoch=epoch + (step % steps_per_epoch or steps_per_epoch) / steps_per_epoch,
                          loss=loss_sum.item(), lr=lr, seconds=time.monotonic() - begin,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), **diagnostics)
            if step % args.probe_interval == 0:
                record["fixed_probe_loss"] = probe(engine, data, rank, world)
            log.write(json.dumps(record) + "\n")
            if rank == 0 and (step <= 5 or step % 25 == 0):
                print(json.dumps(record), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if audit:
            audit.flush()
        engine.save_checkpoint(str(output / "checkpoint"), tag="state", client_state=dict(step=step, epoch=epoch + 1))
        if args.max_steps and step >= args.max_steps:
            break
    log.close()
    dist.barrier()
    if rank == 0:
        (output / "complete.json").write_text(json.dumps(dict(steps=step, epochs_completed=step / steps_per_epoch,
                                                             completed_at=time.time()), indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
