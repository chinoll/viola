"""FP8 wire-codec accuracy, engine resume, and collective latency checks."""

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import viola
from viola.distributed.fp8 import FP8Communication
def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def quantized(x, size, fmt):
    kind = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]
    count = x.numel()
    x = torch.nn.functional.pad(x, (0, -count % size)).view(-1, size)
    amax = x.abs().amax(1, keepdim=True)
    scale = amax.clamp_min(1e-30) / torch.finfo(kind).max
    return ((x / scale).to(kind).float() * scale).flatten()[:count]


def metrics(value, reference):
    error = value.double() - reference.double()
    denominator = reference.double().norm().clamp_min(1e-30)
    return dict(relative_l2=(error.norm() / denominator).item(), max_abs=error.abs().max().item(),
                cosine=torch.nn.functional.cosine_similarity(value.double(), reference.double(), dim=0).item())


def check_codecs(mesh, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    results = []
    for size, n in ((64, 1), (64, 31), (256, 513), (1024, 8193), (256, 65539)):
        for fmt in ("e4m3", "e5m2"):
            torch.manual_seed(300 + rank)
            x = torch.randn(n, device=device)
            x *= torch.logspace(-5, 5, n, device=device)
            x[::31] *= 64
            originals = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(originals, x)
            reference = torch.stack(originals).sum(0)
            first = torch.stack([quantized(value, size, fmt) for value in originals]).sum(0)
            for mode in ("all_reduce", "reduce"):
                transport = FP8Communication(n, mesh, device, mode=mode, fp8_format=fmt, tile_size=size)
                owners = [None] if mode == "all_reduce" else [0, world - 1]
                for owner in owners:
                    value = x.clone()
                    if mode == "all_reduce":
                        transport.all_reduce(value)
                    else:
                        transport.reduce(value, owner)
                    if owner is None or rank == owner:
                        expected = quantized(first, size, fmt) if mode == "all_reduce" else first
                        difference = metrics(value, expected)
                        # FP32 fused multiply-add can move a value on an FP8 tie
                        # in the second quantization; the one-hop check is tighter.
                        assert difference["relative_l2"] < (3e-3 if mode == "all_reduce" else 2e-6), difference
                        row = dict(n=n, tile_size=size, format=fmt, mode=mode, owner=owner,
                                   reference_error=difference, fp32_error=metrics(value, reference),
                                   workspace_bytes=transport.workspace_bytes)
                        results.append(row)
    for mode in ("all_reduce", "reduce"):
        for fmt in ("e4m3", "e5m2"):
            x = torch.ones(1025, device=device)
            if rank == world - 1:
                x[3], x[600] = float("nan"), float("inf")
            transport = FP8Communication(x.numel(), mesh, device, mode=mode, fp8_format=fmt)
            if mode == "all_reduce":
                transport.all_reduce(x)
            else:
                transport.reduce(x, 0)
            if mode == "all_reduce" or rank == 0:
                assert not torch.isfinite(x[3]) and not torch.isfinite(x[600])
                assert torch.isfinite(x[900])
    return results


def check_engine(shard_gradients, output):
    engine = viola.violaEngine(dict(sharding=dict(gradients=shard_gradients), gradient_accumulation_steps=2,
        mixed_precision="no", precision=dict(master_weight_dtype="fp32", gradient_dtype="fp32"),
        communication=dict(dtype="fp8", bucket_mb=.004), ema=dict(decay=.9)))
    torch.manual_seed(42 + engine.rank)
    model = engine.prepare_model(viola.nn.MLP(64, 128, device=engine.device, dtype=torch.bfloat16, mode="fast"))
    optimizer = engine.prepare_optimizer(lambda p: viola.optim.FP8Muon(p, lr=.003))

    def update():
        losses = []
        for micro in range(2):
            x = torch.randn(5 + engine.rank + micro, 64, device=engine.device, dtype=torch.bfloat16)
            with engine.accumulate():
                loss = (model(x).float() - x.float()).square().mean()
                assert torch.isfinite(loss)
                losses.append(loss.item())
                engine.backward(loss, weight=x.shape[0])
                engine.step()
            if shard_gradients:
                assert all(p.grad is None for p in model.parameters())
        for param in model.parameters():
            codes = param.detach().view(torch.uint8).clone()
            dist.broadcast(codes, src=0)
            equal(param.view(torch.uint8), codes)
        return losses

    torch.manual_seed(501 + engine.rank)
    update()
    update()
    path = output / (str(shard_gradients) + "-checkpoint")
    engine.save_checkpoint(path)
    losses = update()
    expected = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), ema=engine.ema.state_dict()))
    engine.load_checkpoint(path)
    equal(update(), losses)
    equal(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), ema=engine.ema.state_dict()), expected)
    with engine.ema_scope() as averaged:
        averaged(torch.zeros(2, 64, device=engine.device, dtype=torch.bfloat16))
    equal(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), ema=engine.ema.state_dict()), expected)
    engine.close()
    return dict(shard_gradients=shard_gradients, exact_resume=True, identical_rank_weights=True, losses=losses)


def benchmark(mesh, device):
    results = []
    for n in (65536, 1048576, 6291456):
        torch.manual_seed(60 + dist.get_rank())
        original = torch.randn(n, device=device)
        for mode in ("reduce", "all_reduce"):
            for fmt in ("fp32", "bf16", "e4m3", "e5m2"):
                kind = torch.bfloat16 if fmt == "bf16" else torch.float32
                value = torch.empty(n, device=device, dtype=kind)
                transport = (FP8Communication(n, mesh, device, mode=mode, fp8_format=fmt)
                             if fmt.startswith("e") else None)
                times = []
                for repeat in range(13):
                    value.copy_(original)
                    dist.barrier()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if transport is not None:
                        transport.reduce(value, 0) if mode == "reduce" else transport.all_reduce(value)
                    elif mode == "reduce":
                        dist.reduce(value, dst=0)
                    else:
                        dist.all_reduce(value)
                    torch.cuda.synchronize()
                    if repeat >= 3:
                        times.append((time.perf_counter() - start) * 1000)
                maximum = torch.tensor(statistics.median(times), device=device)
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                results.append(dict(elements=n, mode=mode, format=fmt, median_ms=maximum.item(),
                                    codec_workspace_bytes=transport.workspace_bytes if transport is not None else 0))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("codec", "engine", "bench"), default="codec")
    parser.add_argument("--output", default="/tmp/viola-fp8-communication")
    args = parser.parse_args()
    import os
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "codec":
        results = check_codecs(mesh, device)
    elif args.phase == "engine":
        results = [check_engine(shard, output) for shard in (False, True)]
    else:
        results = benchmark(mesh, device)
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, results)
    if dist.get_rank() == 0:
        report = dict(phase=args.phase, world_size=dist.get_world_size(), ranks=ranks)
        (output / (args.phase + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(phase=args.phase, world_size=dist.get_world_size(), passed=True)), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
