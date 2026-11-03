"""Compare independent sharding combinations with replicated training arithmetic."""

import argparse
import copy
import gc
import itertools
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

import viola
from viola.distributed.communication import replicate_tensor


def trace(message):
    if os.environ.get("VIOLA_CHECK_TRACE"):
        print(f"rank={dist.get_rank()} {message}", flush=True)


class Block(nn.Module):
    def __init__(self, fp8=False, transformer=False, mode="memory"):
        super().__init__()
        if fp8:
            self.net = viola.nn.MLP(64, 128, weight_dtype="fp8", mode=mode, autotune=False,
                                    device="cuda", dtype=torch.bfloat16)
        elif transformer:
            self.net = nn.TransformerEncoderLayer(64, 4, 128, dropout=0., batch_first=True, norm_first=True)
        else:
            self.net = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))

    def forward(self, x):
        return x + self.net(x) * .1


class Model(nn.Module):
    def __init__(self, fp8=False, transformer=False, checkpoint_blocks=False, mode="memory"):
        super().__init__()
        self.blocks = nn.ModuleList([Block(fp8, transformer, mode), Block(fp8, transformer, mode)])
        self.checkpoint_blocks = checkpoint_blocks
        self.bias = nn.Parameter(torch.zeros(64))
        self.unused = nn.Parameter(torch.ones(7))

    def forward(self, x, branch=True):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.checkpoint_blocks else block(x)
        return x + self.bias if branch else x


class MatrixMomentum(viola.optim.MatrixOptimizer):
    def __init__(self, params):
        super().__init__(params, dict(lr=.003))

    def _step_local(self, parameter, gradient, state, group, index):
        assert parameter.shape == gradient.shape
        state.setdefault("momentum", torch.zeros_like(parameter))
        state["momentum"].mul_(.9).add_(gradient)
        parameter.add_(state["momentum"], alpha=-group["lr"])


def check(args, flags):
    keys = ("parameters", "gradients", "optimizer", "ema")
    sharding = dict(zip(keys, flags))
    config = dict(device=args.device, mixed_precision="no", gradient_accumulation_steps=2,
                  sharding=sharding, precision=dict(master_weight_dtype=args.master, gradient_dtype="fp32"),
                  communication=dict(dtype="fp32", bucket_mb=.02), ema=dict(decay=.9, dtype="fp32"))
    engine = viola.violaEngine(config)
    trace(f"case={flags} init")
    torch.manual_seed(42)
    raw = Model(args.fp8, args.transformer, args.checkpoint_blocks, args.fp8_mode).to(engine.device)
    if args.tied:
        raw.blocks[1].net[0].weight = raw.blocks[0].net[0].weight
    if not args.fp8:
        raw.to(dtype=getattr(torch, args.storage))
    reference = copy.deepcopy(raw)
    model = engine.prepare_model(raw)
    trace("model prepared")
    factory = (lambda p: viola.optim.FP8Muon(p, lr=.003, ns_steps=3)) if args.fp8 else MatrixMomentum
    if args.adamw:
        factory = lambda p: torch.optim.AdamW(p, lr=.003)
    optimizer = engine.prepare_optimizer(factory)
    trace("optimizer prepared")
    ref_optimizer = factory(reference.parameters())
    viola.optim.bind_optimizer(reference, ref_optimizer)
    names = {id(p): n for n, p in reference.named_parameters()}
    masters = {names[id(p)]: m for p, m, _ in ref_optimizer._viola_linear_masters.pairs}
    ema = {n: p.detach().float().clone() for n, p in masters.items()}
    assert all(isinstance(p, DTensor) == flags[0] for p in model.parameters())
    max_error = 0.

    def update(step, micros=2):
        nonlocal max_error
        ref_optimizer.zero_grad()
        accumulated, weight = {}, 0
        for micro in range(micros):
            torch.manual_seed(1000 + step * 10 + micro + engine.rank * 100)
            batch = 2 + engine.rank + micro
            x = torch.randn(batch, 4, 64, device=engine.device,
                            dtype=torch.bfloat16 if args.fp8 else getattr(torch, args.storage))
            target = x.float() * .7
            with engine.accumulate():
                trace(f"forward {step}/{micro}")
                loss = (model(x, engine.rank == 0 or args.same_branch).float() - target).square().mean()
                trace("forward done")
                ref_loss = (reference(x, engine.rank == 0 or args.same_branch).float() - target).square().mean()
                torch.testing.assert_close(loss, ref_loss, rtol=2e-4 if args.fp8 else 1e-5, atol=1e-6)
                engine.backward(loss, weight=batch)
                trace("backward done")
                (ref_loss * batch).backward()
                if flags[1]:
                    for name, p in reference.named_parameters():
                        present = torch.tensor(p.grad is not None, dtype=torch.int32, device=x.device)
                        dist.all_reduce(present, op=dist.ReduceOp.MAX)
                        if not present.item():
                            continue
                        value = p.grad if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
                        value = value.clone()
                        dist.all_reduce(value)
                        accumulated.setdefault(name, torch.zeros_like(value)).add_(value)
                        p.grad = None
                if micro + 1 < micros:
                    assert not engine.step()
            weight += batch
        denominator = torch.tensor(weight, device=engine.device, dtype=torch.float64)
        dist.all_reduce(denominator)
        for name, p in reference.named_parameters():
            if flags[1]:
                p.grad = accumulated.get(name)
            else:
                present = torch.tensor(p.grad is not None, dtype=torch.int32, device=engine.device)
                dist.all_reduce(present, op=dist.ReduceOp.MAX)
                if present.item():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p, dtype=torch.float32)
                    dist.all_reduce(p.grad)
            if p.grad is not None:
                p.grad.div_(denominator)
        trace("step")
        assert engine.step(force=True)
        trace("step done")
        ref_optimizer.step()
        for name, value in ema.items():
            value.mul_(.9).add_(masters[name].detach(), alpha=.1)
        for record in engine.state.parameters.records:
            actual = engine.state.parameters.full(record).float()
            expected = reference.get_parameter(record.name).detach().float()
            error = float((actual - expected).abs().max())
            max_error = max(max_error, error)
            torch.testing.assert_close(actual, expected, rtol=.02 if args.fp8 else 1e-5,
                                       atol=.002 if args.fp8 else 1e-6)
        for name, value in engine.ema.weights.items():
            torch.testing.assert_close(value, ema[name], rtol=1e-4 if args.fp8 else 1e-5, atol=1e-6)

    for step in range(3):
        update(step)
        if flags[0] and args.fp8:
            assert all(m.weight_transpose.numel() == 0 for m in model.modules() if isinstance(m, viola.nn.Linear))
    before = engine.state.parameters.snapshot()
    with engine.ema_scope() as averaged:
        averaged(torch.zeros(1, 4, 64, device=engine.device,
                              dtype=torch.bfloat16 if args.fp8 else getattr(torch, args.storage)))
    for name, value in engine.state.parameters.snapshot().items():
        torch.testing.assert_close(value.float(), before[name].float(), rtol=0, atol=0)
    path = Path(args.output) / ("".join(str(int(v)) for v in flags) + str(args.master) + str(args.fp8))
    engine.save_checkpoint(path, extra="roundtrip")
    update(3, micros=1)
    expected = engine.state.parameters.snapshot()
    assert engine.load_checkpoint(path) == "roundtrip"
    # Continue the engine on the same inputs, without advancing the reference twice.
    torch.manual_seed(1000 + 3 * 10 + engine.rank * 100)
    batch = 2 + engine.rank
    x = torch.randn(batch, 4, 64, device=engine.device,
                    dtype=torch.bfloat16 if args.fp8 else getattr(torch, args.storage))
    engine.backward((model(x, engine.rank == 0 or args.same_branch).float() - x.float() * .7).square().mean(), weight=batch)
    engine.step(force=True)
    for name, value in engine.state.parameters.snapshot().items():
        torch.testing.assert_close(value.float(), expected[name].float(), rtol=0, atol=0)
    counts = dict(parameters=sum(p.to_local().numel() if isinstance(p, DTensor) else p.numel()
                                 for p in model.parameters()),
                  gradients=engine.state.gradients.accumulator.numel(),
                  masters=sum(p.untyped_storage().nbytes() for i, p in engine.state.optimizer.parameters.items()
                              if p is not next(r.parameter for r in engine.state.parameters.records if r.index == i)),
                  ema=sum(p.numel() for p in engine.ema.weights.values()),
                  optimizer=sum(v.numel() for s in optimizer.state.values() for v in s.values()
                                if isinstance(v, torch.Tensor)))
    counts_by_rank = [None] * engine.world_size
    dist.all_gather_object(counts_by_rank, counts)
    total = sum(p.numel() for p in reference.parameters())
    for key, flag in (("parameters", flags[0]), ("gradients", flags[1]), ("ema", flags[3])):
        assert sum(c[key] for c in counts_by_rank) == total * (1 if flag else engine.world_size), (key, counts_by_rank)
    if args.master is None:
        assert all(c["masters"] == 0 for c in counts_by_rank)
    result = dict(sharding=sharding, master=args.master, fp8=args.fp8, max_weight_error=max_error,
                  exact_resume=True, storage=counts_by_rank)
    if engine.is_main_process:
        print(json.dumps(result), flush=True)
    engine.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parameters", choices=("all", "yes", "no"), default="no")
    parser.add_argument("--master-dtype", dest="master", default="fp32")
    parser.add_argument("--storage", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-mode", choices=("memory", "fast"), default="memory")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--checkpoint-blocks", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--tied", action="store_true")
    parser.add_argument("--same-branch", action="store_true")
    parser.add_argument("--case", help="One combination, e.g. 1011")
    parser.add_argument("--output", default="/tmp/viola-sharding-check")
    args = parser.parse_args()
    if os.environ.get("VIOLA_CHECK_TRACE"):
        import faulthandler
        faulthandler.dump_traceback_later(25)
    if args.master == "none":
        args.master = None
    parameter_flags = (False, True) if args.parameters == "all" else (args.parameters == "yes",)
    results = []
    for flags in itertools.product(parameter_flags, (False, True), (False, True), (False, True)):
        if args.case and flags != tuple(c == "1" for c in args.case):
            continue
        results.append(check(args, flags))
        gc.collect()
    if dist.get_rank() == 0:
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        (path / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
