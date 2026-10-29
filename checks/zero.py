"""Two-rank optimizer check: whole matrices, accumulation, skip, checkpoint.

The linear objective gives exactly representable gradients, so a full-matrix
PyTorch oracle can detect incorrect ZeRO slicing without model/loss noise.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import viola
from viola.kernels import dequantize, state_sizes
from quantization import reference_quantize


class Matrices(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.large = torch.nn.Parameter(torch.randn(193, 129) * .02)
        self.conv = torch.nn.Parameter(torch.randn(17, 13, 3, 3) * .02)
        self.small = torch.nn.Parameter(torch.randn(23, 37) * .02)
        self.bias = torch.nn.Parameter(torch.zeros(19))

    def forward(self, gradients):
        return sum((p.float() * g).sum() for p, g in zip(self.parameters(), gradients))


def config(stage):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "bf16": {"enabled": True, "check_grad_overflow": True},
        "data_types": {"grad_accum_dtype": "fp32"},
        "gradient_clipping": 0.0,
        "zero_optimization": {"stage": stage, "overlap_comm": False,
                              "reduce_bucket_size": 65536, "allgather_bucket_size": 65536,
                              "parameter_alignment": True},
        "optimizer": {"type": "Muon", "params": {
            "lr": 1e-3, "momentum": .95, "weight_decay": .01,
            "ns_method": "standard", "adjust_lr_fn": "match_rms_adamw", "torch_adam": True,
        }},
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 1000000,
    }


@torch.no_grad()
def reference_step(masters, grads, states, auxiliary, quant, rounding, step, keys, ns_steps):
    from deepspeed.runtime.zero.muon.original_muon import zeropower_via_newtonschulz5
    for i, (p, g) in enumerate(zip(masters, grads)):
        if p.ndim == 1:
            p.grad = g
            continue
        n = p.numel()
        q, scales = states[i]
        shape = (p.shape[0], n // p.shape[0])
        old = dequantize(q, scales, shape, **quant)
        beta = torch.tensor(.95, device=g.device)
        m = beta * old + (1 - beta) * g.view(-1)
        u = ((1 - beta) * g.view(-1) + beta * m).bfloat16().view(p.shape[0], -1)
        update = zeropower_via_newtonschulz5(u, steps=ns_steps)
        scale = .2 * max(u.shape)**.5
        p.mul_(1 - .001 * .01).add_(update.float().view_as(p), alpha=-.001 * scale)
        states[i] = reference_quantize(m, shape, quant, rounding, step, keys[i])
    auxiliary.step()
    auxiliary.zero_grad(set_to_none=True)


def packed_states(engine):
    return [s["viola_fp8_muon"] for s in engine.optimizer.optimizer.state.values()
            if "viola_fp8_muon" in s]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--client-optimizer", action="store_true")
    parser.add_argument("--mode", choices=("fast", "balanced", "stable"), default="balanced")
    parser.add_argument("--skip-nonfinite", action="store_true")
    parser.add_argument("--rounding", default="nearest", choices=("nearest", "stochastic"))
    parser.add_argument("--granularity", default="tile", choices=("tile", "block"))
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--block-shape", type=int, nargs=2)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(.12)
    import deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    options = dict(mode=args.mode, skip_nonfinite=args.skip_nonfinite, rounding=args.rounding,
                   rounding_seed=19, granularity=args.granularity,
                   tile_size=args.tile_size, block_shape=args.block_shape)
    viola.enable_fp8_muon(**options)
    viola.enable_fp8_muon(**options)
    quant = dict(granularity=args.granularity,
                 tile_size=args.tile_size or dict(fast=256, balanced=256, stable=128)[args.mode],
                 block_shape=tuple(args.block_shape or (16, 16)))
    ns_steps = 3 if args.mode == "fast" else 5
    torch.manual_seed(123)
    def make_engine():
        model = Matrices()
        cfg = config(args.stage)
        optimizer = None
        if args.client_optimizer:
            from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
            cfg.pop("optimizer")
            optimizer = MuonWithAuxAdam([
                dict(params=[p for p in model.parameters() if p.ndim > 1], use_muon=True,
                     lr=.001, momentum=.95, weight_decay=.01, ns_method="standard",
                     adjust_lr_fn="match_rms_adamw"),
                dict(params=[p for p in model.parameters() if p.ndim == 1], use_muon=False,
                     lr=.001, betas=(.9, .95), eps=1e-10, weight_decay=.01),
            ], adam_optimizer=torch.optim.AdamW)
        return deepspeed.initialize(model=model, optimizer=optimizer, config=cfg)[0]
    engine = make_engine()
    assert engine.optimizer.__class__.__name__ == "DeepSpeedZeroOptimizer"
    from deepspeed.utils import safe_get_full_fp32_param
    masters = [safe_get_full_fp32_param(p).detach().clone().requires_grad_() for p in engine.module.parameters()]
    rounding_keys = [p._viola_rounding_key for p in engine.module.parameters()]
    auxiliary = torch.optim.AdamW([p for p in masters if p.ndim == 1], lr=.001,
                                  betas=(.9, .95), eps=1e-10, weight_decay=.01)
    states = {}
    for i, p in enumerate(masters):
        if p.ndim > 1:
            qn, sn = state_sizes((p.shape[0], p.numel() // p.shape[0]), **quant)
            states[i] = (torch.zeros(qn, device=p.device, dtype=torch.uint8), torch.ones(sn, device=p.device))
    output = Path(args.output) / (f"zero{args.stage}-{args.mode}-{args.granularity}-{args.rounding}" + ("-client" if args.client_optimizer else ""))
    output.mkdir(parents=True, exist_ok=True)
    report = []
    successful_steps = 0

    for step in range(10):
        inject = step == (3 if args.skip_nonfinite else 9)
        keys = ("q", "scales")
        before = [{k: s[k].clone() for k in keys} for s in packed_states(engine)]
        weights_before = [p.detach().clone() for p in engine.module.parameters()]
        total = [torch.zeros_like(p) for p in masters]
        for micro in range(2):
            torch.manual_seed(1000 + 20 * step + micro + 100000 * dist.get_rank())
            grads = [torch.randint(-8, 9, p.shape, device=p.device).float() / 1024 for p in masters]
            if inject and dist.get_rank() == 0:
                grads[0].view(-1)[0] = float("inf")
            for acc, g in zip(total, grads):
                acc.add_(g, alpha=.5)
            loss = engine(grads)
            engine.backward(loss)
            engine.step()

        if inject and not args.skip_nonfinite:
            assert not engine.optimizer.overflow
            assert any(not torch.isfinite(p).all() for p in engine.module.parameters()), \
                "Nonfinite gradients must propagate, not be silently skipped"
            report.append(dict(step=step, nonfinite_propagated=True, overflow_skipped=False))
            continue
        if inject:
            assert engine.optimizer.overflow
            for p, old in zip(engine.module.parameters(), weights_before):
                assert torch.equal(p, old), "Overflow modified parameters"
            for current, previous in zip(packed_states(engine), before):
                for k in keys:
                    assert torch.equal(current[k], previous[k]), "Overflow modified momentum"
            report.append(dict(step=step, overflow_skipped=True))
            continue
        for g in total:
            dist.all_reduce(g)
            g.div_(dist.get_world_size())
        reference_step(masters, total, states, auxiliary, quant, args.rounding,
                       successful_steps, rounding_keys, ns_steps)
        successful_steps += 1
        relative = []
        for actual, expected in zip(engine.module.parameters(), masters):
            target = expected.to(actual.dtype).float()
            error = (actual.float() - target).norm() / target.norm().clamp_min(1e-8)
            relative.append(error.item())
            assert error.item() < .003, (step, actual.shape, error.item())
        for state in packed_states(engine):
            assert state["q"].dtype == torch.uint8 and state["scales"].dtype == torch.float32
            assert state["rounding_step"] == successful_steps
        # Boundary matrices exist on both ranks; their stochastic states must
        # agree byte-for-byte, independently of each rank's partition offset.
        hashes = {}
        for state in packed_states(engine):
            qo = so = 0
            for shape in state["shapes"]:
                qn, sn = state_sizes(shape, **quant)
                payload = state["q"][qo:qo+qn].cpu().numpy().tobytes() + state["scales"][so:so+sn].cpu().numpy().tobytes()
                hashes[str(shape)] = hashlib.sha256(payload).hexdigest()
                qo += qn; so += sn
        rank_hashes = [None] * dist.get_world_size()
        dist.all_gather_object(rank_hashes, hashes)
        for other in rank_hashes:
            for shape in hashes.keys() & other.keys():
                assert hashes[shape] == other[shape], (step, shape)
        report.append(dict(step=step, parameter_relative_errors=relative))

        if step == 5:
            saved = [{k: s[k].clone() for k in keys} for s in packed_states(engine)]
            engine.save_checkpoint(str(output / "checkpoint"), tag="resume")
            del engine
            torch.cuda.empty_cache()
            engine = make_engine()
            path, _ = engine.load_checkpoint(str(output / "checkpoint"), tag="resume")
            assert path is not None
            for state, previous in zip(packed_states(engine), saved):
                for k in keys:
                    assert state[k].dtype == previous[k].dtype and torch.equal(state[k], previous[k])
                assert state["rounding_step"] == successful_steps
            report[-1]["checkpoint_bit_exact"] = True

    # Every replicated model must end with the same values on every rank.
    for p in engine.module.parameters():
        first = p.detach().clone()
        dist.broadcast(first, src=0)
        assert torch.equal(first.view(torch.uint8), p.view(torch.uint8))
    state_bytes = sum(t.numel() * t.element_size() for state in packed_states(engine)
                      for t in (state[k] for k in keys))
    report.append(dict(stage=args.stage, rank=dist.get_rank(), state_bytes=state_bytes,
                       world_size=dist.get_world_size(), passed=True))
    (output / f"rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report[-1]), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
