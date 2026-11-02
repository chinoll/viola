"""Two-rank numerical checks for DTensor communication and complete-matrix Muon.

CPU: torchrun --standalone --nproc-per-node=2 checks/dtensor_muon.py --device cpu
GPU: torchrun --standalone --nproc-per-node=2 checks/dtensor_muon.py --device cuda
"""

import argparse
import copy
import io
import json
import os
import warnings

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

import viola
from viola.distributed import MatrixPartition, ShardedEMA, gather_to_owner, partial_gradient, write_from_owner


def roundtrip(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def transfers(mesh, device):
    rank, world = mesh.get_local_rank(), mesh.size()
    for shape in ((5, 7), (1,), (3, 2, 3, 5)):
        full = torch.arange(torch.Size(shape).numel(), device=device, dtype=torch.float32).reshape(shape)
        for placement in (Replicate(), *(Shard(i) for i in range(len(shape)))):
            value = distribute_tensor(full.clone(), mesh, [placement])
            for owner in range(world):
                gathered = gather_to_owner(value, owner)
                if rank == owner:
                    torch.testing.assert_close(gathered, full, rtol=0, atol=0)
                else:
                    assert gathered is None
                write_from_owner(value, full + 1 if rank == owner else None, owner)
                torch.testing.assert_close(value.full_tensor(), full + 1, rtol=0, atol=0)
                full += 1
    local = torch.arange(35, device=device, dtype=torch.float32).view(5, 7) * (rank + 1)
    for average in (False, True):
        contribution = partial_gradient(local, mesh, average=average)
        result = gather_to_owner(contribution, world - 1)
        expected = torch.arange(35, device=device, dtype=torch.float32).view(5, 7)
        expected *= (world + 1) / 2 if average else world * (world + 1) / 2
        if rank == world - 1:
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(contribution.to_local(), local, rtol=0, atol=0)


def ema_check(mesh, device):
    model = nn.Module()
    full = torch.arange(35, device=device, dtype=torch.float32).view(5, 7)
    model.register_parameter("weight", nn.Parameter(distribute_tensor(full.clone(), mesh, [Replicate()])))
    optimizer = torch.optim.SGD(model.parameters(), lr=.25)
    off = ShardedEMA(model, optimizer, decay=0)
    assert off.state_dict() == {} and off._hook is None
    ema = ShardedEMA(model, optimizer, decay=.8)
    expected = full.clone()
    assert ema.weights["weight"].placements == (Shard(0),)
    for _ in range(3):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        full -= .25
        expected.lerp_(full, .2)
    torch.testing.assert_close(ema.weights["weight"].full_tensor(), expected, rtol=0, atol=0)
    saved = roundtrip(ema.state_dict())
    ema.close()
    restored = ShardedEMA(model, optimizer, decay=.8)
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.weights["weight"].full_tensor(), expected, rtol=0, atol=0)
    assert restored.updates == 3
    restored.close()


def make_optimizer(params, quant):
    groups = [{"params": [p], "batched": i == 3} for i, p in enumerate(params)]
    return viola.optim.FP8Muon(groups, lr=.003, weight_decay=.01, **quant)


def muon_check(mesh, device, *, dtype, quant, steps):
    rank, world = mesh.get_local_rank(), mesh.size()
    shapes = [(96, 128), (65, 96), (64, 95), (2, 16, 32), (16, 8, 3, 3), (17,)]
    layouts = [Replicate(), Shard(0), Shard(1), Shard(0), Shard(1), Shard(0)]
    generator = torch.Generator(device=device).manual_seed(724)
    initial = [torch.randn(shape, generator=generator, device=device, dtype=dtype) * .1 for shape in shapes]
    params = [nn.Parameter(distribute_tensor(p.clone(), mesh, [layout])) for p, layout in zip(initial, layouts)]
    refs = [nn.Parameter(p.clone()) for p in initial]
    for p in params + refs:
        p.grad_dtype = torch.float32
    optimizer = make_optimizer(params, quant)
    reference = make_optimizer(refs, quant)
    model = nn.ParameterList(params)
    ema = ShardedEMA(model, optimizer, decay=.9)
    expected_ema = [p.detach().float().clone() for p in refs]
    plan = optimizer.partition
    assert len(plan.owners) == len(params)
    state_sizes = []
    errors = []
    for step in range(steps):
        for index, (p, ref) in enumerate(zip(params, refs)):
            base = torch.randn(shapes[index], generator=generator, device=device, dtype=torch.float32)
            # Power-of-two contributions keep the independent arithmetic oracle
            # exact while checking that the averaging is neither lost nor doubled.
            base = torch.round(base * 16) / 16
            if index % 3 == 0:
                p.grad = partial_gradient(base * (rank + 1), mesh, average=True)
                ref.grad = base * ((world + 1) / 2)
            elif index % 3 == 1:
                grad_layout = Shard(1) if p.ndim > 1 else Shard(0)
                p.grad = distribute_tensor(base.clone(), mesh, [grad_layout])
                ref.grad = base.clone()
            else:
                p.grad = partial_gradient(base * (rank + 1), mesh, average=False)
                ref.grad = base * (world * (world + 1) / 2)
        optimizer.step()
        reference.step()
        for index, (p, ref) in enumerate(zip(params, refs)):
            actual = p.full_tensor()
            difference = (actual.float() - ref.float()).abs().max().item()
            errors.append(difference)
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            expected_ema[index].lerp_(ref.detach().float(), .1)
            torch.testing.assert_close(ema.weights[str(index)].full_tensor(), expected_ema[index], rtol=0, atol=0)
            if plan.is_owner(p):
                for name in ("momentum_q", "momentum_scale", "exp_avg", "exp_avg_sq"):
                    if name in reference.state[ref]:
                        torch.testing.assert_close(optimizer.state[p][name], reference.state[ref][name], rtol=0, atol=0)
            else:
                assert p not in optimizer.state
        if step == steps // 2 - 1:
            saved_opt = roundtrip(optimizer.state_dict())
            saved_ema = roundtrip(ema.state_dict())
            ema.close()
            optimizer = make_optimizer(params, quant)
            optimizer.load_state_dict(saved_opt)
            ema = ShardedEMA(model, optimizer, decay=.9)
            ema.load_state_dict(saved_ema)
    for state in optimizer.state.values():
        state_sizes.append(sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor)))
    local_bytes = sum(state_sizes)
    counters = [None] * world
    dist.all_gather_object(counters, local_bytes, group=mesh.get_group())
    reference_bytes = sum(v.numel() * v.element_size() for s in reference.state.values()
                          for v in s.values() if isinstance(v, torch.Tensor))
    assert sum(counters) == reference_bytes
    ema.close()
    return dict(dtype=str(dtype), quant=quant, steps=steps, max_abs_error=max(errors),
                optimizer_bytes_by_rank=counters, single_copy_optimizer_bytes=reference_bytes)


def binding_check(mesh, device):
    # No master: storage identity is retained. A BF16 model with an FP32 master
    # keeps its existing DTensor layout and uses master values for EMA.
    for master_dtype in (None, torch.float32):
        viola.set_master_weight_dtype(master_dtype)
        viola.set_gradient_dtype(torch.float32)
        full = torch.arange(64 * 64, device=device, dtype=torch.float32).view(64, 64).div_(4096).bfloat16()
        model = nn.Module()
        model.register_parameter("weight", nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)])))
        ref = nn.Linear(64, 64, bias=False, device=device, dtype=torch.bfloat16)
        ref.weight.data.copy_(full)
        optimizer = viola.optim.FP8Muon(model.parameters(), lr=.003)
        reference = viola.optim.FP8Muon(ref.parameters(), lr=.003)
        viola.optim.bind_optimizer(model, optimizer)
        viola.optim.bind_optimizer(ref, reference)
        master = optimizer.param_groups[0]["params"][0]
        assert (master is model.weight) == (master_dtype is None)
        assert master.placements == model.weight.placements
        ema = ShardedEMA(model, optimizer, decay=.9)
        ref_master = reference.param_groups[0]["params"][0]
        expected_ema = ref_master.detach().float().clone()
        for step in range(3):
            gradient = torch.sin(torch.arange(4096, device=device).float() + step).view(64, 64)
            model.weight.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
            ref.weight.grad = gradient.clone()
            optimizer.step()
            reference.step()
            torch.testing.assert_close(model.weight.full_tensor(), ref.weight, rtol=0, atol=0)
            expected_ema.lerp_(ref_master.detach().float(), .1)
            torch.testing.assert_close(ema.weights["weight"].full_tensor(), expected_ema, rtol=0, atol=0)
            optimizer.zero_grad()
            reference.zero_grad()
        saved = roundtrip(optimizer.state_dict())
        assert (saved["viola_masters"]["weights"][0] is None) == (master_dtype is None)
        optimizer.load_state_dict(saved)
        torch.testing.assert_close(model.weight.full_tensor(), ref.weight, rtol=0, atol=0)
        ema.close()
    viola.set_master_weight_dtype(torch.float32)


def optional_skip_check(mesh, device):
    full = torch.ones((16, 16), device=device)
    p = nn.Parameter(distribute_tensor(full.clone(), mesh, [Replicate()]))
    optimizer = viola.optim.FP8Muon([p], skip_nonfinite=True)
    ema = ShardedEMA([("weight", p)], optimizer, decay=.9)
    local = torch.ones_like(full)
    if mesh.get_local_rank() == 0:
        local[0, 0] = float("nan")
    p.grad = partial_gradient(local, mesh)
    optimizer.step()
    assert optimizer.step_was_skipped and ema.updates == 0 and not optimizer.state
    torch.testing.assert_close(p.full_tensor(), full, rtol=0, atol=0)
    ema.close()


def autograd_check(mesh, device):
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 32)).to(device)
    ref = copy.deepcopy(model)
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            module.register_parameter(name, nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Replicate()])))
    optimizer = viola.optim.FP8Muon(model.parameters(), lr=.003)
    reference = viola.optim.FP8Muon(ref.parameters(), lr=.003)
    losses = []
    for step in range(4):
        torch.manual_seed(900 + step)
        full_x = torch.randn(32, 64, device=device)
        full_y = torch.randn(32, 32, device=device)
        x = distribute_tensor(full_x, mesh, [Shard(0)])
        y = distribute_tensor(full_y, mesh, [Shard(0)])
        optimizer.zero_grad()
        reference.zero_grad()
        loss = (model(x) - y).square().mean()
        baseline = (ref(full_x) - full_y).square().mean()
        loss.backward()
        baseline.backward()
        # Reduction order can differ for real matmuls; compare against the
        # synchronized distributed gradient to isolate optimizer equivalence.
        grad_errors = []
        for p, r in zip(model.parameters(), ref.parameters()):
            logical_grad = p.grad.full_tensor()
            grad_errors.append((logical_grad - r.grad).abs().max().item())
            torch.testing.assert_close(logical_grad, r.grad, rtol=1e-4, atol=2e-7)
            r.grad = logical_grad.clone()
        optimizer.step()
        reference.step()
        for p, r in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(p.full_tensor(), r, rtol=0, atol=0)
        losses.append(dict(loss=float(loss.full_tensor()), baseline=float(baseline), grad_abs_max=max(grad_errors)))
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="FP8Muon: .* matrices have partial")
    device = torch.device(args.device)
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    mesh = init_device_mesh(args.device, (dist.get_world_size(),))
    transfers(mesh, device)
    ema_check(mesh, device)
    results = dict(communication="passed", ema="passed")
    if args.device == "cuda":
        results["muon"] = []
        for dtype, quant in (
            (torch.float32, dict(granularity="tile", tile_size=128, rounding="nearest")),
            (torch.bfloat16, dict(granularity="tile", tile_size=256, rounding="stochastic")),
            (torch.float16, dict(granularity="block", block_shape=(16, 16), rounding="nearest")),
        ):
            results["muon"].append(muon_check(mesh, device, dtype=dtype, quant=quant, steps=args.steps))
        binding_check(mesh, device)
        results["master_binding"] = "passed"
        optional_skip_check(mesh, device)
        results["explicit_skip_consensus"] = "passed"
        results["autograd"] = autograd_check(mesh, device)
    if dist.get_rank() == 0:
        print(json.dumps(results, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
