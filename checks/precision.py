"""Check actual gradient buffers, custom backward, independent states and resume."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import viola
from zero import Matrices, config

DTYPES = dict(fp32=torch.float32, bf16=torch.bfloat16, fp16=torch.float16)


class Manual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors):
        n = len(tensors) // 2
        ctx.save_for_backward(*tensors[n:])
        return sum((p.float() * g).sum() for p, g in zip(tensors[:n], tensors[n:]))

    @staticmethod
    def backward(ctx, dy):
        # A real FP32 dW, not a BF16 result subsequently widened to FP32.
        grads = tuple(dy.float() * g for g in ctx.saved_tensors)
        return (*grads, *(None for _ in grads))


class Probe:
    def __init__(self, dtype):
        self.dtype = dtype
        self.calls = 0
        self.expected = None

    def momentum(self, param, gradient, *args):
        assert gradient.dtype == self.dtype
        torch.testing.assert_close(gradient, torch.full_like(gradient, self.expected), rtol=2e-6, atol=1e-9)
        self.calls += 1

    def update(self, *args):
        pass


class PrecisionMatrices(Matrices):
    def forward(self, gradients, manual=True):
        return Manual.apply(*self.parameters(), *gradients) if manual else super().forward(gradients)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, choices=(1, 2), required=True)
    p.add_argument('--master-dtype', dest='master', choices=DTYPES, required=True)
    p.add_argument('--gradient', choices=DTYPES, required=True)
    p.add_argument('--adam-state', choices=DTYPES, default='fp32')
    p.add_argument('--rounding', choices=('nearest', 'stochastic'), default='nearest')
    p.add_argument('--output', default='results/precision-zero')
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.cuda.set_per_process_memory_fraction(.08)
    torch.set_num_threads(2)
    import deepspeed
    deepspeed.init_distributed(dist_backend='nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    viola.set_master_weight_dtype(args.master, rounding=args.rounding, seed=71)
    viola.set_gradient_dtype(args.gradient)
    viola.enable_fp8_muon(ns_method='standard', adam_state_dtype=args.adam_state)
    probe = Probe(DTYPES[args.gradient])

    def make_engine():
        torch.manual_seed(123)
        cfg = config(args.stage)
        cfg['zero_optimization']['overlap_comm'] = True
        cfg['zero_optimization']['reduce_bucket_size'] = 8192  # bucketed + large matrix paths
        engine = deepspeed.initialize(model=PrecisionMatrices(), config=cfg)[0]
        zero = engine.optimizer
        assert zero.gradient_accumulation_dtype == DTYPES[args.gradient]
        assert zero.communication_data_type == DTYPES[args.gradient]
        assert all(p.grad_dtype == DTYPES[args.gradient] for p in engine.module.parameters())
        assert all(p.dtype == DTYPES[args.master] for p in zero.single_partition_of_fp32_groups)
        setup, copy_grads = zero.setup_buckets, zero.copy_grads_in_partition

        def checked_setup():
            setup()
            for bucket in zero.ipg_buckets.values():
                assert all(t.dtype == DTYPES[args.gradient] for t in bucket.buffer)

        def checked_copy(param):
            copy_grads(param)
            assert zero.grads_in_partition.dtype == DTYPES[args.gradient]

        zero.setup_buckets = checked_setup
        zero.copy_grads_in_partition = checked_copy
        zero._viola_observer = probe
        return engine

    engine = make_engine()

    def update(step, manual=True):
        expected = torch.zeros((), device='cuda', dtype=DTYPES[args.gradient])
        for micro in range(2):
            value = .015625 * (rank + 1) * (step + micro + 1) + .0000137
            g = torch.tensor(value, device='cuda') / 2
            if not manual:
                g = g.to(torch.bfloat16)
            g = g.to(DTYPES[args.gradient]) / world
            dist.all_reduce(g)
            expected.add_(g)
            probe.expected = expected.item()
            grads = [torch.full(p.shape, value, dtype=torch.float32, device=p.device)
                     for p in engine.module.parameters()]
            loss = engine(grads, manual=manual)
            engine.backward(loss)
            engine.step()
        for group in engine.optimizer.optimizer.param_groups:
            for param in group['params']:
                state = engine.optimizer.optimizer.state[param]
                if group['use_muon']:
                    assert state['viola_fp8_muon']['q'].dtype == torch.uint8
                else:
                    assert state['exp_avg'].dtype == DTYPES[args.adam_state]
                    assert state['exp_avg_sq'].dtype == DTYPES[args.adam_state]

    def snapshot():
        return copy.deepcopy(dict(
            masters=[p.detach().cpu() for p in engine.optimizer.single_partition_of_fp32_groups],
            state=engine.optimizer.optimizer.state_dict()))

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    for step in range(4):
        update(step, manual=step != 1)
    output = Path(args.output) / f'zero{args.stage}-{args.master}-{args.gradient}-{args.adam_state}-{args.rounding}'
    engine.save_checkpoint(str(output), tag='resume')
    saved = snapshot()
    update(4)
    expected = snapshot()
    del engine
    engine = make_engine()
    engine.load_checkpoint(str(output), tag='resume')
    equal(saved, snapshot())
    update(4)
    equal(expected, snapshot())
    legacy = copy.deepcopy(engine.optimizer.optimizer.state_dict())
    for group in legacy['param_groups']:
        group.pop('viola_precision')
        group.pop('viola_master_key')
    engine.optimizer.optimizer.load_state_dict(legacy)
    equal(expected, snapshot())
    for param in engine.module.parameters():
        reference = param.detach().clone()
        dist.broadcast(reference, src=0)
        assert torch.equal(reference, param)
    result = dict(vars(args), rank=rank, world=world, passed=True, checkpoint_bit_exact=True,
                  raw_gradient_checks=probe.calls, model_dtype='bf16', legacy_precision_metadata=True)
    (output / f'rank{rank}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
