"""Matched single-state FP8/BF16/FP16 DiT-B memory comparison, without shadow states."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from dit import DiT
import viola
import viola.plugin as plugin
from momentum16 import reference_partition



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', choices=('fp8', 'bf16', 'fp16'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    torch.set_num_threads(4)
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    import deepspeed
    from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
    deepspeed.init_distributed(dist_backend='nccl')
    viola.enable_fp8_muon(ns_method='standard')
    if args.dtype != 'fp8':
        plugin._partition = reference_partition(torch.bfloat16 if args.dtype == 'bf16' else torch.float16)
    model = DiT()
    muon, adam = [], []
    for name, p in model.named_parameters():
        (muon if name.startswith('blocks.') and p.ndim == 2 else adam).append(p)
    optimizer = MuonWithAuxAdam([
        dict(params=muon, use_muon=True, lr=1e-4, momentum=.95, weight_decay=.01,
             ns_method='standard', ns_steps=5, nesterov=True, adjust_lr_fn='match_rms_adamw'),
        dict(params=adam, use_muon=False, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=.01),
    ], adam_optimizer=torch.optim.AdamW)
    cfg=dict(train_micro_batch_size_per_gpu=64, gradient_accumulation_steps=1,
        bf16=dict(enabled=True, check_grad_overflow=True), data_types=dict(grad_accum_dtype='fp32'),
        gradient_clipping=0., zero_optimization=dict(stage=2, overlap_comm=False, reduce_bucket_size=8000000,
        allgather_bucket_size=8000000, parameter_alignment=True), zero_allow_untested_optimizer=True,
        steps_per_print=10000000)
    engine = deepspeed.initialize(model=model, optimizer=optimizer, config=cfg)[0]
    root = Path('/data/experiments/viola-fp8-muon/dit-pilot-data')
    n = json.loads((root/'metadata.json').read_text())['samples']
    data = np.memmap(root/'latents.f32', mode='r', dtype='float32', shape=(n, 2, 4, 32, 32))
    labels = np.memmap(root/'labels.i64', mode='r', dtype='int64', shape=(n,))
    times = []
    peak = 0
    for step in range(16):
        x = torch.from_numpy(data[step * 64:step * 64 + 64, 0].copy()).cuda() * .18215
        y = torch.from_numpy(labels[step * 64:step * 64 + 64].copy()).cuda()
        torch.manual_seed(10000 + step)
        eps = torch.randn_like(x)
        t = torch.randn(64, device='cuda').sigmoid()
        z = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * eps
        torch.cuda.synchronize()
        if step >= 4:
            torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        prediction = engine(z.bfloat16(), t, y)
        loss = (prediction.float() - (eps - x)).square().mean()
        engine.backward(loss); engine.step()
        torch.cuda.synchronize()
        assert not engine.optimizer.overflow and torch.isfinite(loss)
        assert not hasattr(engine.optimizer, '_viola_pending')
        if step >= 4:
            times.append(time.perf_counter() - begin)
            peak = max(peak, torch.cuda.max_memory_allocated())
    state_bytes = 0
    for state in engine.optimizer.optimizer.state.values():
        if 'viola_fp8_muon' in state:
            state_bytes += sum(state['viola_fp8_muon'][k].nbytes for k in ('q', 'scales'))
        if 'single_16bit_momentum' in state:
            state_bytes += state['single_16bit_momentum'].nbytes
    result = dict(momentum_dtype=args.dtype, model='DiT-B/4', parameters=sum(p.numel() for p in model.parameters()),
        muon_parameters=sum(p.numel() for p in muon), batch=64, world_size=1, stage=2,
        actual_updates=16, timed_updates=len(times), state_copies=1, momentum_state_bytes=state_bytes,
        peak_allocated_bytes=peak, end_allocated_bytes=torch.cuda.memory_allocated(),
        mean_step_seconds=sum(times)/len(times), final_loss=loss.item(), diagnostic_shadows=False,
        torch=torch.__version__, gpu=torch.cuda.get_device_name())
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
