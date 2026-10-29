"""DiT-B memory comparison with one FP8/BF16/FP16 state, without diagnostic shadows."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
import viola

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from dit import DiT
from momentum16 import Momentum16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp8", "bf16", "fp16"), required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20261029)
    model = DiT().cuda()
    muon, adam = [], []
    for name, p in model.named_parameters():
        (muon if name.startswith("blocks.") and p.ndim == 2 else adam).append(p)
    groups = [dict(params=muon, use_muon=True), dict(params=adam, use_muon=False)]
    optimizer = (viola.optim.FP8Muon(groups, lr=1e-4) if args.dtype == "fp8" else
                 Momentum16(groups, lr=1e-4, momentum_dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]))
    x = torch.randn(args.batch, 4, 32, 32, device="cuda")
    y = torch.randint(1000, (args.batch,), device="cuda")
    t = torch.randn(args.batch, device="cuda").sigmoid()
    times, peak = [], 0
    for step in range(16):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(x, t, y).float() - x).square().mean()
        loss.backward(); optimizer.step()
        torch.cuda.synchronize()
        if step >= 4:
            times.append(time.perf_counter() - begin)
            peak = max(peak, torch.cuda.max_memory_allocated())
    state_bytes = sum(s[k].nbytes for s in optimizer.state.values()
                      for k in ("momentum_q", "momentum_scale", "momentum_buffer") if k in s)
    result = dict(momentum_dtype=args.dtype, model="DiT-B/4", batch=args.batch,
                  momentum_state_bytes=state_bytes, peak_allocated_bytes=peak,
                  mean_step_seconds=sum(times)/len(times), final_loss=loss.item(),
                  input="synthetic tensors; allocation measurement only", diagnostic_shadows=False,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name())
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
