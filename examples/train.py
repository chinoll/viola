"""Small Accelerate example; use --zero-stage 1 or 2."""

import argparse

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim

import viola


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--master-weight-dtype", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--gradient-dtype", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--master-rounding", choices=("nearest", "stochastic"), default="nearest")
    parser.add_argument("--adam-state-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--rounding", default="nearest", choices=("nearest", "stochastic"))
    parser.add_argument("--rounding-seed", type=int, default=0)
    parser.add_argument("--granularity", default="tile", choices=("tile", "block"))
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--block-shape", type=int, nargs=2)
    args = parser.parse_args()

    if args.master_weight_dtype is not None:
        viola.set_master_weight_dtype(args.master_weight_dtype, rounding=args.master_rounding)
    if args.gradient_dtype is not None:
        viola.set_gradient_dtype(args.gradient_dtype)
    viola.enable_fp8_muon(rounding=args.rounding, rounding_seed=args.rounding_seed,
        granularity=args.granularity, tile_size=args.tile_size, block_shape=args.block_shape,
        adam_state_dtype=args.adam_state_dtype)
    config = {
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "data_types": {"grad_accum_dtype": "fp32"},
        "gradient_clipping": 0.0,
        "zero_optimization": {"stage": args.zero_stage, "overlap_comm": False,
                              "reduce_bucket_size": 1048576, "allgather_bucket_size": 1048576},
        "optimizer": {"type": "Muon", "params": {
            "lr": 1e-3, "momentum": .95, "weight_decay": .01,
            "adjust_lr_fn": "match_rms_adamw", "ns_method": "standard", "torch_adam": True,
        }},
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 1000000,
    }
    accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.SiLU(), torch.nn.Linear(256, 64))
    optimizer = DummyOptim(model.parameters())
    model, optimizer = accelerator.prepare(model, optimizer)
    for step in range(args.steps):
        x = torch.randn(8, 128, device=accelerator.device, dtype=torch.bfloat16)
        loss = (model(x).float() - x[:, :64].float()).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if step % 5 == 0:
            accelerator.print(f"step={step} loss={loss.item():.5f}")


if __name__ == "__main__":
    main()
