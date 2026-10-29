"""Single-GPU FP8 Linear training with master weights and gradient accumulation."""
import argparse

import torch
import viola


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("memory", "speed"), default="memory")
    parser.add_argument("--forward-dtype", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument("--backward-dtype", choices=("e4m3", "e5m2"), default="e5m2")
    args = parser.parse_args()
    torch.manual_seed(42)
    viola.set_master_weight_dtype("fp32")
    viola.set_gradient_dtype("bf16")
    model = viola.nn.Linear(256, 256, mode=args.mode, forward_dtype=args.forward_dtype,
                         backward_dtype=args.backward_dtype, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    viola.optim.bind_optimizer(model, optimizer)
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randn_like(x) * .3
    for step in range(100):
        optimizer.zero_grad()
        losses = []
        for inputs, targets in zip(x.chunk(4), target.chunk(4)):
            loss = (model(inputs).float() - targets.float()).square().mean()
            (loss / 4).backward()
            losses.append(loss.detach())
        # Updates the master, then requantizes W (and W.T in speed mode).
        optimizer.step()
        if step % 10 == 0:
            print(f"step={step + 1} loss={torch.stack(losses).mean().item():.5f}")


if __name__ == "__main__":
    main()
