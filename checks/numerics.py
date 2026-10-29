"""Check the fused arithmetic against PyTorch and report actual state storage."""

import json
import torch
import torch.nn.functional as F

from viola.kernels import copy_scaled, momentum_update, weight_update


def unpack(q, scales, tile_size, n):
    return (q.view(torch.float8_e4m3fn).float().view(-1, tile_size) * scales[:, None]).view(-1)[:n]


@torch.no_grad()
def main():
    torch.manual_seed(42)
    report = []
    for tile_size in (128, 256, 512):
        for n, dtype in ((13, torch.float32), (257, torch.bfloat16), (1537, torch.float16),
                         (8192, torch.bfloat16)):
            blocks = (n + tile_size - 1) // tile_size
            q = torch.zeros(blocks * tile_size, dtype=torch.uint8, device="cuda")
            scales = torch.ones(blocks, device="cuda")
            next_q, next_scales = torch.empty_like(q), torch.empty_like(scales)
            out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
            errors, cosines = [], []
            for step in range(12):
                grad = torch.randn(n, device="cuda", dtype=dtype)
                # Exercise loss unscaling and the non-Nesterov branch as well.
                loss_scale = 32.0 if dtype == torch.float16 else 1.0
                nesterov = step % 2 == 0
                g = grad.float() / loss_scale
                beta = torch.tensor(0.95, device="cuda")
                m = beta * unpack(q, scales, tile_size, n) + (1 - beta) * g
                expected = ((1 - beta) * g + beta * m if nesterov else m).bfloat16()
                padded_m = F.pad(m, (0, blocks * tile_size - n)).view(blocks, tile_size)
                expected_scale = padded_m.abs().amax(dim=1).clamp_min(1e-30) / 448
                momentum_update(grad, q, scales, next_q, next_scales, out,
                                beta=0.95, tile_size=tile_size, nesterov=nesterov, loss_scale=loss_scale)
                torch.testing.assert_close(next_scales, expected_scale, rtol=2e-6, atol=1e-30)
                torch.testing.assert_close(out.float(), expected.float(), rtol=0.01, atol=2e-6)
                error = (unpack(next_q, next_scales, tile_size, n) - m).norm() / m.norm()
                assert error.item() < 0.04
                errors.append(error.item())
                cosines.append(F.cosine_similarity(out.float(), expected.float(), dim=0).item())
                q, next_q = next_q, q
                scales, next_scales = next_scales, scales
            report.append(dict(n=n, dtype=str(dtype), tile_size=tile_size,
                               max_momentum_relative_error=max(errors), min_update_cosine=min(cosines)))

    # A zero gradient/state must remain exactly zero, including padded tails.
    q.zero_(); scales.fill_(1)
    momentum_update(torch.zeros_like(grad), q, scales, next_q, next_scales, out,
                    beta=0.95, tile_size=tile_size)
    assert torch.count_nonzero(next_q) == 0 and torch.count_nonzero(out) == 0

    # ZeRO views may have valid element alignment but no 16-byte alignment.
    n, tile_size = 1537, 256
    blocks = (n + tile_size - 1) // tile_size
    grad = torch.randn(n + 1, device="cuda", dtype=torch.bfloat16)[1:]
    q = torch.zeros(blocks * tile_size + 1, device="cuda", dtype=torch.uint8)[1:]
    scales = torch.ones(blocks + 1, device="cuda")[1:]
    next_q = torch.empty(blocks * tile_size + 1, device="cuda", dtype=torch.uint8)[1:]
    next_scales = torch.empty(blocks + 1, device="cuda")[1:]
    out = torch.empty(n + 1, device="cuda", dtype=torch.bfloat16)[1:]
    momentum_update(grad, q, scales, next_q, next_scales, out, beta=.95, tile_size=tile_size)
    torch.testing.assert_close(out.float(), (.0975 * grad.float()).bfloat16().float(), atol=2e-6, rtol=.01)

    master = torch.randn(1537, device="cuda")
    grad = torch.randn_like(master)
    reference = master * (1 - 1e-3 * 0.1) - 1e-3 * grad
    weight_update(master, grad, 1e-3, 0.1)
    torch.testing.assert_close(master, reference, atol=5e-7, rtol=2e-6)
    matrix = torch.randn(17, 31, device="cuda", dtype=torch.bfloat16).T
    partition = torch.zeros(257, device="cuda")
    copy_scaled(matrix, partition, 11, 3, 251, 3.2)
    torch.testing.assert_close(partition[3:254], matrix.reshape(-1)[11:262].float() * 3.2)

    # Warm the production in-place path before measuring its one stored state.
    n = 4 * 1024 * 1024
    tile_size = 256
    q = torch.zeros(n, device="cuda", dtype=torch.uint8)
    scales = torch.ones(n // tile_size, device="cuda")
    grad = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(grad)
    momentum_update(grad, q, scales, q, scales, out, beta=.95, tile_size=tile_size)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        momentum_update(grad, q, scales, q, scales, out, beta=.95, tile_size=tile_size)
    end.record(); end.synchronize()
    report.append(dict(
        benchmark_elements=n, kernel_ms=start.elapsed_time(end) / 50,
        fp8_state_bytes=sum(x.numel() * x.element_size() for x in (q, scales)),
        bf16_state_bytes=2 * n, fp32_state_bytes=4 * n,
        kernel_extra_peak_bytes=torch.cuda.max_memory_allocated() - allocated,
        gpu=torch.cuda.get_device_name(),
    ))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
