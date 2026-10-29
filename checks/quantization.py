"""Quantization geometry, rounding probabilities, replay, and state memory."""
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

from viola.kernels import _ROUNDING_SOURCE, dequantize, momentum_update, state_sizes


@tilelang.jit
def philox_answers():
    @T.prim_func
    def kernel(result: T.Tensor((3,), "uint32")):
        with T.Kernel(1, threads=32):
            T.import_source(_ROUNDING_SOURCE)
            result[0] = T.call_extern("uint32", "viola_philox", T.uint64(0), T.uint64(0), T.uint64(0))
            result[1] = T.call_extern("uint32", "viola_philox", T.uint64(0xffffffffffffffff),
                                     T.uint64(0xffffffffffffffff), T.uint64(0xffffffffffffffff))
            result[2] = T.call_extern("uint32", "viola_philox", T.uint64(0x85a308d3243f6a88),
                                     T.uint64(0x0370734413198a2e), T.uint64(0x299f31d0a4093822))
    return kernel


def expected_scales(m, shape, quant):
    if quant["granularity"] == "tile":
        b = quant["tile_size"]
        grouped = F.pad(m.flatten(), (0, (-m.numel()) % b)).view(-1, b)
    else:
        rows, cols = shape[-2:]
        tr, tc = quant["block_shape"]
        padded = F.pad(m.view(-1, rows, cols), (0, (-cols) % tc, 0, (-rows) % tr))
        grouped = padded.view(-1, padded.shape[1] // tr, tr, padded.shape[2] // tc, tc)
        grouped = grouped.permute(0, 1, 3, 2, 4).reshape(-1, tr * tc)
    return grouped.abs().amax(1).clamp_min(1e-30) / 448


def reference_quantize(m, shape, quant, rounding, step, key):
    """PyTorch layout/rounding oracle with an independent CPU Philox stream."""
    scales = expected_scales(m, shape, quant)
    qn, _ = state_sizes(shape, **quant)
    if quant["granularity"] == "tile":
        expanded = scales.repeat_interleave(quant["tile_size"])
        values = F.pad(m.flatten(), (0, qn - m.numel())) / expanded
    else:
        rows, cols = shape[-2:]
        tr, tc = quant["block_shape"]
        expanded = scales.view(-1, (rows + tr - 1) // tr, (cols + tc - 1) // tc)
        expanded = expanded.repeat_interleave(tr, 1).repeat_interleave(tc, 2)[:, :rows, :cols]
        values = m.flatten() / expanded.flatten()
    values = values.clamp(-448, 448)
    if rounding == "stochastic":
        a = np.arange(qn, dtype=np.uint64)
        b = np.zeros_like(a)
        c = np.full_like(a, step & 0xffffffff)
        d = np.full_like(a, step >> 32)
        k0, k1 = key & 0xffffffff, key >> 32
        for _ in range(10):
            p0, p1 = a * 0xD2511F53, c * 0xCD9E8D57
            a, b, c, d = ((p1 >> 32) ^ b ^ k0, p1 & 0xffffffff,
                          (p0 >> 32) ^ d ^ k1, p0 & 0xffffffff)
            k0, k1 = (k0 + 0x9E3779B9) & 0xffffffff, (k1 + 0xBB67AE85) & 0xffffffff
        uniform = torch.from_numpy((a >> 8).astype(np.float32) * (2. ** -24)).to(m.device)
        table = torch.arange(127, device=m.device, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        idx = torch.searchsorted(table, values.abs()).clamp_max(126)
        lower, upper = table[(idx - 1).clamp_min(0)], table[idx]
        probability = (values.abs() - lower) / (upper - lower).clamp_min(1e-30)
        values = torch.where(uniform < probability, upper, lower).copysign(values)
    return values.to(torch.float8_e4m3fn).view(torch.uint8), scales


@torch.no_grad()
def main():
    torch.manual_seed(79)
    answers = torch.empty(3, dtype=torch.uint32, device="cuda")
    philox_answers()(answers)
    # Random123's published known-answer vectors, independent of our kernel.
    # https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    assert answers.cpu().tolist() == [0x6627e8d5, 0x408f276d, 0xd16cfe09]
    report = []
    for granularity, size in [("tile", b) for b in (64, 128, 256, 512, 1024)] + [
            ("block", t) for t in ((8, 8), (8, 32), (16, 16), (16, 32), (32, 32), (64, 64))]:
        quant = dict(granularity=granularity, tile_size=size if granularity == "tile" else None,
                     block_shape=size if granularity == "block" else None)
        for shape in ((17, 31), (2, 33, 67)):
            qn, sn = state_sizes(shape, **quant)
            for rounding in ("nearest", "stochastic"):
                # Offset every buffer: ZeRO slices need not be 16-byte aligned.
                q = torch.zeros(qn + 1, dtype=torch.uint8, device="cuda")[1:]
                scale = torch.ones(sn + 1, device="cuda")[1:]
                n = math.prod(shape)
                sbf = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
                u = torch.empty(n + 1, dtype=torch.bfloat16, device="cuda")[1:]
                worst = 0.
                for step in range(8):
                    g = torch.randn(n + 1, device="cuda")[1:] * (1e-8 if step < 3 else 1.)
                    old = dequantize(q, scale, shape, **quant)
                    beta = torch.tensor(.95, device="cuda")
                    fresh = beta * old + (1 - beta) * g
                    expected_u = ((1 - beta) * g + beta * fresh).bfloat16()
                    momentum_update(g, q, scale, q, scale, u, beta=.95, matrix_shape=shape,
                                    rounding=rounding, rounding_step=step, rounding_key=123, **quant)
                    torch.testing.assert_close(scale, expected_scales(fresh, shape, quant), rtol=2e-6, atol=1e-30)
                    torch.testing.assert_close(u.float(), expected_u.float(), rtol=.01, atol=2e-7)
                    current = dequantize(q, scale, shape, **quant)
                    assert ((current - fresh).norm() / fresh.norm()).item() < .06
                    sbf = (beta * sbf.float() + (1 - beta) * g).bfloat16()
                    worst = max(worst, ((current - sbf.float()).norm() / sbf.float().norm()).item())
                report.append(dict(granularity=granularity, size=size, shape=shape, rounding=rounding,
                    max_relative_l2_vs_bf16=worst, state_bytes=qn + 4 * sn))

    # Known FP8 interval [1, 1.125], with 27% probability of rounding upward.
    # Every 1-D tile contains an exact endpoint, keeping its scale exactly one.
    n = 256 * 1024
    g = torch.full((n,), 1 + .27 * .125, device="cuda")
    g[::256] = 448
    q = torch.zeros(n, dtype=torch.uint8, device="cuda")
    scales = torch.ones(n // 256, device="cuda")
    out = torch.empty_like(g, dtype=torch.bfloat16)
    random_state = torch.cuda.get_rng_state()
    def draw(step=7, key=42):
        q.zero_(); scales.fill_(1)
        momentum_update(g, q, scales, q, scales, out, beta=0., rounding="stochastic",
                        rounding_step=step, rounding_key=key, nesterov=False)
        return q.clone()
    first = draw()
    assert torch.equal(first, draw())
    assert not torch.equal(first, draw(step=8))
    assert not torch.equal(first, draw(key=43))
    assert torch.equal(random_state, torch.cuda.get_rng_state())
    values = first.view(torch.float8_e4m3fn).float().view(-1, 256)[:, 1:]
    assert ((values == 1.) | (values == 1.125)).all()
    up = (values == 1.125).float().mean().item()
    assert abs(up - .27) < .005, up

    # Both signs, subnormals, exact values, and exponent boundaries: SR must
    # choose one of the immediately adjacent representable E4M3 values.
    table = torch.arange(127, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    values = (table[:-1] * .7 + table[1:] * .3).repeat(1024)
    g = torch.cat((values, -values, table.repeat(1024)))
    g[::256] = 448
    qn, sn = state_sizes(g.shape)
    q = torch.zeros(qn, dtype=torch.uint8, device="cuda")
    scales = torch.ones(sn, device="cuda")
    out = torch.empty_like(g, dtype=torch.bfloat16)
    momentum_update(g, q, scales, q, scales, out, beta=0., rounding="stochastic", nesterov=False)
    actual = dequantize(q, scales, g.shape)
    idx = torch.searchsorted(table, g.abs()).clamp_max(126)
    lower, upper = table[(idx - 1).clamp_min(0)], table[idx]
    assert ((actual.abs() == lower) | (actual.abs() == upper)).all()
    assert ((actual.sign() == g.sign()) | (actual == 0)).all()

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    momentum_update(g, q, scales, q, scales, out, beta=0., rounding="stochastic", nesterov=False)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - allocated
    assert extra == 0, extra
    report.append(dict(philox_known_answers=True, stochastic_round_up_probability=up,
                       expected_probability=.27, deterministic_replay=True,
                       random_tensor_allocated=False, kernel_extra_peak_bytes=extra))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
