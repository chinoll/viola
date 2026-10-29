"""Independent weight/state precision, FP32 arithmetic and stochastic writeback."""
import itertools
import json
import torch
from viola.optim._kernels import adam_update, weight_update


def assert_storage_rounding(actual, value):
    target = value.to(actual.dtype)
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, target, rtol=2e-6, atol=1e-9)
    else:
        # One representable neighbor, including FP16 subnormal spacing.
        upper = torch.nextafter(target, torch.full_like(target, float('inf')))
        lower = torch.nextafter(target, torch.full_like(target, -float('inf')))
        ulp = torch.maximum(upper.float() - target.float(), target.float() - lower.float())
        error = (actual.float() - target.float()).abs()
        assert (error <= ulp).all(), (actual.dtype, (error / ulp).max().item())


@torch.no_grad()
def main():
    torch.manual_seed(107)
    dtypes = (torch.float32, torch.bfloat16, torch.float16)
    report = []
    for wdtype, gdtype in itertools.product(dtypes, repeat=2):
        w = (torch.randn(1030, device='cuda') * .1).to(wdtype)[1:1026]
        g = (torch.randn(1030, device='cuda') * .01).to(gdtype)[1:1026]
        target = (w.float() * (1 - .001 * .01) - .001 * g.float()).to(wdtype)
        weight_update(w, g, .001, .01)
        assert_storage_rounding(w, target)
        for sdtype in dtypes:
            m, v = torch.zeros_like(w, dtype=sdtype), torch.zeros_like(w, dtype=sdtype)
            for step in range(1, 4):
                # FP32 temporary arithmetic; only stored moments and final weight round.
                b1, b2 = torch.tensor(.9, device='cuda'), torch.tensor(.95, device='cuda')
                rm = b1 * m.float() + (1 - b1) * g.float()
                rv = b2 * v.float() + (1 - b2) * g.float().square()
                value = w.float() * (1 - .001 * .01) - .001 * (rm / (1 - .9**step)) / ((rv / (1 - .95**step)).sqrt() + 1e-8)
                adam_update(w, g, m, v, lr=.001, weight_decay=.01, betas=(.9,.95), eps=1e-8, step=step)
                # Fused FP32 arithmetic can select the other neighbor at a
                # BF16/FP16 midpoint; permit one storage ULP, not a fixed percent.
                assert_storage_rounding(m, rm)
                assert_storage_rounding(v, rv)
                assert_storage_rounding(w, value)
            report.append(dict(master=str(wdtype), gradient=str(gdtype), adam_state=str(sdtype), passed=True))

    for dtype in dtypes[1:]:
        # One quarter of the spacing below 1: nearest always drops the update;
        # stochastic rounding must retain it about 25% of the time, without a residual.
        n = 131072
        spacing = 2.**(-8 if dtype == torch.bfloat16 else -11)
        w = torch.ones(n, device='cuda', dtype=dtype)
        g = torch.ones(n, device='cuda')
        rng = torch.cuda.get_rng_state()
        weight_update(w, g, spacing / 4, 0., rounding='stochastic', step=17, key=29)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        fraction = (w != 1).float().mean().item()
        assert abs(fraction - .25) < .005
        replay = torch.ones_like(w)
        cut = 13001
        weight_update(replay[:cut], g[:cut], spacing/4, 0., rounding='stochastic', step=17, key=29)
        weight_update(replay[cut:], g[cut:], spacing/4, 0., rounding='stochastic', step=17, key=29, offset=cut)
        assert torch.equal(w, replay)
        report.append(dict(master=str(dtype), stochastic_fraction=fraction, partition_replay=True))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
