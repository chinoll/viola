"""Check stored BF16 history and its unrounded, same-step NS input."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from audit import FIELDS, advance_shadow, comparison, measure
from viola.kernels import dequantize, momentum_update, state_sizes


@torch.no_grad()
def main():
    torch.manual_seed(91)
    shape = (17, 31)
    beta = torch.tensor(.95, device="cuda")
    for granularity in ("tile", "block"):
        quant = dict(granularity=granularity, tile_size=256, block_shape=(16, 16))
        qn, sn = state_sizes(shape, **quant)
        for magnitude in (1., 1e-4, 1e-9):
            q = torch.zeros(qn, device="cuda", dtype=torch.uint8)
            scale = torch.ones(sn, device="cuda")
            sbf = torch.zeros(shape, device="cuda", dtype=torch.bfloat16).flatten()
            nq, ns = torch.empty_like(q), torch.empty_like(scale)
            u = torch.empty_like(sbf)
            for step in range(4):
                g = torch.randn(sbf.numel(), device="cuda") * magnitude
                newbf, reference_input = advance_shadow(g, sbf, beta)
                fresh = beta * sbf.float() + (1 - beta) * g
                torch.testing.assert_close(newbf, fresh.bfloat16())
                torch.testing.assert_close(reference_input, ((1 - beta) * g + beta * fresh).bfloat16())
                momentum_update(g, q, scale, nq, ns, u, beta=.95, matrix_shape=shape, **quant)
                metrics = measure(g, q, scale, nq, ns, u, newbf, reference_input, beta, shape, quant)
                start = FIELDS.index("fp8_bf16_error_squared_sum")
                expected = torch.stack(comparison(dequantize(nq, ns, shape, **quant), newbf.float()))
                torch.testing.assert_close(metrics[start:start + 6], expected, rtol=1e-4, atol=1e-7)
                sbf = newbf
                q, nq = nq, q
                scale, ns = ns, scale
            print(dict(granularity=granularity, gradient_scale=magnitude, reference="bf16", passed=True), flush=True)


if __name__ == '__main__':
    main()
