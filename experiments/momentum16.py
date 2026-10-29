"""Diagnostic BF16/FP16 momentum reference with the same standalone Muon step."""
import torch
import tilelang
import tilelang.language as T

from viola.optim import FP8Muon


@tilelang.jit
def update16(n, state_dtype, grad_dtype, nesterov):
    @T.prim_func
    def kernel(g: T.Tensor((n,), grad_dtype), m: T.Tensor((n,), state_dtype),
               out: T.Tensor((n,), "bfloat16"), beta: T.float32):
        with T.Kernel(T.ceildiv(n, 1024), threads=128) as block:
            for j in T.Parallel(1024):
                i = block * 1024 + j
                if i < n:
                    grad = T.cast(g[i], "float32")
                    fresh = beta * T.cast(m[i], "float32") + (1.0 - beta) * grad
                    out[i] = (1.0 - beta) * grad + beta * fresh if nesterov else fresh
                    m[i] = fresh
    return kernel


class Momentum16(FP8Muon):
    def __init__(self, params, *, momentum_dtype=torch.bfloat16, **kwargs):
        self.momentum_dtype = momentum_dtype
        super().__init__(params, **kwargs)

    def _momentum(self, param, grad, state, group, key):
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param, dtype=self.momentum_dtype)
        shape = tuple(param.shape) if group["batched"] else (param.shape[0], param.numel() // param.shape[0])
        work = self._workspace.get(param.device)
        if work is None or work.numel() < param.numel():
            work = torch.empty(param.numel(), device=param.device, dtype=torch.bfloat16)
            self._workspace[param.device] = work
        update16(param.numel(), str(self.momentum_dtype).split('.')[-1], str(grad.dtype).split('.')[-1], group["nesterov"])(
            grad.reshape(-1), state["momentum_buffer"].view(-1), work[:param.numel()], group["momentum"])
        return work[:param.numel()].view(shape)
