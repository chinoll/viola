"""Standalone FP8-momentum Muon with optional AdamW parameter groups."""

import os
import warnings

import torch

from ..precision import _dtype, settings
from ._kernels import adam_update, momentum_update, state_sizes, weight_update
from ._orthogonalize import METHODS

_MODES = {"fast": (256, 3), "balanced": (256, 5), "stable": (128, 5)}


def _shape(param, group):
    return tuple(param.shape) if group["batched"] else (param.shape[0], param.numel() // param.shape[0])


def _use_muon(param, group):
    return param.ndim >= 2 if group["use_muon"] is None else group["use_muon"]


class FP8Muon(torch.optim.Optimizer):
    """One FP8 momentum buffer plus FP32 scales; ordinary Optimizer.step().

    Matrices use Muon and vectors use AdamW by default. Set ``use_muon=False``
    on an embedding/output parameter group to use AdamW there as well. Conv
    filters flatten to [out_channels, -1]; ``batched=True`` treats the last two
    dimensions as independent expert matrices.
    """

    def __init__(self, params, lr=1e-3, *, weight_decay=0., momentum=.95,
                 nesterov=True, mode="balanced", granularity="tile", tile_size=None,
                 block_shape=None, rounding="nearest", rounding_seed=0,
                 ns_steps=None, ns_method="standard", adjust_lr_fn="match_rms_adamw",
                 betas=(.9, .95), eps=1e-8, adam_state_dtype="fp32", skip_nonfinite=False):
        tile_default, steps_default = _MODES[mode]
        policy = settings()
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            use_muon=None, batched=False, granularity=granularity,
            tile_size=tile_default if tile_size is None else tile_size,
            block_shape=(16, 16) if block_shape is None else tuple(block_shape),
            rounding=rounding, rounding_seed=rounding_seed,
            ns_steps=steps_default if ns_steps is None else ns_steps, ns_method=ns_method,
            adjust_lr_fn=adjust_lr_fn, betas=betas, eps=eps, adam_state_dtype=_dtype(adam_state_dtype),
            master_rounding=policy["master_rounding"], master_rounding_seed=policy["master_rounding_seed"])
        self.skip_nonfinite = skip_nonfinite
        self.step_was_skipped = False
        self.observer = None
        self._workspace = {}
        super().__init__(params, defaults)
        self.register_load_state_dict_pre_hook(self._before_load)
        self.register_load_state_dict_post_hook(self._after_load)

    def add_param_group(self, group):
        super().add_param_group(group)
        group = self.param_groups[-1]
        assert group["lr"] >= 0 and group["weight_decay"] >= 0
        assert 0 <= group["momentum"] < 1 and group["ns_steps"] > 0
        assert group["ns_method"] in METHODS
        assert group["adjust_lr_fn"] in ("original", "match_rms_adamw")
        assert group["granularity"] in ("tile", "block")
        assert group["rounding"] in ("nearest", "stochastic")
        assert 0 <= group["rounding_seed"] < (1 << 63)
        assert group["tile_size"] in (64, 128, 256, 512, 1024)
        assert len(group["block_shape"]) == 2 and all(v in (8, 16, 32, 64) for v in group["block_shape"])
        group["adam_state_dtype"] = _dtype(group["adam_state_dtype"])
        grad_dtype = settings()["gradient_dtype"]
        unaligned = []
        for param in group["params"]:
            if grad_dtype is not None:
                param.grad_dtype = grad_dtype
            if _use_muon(param, group):
                assert param.ndim >= 2
                shape = _shape(param, group)
                partial = (shape[-1] % group["tile_size"] if group["granularity"] == "tile"
                           else shape[-2] % group["block_shape"][0] or shape[-1] % group["block_shape"][1])
                if partial:
                    unaligned.append(shape)
        if unaligned and int(os.environ.get("RANK", "0")) == 0:
            warnings.warn(f"FP8Muon: {len(unaligned)} matrices have partial {group['granularity']} groups "
                          f"(e.g. {unaligned[:3]}). Original matrix shapes are kept; edge groups use valid elements.",
                          stacklevel=3)

    def _momentum(self, param, grad, state, group, key):
        shape = _shape(param, group)
        quant = {name: group[name] for name in ("granularity", "tile_size", "block_shape")}
        if "momentum_q" not in state:
            qn, sn = state_sizes(shape, **quant)
            state["momentum_q"] = torch.zeros(qn, device=param.device, dtype=torch.uint8)
            state["momentum_scale"] = torch.ones(sn, device=param.device, dtype=torch.float32)
        work = self._workspace.get(param.device)
        if work is None or work.numel() < param.numel():
            work = torch.empty(param.numel(), device=param.device, dtype=torch.bfloat16)
            self._workspace[param.device] = work
        matrix = work[:param.numel()].view(shape)
        q, scale = state["momentum_q"], state["momentum_scale"]
        old_q, old_scale = (q.clone(), scale.clone()) if self.observer is not None else (None, None)
        momentum_update(grad, q, scale, q, scale, matrix.view(-1), beta=group["momentum"],
                        nesterov=group["nesterov"], matrix_shape=shape, **quant,
                        rounding=group["rounding"], rounding_step=state["step"], rounding_key=key)
        if self.observer is not None:
            self.observer.momentum(param, grad, old_q, old_scale, q, scale, matrix, group, 1., quant)
        return matrix

    @torch.no_grad()
    def step(self, closure=None):
        self.step_was_skipped = False
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # No scan or skipped update by default: Inf/NaN propagates normally.
        if self.skip_nonfinite and any(not torch.isfinite(p.grad).all()
                for g in self.param_groups for p in g["params"] if p.grad is not None):
            self.step_was_skipped = True
            return loss
        index = 0
        for group in self.param_groups:
            for param in group["params"]:
                key = (group["rounding_seed"] + index * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)
                master_key = (group["master_rounding_seed"] + index * 0x517CC1B727220A95) & ((1 << 63) - 1)
                index += 1
                if param.grad is None:
                    continue
                grad = param.grad
                assert param.is_cuda and not grad.is_sparse
                assert param.dtype in (torch.float32, torch.bfloat16, torch.float16), \
                    "Bind FP8 modules with viola.optim.bind_optimizer before stepping."
                state = self.state[param]
                state.setdefault("step", 0)
                if _use_muon(param, group):
                    matrix = self._momentum(param, grad, state, group, key)
                    ns = METHODS[group["ns_method"]]
                    update = ns(matrix, steps=group["ns_steps"])
                    if self.observer is not None:
                        self.observer.update(param, update, group, ns)
                    rows, cols = matrix.shape[-2:]
                    scale = (.2 * max(rows, cols)**.5 if group["adjust_lr_fn"] == "match_rms_adamw"
                             else max(1., rows / cols)**.5)
                    # Scale the update, not decoupled weight decay.
                    weight_update(param, update, group["lr"], group["weight_decay"], update_scale=scale,
                                  rounding=group["master_rounding"], step=state["step"], key=master_key)
                    del update, matrix
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(param, dtype=group["adam_state_dtype"])
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=group["adam_state_dtype"])
                    adam_update(param, grad, state["exp_avg"], state["exp_avg_sq"],
                                lr=group["lr"], weight_decay=group["weight_decay"], betas=group["betas"],
                                eps=group["eps"], step=state["step"] + 1,
                                rounding=group["master_rounding"], key=master_key)
                state["step"] += 1
        return loss

    def _before_load(self, optimizer, saved):
        # PyTorch casts optimizer tensor states to parameter dtype. Remove the
        # packed bytes/scales/moments first, preserving their independent dtypes.
        names = {"momentum_q", "momentum_scale", "momentum_buffer", "exp_avg", "exp_avg_sq"}
        self._loading = {pid: {k: v for k, v in state.items() if k in names}
                         for pid, state in saved["state"].items()}
        self._loading_groups = saved["param_groups"]
        return dict(saved, state={pid: {k: v for k, v in state.items() if k not in names}
                                  for pid, state in saved["state"].items()})

    def _after_load(self, optimizer):
        for old, group in zip(self._loading_groups, self.param_groups):
            for pid, param in zip(old["params"], group["params"]):
                for name, value in self._loading.get(pid, {}).items():
                    self.state[param][name] = value.to(device=param.device)
        del self._loading, self._loading_groups
