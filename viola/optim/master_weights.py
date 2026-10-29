"""Bind ordinary optimizers to real master values behind FP8 Linear weights."""

import weakref

import torch
from torch import nn

from ..nn import Linear
from ..precision import settings
from .muon import FP8Muon


class _MasterWeights:
    def __init__(self, model, optimizer):
        assert not optimizer.state, "Bind the optimizer before its first step or load_state_dict."
        config = settings()
        self.master_dtype = config["master_weight_dtype"] or torch.float32
        self.gradient_dtype = config["gradient_dtype"] or torch.float32
        self.rounding = config["master_rounding"]
        self.seed = config["master_rounding_seed"]
        self.optimizer = optimizer
        self.pairs = []
        self.active = []
        layers = {id(layer.weight): layer for layer in model.modules() if isinstance(layer, Linear)}
        masters = {}
        for group in optimizer.param_groups:
            if isinstance(optimizer, FP8Muon):
                group["master_rounding"] = self.rounding
                group["master_rounding_seed"] = self.seed
            params = []
            for param in group["params"]:
                param.grad_dtype = self.gradient_dtype
                if id(param) not in masters:
                    layer = layers.get(id(param))
                    if layer is not None:
                        initial = layer._initial_weight
                        if initial is None:
                            # Model-only checkpoint; full training resumes overwrite
                            # this with the original master from optimizer.state_dict.
                            initial = layer.weight.float() * layer.weight_scale
                        master = nn.Parameter(initial.to(self.master_dtype), requires_grad=param.requires_grad)
                        layer._initial_weight = None
                        layer._master_ref = weakref.ref(master)
                    elif param.dtype != self.master_dtype:
                        master = nn.Parameter(param.detach().to(self.master_dtype), requires_grad=param.requires_grad)
                    else:
                        master = param
                    master.grad_dtype = self.gradient_dtype
                    self.pairs.append((param, master, layer))
                    masters[id(param)] = master
                params.append(masters[id(param)])
            group["params"] = params
        optimizer.register_step_pre_hook(self.before_step)
        optimizer.register_step_post_hook(self.after_step)
        optimizer.register_state_dict_post_hook(self.save_masters)
        optimizer.register_load_state_dict_pre_hook(self.load_masters)
        optimizer.register_load_state_dict_post_hook(self.after_load)
        original_zero_grad = optimizer.zero_grad

        def zero_grad(set_to_none=True):
            original_zero_grad(set_to_none=set_to_none)
            for param, master, _ in self.pairs:
                if param is not master and param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.detach_()
                        param.grad.zero_()
        optimizer.zero_grad = zero_grad

    @torch.no_grad()
    def before_step(self, optimizer, args, kwargs):
        assert len(args) == 1 and kwargs.get("closure") is None, "The master binding uses explicit forward/backward, without an optimizer closure."
        self.active = []
        for param, master, layer in self.pairs:
            if param.grad is not None:
                master.grad = param.grad  # Alias the accumulator: no second gradient buffer.
                self.active.append((param, master, layer, param.grad))

        # Native AdamW lerp_ requires identical master/gradient dtypes. For the
        # independently configured path, reuse Viola's fused TileLang update;
        # do not upcast the entire BF16 gradient or keep an FP32 gradient shadow.
        for group in optimizer.param_groups:
            for master in group["params"]:
                if master.grad is None:
                    continue
                mixed = master.dtype != master.grad.dtype or self.rounding != "nearest"
                if mixed and type(optimizer) is torch.optim.AdamW:
                    assert not any(group.get(key, False) for key in ("amsgrad", "capturable", "differentiable", "fused"))
                    from ._kernels import adam_update
                    state = optimizer.state[master]
                    if not state:
                        state["step"] = torch.tensor(0.)
                        state["exp_avg"] = torch.zeros_like(master)
                        state["exp_avg_sq"] = torch.zeros_like(master)
                    state["step"] += 1
                    adam_update(master, master.grad, state["exp_avg"], state["exp_avg_sq"],
                                lr=group["lr"], weight_decay=group["weight_decay"],
                                betas=group["betas"], eps=group["eps"], step=int(state["step"]),
                                rounding=self.rounding, key=self.seed,
                                maximize=group.get("maximize", False))
                    master.grad = None  # This parameter was updated; native AdamW skips it.
                elif mixed and type(optimizer) is torch.optim.SGD:
                    assert self.rounding == "nearest"
                    group["foreach"] = False

    @torch.no_grad()
    def after_step(self, optimizer, args, kwargs):
        skipped = isinstance(optimizer, FP8Muon) and optimizer.step_was_skipped
        for param, master, layer, gradient in self.active:
            if not skipped and layer is not None:
                layer.refresh_weight(master)
            elif not skipped and param is not master:
                param.copy_(master)
            if param is not master:
                master.grad = None
            else:
                param.grad = gradient
        self.active.clear()

    def save_masters(self, optimizer, state):
        state["viola_masters"] = {
            "weights": [master.detach() for _, master, _ in self.pairs],
            "gradient_dtype": self.gradient_dtype,
            "master_dtype": self.master_dtype,
            "rounding": self.rounding,
            "seed": self.seed,
        }

    @torch.no_grad()
    def load_masters(self, optimizer, state):
        state = dict(state)
        saved = state.pop("viola_masters")
        assert saved["master_dtype"] == self.master_dtype and saved["gradient_dtype"] == self.gradient_dtype
        assert len(saved["weights"]) == len(self.pairs)
        self.rounding, self.seed = saved["rounding"], saved["seed"]
        for (_, master, _), value in zip(self.pairs, saved["weights"]):
            master.copy_(value)
        return state

    @torch.no_grad()
    def after_load(self, optimizer):
        # The model checkpoint contains the exact quantized payload, scales and
        # histories. Do not quantize again here: delayed scaling would advance
        # its history and change the restored model before its next update.
        for param, master, layer in self.pairs:
            if layer is None and param is not master:
                param.copy_(master)
        optimizer.zero_grad()


def bind_optimizer(model, optimizer):
    """Install master writeback hooks; return the same optimizer instance.

    Move the model to its training device before binding. Save/load both model
    and optimizer state_dicts for an exact training resume. Call the independent
    precision setters before binding (defaults: FP32 master and gradient).
    """
    assert not hasattr(optimizer, "_viola_linear_masters"), "This optimizer is already bound."
    optimizer._viola_linear_masters = _MasterWeights(model, optimizer)
    return optimizer
