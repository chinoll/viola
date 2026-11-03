"""Bind optimizers to master weights with independent model storage precision."""

import weakref

import torch
from torch import nn

from ..nn import Linear
from ..precision import settings
from .matrix import MatrixOptimizer


@torch.no_grad()
def step_mixed_adamw(optimizer, precision):
    """Update mixed-dtype AdamW entries; let native step handle other entries."""
    restored = []
    rounding = precision["master_rounding"]
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            mixed = gradient.dtype != parameter.dtype or rounding != "nearest"
            if mixed and type(optimizer) is torch.optim.AdamW:
                assert not any(group.get(key, False) for key in ("amsgrad", "capturable", "differentiable", "fused"))
                from ._kernels import adam_update
                state = optimizer.state[parameter]
                if not state:
                    state.update(step=torch.tensor(0.), exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state["step"] += 1
                adam_update(parameter, gradient, state["exp_avg"], state["exp_avg_sq"],
                            lr=group["lr"], weight_decay=group["weight_decay"],
                            betas=group["betas"], eps=group["eps"], step=int(state["step"]),
                            rounding=rounding, key=precision["master_rounding_seed"],
                            maximize=group.get("maximize", False))
                restored.append((parameter, gradient))
                parameter.grad = None
            elif mixed and type(optimizer) is torch.optim.SGD:
                assert rounding == "nearest"
                group["foreach"] = False
    return restored


class _MasterWeights:
    def __init__(self, model, optimizer):
        assert not optimizer.state, "Bind the optimizer before its first step or load_state_dict."
        config = settings()
        self.master_dtype = config["master_weight_dtype"]
        self.gradient_dtype = config["gradient_dtype"] or torch.float32
        self.rounding = config["master_rounding"]
        self.seed = config["master_rounding_seed"]
        self.optimizer = optimizer
        self.pairs = []
        self.active = []
        layers = {id(layer.weight): layer for layer in model.modules() if isinstance(layer, Linear)}
        masters = {}
        for group in optimizer.param_groups:
            if isinstance(optimizer, MatrixOptimizer):
                group["master_rounding"] = self.rounding
                group["master_rounding_seed"] = self.seed
            params = []
            for param in group["params"]:
                param.grad_dtype = self.gradient_dtype
                if id(param) not in masters:
                    layer = layers.get(id(param))
                    if layer is not None and layer.weight_dtype == "fp8":
                        assert self.master_dtype is not None, \
                            "Disabling the master requires BF16/FP16/FP32 weight storage."
                        initial = layer._initial_weight
                        if initial is None:
                            # Model-only checkpoint; full training resumes overwrite
                            # this with the original master from optimizer.state_dict.
                            initial = layer.dequantize_weight()
                        master = nn.Parameter(initial.to(self.master_dtype), requires_grad=param.requires_grad)
                    elif self.master_dtype is not None and param.dtype != self.master_dtype:
                        master = nn.Parameter(param.detach().to(self.master_dtype), requires_grad=param.requires_grad)
                    else:
                        master = param
                    if layer is not None:
                        layer._initial_weight = None
                        layer._master_ref = weakref.ref(master)
                    master.grad_dtype = self.gradient_dtype
                    self.pairs.append((param, master, layer))
                    masters[id(param)] = master
                params.append(masters[id(param)])
            group["params"] = params
        if isinstance(optimizer, MatrixOptimizer):
            optimizer._partition = None
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

        step_mixed_adamw(optimizer, dict(master_rounding=self.rounding, master_rounding_seed=self.seed))

    @torch.no_grad()
    def after_step(self, optimizer, args, kwargs):
        skipped = isinstance(optimizer, MatrixOptimizer) and optimizer.step_was_skipped
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
            # Aliases are already in the model checkpoint, including the
            # master-disabled path. Do not serialize a second weight copy.
            "weights": [master.detach() if master is not param else None
                        for param, master, _ in self.pairs],
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
            if value is not None:
                master.copy_(value)
        return state

    @torch.no_grad()
    def after_load(self, optimizer):
        # The model checkpoint contains the exact quantized payload, scales and
        # histories. Do not quantize again here: delayed scaling would advance
        # its history and change the restored model before its next update.
        for param, master, layer in self.pairs:
            if (layer is None or layer.weight_dtype != "fp8") and param is not master:
                param.copy_(master)
        optimizer.zero_grad()


def bind_optimizer(model, optimizer):
    """Install master writeback hooks; return the same optimizer instance.

    Move the model to its training device before binding. Save/load both model
    and optimizer state_dicts for an exact training resume. Call the independent
    precision setters before binding (defaults: FP32 master and gradient).
    set_master_weight_dtype(None) aliases the model parameter instead of
    allocating a master; FP8 compute with BF16/FP16 weight storage still works.
    """
    assert not hasattr(optimizer, "_viola_linear_masters"), "This optimizer is already bound."
    optimizer._viola_linear_masters = _MasterWeights(model, optimizer)
    return optimizer
