"""Training precision policy, independent of module and optimizer-state formats."""

import torch

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_settings = dict(master_weight_dtype=None, gradient_dtype=None,
                 master_rounding="nearest", master_rounding_seed=0)


def _dtype(value):
    value = _DTYPES[value] if isinstance(value, str) else value
    assert value in _DTYPES.values()
    return value


def set_master_weight_dtype(dtype, *, rounding="nearest", seed=0):
    """Choose master storage/writeback for subsequent optim.bind_optimizer calls."""
    assert rounding in ("nearest", "stochastic") and 0 <= seed < (1 << 63)
    _settings.update(master_weight_dtype=_dtype(dtype), master_rounding=rounding,
                     master_rounding_seed=seed)


def set_gradient_dtype(dtype):
    """Choose parameter-gradient reception/accumulation for new optimizer bindings.

    FP8Muon also applies this policy when it is constructed without a binding.
    This does not change activation gradients or operator forward/backward math.
    Existing models, optimizers and third-party communication buffers are not
    rewritten. Configure the policy before constructing/binding the optimizer.
    """
    _settings["gradient_dtype"] = _dtype(dtype)


def settings():
    return dict(_settings)
