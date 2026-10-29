"""Explicitly enabled FP8 Muon kernels for DeepSpeed."""

__version__ = "0.1.0"
from .precision import set_gradient_dtype, set_master_weight_dtype

__all__ = ["enable_fp8_muon", "set_master_weight_dtype", "set_gradient_dtype"]


def enable_fp8_muon(*, mode="balanced", tile_size=None, granularity="tile", block_shape=None,
                    rounding="nearest", rounding_seed=0, ns_steps=None, ns_method=None,
                    skip_nonfinite=False, adam_state_dtype="fp32"):
    """Replace DeepSpeed's Muon / ZeRO-1/2 paths in the current process."""
    from .plugin import enable_fp8_muon as enable

    enable(mode=mode, tile_size=tile_size, granularity=granularity, block_shape=block_shape,
           rounding=rounding, rounding_seed=rounding_seed, ns_steps=ns_steps, ns_method=ns_method,
           skip_nonfinite=skip_nonfinite, adam_state_dtype=adam_state_dtype)
