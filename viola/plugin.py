"""Process-local DeepSpeed hooks. Importing viola does not change DeepSpeed."""

import functools
import os
import warnings

import torch

from .kernels import adam_update, copy_scaled, momentum_update, state_sizes, weight_update
from .precision import _dtype, install_zero_precision, settings, step_with_independent_dtypes

_enabled_config = None
_STATE = "viola_fp8_muon"
_MODES = {
    "fast": dict(tile_size=256, ns_steps=3),
    "balanced": dict(tile_size=256, ns_steps=None),
    "stable": dict(tile_size=128, ns_steps=5),
}


def _sync_stage1_accumulators(zero, group_id, tensors):
    """ZeRO-1's legacy reducer reduces param.grad, not its FP32 accumulator."""
    from deepspeed import comm

    process_group = zero.real_dp_process_group[group_id]
    world = comm.get_world_size(group=process_group)
    gradients = {}
    for param in zero.bit16_groups[group_id]:
        grad = zero.get_param_gradient_attribute(param)
        if grad is None:
            grad = torch.zeros_like(param, dtype=zero.gradient_accumulation_dtype)
        comm.all_reduce(grad, group=process_group)
        grad.div_(world)
        gradients[id(param)] = grad
    zero.all_grad_tensors[group_id] = [gradients[id(param)] for param in tensors]


def _matrix_shape(param):
    return tuple(param.shape) if getattr(param, "is_expert_group", False) else (param.shape[0], param.numel() // param.shape[0])


def _restore_quantization_names(saved):
    """Version 1 named flat 1-D groups 'block' and 2-D groups 'tile'."""
    state = dict(saved)
    if state.get("quantization_version", 1) == 1:
        state["granularity"] = "tile" if state.get("granularity", "block") == "block" else "block"
        state["tile_size"] = state.pop("block_size", None)
        state["block_shape"] = state.pop("tile_shape", None)
        state["quantization_version"] = 2
    return state


def _warn_unaligned(shapes, quant):
    """Warn once per initialization, without changing any matrix dimensions."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    if quant["granularity"] == "tile":
        unaligned = [s for s in shapes if s[-1] % quant["tile_size"]]
        boundary = f"1-D tile_size={quant['tile_size']}"
        detail = "Flat contiguous tiles may cross row boundaries; a partial final tile uses valid elements only."
    else:
        br, bc = quant["block_shape"]
        unaligned = [s for s in shapes if s[-2] % br or s[-1] % bc]
        boundary = f"2-D block_shape={quant['block_shape']}"
        detail = "Partial edge blocks use valid elements only."
    if unaligned:
        warnings.warn(
            f"Viola FP8 Muon: {len(unaligned)} matrices are not aligned to {boundary}; "
            f"example shapes: {unaligned[:4]}. Keeping original matrix shapes. {detail}",
            UserWarning, stacklevel=3,
        )


def _state(zero, group_id, tensors, quant, rounding, device):
    group = zero.optimizer.param_groups[group_id]
    param = group["params"][0]
    state = zero.optimizer.state[param]
    numels = [p.numel() for p in tensors]
    shapes = [_matrix_shape(p) for p in tensors]
    sizes = [state_sizes(shape, **quant) for shape in shapes]
    if _STATE not in state:
        state[_STATE] = dict(
            q=torch.zeros(sum(q for q, _ in sizes), dtype=torch.uint8, device=device),
            scales=torch.ones(sum(s for _, s in sizes), dtype=torch.float32, device=device),
            numels=numels, shapes=shapes, **quant, rounding_step=0, quantization_version=2,
        )
    packed = state[_STATE]
    assert packed["numels"] == numels and packed["granularity"] == quant["granularity"], \
        "Resume with the same parameter partitioning and quantization layout."
    if quant["granularity"] == "tile":
        assert packed["tile_size"] == quant["tile_size"]
    else:
        assert packed["shapes"] == shapes and tuple(packed["block_shape"]) == quant["block_shape"]

    needed = max(numels)
    work = zero.__dict__.get("_viola_workspace")
    if work is None or work.numel() < needed:
        work = torch.empty(needed, dtype=torch.bfloat16, device=device)
        zero._viola_workspace = work

    # Converting an existing full-precision checkpoint is streamed per matrix;
    # there is never an additional full-model FP32 momentum copy.
    if "momentum_buffer" in state:
        previous = state.pop("momentum_buffer")
        offset = q_start = s_start = 0
        for tensor, shape, n, (qn, sn) in zip(tensors, shapes, numels, sizes):
            q = packed["q"][q_start:q_start + qn]
            scales = packed["scales"][s_start:s_start + sn]
            momentum_update(previous[offset:offset + n], q, scales, q, scales, work[:n],
                            beta=0.0, nesterov=False, matrix_shape=shape, **quant,
                            rounding=rounding, rounding_step=(1 << 63) - 1,
                            rounding_key=tensor._viola_rounding_key)
            offset += n
            q_start += qn
            s_start += sn
    return packed, sizes, work


def _raw_partition(zero, tensors, gradients, partition_size, dtype, device, group_id,
                   return_tensor_list):
    """Keep views of reduced gradients; no momentum update during backward."""
    from deepspeed import comm
    rank = comm.get_rank(group=zero.real_dp_process_group[group_id])
    pieces, cursor = [], 0
    for tensor, grad in zip(tensors, gradients):
        param_id = zero.get_param_id(tensor)
        src = int(zero.grad_start_offset[group_id][rank][param_id])
        dst = int(zero.grad_partition_insertion_offset[group_id][rank][param_id])
        count = min(tensor.numel() - src, int(partition_size) - dst)
        if dst > cursor:
            pad = torch.zeros(dst - cursor, dtype=dtype, device=device)
            pad._zero_padding = True
            pieces.append(pad)
        pieces.append(grad.reshape(-1)[src:src + count])
        cursor = dst + count
    if cursor < int(partition_size):
        pad = torch.zeros(int(partition_size) - cursor, dtype=dtype, device=device)
        pad._zero_padding = True
        pieces.append(pad)
    return pieces if return_tensor_list else zero.flatten(pieces)


def _partition(zero, tensors, partition_size, dtype, device, group_id, return_tensor_list,
               quant, rounding):
    from deepspeed import comm
    from deepspeed.runtime.zero.muon.original_muon import (
        zeropower_via_gram_newtonschulz, zeropower_via_newtonschulz5,
    )

    group = zero.optimizer.param_groups[group_id]
    gradients = zero.all_grad_tensors[group_id]
    packed, sizes, work = _state(zero, group_id, tensors, quant, rounding, device)
    rank = comm.get_rank(group=zero.real_dp_process_group[group_id])
    # FP16 loss scaling can exceed the representable range of an FP16 update.
    # Store scaled updates in FP32; DeepSpeed unscales them in its usual step.
    out_dtype = torch.float32 if gradients[0].dtype == torch.float16 else gradients[0].dtype
    flat = torch.zeros(int(partition_size), dtype=out_dtype, device=device)
    pieces = []
    cursor = q_offset = s_offset = 0
    rounding_step = packed.get("rounding_step", 0)
    ns = (zeropower_via_gram_newtonschulz if group.get("ns_method", "gram") == "gram"
          else zeropower_via_newtonschulz5)
    observer = getattr(zero, "_viola_observer", None)

    for tensor, grad, (q_count, s_count) in zip(tensors, gradients, sizes):
        n = tensor.numel()
        shape = _matrix_shape(tensor)
        q_slice = slice(q_offset, q_offset + q_count)
        s_slice = slice(s_offset, s_offset + s_count)
        # Only diagnostic mode needs the pre-update bytes. The optimizer itself
        # owns exactly one FP8 state and writes it in place at optimizer.step().
        old_q = packed["q"][q_slice].clone() if observer is not None else None
        old_scale = packed["scales"][s_slice].clone() if observer is not None else None
        momentum_update(
            grad, packed["q"][q_slice], packed["scales"][s_slice],
            packed["q"][q_slice], packed["scales"][s_slice], work[:n],
            beta=group["momentum"], matrix_shape=shape, **quant, rounding=rounding,
            rounding_step=rounding_step, rounding_key=tensor._viola_rounding_key,
            nesterov=group.get("nesterov", True), loss_scale=zero.loss_scale,
        )
        matrix = work[:n].view(shape)
        if observer is not None:
            observer.momentum(tensor, grad, old_q, old_scale,
                              packed["q"][q_slice], packed["scales"][s_slice], matrix,
                              group, zero.loss_scale, quant)
        update = ns(matrix, steps=group.get("ns_steps", 5))
        if observer is not None:
            observer.update(tensor, update, group, ns)
        rows, cols = shape[-2:]
        scaling = group.get("adjust_lr_fn", "match_rms_adamw")
        scale = 0.2 * max(rows, cols)**0.5 if scaling == "match_rms_adamw" else max(1.0, rows / cols)**0.5

        param_id = zero.get_param_id(tensor)
        src = int(zero.grad_start_offset[group_id][rank][param_id])
        dst = int(zero.grad_partition_insertion_offset[group_id][rank][param_id])
        count = min(n - src, int(partition_size) - dst)
        if dst > cursor:
            pad = flat[cursor:dst]
            pad._zero_padding = True
            pieces.append(pad)
        copy_scaled(update, flat, src, dst, count, scale * zero.loss_scale)
        pieces.append(flat[dst:dst + count])
        cursor = dst + count
        q_offset += q_count
        s_offset += s_count
        # Only this rank's slice survives; don't retain each full NS output.
        del update, matrix

    packed["rounding_step"] = rounding_step + 1
    if cursor < flat.numel():
        pad = flat[cursor:]
        pad._zero_padding = True
        pieces.append(pad)
    return pieces if return_tensor_list else flat


def enable_fp8_muon(*, mode="balanced", tile_size=None, granularity="tile", block_shape=None,
                    rounding="nearest", rounding_seed=0, ns_steps=None, ns_method=None,
                    skip_nonfinite=False, adam_state_dtype="fp32"):
    global _enabled_config
    defaults = _MODES[mode]
    assert granularity in ("tile", "block")
    assert rounding in ("nearest", "stochastic")
    assert 0 <= rounding_seed < (1 << 63)
    if granularity == "tile":
        assert block_shape is None, "block_shape requires granularity='block'."
        tile_size = defaults["tile_size"] if tile_size is None else tile_size
        assert tile_size in (64, 128, 256, 512, 1024)
    else:
        assert tile_size is None, "Use block_shape for granularity='block'."
        block_shape = (16, 16) if block_shape is None else tuple(block_shape)
        assert len(block_shape) == 2 and all(v in (8, 16, 32, 64) for v in block_shape)
    quant = dict(granularity=granularity, tile_size=tile_size, block_shape=block_shape)
    ns_steps = defaults["ns_steps"] if ns_steps is None else ns_steps
    adam_state_dtype = _dtype(adam_state_dtype)
    config = (granularity, tile_size, block_shape, rounding, rounding_seed, ns_steps, ns_method,
              skip_nonfinite, adam_state_dtype)
    if _enabled_config is not None:
        assert _enabled_config == config, "Viola FP8 Muon is already enabled with different settings."
        return
    assert ns_steps is None or ns_steps > 0
    assert ns_method in (None, "standard", "gram")

    from deepspeed.runtime.engine import DeepSpeedEngine, ZERO_OPTIMIZATION
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam

    original_sanity = DeepSpeedEngine._do_optimizer_sanity_check
    original_partition = DeepSpeedZeroOptimizer.get_flat_partition
    original_zero_step = DeepSpeedZeroOptimizer.step
    original_check_overflow = DeepSpeedZeroOptimizer.check_overflow
    original_muon_step = MuonWithAuxAdam.step
    original_load = MuonWithAuxAdam.load_state_dict

    @functools.wraps(original_sanity)
    def sanity(engine, optimizer):
        if not isinstance(optimizer, MuonWithAuxAdam):
            return original_sanity(engine, optimizer)
        assert engine.zero_optimization_stage() in (1, 2), "Viola FP8 Muon uses ZeRO-1 or ZeRO-2."
        assert not engine.zero_cpu_offload(), "FP8 Muon kernels and states run on GPU."
        precision = settings()
        precision["adam_state_dtype"] = adam_state_dtype
        optimizer._viola_precision = precision
        if precision["gradient_dtype"] is not None:
            engine._config.grad_accum_dtype = precision["gradient_dtype"]
            engine.communication_data_type = precision["gradient_dtype"]
        original_sanity(engine, optimizer)
        parameter_index = 0
        matrix_shapes = []
        for group_index, group in enumerate(optimizer.param_groups):
            if group["use_muon"]:
                if ns_steps is not None:
                    group["ns_steps"] = ns_steps
                if ns_method is not None:
                    group["ns_method"] = ns_method
            for param in group["params"]:
                # Client-created optimizers bypass DeepSpeed's config-based tags.
                # These tags tell the reducer which ranks need a whole matrix.
                param.use_muon = bool(group["use_muon"])
                if precision["gradient_dtype"] is not None:
                    param.grad_dtype = precision["gradient_dtype"]
                if group["use_muon"]:
                    matrix_shapes.append(_matrix_shape(param))
                # Global parameter order, never rank-local offsets: duplicated
                # boundary matrices must round identically on every rank.
                param._viola_rounding_key = (rounding_seed + parameter_index * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)
                parameter_index += 1
            if group["use_muon"]:
                group["viola_rounding"] = rounding
                group["viola_rounding_seed"] = rounding_seed
            group["viola_precision"] = dict(precision)
            group["viola_master_key"] = (precision["master_rounding_seed"] +
                (group_index + 1) * 0x517CC1B727220A95) & ((1 << 63) - 1)
            if not group["use_muon"] and (precision["master_weight_dtype"] is not None
                    or precision["gradient_dtype"] is not None or adam_state_dtype != torch.float32):
                assert not group.get("amsgrad", False), "Viola's independent-dtype Adam kernel uses Adam/AdamW."
        _warn_unaligned(matrix_shapes, quant)
        optimizer._viola_fp8_muon = True
        # In particular, override ZeRO-1 + BF16 + FP32 grad accumulation's
        # BF16_Optimizer route, which cannot perform full-matrix Muon updates.
        return ZERO_OPTIMIZATION

    @functools.wraps(original_partition)
    def partition(zero, tensor_list, first_offset, partition_size, dtype, device,
                  param_group_idx, return_tensor_list=False):
        group = zero.optimizer.param_groups[param_group_idx]
        enabled = getattr(zero.optimizer, "_viola_fp8_muon", False)
        if not enabled:
            return original_partition(zero, tensor_list, first_offset, partition_size, dtype,
                                      device, param_group_idx, return_tensor_list)
        gradients = zero.all_grad_tensors[param_group_idx]
        # These are references to existing gradient accumulators, not copies.
        # Reentrant backward may replace this entry; momentum advances only once
        # in step(), after the complete backward and global overflow decision.
        zero.__dict__.setdefault("_viola_gradients", {})[param_group_idx] = (
            tensor_list, gradients, partition_size, dtype, device)
        return _raw_partition(zero, tensor_list, gradients, partition_size, dtype, device,
                              param_group_idx, return_tensor_list)

    @functools.wraps(original_check_overflow)
    def check_overflow(zero, partition_gradients=True):
        if "_viola_overflow_checked" in zero.__dict__:
            zero.overflow = zero._viola_overflow_checked
            return
        return original_check_overflow(zero, partition_gradients)

    @functools.wraps(original_zero_step)
    def zero_step(zero, *args, **kwargs):
        if not getattr(zero.optimizer, "_viola_fp8_muon", False):
            return original_zero_step(zero, *args, **kwargs)
        deferred = zero.__dict__.get("_viola_gradients", {})
        if not zero.partition_gradients and zero.use_grad_accum_attribute:
            for group_id, (tensors, gradients, size, dtype, device) in deferred.items():
                _sync_stage1_accumulators(zero, group_id, tensors)
                gradients = zero.all_grad_tensors[group_id]
                deferred[group_id] = (tensors, gradients, size, dtype, device)
                zero.averaged_gradients[group_id] = _raw_partition(
                    zero, tensors, gradients, size, dtype, device, group_id, True)
        # Default: no scan, collective, or skipped step. Inf/NaN propagates into
        # the state/parameters/loss instead of being hidden by a skipped update.
        # An explicitly requested skip policy checks raw gradients before writes.
        if skip_nonfinite:
            original_check_overflow(zero, partition_gradients=True)
        else:
            zero.overflow = False
        zero._viola_overflow_checked = zero.overflow
        try:
            if not zero.overflow:
                for group_id, (tensors, gradients, size, dtype, device) in list(deferred.items()):
                    if tensors and zero.optimizer.param_groups[group_id]["use_muon"]:
                        zero.all_grad_tensors[group_id] = gradients
                        zero.averaged_gradients[group_id] = _partition(
                            zero, tensors, size, dtype, device, group_id, True, quant, rounding)
                        zero.all_grad_tensors[group_id] = None
                    del deferred[group_id]
                if getattr(zero, "_viola_independent_step", False):
                    return step_with_independent_dtypes(zero)
            return original_zero_step(zero, *args, **kwargs)
        finally:
            deferred.clear()
            zero.__dict__.pop("_viola_overflow_checked", None)

    @torch.no_grad()
    @functools.wraps(original_muon_step)
    def muon_step(optimizer, closure=None, step_id=None):
        if not getattr(optimizer, "_viola_fp8_muon", False):
            return original_muon_step(optimizer, closure=closure, step_id=step_id)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = optimizer.param_groups
        precision = optimizer._viola_precision
        independent = (precision["master_weight_dtype"] is not None or precision["gradient_dtype"] is not None
                       or precision["adam_state_dtype"] != torch.float32)
        for group in groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                options = dict(rounding=precision["master_rounding"], key=group["viola_master_key"],
                               offset=getattr(param, "_viola_gradient_offset", 0))
                state = optimizer.state[param]
                if group["use_muon"]:
                    step = state.get("viola_master_step", 0)
                    weight_update(param, param.grad, group["lr"], group["weight_decay"], step=step, **options)
                    state["viola_master_step"] = step + 1
                elif independent:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(param, dtype=adam_state_dtype)
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=adam_state_dtype)
                        state["step"] = 0
                    state["step"] = int(state["step"]) + 1
                    adam_update(param, param.grad, state["exp_avg"], state["exp_avg_sq"],
                        lr=group["lr"], weight_decay=group["weight_decay"], betas=group["betas"],
                        eps=group["eps"], step=state["step"], adamw=optimizer.adam_w_mode,
                        maximize=group.get("maximize", False), bias_correction=group.get("bias_correction", True),
                        **options)
        if independent:
            return loss
        # Auxiliary Adam groups continue to use the configured DeepSpeed optimizer.
        optimizer.param_groups = [g for g in groups if not g["use_muon"]]
        try:
            original_muon_step(optimizer, step_id=step_id)
        finally:
            optimizer.param_groups = groups
        return loss

    @functools.wraps(original_load)
    def load_state_dict(optimizer, saved):
        if not getattr(optimizer, "_viola_fp8_muon", False):
            return original_load(optimizer, saved)
        # Optimizer.load_state_dict casts even integer tensor states to the
        # floating parameter dtype. Remove FP8 payloads *before* that traversal
        # so loading never materializes a model-sized FP32 byte buffer.
        for group in saved["param_groups"]:
            if "viola_precision" in group:
                assert group["viola_precision"] == optimizer._viola_precision, "Resume with the saved precision settings."
            if group["use_muon"] and any(_STATE in saved["state"].get(pid, {}) for pid in group["params"]):
                assert group.get("viola_rounding", "nearest") == rounding, "Resume with the saved rounding mode."
                assert group.get("viola_rounding_seed", 0) == rounding_seed, "Resume with the saved rounding seed."
        preserved = {_STATE, "exp_avg", "exp_avg_sq"}
        packed = {pid: {k: v for k, v in state.items() if k in preserved}
                  for pid, state in saved["state"].items()}
        ordinary = dict(saved)
        ordinary["state"] = {pid: {k: v for k, v in state.items() if k not in preserved}
                             for pid, state in saved["state"].items()}
        master_keys = [group["viola_master_key"] for group in optimizer.param_groups]
        result = original_load(optimizer, ordinary)
        for saved_group, group, master_key in zip(saved["param_groups"], optimizer.param_groups, master_keys):
            # Reattach the engine's precision policy after PyTorch replaces
            # param_groups, including checkpoints made before these settings.
            group["viola_precision"] = dict(optimizer._viola_precision)
            group["viola_master_key"] = master_key
            for pid, param in zip(saved_group["params"], group["params"]):
                payload = packed.get(pid, {})
                if _STATE in payload:
                    state = _restore_quantization_names(payload[_STATE])
                    state["q"] = state["q"].to(device=param.device, dtype=torch.uint8)
                    state["scales"] = state["scales"].to(device=param.device, dtype=torch.float32)
                    optimizer.state[param][_STATE] = state
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in payload:
                        optimizer.state[param][key] = payload[key].to(device=param.device, dtype=adam_state_dtype)
        return result

    DeepSpeedEngine._do_optimizer_sanity_check = sanity
    DeepSpeedZeroOptimizer.get_flat_partition = partition
    DeepSpeedZeroOptimizer.step = zero_step
    DeepSpeedZeroOptimizer.check_overflow = check_overflow
    MuonWithAuxAdam.step = muon_step
    MuonWithAuxAdam.load_state_dict = load_state_dict
    install_zero_precision()
    DeepSpeedEngine._MUON_HALF_KEYS = tuple(dict.fromkeys(
        (*DeepSpeedEngine._MUON_HALF_KEYS, "adjust_lr_fn", "ns_steps", "nesterov")
    ))
    _enabled_config = config
    if int(os.environ.get("RANK", "0")) == 0:
        warnings.warn(
            "Viola FP8 Muon enabled: replacing DeepSpeed Muon and ZeRO-1/2 update/checkpoint paths "
            "in this process (including ZeRO-1 BF16/FP32-gradient routing). "
            f"Momentum: single-buffer E4M3, granularity={granularity}, "
            f"tile_size={tile_size}, block_shape={block_shape}, rounding={rounding}, "
            f"rounding_seed={rounding_seed}, skip_nonfinite={skip_nonfinite}; "
            f"NS method={ns_method or 'optimizer group'}, steps={ns_steps or 'optimizer group'}.",
            UserWarning, stacklevel=3,
        )
