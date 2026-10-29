"""Training precision settings, independent of Muon state quantization."""
import torch

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_settings = dict(master_weight_dtype=None, gradient_dtype=None,
                 master_rounding="nearest", master_rounding_seed=0)


def _dtype(value):
    value = _DTYPES[value] if isinstance(value, str) else value
    assert value in _DTYPES.values()
    return value


def set_master_weight_dtype(dtype, *, rounding="nearest", seed=0):
    """Set storage/writeback for subsequently initialized Viola ZeRO engines."""
    assert rounding in ("nearest", "stochastic") and 0 <= seed < (1 << 63)
    _settings.update(master_weight_dtype=_dtype(dtype), master_rounding=rounding,
                     master_rounding_seed=seed)


def set_gradient_dtype(dtype):
    """Set parameter-gradient reception, accumulation, reduction and update input.

    This does not change operator forward/backward math or activation gradients.
    A kernel's already rounded output cannot regain precision by casting it here.
    Call before constructing each DeepSpeed/Accelerate engine, on every rank.
    """
    _settings["gradient_dtype"] = _dtype(dtype)


def settings():
    return dict(_settings)


def install_zero_precision():
    """Install precision hooks only for explicitly enabled Viola ZeRO-1/2 engines."""
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer, IPGBucket

    original_init = DeepSpeedZeroOptimizer.__init__
    original_master = DeepSpeedZeroOptimizer._configure_master_weights
    original_setup = DeepSpeedZeroOptimizer.setup_buckets
    original_copy = DeepSpeedZeroOptimizer.copy_grads_in_partition
    original_comm = DeepSpeedZeroOptimizer.get_param_comm_dtype

    def configure_master(zero, *args, **kwargs):
        config = getattr(zero.optimizer, "_viola_precision", {})
        dtype = config.get("master_weight_dtype")
        if dtype is None:
            return original_master(zero, *args, **kwargs)
        # Native DeepSpeed couples low-precision masters to CPUAdam/offload.
        # Viola's GPU kernels accept master, gradient and state dtypes separately.
        zero.fp16_master_weights_and_gradients = dtype == torch.float16
        zero.bf16_master_weights_and_gradients = dtype == torch.bfloat16
        zero.bf16_optimizer_states = config["adam_state_dtype"] == torch.bfloat16
        return dtype

    def initialize(zero, *args, **kwargs):
        original_init(zero, *args, **kwargs)
        config = getattr(zero.optimizer, "_viola_precision", None)
        if config is None:
            return
        zero._viola_independent_step = (
            config["master_weight_dtype"] is not None or config["gradient_dtype"] is not None
            or config["adam_state_dtype"] != torch.float32)
        if zero._viola_independent_step:
            assert zero.torch_autocast_gradscaler is None, "Use DeepSpeed's fp16/bf16 mode for Viola precision settings."
        grad_dtype = config["gradient_dtype"]
        if grad_dtype is not None:
            # PyTorch leaf grad_dtype keeps a custom backward's FP32 dW intact.
            # It also handles ordinary kernels returning lower-precision dW.
            for group in zero.bit16_groups:
                for param in group:
                    param.grad_dtype = grad_dtype
            # param.grad already has the selected dtype. No second accumulator
            # or ZeRO-1's legacy cast back to the model dtype is necessary.
            zero.use_grad_accum_attribute = False
            zero.use_separate_grad_accum = False
            zero.ipg_buckets = {grad_dtype: IPGBucket()}
            zero.previous_reduced_grads = {grad_dtype: []}

    def setup_buckets(zero):
        grad_dtype = getattr(zero.optimizer, "_viola_precision", {}).get("gradient_dtype")
        if grad_dtype is None:
            return original_setup(zero)
        if not zero.ready_for_gradients:
            zero.micro_step_id += 1
            if zero.contiguous_gradients:
                for bucket in zero.ipg_buckets.values():
                    bucket.buffer.clear()
                    bucket.buffer.extend(torch.empty(int(zero.reduce_bucket_size),
                        dtype=grad_dtype, device=zero.device) for _ in range(2 if zero.overlap_comm else 1))
                    bucket.index = 0
            zero.ready_for_gradients = True

    def copy_grads(zero, param):
        grad_dtype = getattr(zero.optimizer, "_viola_precision", {}).get("gradient_dtype")
        if grad_dtype is not None and zero.grads_in_partition is None:
            size = sum(p.numel() for group in zero.params_in_partition for p in group)
            zero.grads_in_partition_offset = 0
            zero.grads_in_partition = torch.empty(size, dtype=grad_dtype, device=zero.device)
        return original_copy(zero, param)

    def comm_dtype(zero, param):
        dtype = getattr(zero.optimizer, "_viola_precision", {}).get("gradient_dtype")
        return dtype if dtype is not None else original_comm(zero, param)

    DeepSpeedZeroOptimizer._configure_master_weights = configure_master
    DeepSpeedZeroOptimizer.__init__ = initialize
    DeepSpeedZeroOptimizer.setup_buckets = setup_buckets
    DeepSpeedZeroOptimizer.copy_grads_in_partition = copy_grads
    DeepSpeedZeroOptimizer.get_param_comm_dtype = comm_dtype


@torch.no_grad()
def step_with_independent_dtypes(zero):
    """ZeRO-1/2 GPU step without casting gradients to the master-weight dtype.

    A flat master's grad_dtype is independent of its storage dtype. This avoids
    DeepSpeed's native cast of the update gradient to the master-weight dtype.
    Model synchronization and parameter partitioning still use DeepSpeed.
    """
    from deepspeed.runtime.utils import all_gather_dp_groups
    from deepspeed import comm

    zero.micro_step_id = -1
    previous_scale = zero.loss_scale
    zero._update_scale(False)
    norm = zero.scaled_global_norm() if zero.compute_grad_norm else None
    zero._global_grad_norm = norm / previous_scale if norm is not None else None
    grad_dtype = zero.optimizer._viola_precision["gradient_dtype"] or zero.gradient_accumulation_dtype

    for i, group in enumerate(zero.optimizer.param_groups):
        zero.timers('optimizer_gradients').start()
        zero.free_grad_in_param_list(zero.params_not_in_partition[i])
        flat = zero._get_preflattened_grad_partition(i)
        if flat is None:
            flat = zero.flatten_dense_tensors_aligned(zero.averaged_gradients[i], int(zero.partition_size[i]))
        # Muon may retain a scaled update in FP32 to avoid FP16 overflow.
        # Unscale before any narrowing cast to the requested gradient dtype.
        zero.unscale_and_clip_grads([flat], norm)
        gradient = flat.to(grad_dtype)
        del flat
        master = zero.single_partition_of_fp32_groups[i]
        assert gradient.numel() == master.numel()
        rank = comm.get_rank(group=zero.real_dp_process_group[i])
        master.grad_dtype = grad_dtype
        master.grad = gradient
        master._viola_gradient_offset = rank * master.numel()
        zero.free_grad_in_param_list(zero.params_in_partition[i])
        zero._release_preflattened_grad_buffers(i)
        zero.averaged_gradients[i] = None
        zero.all_grad_tensors[i] = None
        zero.timers('optimizer_gradients').stop()

        zero.timers('optimizer_step').start()
        try:
            zero._optimizer_step(i)
        finally:
            master.grad = None
            del master._viola_gradient_offset
        del gradient
        zero.parallel_partitioned_bit16_groups[i][rank].copy_(master.data)
        zero.timers('optimizer_step').stop()

    zero.timers('optimizer_allgather').start()
    all_gather_dp_groups(groups_flat=zero.bit16_groups_flat,
                        partitioned_param_groups=zero.parallel_partitioned_bit16_groups,
                        dp_process_group=zero.real_dp_process_group,
                        start_alignment_factor=zero.nccl_start_alignment_factor,
                        allgather_bucket_size=zero.allgather_bucket_size)
    zero.timers('optimizer_allgather').stop()
    for i in range(len(zero.bit16_groups)):
        zero._update_model_bit16_weights(i)
    zero.timers.log(['optimizer_gradients', 'optimizer_step', 'optimizer_allgather'])
