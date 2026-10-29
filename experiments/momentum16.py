"""Single BF16/FP16 momentum reference, using the same ZeRO and NS paths."""
import torch
import tilelang
import tilelang.language as T
from viola.kernels import copy_scaled


@tilelang.jit
def update16(n, state_dtype, grad_dtype):
    @T.prim_func
    def kernel(g: T.Tensor((n,), grad_dtype), m: T.Tensor((n,), state_dtype),
               out: T.Tensor((n,), "bfloat16"), beta: T.float32, inv_scale: T.float32):
        with T.Kernel(T.ceildiv(n, 1024), threads=128) as block:
            for j in T.Parallel(1024):
                i = block * 1024 + j
                if i < n:
                    grad = T.cast(g[i], "float32") * inv_scale
                    new_m = beta * T.cast(m[i], "float32") + (1.0 - beta) * grad
                    out[i] = (1.0 - beta) * grad + beta * new_m
                    m[i] = new_m
    return kernel


def reference_partition(state_dtype):
    """Same deferred ZeRO path and BF16 NS; only storage/update kernel differs."""
    def partition(zero, tensors, partition_size, dtype, device, group_id, return_list, quant, rounding):
        from deepspeed import comm
        from deepspeed.runtime.zero.muon.original_muon import zeropower_via_newtonschulz5
        group = zero.optimizer.param_groups[group_id]
        state = zero.optimizer.state[group['params'][0]]
        if 'single_16bit_momentum' not in state:
            state['single_16bit_momentum'] = torch.zeros(sum(p.numel() for p in tensors),
                                                        device=device, dtype=state_dtype)
        momentum = state['single_16bit_momentum']
        needed = max(p.numel() for p in tensors)
        work = getattr(zero, '_viola_workspace', None)
        if work is None or work.numel() < needed:
            work = zero._viola_workspace = torch.empty(needed, dtype=torch.bfloat16, device=device)
        flat = torch.zeros(int(partition_size), dtype=dtype, device=device)
        rank = comm.get_rank(group=zero.real_dp_process_group[group_id])
        pieces, cursor, offset = [], 0, 0
        for p, g in zip(tensors, zero.all_grad_tensors[group_id]):
            n = p.numel()
            update16(n, str(state_dtype).split('.')[-1], str(g.dtype).split('.')[-1])(
                g.reshape(-1), momentum[offset:offset + n], work[:n], float(group['momentum']), 1. / zero.loss_scale)
            matrix = work[:n].view(p.shape[0], -1)
            update = zeropower_via_newtonschulz5(matrix, steps=group.get('ns_steps', 5))
            scale = .2 * max(matrix.shape) ** .5
            pid = zero.get_param_id(p)
            src = int(zero.grad_start_offset[group_id][rank][pid])
            dst = int(zero.grad_partition_insertion_offset[group_id][rank][pid])
            count = min(n - src, int(partition_size) - dst)
            if dst > cursor:
                pad = flat[cursor:dst]; pad._zero_padding = True; pieces.append(pad)
            copy_scaled(update, flat, src, dst, count, scale * zero.loss_scale)
            pieces.append(flat[dst:dst + count])
            cursor = dst + count; offset += n
            del update, matrix
        if cursor < flat.numel():
            pad = flat[cursor:]; pad._zero_padding = True; pieces.append(pad)
        return pieces if return_list else flat
    return partition
