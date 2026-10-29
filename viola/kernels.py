"""Fused momentum update; FP32 intermediates stay in registers."""

import math
import torch
import tilelang
import tilelang.language as T


# Counter-based Philox4x32-10: one random value per (element, step, matrix key).
# No RNG tensor, global generator consumption, or per-element persistent state.
# Algorithm: https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
_ROUNDING_SOURCE = r"""
__device__ __forceinline__ unsigned int viola_philox(
    unsigned long long element, unsigned long long step, unsigned long long key) {
    unsigned int a = element, b = element >> 32, c = step, d = step >> 32;
    unsigned int k0 = key, k1 = key >> 32;
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        unsigned int lo0 = 0xD2511F53u * a, hi0 = __umulhi(0xD2511F53u, a);
        unsigned int lo1 = 0xCD9E8D57u * c, hi1 = __umulhi(0xCD9E8D57u, c);
        a = hi1 ^ b ^ k0; b = lo1; c = hi0 ^ d ^ k1; d = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return a;
}

__device__ __forceinline__ float viola_stochastic_round(
    float x, unsigned long long element, unsigned long long step, unsigned long long key) {
    if (!isfinite(x)) return x;
    float a = fabsf(x);
    // E4M3 subnormal spacing is 2^-9; normal spacing is 2^(exponent-3).
    int exponent = int((__float_as_uint(a) >> 23) & 255) - 127;
    float spacing = __uint_as_float((unsigned int)(max(exponent - 3, -9) + 127) << 23);
    float lower = floorf(a / spacing) * spacing;
    float probability = (a - lower) / spacing;
    float uniform = float(viola_philox(element, step, key) >> 8) * 0x1p-24f;
    return copysignf(lower + (uniform < probability ? spacing : 0.0f), x);
}

__device__ __forceinline__ float viola_stochastic_bf16(
    float x, unsigned long long element, unsigned long long step, unsigned long long key) {
    if (!isfinite(x)) return x;
    unsigned int bits = __float_as_uint(x);
    bits = (bits + (viola_philox(element, step, key) >> 16)) & 0xffff0000u;
    return __uint_as_float(bits);
}

__device__ __forceinline__ float viola_stochastic_fp16(
    float x, unsigned long long element, unsigned long long step, unsigned long long key) {
    if (!isfinite(x) || fabsf(x) > 65504.0f) return x;
    float a = fabsf(x);
    int exponent = int((__float_as_uint(a) >> 23) & 255) - 127;
    float spacing = __uint_as_float((unsigned int)(max(exponent - 10, -24) + 127) << 23);
    float lower = floorf(a / spacing) * spacing;
    float uniform = float(viola_philox(element, step, key) >> 8) * 0x1p-24f;
    return copysignf(lower + (uniform < (a - lower) / spacing ? spacing : 0.0f), x);
}
"""


def state_sizes(shape, *, granularity="tile", tile_size=256, block_shape=(16, 16)):
    """FP8 element count and scale count; 2-D blocks keep an unpadded payload."""
    n = math.prod(shape)
    if granularity == "tile":
        blocks = (n + tile_size - 1) // tile_size
        return blocks * tile_size, blocks
    rows, cols = shape[-2:]
    tr, tc = block_shape
    tiles = (rows + tr - 1) // tr * ((cols + tc - 1) // tc)
    return n, math.prod(shape[:-2]) * tiles


def dequantize(q, scales, shape, *, granularity="tile", tile_size=256, block_shape=(16, 16)):
    """Diagnostic helper only; the optimizer never materializes a full FP32 state."""
    n = math.prod(shape)
    if granularity == "tile":
        return (q.view(torch.float8_e4m3fn).float().view(-1, tile_size)
                * scales[:, None]).flatten()[:n]
    rows, cols = shape[-2:]
    tr, tc = block_shape
    expanded = scales.view(-1, (rows + tr - 1) // tr, (cols + tc - 1) // tc)
    expanded = expanded.repeat_interleave(tr, 1).repeat_interleave(tc, 2)[:, :rows, :cols]
    return (q.view(torch.float8_e4m3fn).float().view(-1, rows, cols) * expanded).flatten()


# ZeRO gradients and scale slices can begin at arbitrary element offsets.
# Scalar global accesses remain coalesced without requiring an aligned clone.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
def _momentum_1d_kernel(n, tile_size, grad_dtype, nesterov, rounding):
    blocks = T.ceildiv(n, tile_size)
    padded = blocks * tile_size
    rows = 4

    @T.prim_func
    def kernel(
        grad: T.Tensor((n,), grad_dtype),
        old_q: T.Tensor((padded,), "float8_e4m3"),
        old_scale: T.Tensor((blocks,), "float32"),
        new_q: T.Tensor((padded,), "float8_e4m3"),
        new_scale: T.Tensor((blocks,), "float32"),
        update: T.Tensor((n,), "bfloat16"),
        beta: T.float32,
        inv_loss_scale: T.float32,
        rounding_step: T.int64,
        rounding_key: T.int64,
    ):
        with T.Kernel(T.ceildiv(blocks, rows), threads=128) as pid:
            T.import_source(_ROUNDING_SOURCE if rounding == "stochastic" else None)
            g = T.alloc_fragment((rows, tile_size), "float32")
            m = T.alloc_fragment((rows, tile_size), "float32")
            amax = T.alloc_fragment((rows,), "float32")
            scale = T.alloc_fragment((rows,), "float32")

            for r, c in T.Parallel(rows, tile_size):
                b = pid * rows + r
                i = b * tile_size + c
                g[r, c] = T.if_then_else(i < n, T.cast(grad[i], "float32") * inv_loss_scale, 0.0)
                m[r, c] = T.if_then_else(
                    i < n, T.cast(old_q[i], "float32") * old_scale[b], 0.0
                )
                m[r, c] = beta * m[r, c] + (1.0 - beta) * g[r, c]

            T.reduce_absmax(m, amax, dim=1)
            for r in T.Parallel(rows):
                scale[r] = T.max(amax[r], 1e-30) / 448.0
                if pid * rows + r < blocks:
                    new_scale[pid * rows + r] = scale[r]

            for r, c in T.Parallel(rows, tile_size):
                i = (pid * rows + r) * tile_size + c
                if i < padded:
                    if rounding == "stochastic":
                        new_q[i] = T.call_extern("float32", "viola_stochastic_round",
                            T.clamp(m[r, c] / scale[r], -448.0, 448.0),
                            T.cast(i, "uint64"), rounding_step, rounding_key)
                    else:
                        new_q[i] = T.clamp(m[r, c] / scale[r], -448.0, 448.0)
                if i < n:
                    if nesterov:
                        update[i] = (1.0 - beta) * g[r, c] + beta * m[r, c]
                    else:
                        update[i] = m[r, c]

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
def _momentum_2d_kernel(planes, rows, cols, tr, tc, grad_dtype, nesterov, rounding):
    n = planes * rows * cols
    nr, nc = T.ceildiv(rows, tr), T.ceildiv(cols, tc)
    tiles = planes * nr * nc

    @T.prim_func
    def kernel(
        grad: T.Tensor((n,), grad_dtype),
        old_q: T.Tensor((n,), "float8_e4m3"),
        old_scale: T.Tensor((tiles,), "float32"),
        new_q: T.Tensor((n,), "float8_e4m3"),
        new_scale: T.Tensor((tiles,), "float32"),
        update: T.Tensor((n,), "bfloat16"),
        beta: T.float32,
        inv_loss_scale: T.float32,
        rounding_step: T.int64,
        rounding_key: T.int64,
    ):
        with T.Kernel(nc, nr, planes, threads=128) as (bc, br, plane):
            T.import_source(_ROUNDING_SOURCE if rounding == "stochastic" else None)
            g = T.alloc_fragment((tr, tc), "float32")
            m = T.alloc_fragment((tr, tc), "float32")
            row_max = T.alloc_fragment((tr,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            tile = (plane * nr + br) * nc + bc
            for r, c in T.Parallel(tr, tc):
                y, x = br * tr + r, bc * tc + c
                i = (plane * rows + y) * cols + x
                g[r, c] = T.if_then_else(y < rows and x < cols,
                    T.cast(grad[i], "float32") * inv_loss_scale, 0.0)
                m[r, c] = T.if_then_else(y < rows and x < cols,
                    T.cast(old_q[i], "float32") * old_scale[tile], 0.0)
                m[r, c] = beta * m[r, c] + (1.0 - beta) * g[r, c]
            T.reduce_absmax(m, row_max, dim=1)
            T.reduce_max(row_max, maximum, dim=0)
            maximum[0] = T.max(maximum[0], 1e-30) / 448.0
            new_scale[tile] = maximum[0]
            for r, c in T.Parallel(tr, tc):
                y, x = br * tr + r, bc * tc + c
                i = (plane * rows + y) * cols + x
                if y < rows and x < cols:
                    if rounding == "stochastic":
                        new_q[i] = T.call_extern("float32", "viola_stochastic_round",
                            T.clamp(m[r, c] / maximum[0], -448.0, 448.0),
                            T.cast(i, "uint64"), rounding_step, rounding_key)
                    else:
                        new_q[i] = T.clamp(m[r, c] / maximum[0], -448.0, 448.0)
                    if nesterov:
                        update[i] = (1.0 - beta) * g[r, c] + beta * m[r, c]
                    else:
                        update[i] = m[r, c]

    return kernel


@tilelang.jit
def _copy_scaled_kernel(n, size, dtype, out_dtype):
    tile = 1024

    @T.prim_func
    def kernel(
        update: T.Tensor((n,), dtype),
        out: T.Tensor((size,), out_dtype),
        src: T.int32,
        dst: T.int32,
        count: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(count, tile), threads=128) as pid:
            for i in T.Parallel(tile):
                j = pid * tile + i
                if j < count:
                    out[dst + j] = T.cast(update[src + j], "float32") * scale

    return kernel


# ZeRO partitions and mixed element widths do not guarantee vector-load alignment.
# Scalar accesses across adjacent lanes remain coalesced, without padding copies.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
def _weight_kernel(n, grad_dtype, weight_dtype, rounding):
    tile = 1024

    @T.prim_func
    def kernel(
        weight: T.Tensor((n,), weight_dtype),
        grad: T.Tensor((n,), grad_dtype),
        lr: T.float32,
        decay: T.float32,
        rounding_step: T.int64,
        rounding_key: T.int64,
        offset: T.int64,
    ):
        with T.Kernel(T.ceildiv(n, tile), threads=128) as pid:
            T.import_source(_ROUNDING_SOURCE if rounding == "stochastic" else None)
            for i in T.Parallel(tile):
                j = pid * tile + i
                if j < n:
                    value = T.cast(weight[j], "float32") * (1.0 - lr * decay) - lr * T.cast(grad[j], "float32")
                    if rounding == "stochastic" and weight_dtype != "float32":
                        weight[j] = T.call_extern("float32",
                            "viola_stochastic_bf16" if weight_dtype == "bfloat16" else "viola_stochastic_fp16",
                            value, T.cast(j + offset, "uint64"), rounding_step, rounding_key)
                    else:
                        weight[j] = value

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
def _adam_kernel(n, weight_dtype, grad_dtype, state_dtype, rounding, adamw, maximize):
    tile = 1024

    @T.prim_func
    def kernel(weight: T.Tensor((n,), weight_dtype), grad: T.Tensor((n,), grad_dtype),
               first: T.Tensor((n,), state_dtype), second: T.Tensor((n,), state_dtype),
               lr: T.float32, decay: T.float32, b1: T.float32, b2: T.float32,
               inv_c1: T.float32, inv_c2: T.float32, eps: T.float32,
               rounding_step: T.int64, rounding_key: T.int64, offset: T.int64):
        with T.Kernel(T.ceildiv(n, tile), threads=128) as pid:
            T.import_source(_ROUNDING_SOURCE if rounding == "stochastic" else None)
            for i in T.Parallel(tile):
                j = pid * tile + i
                if j < n:
                    w = T.cast(weight[j], "float32")
                    g = T.cast(grad[j], "float32")
                    if maximize:
                        g = -g
                    if not adamw:
                        g = g + decay * w
                    m = b1 * T.cast(first[j], "float32") + (1.0 - b1) * g
                    v = b2 * T.cast(second[j], "float32") + (1.0 - b2) * g * g
                    first[j] = m
                    second[j] = v
                    if adamw:
                        w = w * (1.0 - lr * decay)
                    value = w - lr * (m * inv_c1) / (T.sqrt(v * inv_c2) + eps)
                    if rounding == "stochastic" and weight_dtype != "float32":
                        weight[j] = T.call_extern("float32",
                            "viola_stochastic_bf16" if weight_dtype == "bfloat16" else "viola_stochastic_fp16",
                            value, T.cast(j + offset, "uint64"), rounding_step, rounding_key)
                    else:
                        weight[j] = value

    return kernel


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def momentum_update(grad, q, scales, next_q, next_scales, out, *, beta, tile_size=256,
                    nesterov=True, loss_scale=1.0, granularity="tile", block_shape=(16, 16),
                    matrix_shape=None, rounding="nearest", rounding_step=0, rounding_key=0):
    # The plugin's load hook preserves these raw bytes instead of letting
    # Optimizer.load_state_dict cast them. FP8 views allocate no memory.
    if granularity == "tile":
        kernel = _momentum_1d_kernel(grad.numel(), tile_size, _dtype(grad), nesterov, rounding)
    else:
        rows, cols = matrix_shape[-2:]
        kernel = _momentum_2d_kernel(math.prod(matrix_shape[:-2]), rows, cols, *block_shape,
                                       _dtype(grad), nesterov, rounding)
    kernel(
        grad.reshape(-1), q.view(torch.float8_e4m3fn), scales,
        next_q.view(torch.float8_e4m3fn), next_scales, out,
        float(beta), 1.0 / float(loss_scale), int(rounding_step), int(rounding_key),
    )


def copy_scaled(update, partition, src, dst, count, scale):
    flat = update.contiguous().view(-1)
    _copy_scaled_kernel(flat.numel(), partition.numel(), _dtype(flat), _dtype(partition))(
        flat, partition, int(src), int(dst), int(count), float(scale),
    )


def weight_update(weight, grad, lr, weight_decay, *, rounding="nearest", step=0, key=0, offset=0):
    _weight_kernel(weight.numel(), _dtype(grad), _dtype(weight), rounding)(
        weight.view(-1), grad.view(-1), float(lr), float(weight_decay), int(step), int(key), int(offset),
    )


def adam_update(weight, grad, first, second, *, lr, weight_decay, betas, eps, step,
                rounding="nearest", key=0, offset=0, adamw=True, maximize=False, bias_correction=True):
    b1, b2 = betas
    _adam_kernel(weight.numel(), _dtype(weight), _dtype(grad), _dtype(first), rounding, adamw, maximize)(
        weight.view(-1), grad.view(-1), first.view(-1), second.view(-1),
        float(lr), float(weight_decay), float(b1), float(b2), 1. / (1. - b1**step) if bias_correction else 1.,
        1. / (1. - b2**step) if bias_correction else 1., float(eps), int(step - 1), int(key), int(offset))
