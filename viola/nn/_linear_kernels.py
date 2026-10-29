"""TileLang kernels for viola.nn.Linear."""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter
from tilelang.layout import make_swizzled_layout


COPY_PASSES = {
    "default": {},
    "no_warp_specialization": {"tl.disable_warp_specialized": True},
    "cp_async": {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True,
                  "tl.enable_async_copy": True},
}


@lru_cache(maxsize=256)
def _configured_kernel(mixed, args, options, copy_mode, device_index):
    factory = _mixed_gemm_kernel if mixed else _gemm_kernel
    with torch.cuda.device(device_index):
        return tilelang.compile(factory.get_tir(*args, **dict(options)), target="cuda",
                                pass_configs=COPY_PASSES[copy_mode])


def _dtype(tensor):
    return {torch.float8_e4m3fn: "float8_e4m3", torch.float8_e5m2: "float8_e5m2"}.get(
        tensor.dtype, str(tensor.dtype).removeprefix("torch."))


@tilelang.jit
def _amax_kernel(n, dtype):
    block = 2048

    @T.prim_func
    def kernel(x: T.Tensor((n,), dtype), maxima: T.Tensor((T.ceildiv(n, block),), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=128) as pid:
            values = T.alloc_fragment((block,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(block):
                values[i] = T.if_then_else(pid * block + i < n, T.cast(x[pid * block + i], "float32"), 0.0)
            T.reduce_absmax(values, maximum, dim=0)
            maxima[pid] = maximum[0]
    return kernel


@tilelang.jit
def _scale_kernel(n, history_len, fp8_max, delayed):
    size = 1 << (max(n, history_len) - 1).bit_length()

    @T.prim_func
    def kernel(maxima: T.Tensor((n,), "float32"), history: T.Tensor((history_len,), "float32"),
               scale: T.Tensor((1,), "float32"), index: T.int32):
        with T.Kernel(1, threads=128):
            current = T.alloc_fragment((size,), "float32")
            previous = T.alloc_fragment((size,), "float32")
            amax = T.alloc_fragment((1,), "float32")
            old_amax = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(size):
                current[i] = T.if_then_else(i < n, maxima[i], 0.0)
                previous[i] = T.if_then_else(i < history_len, history[i], 0.0)
            T.reduce_max(current, amax, dim=0)
            T.reduce_max(previous, old_amax, dim=0)
            if delayed:
                scale[0] = T.max(T.if_then_else(index == 0, amax[0], old_amax[0]), 1e-30) / fp8_max
            else:
                scale[0] = T.max(amax[0], 1e-30) / fp8_max
            history[index % history_len] = amax[0]
    return kernel


@tilelang.jit
def _quantize_kernel(rows, cols, dtype, fp8_dtype, rowwise, transpose, quantization_limit=None):
    block = 64
    fp8_max = quantization_limit if quantization_limit is not None else (448.0 if fp8_dtype == "float8_e4m3" else 57344.0)

    @T.prim_func
    def kernel(x: T.Tensor((rows, cols), dtype), scale: T.Tensor((1,), "float32"),
               q: T.Tensor((rows, cols) if rowwise else (0, 0), fp8_dtype),
               qt: T.Tensor((cols, rows) if transpose else (0, 0), fp8_dtype)):
        with T.Kernel(T.ceildiv(cols, block), T.ceildiv(rows, block), threads=128) as (bc, br):
            local = T.alloc_fragment((block, block), fp8_dtype)
            if transpose:
                shared = T.alloc_shared((block, block), fp8_dtype)
                transposed = T.alloc_fragment((block, block), fp8_dtype)
                T.annotate_layout({shared: make_swizzled_layout(shared)})
            for i, j in T.Parallel(block, block):
                r, c = br * block + i, bc * block + j
                value = T.if_then_else(r < rows and c < cols, T.cast(x[r, c], "float32"), 0.0)
                # Preserve NaNs instead of letting min/max turn them into finite values.
                local[i, j] = T.if_then_else(T.isnan(value), value,
                                             T.clamp(value / scale[0], -fp8_max, fp8_max))
            if rowwise:
                T.copy(local, q[br * block, bc * block])
            if transpose:
                # Redistribute lanes through shared memory so both layouts use
                # contiguous global stores instead of byte-wise scatter writes.
                T.copy(local, shared)
                for i, j in T.Parallel(block, block):
                    transposed[i, j] = shared[j, i]
                T.copy(transposed, qt[bc * block, br * block])
    return kernel


@tilelang.jit
def _transpose_kernel(rows, cols, dtype):
    block = 64

    @T.prim_func
    def kernel(x: T.Tensor((rows, cols), dtype), out: T.Tensor((cols, rows), dtype)):
        with T.Kernel(T.ceildiv(cols, block), T.ceildiv(rows, block), threads=128) as (bc, br):
            shared = T.alloc_shared((block, block), dtype)
            result = T.alloc_fragment((block, block), dtype)
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(x[br * block, bc * block], shared)
            for i, j in T.Parallel(block, block):
                result[i, j] = shared[j, i]
            T.copy(result, out[bc * block, br * block])
    return kernel


@tilelang.jit
def _gemm_kernel(m, n, k, a_dtype, b_dtype, out_dtype, bias_dtype, has_bias,
                 bm=64, bn=64, bk=64, threads=128, stages=3, accum_dtype="float32",
                 swizzle=0, swizzle_order="row", accum_block_size=0, accum_reduce_dtype="float32"):

    @T.prim_func
    def kernel(a: T.Tensor((m, k), a_dtype), b: T.Tensor((n, k), b_dtype),
               sa: T.Tensor((1,), "float32"), sb: T.Tensor((1,), "float32"),
               bias: T.Tensor((n,) if has_bias else (0,), bias_dtype),
               out: T.Tensor((m, n), out_dtype)):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            if swizzle:
                T.use_swizzle(swizzle, order=swizzle_order)
            a_shared = T.alloc_shared((bm, bk), a_dtype)
            b_shared = T.alloc_shared((bn, bk), b_dtype)
            acc = T.alloc_fragment((bm, bn), accum_dtype)
            if accum_block_size:
                total = T.alloc_fragment((bm, bn), accum_reduce_dtype)
                T.clear(total)
            result = T.alloc_fragment((bm, bn), out_dtype)
            T.clear(acc)
            for i in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, i * bk], a_shared)
                T.copy(b[bx * bn, i * bk], b_shared)
                T.gemm(a_shared, b_shared, acc, transpose_B=True)
                if accum_block_size:
                    if (i + 1) % (accum_block_size // bk) == 0 or i == T.ceildiv(k, bk) - 1:
                        for r, c in T.Parallel(bm, bn):
                            total[r, c] = T.cast(total[r, c], "float32") + T.cast(acc[r, c], "float32")
                        T.clear(acc)
            for i, j in T.Parallel(bm, bn):
                if accum_block_size:
                    value = T.cast(total[i, j], "float32") * sa[0] * sb[0]
                else:
                    value = T.cast(acc[i, j], "float32") * sa[0] * sb[0]
                if has_bias:
                    value += T.if_then_else(bx * bn + j < n, T.cast(bias[bx * bn + j], "float32"), 0.0)
                result[i, j] = value
            T.copy(result, out[by * bm, bx * bn])
    return kernel


@tilelang.jit
def _mixed_gemm_kernel(m, n, k, a_dtype, b_dtype, out_dtype, bias_dtype, has_bias,
                       bm=64, bn=64, bk=64, threads=128, stages=3, accum_dtype="float32",
                       accum_block_size=0, accum_reduce_dtype="float32", row_warps=0,
                       swizzle=0, swizzle_order="row"):
    """FP8 MMA, optionally clearing FP16 partials into a separate running sum."""
    row_warps = row_warps or (2 if threads == 128 else 4)
    col_warps = threads // 32 // row_warps
    mma = TensorCoreIntrinEmitter(a_dtype=a_dtype, b_dtype=b_dtype, accum_dtype=accum_dtype,
                                  a_transposed=False, b_transposed=True,
                                  block_row_warps=row_warps, block_col_warps=col_warps,
                                  warp_row_tiles=bm // row_warps, warp_col_tiles=bn // col_warps, chunk=bk)
    store_map = mma.get_store_index_map()

    @T.prim_func
    def kernel(a: T.Tensor((m, k), a_dtype), b: T.Tensor((n, k), b_dtype),
               sa: T.Tensor((1,), "float32"), sb: T.Tensor((1,), "float32"),
               bias: T.Tensor((n,) if has_bias else (0,), bias_dtype),
               out: T.Tensor((m, n), out_dtype)):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            if swizzle:
                T.use_swizzle(swizzle, order=swizzle_order)
            a_shared = T.alloc_shared((bm, bk), a_dtype)
            b_shared = T.alloc_shared((bn, bk), b_dtype)
            a_local = T.alloc_local((mma.warp_rows * mma.local_size_a,), a_dtype)
            b_local = T.alloc_local((mma.warp_cols * mma.local_size_b,), b_dtype)
            acc = T.alloc_local((bm * bn // threads,), accum_dtype)
            if accum_block_size:
                total = T.alloc_local((bm * bn // threads,), accum_reduce_dtype)
                T.clear(total)
            T.annotate_layout({a_shared: make_swizzled_layout(a_shared),
                               b_shared: make_swizzled_layout(b_shared)})
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, ko * bk], a_shared)
                T.copy(b[bx * bn, ko * bk], b_shared)
                for ki in T.serial(bk // 32):
                    mma.ldmatrix_a(a_local, a_shared, ki)
                    mma.ldmatrix_b(b_local, b_shared, ki)
                    mma.mma(a_local, b_local, acc)
                if accum_block_size:
                    if (ko + 1) % (accum_block_size // bk) == 0 or ko == T.ceildiv(k, bk) - 1:
                        for index in T.unroll(bm * bn // threads):
                            total[index] = T.cast(total[index], "float32") + T.cast(acc[index], "float32")
                        T.clear(acc)
            # Keep the epilogue in registers. Avoid a full FP32 shared-memory
            # output tile and its barrier, leaving more space for pipelining.
            lane, warp_n, warp_m = T.meta_var(mma.extract_thread_binding(T.get_thread_binding()))
            for i, j in T.grid(mma.warp_rows, mma.warp_cols):
                for local_id in T.unroll(mma.local_size_out):
                    r, c = T.meta_var(store_map.map_indices([lane, local_id]))
                    row = by * bm + warp_m * (bm // row_warps) + i * mma.micro_size_x + r
                    col = bx * bn + warp_n * (bn // col_warps) + j * mma.micro_size_y + c
                    if row < m and col < n:
                        index = i * mma.warp_cols * mma.local_size_out + j * mma.local_size_out + local_id
                        if accum_block_size:
                            value = T.cast(total[index], "float32") * sa[0] * sb[0]
                        else:
                            value = T.cast(acc[index], "float32") * sa[0] * sb[0]
                        if has_bias:
                            value += T.cast(bias[col], "float32")
                        out[row, col] = value
    return kernel


def tensor_scale(x, *, history=None, index=0, delayed=False, fp8_dtype=torch.float8_e4m3fn,
                 quantization_limit=None):
    """No full-sized FP32 temporary: reduce to one scalar through small scratch."""
    maxima = torch.empty((x.numel() + 2047) // 2048, device=x.device, dtype=torch.float32)
    if history is None:
        history = torch.zeros(1, device=x.device, dtype=torch.float32)
    scale = torch.empty(1, device=x.device, dtype=torch.float32)
    _amax_kernel(x.numel(), _dtype(x))(x.view(-1), maxima)
    limit = torch.finfo(fp8_dtype).max if quantization_limit is None else quantization_limit
    _scale_kernel(maxima.numel(), history.numel(), limit, delayed)(
        maxima, history, scale, index)
    return scale


def quantize(x, scale, *, rowwise=True, transpose=False, fp8_dtype=torch.float8_e4m3fn,
             out=None, out_t=None, quantization_limit=None):
    q = out if out is not None else torch.empty(x.shape if rowwise else (0, 0), device=x.device, dtype=fp8_dtype)
    qt = out_t if out_t is not None else torch.empty(x.shape[::-1] if transpose else (0, 0), device=x.device, dtype=fp8_dtype)
    _quantize_kernel(*x.shape, _dtype(x), _dtype(q), rowwise, transpose, quantization_limit)(x, scale, q, qt)
    return q, qt


def transpose(x, out=None):
    if out is None:
        out = torch.empty(x.shape[::-1], device=x.device, dtype=x.dtype)
    _transpose_kernel(*x.shape, _dtype(x))(x, out)
    return out


def gemm(a, b, sa, sb, *, out_dtype, bias=None, autotune=False, accum_dtype="fp32",
          accum_block_size=0, accum_reduce_dtype="fp32"):
    """Compute (a * sa) @ (b * sb).T, including scaling/bias in the epilogue."""
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    empty_bias = bias if bias is not None else torch.empty(0, device=a.device, dtype=out_dtype)
    kernel = _mixed_gemm_kernel if a.dtype != b.dtype else _gemm_kernel
    accumulator = {"fp32": "float32", "fp16": "float16"}[accum_dtype]
    reducer = {"fp32": "float32", "bf16": "bfloat16"}[accum_reduce_dtype]
    config = {}
    if accum_block_size and accum_block_size < 64:
        config["bk"] = accum_block_size
    if autotune:
        from ._linear_tuning import get_config
        config = dict(get_config(m, n, k, _dtype(a), _dtype(b), _dtype(out), _dtype(empty_bias),
                                 bias is not None, a.device.index, accumulator, accum_block_size, reducer))
    if config.pop("explicit_mma", False):
        kernel = _mixed_gemm_kernel
    copy_mode = config.pop("copy_mode", "default")
    config.update(accum_block_size=accum_block_size, accum_reduce_dtype=reducer)
    if kernel is _gemm_kernel:
        config.pop("row_warps", None)
    args = (m, n, k, _dtype(a), _dtype(b), _dtype(out), _dtype(empty_bias), bias is not None)
    config["accum_dtype"] = accumulator
    if copy_mode == "default":
        compiled = kernel(*args, **config)
    else:
        compiled = _configured_kernel(kernel is _mixed_gemm_kernel, args, tuple(sorted(config.items())),
                                      copy_mode, a.device.index)
    compiled(a, b, sa, sb, empty_bias, out)
    return out
