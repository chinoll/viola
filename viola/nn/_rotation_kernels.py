"""Regular Hadamard-64 preprocessing without full-size floating scratch."""

import torch
import tilelang
import tilelang.language as T
from tilelang.layout import make_swizzled_layout

from ._linear_kernels import _dtype, _scale_kernel


@T.macro
def _regular64(values, scratch, rows, cols):
    # H4 = (ones(4, 4) - 2 * flip(eye(4))) / 2.
    # Three Kronecker factors give a symmetric, orthogonal H64.
    for stage in T.unroll(3):
        T.sync_threads()
        for i, j in T.Parallel(rows, cols):
            stride = 1 << (2 * stage)
            base = j // (4 * stride) * (4 * stride) + j % stride
            digit = j // stride % 4
            total = ((values[i, base] + values[i, base + stride])
                     + (values[i, base + 2 * stride] + values[i, base + 3 * stride]))
            scratch[i, j] = total * 0.5 - values[i, base + (3 - digit) * stride]
        T.copy(scratch, values)
    T.sync_threads()


@tilelang.jit
def _rotation_amax(rows, cols, dtype, br=16, bc=128):
    grid_c = T.ceildiv(cols, bc)

    @T.prim_func
    def kernel(x: T.Tensor((rows, cols), dtype),
               maxima: T.Tensor((T.ceildiv(rows, br) * grid_c,), "float32")):
        with T.Kernel(grid_c, T.ceildiv(rows, br), threads=128) as (cb, rb):
            values = T.alloc_shared((br, bc), "float32")
            scratch = T.alloc_fragment((br, bc), "float32")
            row_max = T.alloc_fragment((br,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            for i, j in T.Parallel(br, bc):
                values[i, j] = T.if_then_else(rb * br + i < rows and cb * bc + j < cols,
                                              T.cast(x[rb * br + i, cb * bc + j], "float32"), 0.0)
            _regular64(values, scratch, br, bc)
            T.reduce_absmax(scratch, row_max, dim=1)
            T.reduce_max(row_max, maximum, dim=0)
            maxima[rb * grid_c + cb] = maximum[0]
    return kernel


@tilelang.jit
def _rotation_quantize(rows, cols, dtype, fp8_dtype, rowwise, transpose, limit, br=32, bc=64):

    @T.prim_func
    def kernel(x: T.Tensor((rows, cols), dtype), scale: T.Tensor((1,), "float32"),
               q: T.Tensor((rows, cols) if rowwise else (0, 0), fp8_dtype),
               qt: T.Tensor((cols, rows) if transpose else (0, 0), fp8_dtype)):
        with T.Kernel(T.ceildiv(cols, bc), T.ceildiv(rows, br), threads=128) as (cb, rb):
            values = T.alloc_shared((br, bc), "float32")
            scratch = T.alloc_fragment((br, bc), "float32")
            quantized = T.alloc_fragment((br, bc), fp8_dtype)
            if transpose:
                shared = T.alloc_shared((br, bc), fp8_dtype)
                transposed = T.alloc_fragment((bc, br), fp8_dtype)
                T.annotate_layout({shared: make_swizzled_layout(shared)})
            for i, j in T.Parallel(br, bc):
                values[i, j] = T.if_then_else(rb * br + i < rows and cb * bc + j < cols,
                                              T.cast(x[rb * br + i, cb * bc + j], "float32"), 0.0)
            _regular64(values, scratch, br, bc)
            for i, j in T.Parallel(br, bc):
                quantized[i, j] = T.if_then_else(T.isnan(scratch[i, j]), scratch[i, j],
                                                T.clamp(scratch[i, j] / scale[0], -limit, limit))
            if rowwise:
                T.copy(quantized, q[rb * br, cb * bc])
            if transpose:
                T.copy(quantized, shared)
                for i, j in T.Parallel(bc, br):
                    transposed[i, j] = shared[j, i]
                T.copy(transposed, qt[cb * bc, rb * br])
    return kernel


@tilelang.jit
def _rotation_inplace(rows, cols, dtype, br=16, bc=128):

    @T.prim_func
    def kernel(x: T.Tensor((rows, cols), dtype)):
        with T.Kernel(T.ceildiv(cols, bc), T.ceildiv(rows, br), threads=128) as (cb, rb):
            values = T.alloc_shared((br, bc), "float32")
            scratch = T.alloc_fragment((br, bc), "float32")
            for i, j in T.Parallel(br, bc):
                values[i, j] = T.if_then_else(rb * br + i < rows and cb * bc + j < cols,
                                              T.cast(x[rb * br + i, cb * bc + j], "float32"), 0.0)
            _regular64(values, scratch, br, bc)
            for i, j in T.Parallel(br, bc):
                if rb * br + i < rows and cb * bc + j < cols:
                    x[rb * br + i, cb * bc + j] = scratch[i, j]
    return kernel


def tensor_scale(x, *, history, index, delayed, fp8_dtype, quantization_limit=None):
    rows, cols = x.shape
    maxima = torch.empty(((rows + 15) // 16) * ((cols + 127) // 128),
                         device=x.device, dtype=torch.float32)
    scale = torch.empty(1, device=x.device, dtype=torch.float32)
    _rotation_amax(rows, cols, _dtype(x))(x, maxima)
    limit = torch.finfo(fp8_dtype).max if quantization_limit is None else quantization_limit
    _scale_kernel(maxima.numel(), history.numel(), limit, delayed)(maxima, history, scale, index)
    return scale


def quantize(x, scale, *, rowwise=True, transpose=False, fp8_dtype=torch.float8_e4m3fn,
             out=None, out_t=None, quantization_limit=None):
    q = out if out is not None else torch.empty(x.shape if rowwise else (0, 0), device=x.device, dtype=fp8_dtype)
    qt = out_t if out_t is not None else torch.empty(x.shape[::-1] if transpose else (0, 0), device=x.device, dtype=fp8_dtype)
    limit = torch.finfo(fp8_dtype).max if quantization_limit is None else quantization_limit
    _rotation_quantize(*x.shape, _dtype(x), _dtype(q), rowwise, transpose, limit)(x, scale, q, qt)
    return q, qt


def inverse_(x):
    """H64 is its own inverse; each CTA owns complete, disjoint channel groups."""
    _rotation_inplace(*x.shape, _dtype(x))(x)
    return x


def rotate_cpu(x):
    """CPU initialization only; training uses the fused TileLang kernels."""
    h = (torch.ones(4, 4) - 2 * torch.eye(4).flip(1)) / 2
    h = torch.kron(torch.kron(h, h), h)
    return (x.float().reshape(-1, 64) @ h).reshape(x.shape)
