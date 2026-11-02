"""Grouped amax quantization; software FP32 scales or native SM120 MXFP8."""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter
from tilelang.layout import make_swizzled_layout

from ._linear_kernels import _dtype, transpose as transpose_payload, COPY_PASSES


def scale_shape(shape, group):
    return tuple((n + g - 1) // g for n, g in zip(shape, group))


def decode_scale(scale):
    return torch.exp2(scale.float() - 127) if scale.dtype == torch.uint8 else scale


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
def decode_kernel(rows, cols, dtype, gr, gc, native):
    sf_dtype = 'uint8' if native else 'float32'

    @T.prim_func
    def kernel(q: T.Tensor((rows, cols), dtype),
               scale: T.Tensor((T.ceildiv(rows, gr), T.ceildiv(cols, gc)), sf_dtype),
               out: T.Tensor((rows, cols), 'float32')):
        with T.Kernel(T.ceildiv(rows * cols, 1024), threads=128) as block:
            for i in T.Parallel(1024):
                offset = block * 1024 + i
                if offset < rows * cols:
                    r, c = offset // cols, offset % cols
                    s = T.cast(scale[r // gr, c // gc], 'float32')
                    if native:
                        s = T.exp2(s - 127.)
                    out[r, c] = T.cast(q[r, c], 'float32') * s
    return kernel


def dequantize(q, scale, group):
    if q.is_cuda:
        out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        decode_kernel(*q.shape, _dtype(q), *group, scale.dtype == torch.uint8)(q, scale, out)
        return out
    expanded = decode_scale(scale).repeat_interleave(group[0], 0).repeat_interleave(group[1], 1)
    return q.float() * expanded[:q.shape[0], :q.shape[1]]


@tilelang.jit
def encode_kernel(rows, cols, input_dtype, fp8_dtype, gr, gc, native, limit,
                  transposed=False, encoded=False, source_gr=1, source_gc=32):
    out_rows, out_cols = (cols, rows) if transposed else (rows, cols)
    nr, nc = (out_rows + gr - 1) // gr, (out_cols + gc - 1) // gc
    groups = max(1, 1024 // (gr * gc))
    sf_dtype = 'uint8' if native else 'float32'
    sr, sc = ((rows + source_gr - 1) // source_gr, (cols + source_gc - 1) // source_gc) if encoded else (1, 1)

    @T.prim_func
    def kernel(x: T.Tensor((rows, cols), input_dtype), source_scale: T.Tensor((sr, sc), sf_dtype),
               q: T.Tensor((out_rows, out_cols), fp8_dtype), scale: T.Tensor((nr, nc), sf_dtype)):
        with T.Kernel(T.ceildiv(nr * nc, groups), threads=128) as block:
            value = T.alloc_fragment((groups, gr * gc), 'float32')
            maximum = T.alloc_fragment((groups,), 'float32')
            s = T.alloc_fragment((groups,), 'float32')
            for i, j in T.Parallel(groups, gr * gc):
                group = block * groups + i
                r, c = group // nc * gr + j // gc, group % nc * gc + j % gc
                xr, xc = (c, r) if transposed else (r, c)
                v = T.if_then_else(xr < rows and xc < cols, T.cast(x[xr, xc], 'float32'), 0.)
                if encoded:
                    old = T.if_then_else(xr < rows and xc < cols, source_scale[xr // source_gr, xc // source_gc], 127 if native else 1.)
                    v *= T.exp2(T.cast(old, 'float32') - 127.) if native else old
                value[i, j] = v
            T.reduce_absmax(value, maximum, dim=1)
            for i in T.Parallel(groups):
                s[i] = T.max(maximum[i], 1e-30) / limit
                if native:
                    s[i] = T.exp2(T.ceil(T.log2(s[i])))
                group = block * groups + i
                if group < nr * nc:
                    scale[group // nc, group % nc] = T.log2(s[i]) + 127. if native else s[i]
            for i, j in T.Parallel(groups, gr * gc):
                group = block * groups + i
                r, c = group // nc * gr + j // gc, group % nc * gc + j % gc
                if r < out_rows and c < out_cols:
                    v = value[i, j] / s[i]
                    q[r, c] = T.if_then_else(T.isnan(v), v, T.clamp(v, -limit, limit))
    return kernel


@tilelang.jit
def transpose_encode_kernel(rows, cols, input_dtype, fp8_dtype, gc, native, limit):
    """Coalesced reads and writes for the common X.T/dY.T 1-D encoding."""
    sf_dtype = 'uint8' if native else 'float32'

    @T.prim_func
    def kernel(x: T.Tensor((rows, cols), input_dtype), q: T.Tensor((cols, rows), fp8_dtype),
               scale: T.Tensor((cols, T.ceildiv(rows, gc)), sf_dtype)):
        with T.Kernel(T.ceildiv(cols, 32), T.ceildiv(rows, gc), threads=128) as (br, bc):
            shared = T.alloc_shared((gc, 32), input_dtype)
            value = T.alloc_fragment((32, gc), 'float32')
            maximum = T.alloc_fragment((32,), 'float32')
            s = T.alloc_fragment((32,), 'float32')
            result = T.alloc_fragment((32, gc), fp8_dtype)
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(x[bc * gc, br * 32], shared)
            for i, j in T.Parallel(32, gc):
                value[i, j] = T.cast(shared[j, i], 'float32')
            T.reduce_absmax(value, maximum, dim=1)
            for i in T.Parallel(32):
                s[i] = T.max(maximum[i], 1e-30) / limit
                if native:
                    s[i] = T.exp2(T.ceil(T.log2(s[i])))
                if br * 32 + i < cols:
                    scale[br * 32 + i, bc] = T.log2(s[i]) + 127. if native else s[i]
            for i, j in T.Parallel(32, gc):
                v = value[i, j] / s[i]
                result[i, j] = T.if_then_else(T.isnan(v), v, T.clamp(v, -limit, limit))
            T.copy(result, q[br * 32, bc * gc])
    return kernel


def quantize(x, dtype, group, *, native=False, limit=None, transposed=False,
             source_scale=None, source_group=(1, 32), out=None, scale_out=None):
    """Transpose directly from the floating input, without a decoded matrix."""
    shape = x.shape[::-1] if transposed else x.shape
    limit = torch.finfo(dtype).max if limit is None else limit
    sf_dtype = torch.uint8 if native else torch.float32
    q = out if out is not None else torch.empty(shape, device=x.device, dtype=dtype)
    s = scale_out if scale_out is not None else torch.empty(scale_shape(shape, group), device=x.device, dtype=sf_dtype)
    if not x.is_cuda:
        # CPU initialization only. Linear execution always uses CUDA kernels.
        value = dequantize(x, source_scale, source_group) if source_scale is not None else x.float()
        value = value.T if transposed else value
        gr, gc = group
        nr, nc = s.shape
        blocks = F.pad(value, (0, nc * gc - shape[1], 0, nr * gr - shape[0])).reshape(nr, gr, nc, gc)
        scales = blocks.abs().amax((1, 3)).clamp_min(1e-30) / limit
        if native:
            exponent = scales.log2().ceil()
            s.copy_((exponent + 127).to(torch.uint8))
            scales = exponent.exp2()
        else:
            s.copy_(scales)
        codes = (blocks / scales[:, None, :, None]).clamp(-limit, limit)
        q.copy_(codes.reshape(nr * gr, nc * gc)[:shape[0], :shape[1]].to(dtype))
        return q, s
    if transposed and group[0] == 1 and source_scale is None:
        transpose_encode_kernel(*x.shape, _dtype(x), _dtype(q), group[1], native, limit)(x, q, s)
        return q, s
    dummy = source_scale if source_scale is not None else torch.empty((1, 1), device=x.device, dtype=sf_dtype)
    encode_kernel(*x.shape, _dtype(x), _dtype(q), *group, native, limit,
                  transposed, source_scale is not None, *source_group)(x, dummy, q, s)
    return q, s


def transpose_group(group):
    # A 2-D block transposes exactly. A 1-D row group needs new K groups.
    return group[::-1] if group[0] >= 32 else group


def transpose(q, scale, group, *, native=False, limit=None):
    if group[0] >= 32:
        qt = transpose_payload(q) if q.is_cuda else q.T.contiguous()
        return qt, scale.T.contiguous()
    return quantize(q, q.dtype, group, native=native, limit=limit, transposed=True,
                    source_scale=scale, source_group=group)


def native_mma_source(a_dtype, b_dtype):
    ad, bd = ({'float8_e4m3': 'e4m3', 'float8_e5m2': 'e5m2'}[d] for d in (a_dtype, b_dtype))
    fn = f'viola_grouped_mx_{ad}_{bd}'
    source = '''
__device__ __forceinline__ void FN(const void *ap, const void *bp, float *c,
                                  unsigned sa, unsigned sb) {
    const unsigned *a=static_cast<const unsigned*>(ap);
    const unsigned *b=static_cast<const unsigned*>(bp);
    const unsigned short selector=0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.AD.BD.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3},"
        "{%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sa), "h"(selector), "h"(selector), "r"(sb), "h"(selector), "h"(selector));
}
'''.replace('FN', fn).replace('AD', ad).replace('BD', bd)

    return fn, source


@tilelang.jit
def gemm_kernel(m, n, k, a_dtype, b_dtype, out_dtype, bias_dtype, has_bias,
                ar, ak, br, bk_scale, native, accum_dtype='float32', partial_k=32,
                reduce_dtype='float32', bm=64, bn=64, bk=64, threads=128, stages=2,
                cache_scales=True):
    row_warps = 2 if threads == 128 else 4
    col_warps = threads // 32 // row_warps
    mma = TensorCoreIntrinEmitter(a_dtype=a_dtype, b_dtype=b_dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True, block_row_warps=row_warps,
        block_col_warps=col_warps, warp_row_tiles=bm // row_warps,
        warp_col_tiles=bn // col_warps, chunk=bk)
    store = mma.get_store_index_map()
    sf_dtype = 'uint8' if native else 'float32'
    sa_shape, sb_shape = scale_shape((m, k), (ar, ak)), scale_shape((n, k), (br, bk_scale))
    fn, source = native_mma_source(a_dtype, b_dtype)

    @T.prim_func
    def kernel(a: T.Tensor((m, k), a_dtype), b: T.Tensor((n, k), b_dtype),
               sa: T.Tensor(sa_shape, sf_dtype), sb: T.Tensor(sb_shape, sf_dtype),
               bias: T.Tensor((n,) if has_bias else (0,), bias_dtype),
               out: T.Tensor((m, n), out_dtype)):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            if native:
                T.import_source(source)
            T.use_swizzle(8, order='row')
            ash = T.alloc_shared((bm, bk), a_dtype)
            bsh = T.alloc_shared((bn, bk), b_dtype)
            if cache_scales:
                # Expand only the current MMA tile's scales into shared memory.
                # K-group first makes adjacent rows bank-friendly for FP32 loads.
                ssa = T.alloc_shared((bk // 32, bm), sf_dtype)
                ssb = T.alloc_shared((bk // 32, bn), sf_dtype)
            al = T.alloc_local((mma.warp_rows * mma.local_size_a,), a_dtype)
            bl = T.alloc_local((mma.warp_cols * mma.local_size_b,), b_dtype)
            acc = T.alloc_local((bm * bn // threads,), accum_dtype)
            if not native:
                total = T.alloc_local((bm * bn // threads,), reduce_dtype)
                T.clear(total)
            T.annotate_layout({ash: make_swizzled_layout(ash), bsh: make_swizzled_layout(bsh)})
            lane, wn, wm = T.meta_var(mma.extract_thread_binding(T.get_thread_binding()))
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, ko * bk], ash)
                T.copy(b[bx * bn, ko * bk], bsh)
                if cache_scales:
                    for sk, r in T.Parallel(bk // 32, bm):
                        rr, kk = by * bm + r, ko * bk + sk * 32
                        ssa[sk, r] = T.if_then_else(rr < m and kk < k, sa[rr // ar, kk // ak], 127 if native else 1.)
                    for sk, r in T.Parallel(bk // 32, bn):
                        rr, kk = bx * bn + r, ko * bk + sk * 32
                        ssb[sk, r] = T.if_then_else(rr < n and kk < k, sb[rr // br, kk // bk_scale], 127 if native else 1.)
                for ki in T.serial(bk // 32):
                    mma.ldmatrix_a(al, ash, ki)
                    mma.ldmatrix_b(bl, bsh, ki)
                    kk = ko * bk + ki * 32
                    if native:
                        for ij in T.unroll(mma.warp_rows * mma.warp_cols):
                            i, j = ij // mma.warp_cols, ij % mma.warp_cols
                            ao, bo, co = i * mma.local_size_a, j * mma.local_size_b, ij * mma.local_size_out
                            ra = by * bm + wm * (bm // row_warps) + i * 16 + lane // 4 + (lane % 2) * 8
                            rb = bx * bn + wn * (bn // col_warps) + j * 16 + lane // 4
                            if cache_scales:
                                av = T.cast(ssa[ki, ra - by * bm], 'uint32')
                                bv = T.cast(ssb[ki, rb - bx * bn], 'uint32')
                                bv2 = T.cast(ssb[ki, rb + 8 - bx * bn], 'uint32')
                            else:
                                av = T.cast(T.if_then_else(ra < m and kk < k, sa[ra // ar, kk // ak], 127), 'uint32')
                                bv = T.cast(T.if_then_else(rb < n and kk < k, sb[rb // br, kk // bk_scale], 127), 'uint32')
                                bv2 = T.cast(T.if_then_else(rb + 8 < n and kk < k, sb[(rb + 8) // br, kk // bk_scale], 127), 'uint32')
                            T.call_extern('void', fn, T.address_of(al[ao]), T.address_of(bl[bo]), T.address_of(acc[co]), av, bv)
                            T.call_extern('void', fn, T.address_of(al[ao]), T.address_of(bl[bo + mma.local_size_b // 2]),
                                          T.address_of(acc[co + mma.local_size_out // 2]), av, bv2)
                    else:
                        mma.mma(al, bl, acc)
                        if (kk + 32) % partial_k == 0 or kk + 32 >= k:
                            for i, j in T.grid(mma.warp_rows, mma.warp_cols):
                                for lid in T.unroll(mma.local_size_out):
                                    r, c = T.meta_var(store.map_indices([lane, lid]))
                                    row = by * bm + wm * (bm // row_warps) + i * mma.micro_size_x + r
                                    col = bx * bn + wn * (bn // col_warps) + j * mma.micro_size_y + c
                                    index = (i * mma.warp_cols + j) * mma.local_size_out + lid
                                    if row < m and col < n and kk < k:
                                        if cache_scales:
                                            total[index] = T.cast(total[index], 'float32') + T.cast(acc[index], 'float32') * ssa[ki, row - by * bm] * ssb[ki, col - bx * bn]
                                        else:
                                            total[index] = T.cast(total[index], 'float32') + T.cast(acc[index], 'float32') * sa[row // ar, kk // ak] * sb[col // br, kk // bk_scale]
                            T.clear(acc)
            for i, j in T.grid(mma.warp_rows, mma.warp_cols):
                for lid in T.unroll(mma.local_size_out):
                    r, c = T.meta_var(store.map_indices([lane, lid]))
                    row = by * bm + wm * (bm // row_warps) + i * mma.micro_size_x + r
                    col = bx * bn + wn * (bn // col_warps) + j * mma.micro_size_y + c
                    if row < m and col < n:
                        index = (i * mma.warp_cols + j) * mma.local_size_out + lid
                        value = T.cast(acc[index], 'float32') if native else T.cast(total[index], 'float32')
                        if has_bias:
                            value += T.cast(bias[col], 'float32')
                        out[row, col] = value
    return kernel


@lru_cache(maxsize=256)
def compiled_gemm(args, config, device):
    options = dict(config)
    copy_mode = options.pop('copy_mode', 'default')
    passes = dict(COPY_PASSES[copy_mode])
    if args[12] or options.get('cache_scales', True):
        # Native extern-MMA and shared scale scatter use a cooperative CTA
        # pipeline: automatic warp specialization produced repeat races on SM120.
        passes['tl.disable_warp_specialized'] = True
    with torch.cuda.device(device):
        return tilelang.compile(gemm_kernel.get_tir(*args, **options), target='cuda',
                                pass_configs=passes)


def gemm(a, b, sa, sb, group_a, group_b, *, out_dtype, native=False, bias=None,
         autotune=False, accum_dtype='fp32', accum_block_size=0, accum_reduce_dtype='fp32'):
    m, k = a.shape
    n = b.shape[0]
    partial = math.gcd(group_a[1], group_b[1])
    if accum_block_size:
        partial = math.gcd(partial, accum_block_size)
    accumulator = {'fp32': 'float32', 'fp16': 'float16'}[accum_dtype]
    reducer = {'fp32': 'float32', 'bf16': 'bfloat16'}[accum_reduce_dtype]
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    bias_tensor = bias if bias is not None else out.new_empty(0)
    args = (m, n, k, _dtype(a), _dtype(b), _dtype(out), _dtype(bias_tensor), bias is not None,
            *group_a, *group_b, native, accumulator, partial, reducer)
    config = {}
    if autotune:
        from ._grouped_tuning import get_config
        config = get_config(args, a.device.index)
    compiled_gemm(args, tuple(sorted(config.items())), a.device.index)(a, b, sa, sb, bias_tensor, out)
    return out
