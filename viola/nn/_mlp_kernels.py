"""FP8 GEMMs with gated forward/backward epilogues, implemented in TileLang."""

from functools import lru_cache
import math

import torch
import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter
from tilelang.layout import make_swizzled_layout

from ._linear_kernels import _dtype, COPY_PASSES
from ._grouped_kernels import scale_shape, native_mma_source


@tilelang.jit
def kernel(m, h, k, ad, bd, od, bias_dtype, has_bias, ar, ak, br, bgroup_k,
           scaling, accum_dtype, partial_k, reduce_dtype, activation, backward, save,
           encode_output=False, encode_limit=448., save_output_transpose=False,
           bm=64, bn=64, bk=64, threads=128, stages=2):
    """Forward: two GEMMs share A. Backward: dH GEMM immediately produces dG/dU."""
    native, grouped = scaling == 'mxfp8', scaling in ('grouped', 'mxfp8')
    sf = 'uint8' if native else 'float32'
    n = h if backward else 2 * h
    rows = 2 if threads == 128 else 4
    cols = threads // 32 // rows
    mma = TensorCoreIntrinEmitter(a_dtype=ad, b_dtype=bd, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True, block_row_warps=rows, block_col_warps=cols,
        warp_row_tiles=bm // rows, warp_col_tiles=bn // cols, chunk=bk)
    store = mma.get_store_index_map()
    fn, source = native_mma_source(ad, bd)
    split = grouped and not native or partial_k > 0
    flush_k = partial_k if partial_k else k
    sas = scale_shape((m, k), (ar, ak)) if grouped else (1, 1)
    sbs = scale_shape((n, k), (br, bgroup_k)) if grouped else (1, 1)
    # An auxiliary tensor is written only for training; inference stores just H.
    aux_shape = (m, 2 * h) if backward or save else (0, 0)

    @T.prim_func
    def main(a: T.Tensor((m, k), ad), b: T.Tensor((n, k), bd),
             sa: T.Tensor(sas, sf), sb: T.Tensor(sbs, sf),
             bias: T.Tensor((n,) if has_bias else (0,), bias_dtype),
             aux: T.Tensor(aux_shape, od),
             out_scale: T.Tensor((m, T.ceildiv(h, 32)) if encode_output else (0, 0), 'uint8'),
             out_t: T.Tensor((h, m) if save_output_transpose else (0, 0), ad),
             out_t_scale: T.Tensor((h, T.ceildiv(m, 32)) if save_output_transpose else (0, 0), 'uint8'),
             out: T.Tensor((m, 2 * h if backward else h), ad if encode_output else od)):
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            if native:
                T.import_source(source)
            T.use_swizzle(8, order='row')
            ash = T.alloc_shared((bm, bk), ad)
            bsh = T.alloc_shared((bn, bk), bd)
            al = T.alloc_local((mma.warp_rows * mma.local_size_a,), ad)
            bl = T.alloc_local((mma.warp_cols * mma.local_size_b,), bd)
            acc = T.alloc_local((bm * bn // threads,), accum_dtype)
            T.annotate_layout({ash: make_swizzled_layout(ash), bsh: make_swizzled_layout(bsh)})
            T.clear(acc)
            if encode_output:
                hidden = T.alloc_shared((bm, bn), od)
                T.annotate_layout({hidden: make_swizzled_layout(hidden)})
                T.clear(hidden)
            if not backward:
                ush = T.alloc_shared((bn, bk), bd)
                ul = T.alloc_local((mma.warp_cols * mma.local_size_b,), bd)
                uacc = T.alloc_local((bm * bn // threads,), accum_dtype)
                T.annotate_layout({ush: make_swizzled_layout(ush)})
                T.clear(uacc)
            if split:
                total = T.alloc_local((bm * bn // threads,), reduce_dtype)
                T.clear(total)
                if not backward:
                    utotal = T.alloc_local((bm * bn // threads,), reduce_dtype)
                    T.clear(utotal)
            if grouped:
                ssa = T.alloc_shared((bk // 32, bm), sf)
                ssb = T.alloc_shared((bk // 32, bn), sf)
                if not backward:
                    ssu = T.alloc_shared((bk // 32, bn), sf)
            lane, wn, wm = T.meta_var(mma.extract_thread_binding(T.get_thread_binding()))
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, ko * bk], ash)
                if h % bn == 0:
                    T.copy(b[bx * bn, ko * bk], bsh)
                    if not backward:
                        T.copy(b[h + bx * bn, ko * bk], ush)
                else:
                    # Mask the branch boundary as well as the weight boundary.
                    for i, j in T.Parallel(bn, bk):
                        col, kk = bx * bn + i, ko * bk + j
                        bsh[i, j] = T.if_then_else(col < h and kk < k, b[col, kk], 0.)
                        if not backward:
                            ush[i, j] = T.if_then_else(col < h and kk < k, b[h + col, kk], 0.)
                if grouped:
                    for sk, r in T.Parallel(bk // 32, bm):
                        rr, kk = by * bm + r, ko * bk + sk * 32
                        ssa[sk, r] = T.if_then_else(rr < m and kk < k, sa[rr // ar, kk // ak], 127 if native else 1.)
                    for sk, c in T.Parallel(bk // 32, bn):
                        cc, kk = bx * bn + c, ko * bk + sk * 32
                        ssb[sk, c] = T.if_then_else(cc < h and kk < k, sb[cc // br, kk // bgroup_k], 127 if native else 1.)
                        if not backward:
                            ssu[sk, c] = T.if_then_else(cc < h and kk < k, sb[(h + cc) // br, kk // bgroup_k], 127 if native else 1.)
                for ki in T.serial(bk // 32):
                    mma.ldmatrix_a(al, ash, ki)
                    mma.ldmatrix_b(bl, bsh, ki)
                    if not backward:
                        mma.ldmatrix_b(ul, ush, ki)
                    kk = ko * bk + ki * 32
                    if native:
                        for ij in T.unroll(mma.warp_rows * mma.warp_cols):
                            i, j = ij // mma.warp_cols, ij % mma.warp_cols
                            ao, bo, co = i * mma.local_size_a, j * mma.local_size_b, ij * mma.local_size_out
                            ra = wm * (bm // rows) + i * 16 + lane // 4 + (lane % 2) * 8
                            rb = wn * (bn // cols) + j * 16 + lane // 4
                            av = T.cast(ssa[ki, ra], 'uint32')
                            bv, bv2 = T.cast(ssb[ki, rb], 'uint32'), T.cast(ssb[ki, rb + 8], 'uint32')
                            T.call_extern('void', fn, T.address_of(al[ao]), T.address_of(bl[bo]), T.address_of(acc[co]), av, bv)
                            T.call_extern('void', fn, T.address_of(al[ao]), T.address_of(bl[bo + mma.local_size_b // 2]),
                                          T.address_of(acc[co + mma.local_size_out // 2]), av, bv2)
                            if not backward:
                                uv, uv2 = T.cast(ssu[ki, rb], 'uint32'), T.cast(ssu[ki, rb + 8], 'uint32')
                                T.call_extern('void', fn, T.address_of(al[ao]), T.address_of(ul[bo]), T.address_of(uacc[co]), av, uv)
                                T.call_extern('void', fn, T.address_of(al[ao]), T.address_of(ul[bo + mma.local_size_b // 2]),
                                              T.address_of(uacc[co + mma.local_size_out // 2]), av, uv2)
                    else:
                        mma.mma(al, bl, acc)
                        if not backward:
                            mma.mma(al, ul, uacc)
                        if split:
                            if (kk + 32) % flush_k == 0 or kk + 32 >= k:
                                for i, j in T.grid(mma.warp_rows, mma.warp_cols):
                                    for lid in T.unroll(mma.local_size_out):
                                        r, c = T.meta_var(store.map_indices([lane, lid]))
                                        ri = wm * (bm // rows) + i * mma.micro_size_x + r
                                        ci = wn * (bn // cols) + j * mma.micro_size_y + c
                                        index = (i * mma.warp_cols + j) * mma.local_size_out + lid
                                        if grouped:
                                            total[index] = T.cast(total[index], 'float32') + T.cast(acc[index], 'float32') * ssa[ki, ri] * ssb[ki, ci]
                                            if not backward:
                                                utotal[index] = T.cast(utotal[index], 'float32') + T.cast(uacc[index], 'float32') * ssa[ki, ri] * ssu[ki, ci]
                                        else:
                                            total[index] = T.cast(total[index], 'float32') + T.cast(acc[index], 'float32')
                                            if not backward:
                                                utotal[index] = T.cast(utotal[index], 'float32') + T.cast(uacc[index], 'float32')
                                T.clear(acc)
                                if not backward:
                                    T.clear(uacc)
            for i, j in T.grid(mma.warp_rows, mma.warp_cols):
                for lid in T.unroll(mma.local_size_out):
                    r, c = T.meta_var(store.map_indices([lane, lid]))
                    row = by * bm + wm * (bm // rows) + i * mma.micro_size_x + r
                    col = bx * bn + wn * (bn // cols) + j * mma.micro_size_y + c
                    index = (i * mma.warp_cols + j) * mma.local_size_out + lid
                    if row < m and col < h:
                        value = T.cast(total[index], 'float32') if split else T.cast(acc[index], 'float32')
                        if not grouped:
                            value *= sa[0, 0] * sb[0, 0]
                        if backward:
                            # Match the ordinary Linear's output rounding before
                            # differentiating the activation in FP32.
                            d = T.cast(T.cast(value, od), 'float32')
                            g, u = T.cast(aux[row, col], 'float32'), T.cast(aux[row, h + col], 'float32')
                        else:
                            uv = T.cast(utotal[index], 'float32') if split else T.cast(uacc[index], 'float32')
                            if not grouped:
                                uv *= sa[0, 0] * sb[0, 0]
                            if has_bias:
                                value += T.cast(bias[col], 'float32')
                                uv += T.cast(bias[h + col], 'float32')
                            g, u = T.cast(T.cast(value, od), 'float32'), T.cast(T.cast(uv, od), 'float32')
                            if save:
                                aux[row, col], aux[row, h + col] = g, u
                        if activation == 'silu':
                            s = 1. / (1. + T.exp(-g))
                            act = g * s
                            if backward:
                                derivative = s * (1. + g * (1. - s))
                        elif activation == 'gelu':
                            t = T.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))
                            act = .5 * g * (1. + t)
                            if backward:
                                derivative = .5 * (1. + t) + .5 * g * (1. - t * t) * 0.7978845608028654 * (1. + .134145 * g * g)
                        else:
                            act = T.max(g, 0.)
                            if backward:
                                derivative = T.if_then_else(g > 0., 1., 0.)
                        if backward:
                            out[row, col] = d * u * derivative
                            out[row, h + col] = d * act
                        else:
                            if encode_output:
                                hidden[row - by * bm, col - bx * bn] = act * u
                            else:
                                out[row, col] = act * u
            if encode_output:
                # Quantize both orientations from the same rounded floating H
                # tile. H never makes a round trip through global BF16 storage.
                rv = T.alloc_fragment((bm * (bn // 32), 32), 'float32')
                rm = T.alloc_fragment((bm * (bn // 32),), 'float32')
                rs = T.alloc_fragment((bm * (bn // 32),), 'float32')
                for i, j in T.Parallel(bm * (bn // 32), 32):
                    rv[i, j] = T.cast(hidden[i // (bn // 32), i % (bn // 32) * 32 + j], 'float32')
                T.reduce_absmax(rv, rm, dim=1)
                for i in T.Parallel(bm * (bn // 32)):
                    rs[i] = T.exp2(T.ceil(T.log2(T.max(rm[i], 1e-30) / encode_limit)))
                    row, col = by * bm + i // (bn // 32), bx * bn // 32 + i % (bn // 32)
                    if row < m and col < T.ceildiv(h, 32):
                        out_scale[row, col] = T.cast(T.log2(rs[i]) + 127., 'uint8')
                for i, j in T.Parallel(bm * (bn // 32), 32):
                    row, col = by * bm + i // (bn // 32), bx * bn + i % (bn // 32) * 32 + j
                    if row < m and col < h:
                        v = rv[i, j] / rs[i]
                        out[row, col] = T.if_then_else(T.isnan(v), v, T.clamp(v, -encode_limit, encode_limit))
                if save_output_transpose:
                    tv = T.alloc_fragment((bn * (bm // 32), 32), 'float32')
                    tm = T.alloc_fragment((bn * (bm // 32),), 'float32')
                    ts = T.alloc_fragment((bn * (bm // 32),), 'float32')
                    for i, j in T.Parallel(bn * (bm // 32), 32):
                        tv[i, j] = T.cast(hidden[i % (bm // 32) * 32 + j, i // (bm // 32)], 'float32')
                    T.reduce_absmax(tv, tm, dim=1)
                    for i in T.Parallel(bn * (bm // 32)):
                        ts[i] = T.exp2(T.ceil(T.log2(T.max(tm[i], 1e-30) / encode_limit)))
                        row, col = bx * bn + i // (bm // 32), by * bm // 32 + i % (bm // 32)
                        if row < h and col < T.ceildiv(m, 32):
                            out_t_scale[row, col] = T.cast(T.log2(ts[i]) + 127., 'uint8')
                    for i, j in T.Parallel(bn * (bm // 32), 32):
                        row, col = bx * bn + i // (bm // 32), by * bm + i % (bm // 32) * 32 + j
                        if row < h and col < m:
                            v = tv[i, j] / ts[i]
                            out_t[row, col] = T.if_then_else(T.isnan(v), v, T.clamp(v, -encode_limit, encode_limit))
    return main


@lru_cache(maxsize=256)
def compiled(args, config, device):
    config = dict(config)
    copy_mode = config.pop('copy_mode', 'default')
    passes = dict(COPY_PASSES[copy_mode], **{'tl.disable_warp_specialized': True})
    with torch.cuda.device(device):
        return tilelang.compile(kernel.get_tir(*args, **config), target='cuda', pass_configs=passes)


def gemm(a, b, sa, sb, *, layer, out_dtype, activation, backward=False, aux=None,
         bias=None, save=True, group_a=None, group_b=None, encode_output=False,
         save_output_transpose=False):
    m, k = a.shape
    h = b.shape[0] if backward else b.shape[0] // 2
    grouped = layer._grouped_scaling
    ga = group_a or layer.input_group_shape
    gb = group_b or layer.weight_group_shape
    partial = math.gcd(ga[1], gb[1]) if grouped else 0
    if layer.accum_block_size:
        partial = math.gcd(partial, layer.accum_block_size) if partial else layer.accum_block_size
    if layer._native_scaling:
        partial = 0
    out = torch.empty((m, 2 * h if backward else h), device=a.device,
                       dtype=layer.fp8_dtype if encode_output else out_dtype)
    if aux is None:
        aux = torch.empty((m, 2 * h) if save else (0, 0), device=a.device, dtype=out_dtype)
    bias = bias if bias is not None else torch.empty(0, device=a.device, dtype=out_dtype)
    out_scale = torch.empty((m, (h + 31) // 32) if encode_output else (0, 0), device=a.device, dtype=torch.uint8)
    out_t = out.new_empty((h, m) if save_output_transpose else (0, 0), dtype=a.dtype)
    out_t_scale = torch.empty((h, (m + 31) // 32) if save_output_transpose else (0, 0), device=a.device, dtype=torch.uint8)
    output_limit = layer._group_limit(layer.input_group_shape, layer.fp8_dtype) or torch.finfo(layer.fp8_dtype).max
    args = (m, h, k, _dtype(a), _dtype(b), str(out_dtype).removeprefix('torch.'), _dtype(bias), bool(bias.numel()),
            *ga, *gb, layer.input_quantizer.scaling,
            {'fp32':'float32', 'fp16':'float16'}[layer.accum_dtype], partial,
            {'fp32':'float32', 'bf16':'bfloat16'}[layer.accum_reduce_dtype], activation, backward, save,
            encode_output, output_limit, save_output_transpose)
    operands = (a, b, sa if grouped else sa.reshape(1, 1), sb if grouped else sb.reshape(1, 1),
                bias, aux, out_scale, out_t, out_t_scale, out)
    config = {}
    if layer.autotune:
        from ._mlp_tuning import get_config
        config = get_config(args, operands, a.device.index)
    compiled(args, tuple(sorted(config.items())), a.device.index)(*operands)
    return out, aux, (out_scale, out_t, out_t_scale)
