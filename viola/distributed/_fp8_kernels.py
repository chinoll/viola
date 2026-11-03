"""TileLang FP8 communication codecs; packets contain scales and FP8 codes."""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


def views(packet, tile_size, fp8_dtype):
    groups = packet.shape[1] // (tile_size + 4)
    width = groups * tile_size
    codes = packet[:, :width].view(fp8_dtype)
    scales = packet[:, width:width + groups * 4].view(torch.float32)
    return codes, scales


@tilelang.jit
def _pack(n, rows, groups, size, fp8):
    width = groups * size
    stride = (groups * (size + 4) + 15) // 16 * 16
    maximum = 448.0 if fp8 == "float8_e4m3" else 57344.0

    @T.prim_func
    def kernel(x: T.Tensor((n,), "float32"),
               q: T.StridedTensor((rows, width), (stride, 1), fp8),
               scales: T.StridedTensor((rows, groups), (stride // 4, 1), "float32")):
        with T.Kernel(groups, rows, threads=128) as (group, row):
            values = T.alloc_fragment((size,), "float32")
            magnitudes = T.alloc_fragment((size,), "float32")
            amax = T.alloc_fragment((1,), "float32")
            for i in T.Parallel(size):
                index = row * width + group * size + i
                values[i] = T.if_then_else(index < n, x[index], 0.0)
                # Preserve nonfinite inputs through the codec. NaN must not be
                # silently dropped by a max reduction or a saturating cast.
                magnitudes[i] = T.if_then_else(T.isnan(values[i]), T.infinity("float32"), T.abs(values[i]))
            T.reduce_max(magnitudes, amax, dim=0)
            amax[0] = T.max(amax[0], 1e-30) / maximum
            scales[row, group] = amax[0]
            for i in T.Parallel(size):
                q[row, group * size + i] = values[i] / amax[0]
    return kernel


@tilelang.jit
def _unpack(n, rows, groups, size, fp8, reduce):
    width = groups * size
    stride = (groups * (size + 4) + 15) // 16 * 16

    @T.prim_func
    def kernel(q: T.StridedTensor((rows, width), (stride, 1), fp8),
               scales: T.StridedTensor((rows, groups), (stride // 4, 1), "float32"),
               out: T.Tensor((n,), "float32")):
        with T.Kernel(groups, 1 if reduce else rows, threads=128) as (group, row):
            result = T.alloc_fragment((size,), "float32")
            T.clear(result)
            if reduce:
                for rank in T.serial(rows):
                    for i in T.Parallel(size):
                        result[i] += T.cast(q[rank, group * size + i], "float32") * scales[rank, group]
            else:
                for i in T.Parallel(size):
                    result[i] = T.cast(q[row, group * size + i], "float32") * scales[row, group]
            for i in T.Parallel(size):
                index = (0 if reduce else row * width) + group * size + i
                if index < n:
                    out[index] = result[i]
    return kernel


@lru_cache(maxsize=256)
def _compiled_pack(n, rows, groups, size, fp8, device):
    with torch.cuda.device(device):
        return _pack.compile(n, rows, groups, size, fp8)


@lru_cache(maxsize=256)
def _compiled_unpack(n, rows, groups, size, fp8, reduce, device):
    with torch.cuda.device(device):
        return _unpack.compile(n, rows, groups, size, fp8, reduce)


def pack(source, packet, tile_size, fp8_dtype):
    q, scales = views(packet, tile_size, fp8_dtype)
    name = "float8_e4m3" if fp8_dtype == torch.float8_e4m3fn else "float8_e5m2"
    _compiled_pack(source.numel(), packet.shape[0], scales.shape[1], tile_size, name, source.device.index)(source, q, scales)


def unpack(packet, out, tile_size, fp8_dtype, *, reduce=False):
    q, scales = views(packet, tile_size, fp8_dtype)
    name = "float8_e4m3" if fp8_dtype == torch.float8_e4m3fn else "float8_e5m2"
    _compiled_unpack(out.numel(), packet.shape[0], scales.shape[1], tile_size, name, reduce, out.device.index)(q, scales, out)
