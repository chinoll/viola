"""Autograd for grouped FP8 and native MXFP8; saved inputs remain FP8."""

import torch

from . import _grouped_kernels as kernels


class GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, layer, need_backward):
        shape = x.shape
        x = x.reshape(-1, layer.in_features).contiguous()
        gx, gw = layer.input_group_shape, layer.weight_group_shape
        limit = layer._group_limit(gx, layer.fp8_dtype)
        qx, sx = kernels.quantize(x, layer.fp8_dtype, gx, native=layer._native_scaling, limit=limit)
        y = layer._group_gemm(qx, weight, sx, layer.weight_scale, gx, gw, out_dtype=x.dtype, bias=bias)
        del qx, sx
        if need_backward:
            save_fp8 = weight.requires_grad and layer.saved_input_dtype == 'fp8'
            if save_fp8:
                saved_x, saved_scale = kernels.quantize(x, layer.fp8_dtype, gx, native=layer._native_scaling,
                                                        limit=limit, transposed=True)
            elif weight.requires_grad:
                dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[layer.saved_input_dtype]
                saved_x, saved_scale = x.to(dtype), x.new_empty(0)
            else:
                saved_x, saved_scale = x.new_empty(0), x.new_empty(0)
            ctx.save_for_backward(saved_x, saved_scale, weight, layer.weight_scale,
                                  layer.weight_transpose, layer.weight_transpose_scale)
            ctx.layer, ctx.shape, ctx.x_dtype = layer, shape, x.dtype
            ctx.save_fp8, ctx.input_limit = save_fp8, limit
            ctx.weight_grad_dtype = weight.grad_dtype
            ctx.bias_grad_dtype = bias.grad_dtype if bias is not None else None
        return y.reshape(*shape[:-1], layer.out_features)

    @staticmethod
    def backward(ctx, dy):
        saved_x, saved_scale, w, sw, cached_wt, cached_st = ctx.saved_tensors
        layer = ctx.layer
        dy = dy.reshape(-1, layer.out_features).contiguous()
        gg, gx, gw = layer.grad_group_shape, layer.input_group_shape, layer.weight_group_shape
        native = layer._native_scaling
        limit = layer._group_limit(gg, layer.backward_fp8_dtype)
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            qg, sg = kernels.quantize(dy, layer.backward_fp8_dtype, gg, native=native, limit=limit)
            if cached_wt.numel():
                wt, st = cached_wt, cached_st
            else:
                wt, st = kernels.transpose(w, sw, gw, native=native, limit=layer._group_limit(gw, layer.fp8_dtype, weight=True))
            dx = layer._group_gemm(qg, wt, sg, st, gg, kernels.transpose_group(gw), out_dtype=ctx.x_dtype).reshape(ctx.shape)
            del qg, sg, wt, st
        if ctx.needs_input_grad[1]:
            # Both directions are quantized from the incoming floating dY.
            gt, sgt = kernels.quantize(dy, layer.backward_fp8_dtype, gg, native=native, limit=limit, transposed=True)
            if ctx.save_fp8:
                xt, sxt = saved_x, saved_scale
            else:
                xt, sxt = kernels.quantize(saved_x, layer.fp8_dtype, gx, native=native,
                                           limit=ctx.input_limit, transposed=True)
            dw = layer._group_gemm(gt, xt, sgt, sxt, gg, gx, out_dtype=ctx.weight_grad_dtype)
        if ctx.needs_input_grad[2]:
            db = dy.sum(0, dtype=ctx.bias_grad_dtype)
        return dx, dw, db, None, None
