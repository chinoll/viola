"""Grouped/MXFP8 autograd with FP8 or BF16/FP16 weight storage."""

import torch

from . import _grouped_kernels as kernels


class GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, layer, need_backward):
        shape = x.shape
        encoded = getattr(ctx, 'encoded_input', None)
        x_dtype = ctx.input_dtype if encoded is not None else x.dtype
        x = x.reshape(-1, layer.in_features)
        gx, gw = layer.input_group_shape, layer.weight_group_shape
        limit = layer._group_limit(gx, layer.fp8_dtype)
        if encoded is not None:
            qx, sx = x, encoded[0]
        else:
            x = x.contiguous()
            qx, sx = kernels.quantize(x, layer.fp8_dtype, gx, native=layer._native_scaling, limit=limit)
        if layer.weight_dtype != 'fp8':
            weight_limit = layer._group_limit(gw, layer.fp8_dtype, weight=True)
            qw, sw = kernels.quantize(weight, layer.fp8_dtype, gw, native=layer._native_scaling, limit=weight_limit)
        else:
            qw, sw = weight, layer.weight_scale
        if getattr(ctx, 'gated_forward', None):
            from ._mlp_kernels import gemm
            y, ctx.gate_aux, ctx.gate_encoded = gemm(qx, qw, sx, sw, layer=layer, out_dtype=x_dtype,
                                  bias=bias, activation=ctx.gated_forward, save=need_backward,
                                  encode_output=getattr(ctx, 'encode_output', False),
                                  save_output_transpose=getattr(ctx, 'encode_transpose', False))
        else:
            y = layer._group_gemm(qx, qw, sx, sw, gx, gw, out_dtype=x_dtype, bias=bias)
        del qx, sx, qw
        if need_backward:
            wt, st = layer.weight_transpose, layer.weight_transpose_scale
            if layer.weight_dtype != 'fp8' and ctx.needs_input_grad[0] and layer.cache_weight_transpose:
                wt, st = kernels.quantize(weight, layer.fp8_dtype, kernels.transpose_group(gw),
                                          native=layer._native_scaling, limit=weight_limit, transposed=True)
            save_fp8 = weight.requires_grad and layer.saved_input_dtype == 'fp8'
            if save_fp8:
                if encoded is not None:
                    saved_x, saved_scale = encoded[1:]
                else:
                    saved_x, saved_scale = kernels.quantize(x, layer.fp8_dtype, gx, native=layer._native_scaling,
                                                        limit=limit, transposed=True)
            elif weight.requires_grad:
                dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[layer.saved_input_dtype]
                saved_x, saved_scale = x.to(dtype), x.new_empty(0)
            else:
                saved_x, saved_scale = x.new_empty(0), x.new_empty(0)
            ctx.save_for_backward(saved_x, saved_scale, weight, sw, wt, st)
            ctx.layer, ctx.shape, ctx.x_dtype = layer, shape, x_dtype
            ctx.save_fp8, ctx.input_limit = save_fp8, limit
            ctx.weight_grad_dtype = weight.grad_dtype
            ctx.bias_grad_dtype = bias.grad_dtype if bias is not None else None
        return y.reshape(*shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        saved_x, saved_scale, w, sw, cached_wt, cached_st = ctx.saved_tensors
        layer = ctx.layer
        dy = dy.reshape(-1, layer.out_features).contiguous()
        gg, gx, gw = layer.grad_group_shape, layer.input_group_shape, layer.weight_group_shape
        native = layer._native_scaling
        limit = layer._group_limit(gg, layer.backward_fp8_dtype)
        dx = dw = db = None
        # The fused dX is two branches wide; produce it after dW's temporaries.
        order = (1, 0) if getattr(ctx, 'gated_backward', None) else (0, 1)
        for operation in order:
            if operation == 0 and ctx.needs_input_grad[0]:
                qg, sg = kernels.quantize(dy, layer.backward_fp8_dtype, gg, native=native, limit=limit)
                if cached_wt.numel():
                    wt, st = cached_wt, cached_st
                elif layer.weight_dtype != 'fp8':
                    wt, st = kernels.quantize(w, layer.fp8_dtype, kernels.transpose_group(gw), native=native,
                                              limit=layer._group_limit(gw, layer.fp8_dtype, weight=True), transposed=True)
                else:
                    wt, st = kernels.transpose(w, sw, gw, native=native, limit=layer._group_limit(gw, layer.fp8_dtype, weight=True))
                if getattr(ctx, 'gated_backward', None):
                    from ._mlp_kernels import gemm
                    dx, _, _ = gemm(qg, wt, sg, st, layer=layer, out_dtype=ctx.x_dtype, backward=True,
                                 activation=ctx.gated_backward, aux=ctx.gate_aux,
                                 group_a=gg, group_b=kernels.transpose_group(gw))
                else:
                    dx = layer._group_gemm(qg, wt, sg, st, gg, kernels.transpose_group(gw), out_dtype=ctx.x_dtype)
                dx = dx.reshape(*ctx.shape[:-1], dx.shape[-1])
                del qg, sg, wt, st
            if operation == 1 and ctx.needs_input_grad[1]:
                # Both directions are quantized from the incoming floating dY.
                gt, sgt = kernels.quantize(dy, layer.backward_fp8_dtype, gg, native=native, limit=limit, transposed=True)
                if ctx.save_fp8:
                    xt, sxt = saved_x, saved_scale
                else:
                    xt, sxt = kernels.quantize(saved_x, layer.fp8_dtype, gx, native=native,
                                               limit=ctx.input_limit, transposed=True)
                dw = layer._group_gemm(gt, xt, sgt, sxt, gg, gx, out_dtype=ctx.weight_grad_dtype)
                del gt, sgt, xt, sxt
        if ctx.needs_input_grad[2]:
            db = dy.sum(0, dtype=ctx.bias_grad_dtype)
        return dx, dw, db, None, None
