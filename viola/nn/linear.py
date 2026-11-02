"""FP8/BF16/FP16 weight storage with TileLang FP8 forward/backward."""

import math

import torch
from torch import nn

from . import _linear_kernels as kernels
from . import _rotation_kernels as rotation
from . import _grouped_kernels as grouped
from ..precision import settings


class _Quantizer(nn.Module):
    def __init__(self, scaling, history_len, device):
        super().__init__()
        self.scaling = scaling
        self.index = 0
        self.register_buffer("history", torch.zeros(history_len, device=device, dtype=torch.float32))

    def scale(self, x, fp8_dtype=torch.float8_e4m3fn, quantization_limit=None, rotation_size=0):
        implementation = rotation if rotation_size else kernels
        scale = implementation.tensor_scale(x, history=self.history, index=self.index,
                                     delayed=self.scaling == "delayed", fp8_dtype=fp8_dtype,
                                     quantization_limit=quantization_limit)
        self.index += 1
        return scale

    def get_extra_state(self):
        return {"index": self.index, "scaling": self.scaling}

    def set_extra_state(self, state):
        self.index, self.scaling = state["index"], state["scaling"]


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, layer, need_backward):
        shape = x.shape
        x = x.reshape(-1, layer.in_features).contiguous()
        input_limit = layer._quantization_limit(max(x.shape), layer.fp8_dtype)
        sx = layer.input_quantizer.scale(x, layer.fp8_dtype, input_limit, layer.rotation_size)
        quantize = rotation.quantize if layer.rotation_size else kernels.quantize
        save_fp8 = need_backward and weight.requires_grad and layer.saved_input_dtype == "fp8"
        qx, qxt = quantize(x, sx, transpose=save_fp8, fp8_dtype=layer.fp8_dtype,
                                  quantization_limit=input_limit)
        if layer.weight_dtype != "fp8":
            weight_limit = layer._quantization_limit(max(weight.shape), layer.fp8_dtype)
            sw = layer.weight_quantizer.scale(weight, layer.fp8_dtype, weight_limit, layer.rotation_size)
            qw, wt = quantize(weight, sw, fp8_dtype=layer.fp8_dtype,
                              transpose=need_backward and ctx.needs_input_grad[0] and layer.cache_weight_transpose,
                              quantization_limit=weight_limit)
        else:
            qw, sw, wt = weight, layer.weight_scale, layer.weight_transpose
        y = kernels.gemm(qx, qw, sx, sw, out_dtype=x.dtype, bias=bias,
                         autotune=layer.autotune, accum_dtype=layer.accum_dtype,
                         accum_block_size=layer.accum_block_size, accum_reduce_dtype=layer.accum_reduce_dtype)
        del qw

        if need_backward:
            if save_fp8:
                saved_x = qxt
            elif weight.requires_grad:
                saved_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[layer.saved_input_dtype]
                saved_x = x.to(saved_dtype)
            else:
                saved_x = x.new_empty(0)
            # Save the exact scale used for this tensor, not a mutable quantizer state.
            ctx.save_for_backward(saved_x, sx, weight, sw, wt)
            ctx.layer, ctx.shape, ctx.x_dtype = layer, shape, x.dtype
            ctx.save_fp8 = save_fp8
            ctx.input_limit = input_limit
            ctx.weight_grad_dtype = weight.grad_dtype
            ctx.bias_grad_dtype = bias.grad_dtype if bias is not None else None
        return y.reshape(*shape[:-1], layer.out_features)

    @staticmethod
    def backward(ctx, dy):
        saved_x, sx, weight, sw, cached_wt = ctx.saved_tensors
        layer = ctx.layer
        dy = dy.reshape(-1, layer.out_features).contiguous()
        needs_x, needs_w, needs_b = ctx.needs_input_grad[:3]
        grad_limit = layer._quantization_limit(max(dy.shape), layer.backward_fp8_dtype)
        sg = layer.grad_quantizer.scale(dy, layer.backward_fp8_dtype, grad_limit)
        # memory: do dX first, release W.T, then materialize dY.T for dW.
        # fast: quantize both dY layouts together and reuse the persistent W.T.
        qg, qgt = kernels.quantize(dy, sg, rowwise=needs_x,
                                  transpose=needs_w and (layer.mode == "fast" or not needs_x),
                                  fp8_dtype=layer.backward_fp8_dtype, quantization_limit=grad_limit)
        dx = dw = db = None
        if needs_x:
            if cached_wt.numel():
                wt = cached_wt
            elif layer.weight_dtype != "fp8":
                quantize = rotation.quantize if layer.rotation_size else kernels.quantize
                # Reuse this forward's scale; do not advance delayed history in backward.
                _, wt = quantize(weight, sw, rowwise=False, transpose=True, fp8_dtype=layer.fp8_dtype,
                                  quantization_limit=layer._quantization_limit(max(weight.shape), layer.fp8_dtype))
            else:
                wt = kernels.transpose(weight)
            dx = kernels.gemm(qg, wt, sg, sw, out_dtype=ctx.x_dtype, autotune=layer.autotune,
                              accum_dtype=layer.accum_dtype, accum_block_size=layer.accum_block_size,
                              accum_reduce_dtype=layer.accum_reduce_dtype)
            if layer.rotation_size:
                rotation.inverse_(dx)
            dx = dx.reshape(ctx.shape)
            del wt
        if needs_w:
            if not qgt.numel():
                # Requantize from the existing dY with the same scale after
                # releasing its row layout, avoiding two live FP8 dY copies.
                del qg
                _, qgt = kernels.quantize(dy, sg, rowwise=False, transpose=True, fp8_dtype=layer.backward_fp8_dtype,
                                          quantization_limit=grad_limit)
            else:
                del qg
            if ctx.save_fp8:
                xt = saved_x
            else:
                quantize = rotation.quantize if layer.rotation_size else kernels.quantize
                _, xt = quantize(saved_x, sx, rowwise=False, transpose=True, fp8_dtype=layer.fp8_dtype,
                                         quantization_limit=ctx.input_limit)
            dw = kernels.gemm(qgt, xt, sg, sx, out_dtype=ctx.weight_grad_dtype, autotune=layer.autotune,
                              accum_dtype=layer.accum_dtype, accum_block_size=layer.accum_block_size,
                              accum_reduce_dtype=layer.accum_reduce_dtype)
            if layer.rotation_size:
                rotation.inverse_(dw)
        if needs_b:
            db = dy.sum(dim=0, dtype=ctx.bias_grad_dtype)
        # Derivatives refer to real weights, including when storage uses FP8 codes.
        return dx, dw, db, None, None


class Linear(nn.Module):
    """A CUDA FP8 Linear with configurable weight storage and caching.

    Construct/move the model, construct its optimizer, then call
    ``viola.optim.bind_optimizer(model, optimizer)`` before training. Forward inputs
    and outputs are ordinary FP16/BF16/FP32 tensors. ``weight_dtype`` selects FP8
    codes or BF16/FP16 real weights. All storage formats compute with FP8 GEMMs.
    ``dequantize_weight()`` returns real weights in the original coordinates.
    """

    def __init__(self, in_features, out_features, bias=True, *, mode="memory",
                 weight_dtype="fp8", forward_dtype="e4m3", backward_dtype="e5m2",
                 cache_weight_transpose=None, saved_input_dtype="fp8",
                 scaling="current", history_len=16, autotune=False, accum_dtype="fp32",
                 accum_block_size=0, accum_reduce_dtype="fp32", quantization_limit=None,
                 rotation_size=0,
                 weight_group_shape=(32, 32), input_group_shape=(1, 32), grad_group_shape=(1, 32),
                 device=None, dtype=None):
        super().__init__()
        assert mode in ("memory", "fast")
        assert weight_dtype in ("fp8", "bf16", "fp16")
        self.weight_dtype = weight_dtype
        formats = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
        self.forward_dtype, self.backward_dtype = forward_dtype, backward_dtype
        self.fp8_dtype, self.backward_fp8_dtype = formats[forward_dtype], formats[backward_dtype]
        assert saved_input_dtype in ("fp8", "bf16", "fp16", "fp32")
        assert scaling in ("current", "delayed", "grouped", "mxfp8") and history_len > 0
        self._grouped_scaling = scaling in ("grouped", "mxfp8")
        self._native_scaling = scaling == "mxfp8"
        self.weight_group_shape = tuple(weight_group_shape)
        self.input_group_shape = tuple(input_group_shape)
        self.grad_group_shape = tuple(grad_group_shape)
        if self._grouped_scaling:
            assert rotation_size == 0, "Grouped scaling does not combine with rotation_size."
            for gr, gc in (self.weight_group_shape, self.input_group_shape, self.grad_group_shape):
                assert gr > 0 and gr & (gr - 1) == 0
                assert gc >= 32 and gc & (gc - 1) == 0
            if self._native_scaling:
                assert accum_dtype == "fp32" and accum_block_size == 0 and accum_reduce_dtype == "fp32", "MXFP8 MMA uses FP32 accumulation."
        self.in_features, self.out_features = in_features, out_features
        self.mode, self.saved_input_dtype = mode, saved_input_dtype
        self.autotune = autotune
        assert accum_dtype in ("fp32", "fp16")
        assert accum_reduce_dtype in ("fp32", "bf16")
        assert accum_block_size >= 0 and accum_block_size % 32 == 0
        self.accum_dtype = accum_dtype
        self.accum_block_size = accum_block_size
        self.accum_reduce_dtype = accum_reduce_dtype
        self.quantization_limit = quantization_limit
        assert rotation_size in (0, 64)
        assert not rotation_size or in_features % rotation_size == 0
        self.rotation_size = rotation_size
        self.cache_weight_transpose = mode == "fast" if cache_weight_transpose is None else cache_weight_transpose
        config = settings()
        master_dtype = config["master_weight_dtype"] or torch.float32
        gradient_dtype = config["gradient_dtype"] or torch.float32
        dtype = dtype or torch.get_default_dtype()
        # FP8 initialization is handed to the optimizer as its master. BF16/FP16
        # storage starts directly in the parameter, without an initializer copy.
        self._initial_weight = (torch.empty((out_features, in_features), device=device, dtype=master_dtype)
                                if weight_dtype == "fp8" else None)
        self._master_ref = None
        storage_dtype = {"fp8": self.fp8_dtype, "bf16": torch.bfloat16, "fp16": torch.float16}[weight_dtype]
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=storage_dtype))
        self.weight.grad_dtype = gradient_dtype
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        if self.bias is not None:
            self.bias.grad_dtype = gradient_dtype
        scale_shape = grouped.scale_shape((out_features, in_features), self.weight_group_shape) if self._grouped_scaling else (1,)
        if weight_dtype != "fp8":
            scale_shape = (0, 0) if self._grouped_scaling else (0,)
        scale_dtype = torch.uint8 if self._native_scaling else torch.float32
        self.register_buffer("weight_scale", torch.full(scale_shape, 127 if self._native_scaling else 1,
                                                       device=device, dtype=scale_dtype), persistent=weight_dtype == "fp8")
        self.register_buffer("weight_transpose", torch.empty(0, 0, device=device, dtype=self.fp8_dtype), persistent=False)
        if self._grouped_scaling:
            self.register_buffer("weight_transpose_scale", torch.empty(0, 0, device=device, dtype=scale_dtype), persistent=False)
        self.weight_quantizer = _Quantizer(scaling, history_len, device)
        self.input_quantizer = _Quantizer(scaling, history_len, device)
        self.grad_quantizer = _Quantizer(scaling, history_len, device)
        self.reset_parameters()

    def _quantization_limit(self, reduction, fp8_dtype):
        if self.quantization_limit is not None:
            return min(self.quantization_limit, torch.finfo(fp8_dtype).max)
        if self.accum_dtype == "fp32":
            return None
        # FP8 codes are accumulated before their scales are restored. Reserve
        # FP16 range for the reduction, not just for each individual operand.
        if self.accum_block_size:
            reduction = self.accum_block_size
        limit = 2.0 ** math.floor(math.log2(math.sqrt(32768.0 / reduction)))
        return min(limit, torch.finfo(fp8_dtype).max)

    def _group_limit(self, shape, dtype, weight=False):
        if self.quantization_limit is not None:
            return min(self.quantization_limit, torch.finfo(dtype).max)
        if self.accum_dtype == "fp32":
            return None
        # W's 2-D block is also used transposed; protect both K directions.
        reduction = max(shape) if weight else shape[1]
        if self.accum_block_size:
            reduction = math.gcd(reduction, self.accum_block_size)
        return min(2.0 ** math.floor(math.log2(math.sqrt(32768.0 / reduction))), torch.finfo(dtype).max)

    def _group_gemm(self, a, b, sa, sb, ga, gb, *, out_dtype, bias=None):
        return grouped.gemm(a, b, sa, sb, ga, gb, out_dtype=out_dtype, bias=bias,
                            native=self._native_scaling, autotune=self.autotune,
                            accum_dtype=self.accum_dtype, accum_block_size=self.accum_block_size,
                            accum_reduce_dtype=self.accum_reduce_dtype)

    @torch.no_grad()
    def reset_parameters(self):
        master = self._master_ref() if self._master_ref is not None else (
            self.weight if self.weight_dtype != "fp8" else self._initial_weight)
        nn.init.kaiming_uniform_(master, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))
        self.refresh_weight(master)

    @torch.no_grad()
    def refresh_weight(self, master):
        """Called after an optimizer update, directly from its updated master."""
        if self.weight_dtype != "fp8":
            if master is not self.weight:
                self.weight.copy_(master)
            return
        if self._grouped_scaling:
            if self._native_scaling and master.is_cuda:
                assert torch.cuda.get_device_capability(master.device)[0] == 12, "Native MXFP8 backend targets SM120/SM121."
            limit = self._group_limit(self.weight_group_shape, self.fp8_dtype, weight=True)
            grouped.quantize(master, self.fp8_dtype, self.weight_group_shape, native=self._native_scaling,
                             limit=limit, out=self.weight, scale_out=self.weight_scale)
            if self.cache_weight_transpose:
                self.weight_transpose, self.weight_transpose_scale = grouped.transpose(
                    self.weight, self.weight_scale, self.weight_group_shape, native=self._native_scaling, limit=limit)
            return
        weight_limit = self._quantization_limit(max(self.in_features, self.out_features), self.fp8_dtype)
        if self.cache_weight_transpose and self.weight_transpose.numel() == 0:
            self.weight_transpose = torch.empty((self.in_features, self.out_features), device=master.device,
                                                dtype=self.fp8_dtype)
        if master.is_cuda:
            scale = self.weight_quantizer.scale(master, self.fp8_dtype, weight_limit, self.rotation_size)
            self.weight_scale.copy_(scale)
            quantize = rotation.quantize if self.rotation_size else kernels.quantize
            quantize(master, scale, transpose=self.cache_weight_transpose,
                             fp8_dtype=self.fp8_dtype, out=self.weight, out_t=self.weight_transpose,
                             quantization_limit=weight_limit)
        else:
            # CPU construction only; CUDA training always uses the TileLang kernels.
            if self.rotation_size:
                master = rotation.rotate_cpu(master)
            fp8_max = torch.finfo(self.fp8_dtype).max if weight_limit is None else weight_limit
            amax = master.float().abs().max()
            quantizer = self.weight_quantizer
            maximum = quantizer.history.max() if quantizer.scaling == "delayed" and quantizer.index else amax
            scale = maximum.clamp_min(1e-30) / fp8_max
            quantizer.history[quantizer.index % quantizer.history.numel()] = amax
            quantizer.index += 1
            self.weight_scale.copy_(scale)
            self.weight.copy_((master.float() / scale).clamp(-fp8_max, fp8_max).to(self.weight.dtype))
            if self.cache_weight_transpose:
                self.weight_transpose.copy_(self.weight.T)

    def dequantize_weight(self):
        if self.weight_dtype != "fp8":
            return self.weight.float()
        if self._grouped_scaling:
            return grouped.dequantize(self.weight, self.weight_scale, self.weight_group_shape)
        weight = self.weight.float() * self.weight_scale
        if self.rotation_size:
            return rotation.inverse_(weight) if weight.is_cuda else rotation.rotate_cpu(weight)
        return weight

    @classmethod
    @torch.no_grad()
    def from_float(cls, linear, **kwargs):
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None,
                    device=linear.weight.device, dtype=linear.weight.dtype, **kwargs)
        master = layer.weight if layer.weight_dtype != "fp8" else layer._initial_weight
        master.copy_(linear.weight)
        # A pretrained weight must not inherit the temporary initializer's amax.
        layer.weight_quantizer.history.zero_()
        layer.weight_quantizer.index = 0
        layer.refresh_weight(master)
        if linear.bias is not None:
            layer.bias.copy_(linear.bias)
        layer.weight.requires_grad_(linear.weight.requires_grad)
        if layer.bias is not None:
            layer.bias.requires_grad_(linear.bias.requires_grad)
        return layer

    def forward(self, x):
        assert x.is_cuda, "viola.nn.Linear executes on CUDA."
        if torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        need_backward = torch.is_grad_enabled() and (x.requires_grad or self.weight.requires_grad
                                                    or (self.bias is not None and self.bias.requires_grad))
        if self._grouped_scaling:
            if self._native_scaling:
                assert torch.cuda.get_device_capability(x.device)[0] == 12, "Native MXFP8 backend targets SM120/SM121."
            from ._grouped_linear import GroupedLinear
            return GroupedLinear.apply(x, self.weight, self.bias, self, need_backward)
        return _Linear.apply(x, self.weight, self.bias, self, need_backward)

    def _apply(self, fn, recurse=True):
        # Device moves preserve the selected weight storage and scale/history
        # dtypes. Move the model before binding its optimizer.
        weight = self._parameters["weight"]
        self._parameters["weight"] = None
        grad_dtype = weight.grad_dtype
        fixed = [(self, name, self._buffers.pop(name)) for name in ("weight_scale", "weight_transpose")]
        if self._grouped_scaling:
            fixed.append((self, "weight_transpose_scale", self._buffers.pop("weight_transpose_scale")))
        for quantizer in (self.weight_quantizer, self.input_quantizer, self.grad_quantizer):
            fixed.append((quantizer, "history", quantizer._buffers.pop("history")))
        super()._apply(fn, recurse)
        def move(t):
            return t.to(device=fn(torch.empty(0, device=t.device, dtype=t.dtype)).device)
        weight.data = move(weight.data)
        if weight.grad is not None:
            weight.grad = move(weight.grad)
        weight.grad_dtype = grad_dtype
        self._parameters["weight"] = weight
        for module, name, tensor in fixed:
            module._buffers[name] = move(tensor)
        if self._initial_weight is not None:
            self._initial_weight = move(self._initial_weight)
        if self.bias is not None:
            self.bias.grad_dtype = grad_dtype
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        expected = {"fp8": self.fp8_dtype, "bf16": torch.bfloat16, "fp16": torch.float16}[self.weight_dtype]
        assert state_dict[prefix + "weight"].dtype == expected, "Construct Linear with the checkpoint's weight storage and FP8 format."
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._initial_weight = None
        if self.cache_weight_transpose and self.weight_dtype == "fp8":
            if self._grouped_scaling:
                self.weight_transpose, self.weight_transpose_scale = grouped.transpose(
                    self.weight, self.weight_scale, self.weight_group_shape, native=self._native_scaling,
                    limit=self._group_limit(self.weight_group_shape, self.fp8_dtype, weight=True))
            elif self.weight.is_cuda:
                self.weight_transpose = kernels.transpose(self.weight)
            else:
                self.weight_transpose = self.weight.T.contiguous()

    def extra_repr(self):
        groups = (f", weight_group_shape={self.weight_group_shape}, input_group_shape={self.input_group_shape}, "
                  f"grad_group_shape={self.grad_group_shape}" if self._grouped_scaling else "")
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, mode={self.mode!r}, weight_dtype={self.weight_dtype!r}, "
                f"forward_dtype={self.forward_dtype!r}, backward_dtype={self.backward_dtype!r}, "
                f"cache_weight_transpose={self.cache_weight_transpose}, "
                f"saved_input_dtype={self.saved_input_dtype!r}, autotune={self.autotune}, "
                f"scaling={self.weight_quantizer.scaling!r}, "
                f"accum_dtype={self.accum_dtype!r}, accum_block_size={self.accum_block_size}, "
                f"accum_reduce_dtype={self.accum_reduce_dtype!r}, rotation_size={self.rotation_size}{groups}")
