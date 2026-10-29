"""FP8 model weights, FP8 saved activations, and TileLang forward/backward."""

import math

import torch
from torch import nn

from . import _linear_kernels as kernels
from ..precision import settings


class _Quantizer(nn.Module):
    def __init__(self, scaling, history_len, device):
        super().__init__()
        self.scaling = scaling
        self.index = 0
        self.register_buffer("history", torch.zeros(history_len, device=device, dtype=torch.float32))

    def scale(self, x, fp8_dtype=torch.float8_e4m3fn):
        scale = kernels.tensor_scale(x, history=self.history, index=self.index,
                                     delayed=self.scaling == "delayed", fp8_dtype=fp8_dtype)
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
        sx = layer.input_quantizer.scale(x, layer.fp8_dtype)
        save_fp8 = need_backward and weight.requires_grad and layer.saved_input_dtype == "fp8"
        qx, qxt = kernels.quantize(x, sx, transpose=save_fp8, fp8_dtype=layer.fp8_dtype)
        y = kernels.gemm(qx, weight, sx, layer.weight_scale, out_dtype=x.dtype, bias=bias,
                         autotune=layer.autotune)

        if need_backward:
            if save_fp8:
                saved_x = qxt
            elif weight.requires_grad:
                saved_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[layer.saved_input_dtype]
                saved_x = x.to(saved_dtype)
            else:
                saved_x = x.new_empty(0)
            # Save the exact scale used for this tensor, not a mutable quantizer state.
            ctx.save_for_backward(saved_x, sx, weight, layer.weight_scale,
                                  layer.weight_transpose)
            ctx.layer, ctx.shape, ctx.x_dtype = layer, shape, x.dtype
            ctx.save_fp8 = save_fp8
            ctx.weight_grad_dtype = weight.grad_dtype
            ctx.bias_grad_dtype = bias.grad_dtype if bias is not None else None
        return y.reshape(*shape[:-1], layer.out_features)

    @staticmethod
    def backward(ctx, dy):
        saved_x, sx, weight, sw, cached_wt = ctx.saved_tensors
        layer = ctx.layer
        dy = dy.reshape(-1, layer.out_features).contiguous()
        needs_x, needs_w, needs_b = ctx.needs_input_grad[:3]
        sg = layer.grad_quantizer.scale(dy, layer.backward_fp8_dtype)
        # memory: do dX first, release W.T, then materialize dY.T for dW.
        # speed: quantize both dY layouts together and reuse the persistent W.T.
        qg, qgt = kernels.quantize(dy, sg, rowwise=needs_x,
                                  transpose=needs_w and (layer.mode == "speed" or not needs_x),
                                  fp8_dtype=layer.backward_fp8_dtype)
        dx = dw = db = None
        if needs_x:
            wt = cached_wt if cached_wt.numel() else kernels.transpose(weight)
            dx = kernels.gemm(qg, wt, sg, sw, out_dtype=ctx.x_dtype, autotune=layer.autotune).reshape(ctx.shape)
            del wt
        if needs_w:
            if not qgt.numel():
                # Requantize from the existing dY with the same scale after
                # releasing its row layout, avoiding two live FP8 dY copies.
                del qg
                _, qgt = kernels.quantize(dy, sg, rowwise=False, transpose=True, fp8_dtype=layer.backward_fp8_dtype)
            else:
                del qg
            if ctx.save_fp8:
                xt = saved_x
            else:
                _, xt = kernels.quantize(saved_x, sx, rowwise=False, transpose=True, fp8_dtype=layer.fp8_dtype)
            dw = kernels.gemm(qgt, xt, sg, sx, out_dtype=ctx.weight_grad_dtype, autotune=layer.autotune)
        if needs_b:
            db = dy.sum(dim=0, dtype=ctx.bias_grad_dtype)
        # These are derivatives w.r.t. real weights; the optimizer updates their
        # master values, never the scaled FP8 codes.
        return dx, dw, db, None, None


class Linear(nn.Module):
    """A CUDA FP8 Linear with two storage presets and independent overrides.

    Construct/move the model, construct its optimizer, then call
    ``viola.optim.bind_optimizer(model, optimizer)`` before training. Forward inputs
    and outputs are ordinary FP16/BF16/FP32 tensors. ``weight`` stores FP8 codes;
    real weights are ``weight * weight_scale``.
    """

    def __init__(self, in_features, out_features, bias=True, *, mode="memory",
                 forward_dtype="e4m3", backward_dtype="e5m2",
                 cache_weight_transpose=None, saved_input_dtype="fp8",
                 scaling="current", history_len=16, autotune=False, device=None, dtype=None):
        super().__init__()
        assert mode in ("memory", "speed")
        formats = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
        self.forward_dtype, self.backward_dtype = forward_dtype, backward_dtype
        self.fp8_dtype, self.backward_fp8_dtype = formats[forward_dtype], formats[backward_dtype]
        assert saved_input_dtype in ("fp8", "bf16", "fp16", "fp32")
        assert scaling in ("current", "delayed") and history_len > 0
        self.in_features, self.out_features = in_features, out_features
        self.mode, self.saved_input_dtype = mode, saved_input_dtype
        self.autotune = autotune
        self.cache_weight_transpose = mode == "speed" if cache_weight_transpose is None else cache_weight_transpose
        config = settings()
        master_dtype = config["master_weight_dtype"] or torch.float32
        gradient_dtype = config["gradient_dtype"] or torch.float32
        dtype = dtype or torch.get_default_dtype()
        # This allocation becomes the optimizer's master without cloning it.
        # It is not a second high-precision model weight retained after binding.
        self._initial_weight = torch.empty((out_features, in_features), device=device, dtype=master_dtype)
        self._master_ref = None
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=self.fp8_dtype))
        self.weight.grad_dtype = gradient_dtype
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        if self.bias is not None:
            self.bias.grad_dtype = gradient_dtype
        self.register_buffer("weight_scale", torch.ones(1, device=device, dtype=torch.float32))
        self.register_buffer("weight_transpose", torch.empty(0, 0, device=device, dtype=self.fp8_dtype), persistent=False)
        self.weight_quantizer = _Quantizer(scaling, history_len, device)
        self.input_quantizer = _Quantizer(scaling, history_len, device)
        self.grad_quantizer = _Quantizer(scaling, history_len, device)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        master = self._master_ref() if self._master_ref is not None else self._initial_weight
        nn.init.kaiming_uniform_(master, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))
        self.refresh_weight(master)

    @torch.no_grad()
    def refresh_weight(self, master):
        """Called after an optimizer update, directly from its updated master."""
        if self.cache_weight_transpose and self.weight_transpose.numel() == 0:
            self.weight_transpose = torch.empty((self.in_features, self.out_features), device=master.device,
                                                dtype=self.fp8_dtype)
        if master.is_cuda:
            scale = self.weight_quantizer.scale(master, self.fp8_dtype)
            self.weight_scale.copy_(scale)
            kernels.quantize(master, scale, transpose=self.cache_weight_transpose,
                             fp8_dtype=self.fp8_dtype, out=self.weight, out_t=self.weight_transpose)
        else:
            # CPU construction only; CUDA training always uses the TileLang kernels.
            fp8_max = torch.finfo(self.fp8_dtype).max
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

    @classmethod
    @torch.no_grad()
    def from_float(cls, linear, **kwargs):
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None,
                    device=linear.weight.device, dtype=linear.weight.dtype, **kwargs)
        layer._initial_weight.copy_(linear.weight)
        # A pretrained weight must not inherit the temporary initializer's amax.
        layer.weight_quantizer.history.zero_()
        layer.weight_quantizer.index = 0
        layer.refresh_weight(layer._initial_weight)
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
        return _Linear.apply(x, self.weight, self.bias, self, need_backward)

    def _apply(self, fn, recurse=True):
        # .cuda()/.to() move payloads, but .bfloat16() must not cast the FP8 codes
        # or scale/history tensors. Move the model before binding its optimizer.
        weight = self._parameters["weight"]
        self._parameters["weight"] = None
        grad_dtype = weight.grad_dtype
        fixed = [(self, name, self._buffers.pop(name)) for name in ("weight_scale", "weight_transpose")]
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
        assert state_dict[prefix + "weight"].dtype == self.fp8_dtype, "Construct Linear with the checkpoint's FP8 format."
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._initial_weight = None
        if self.cache_weight_transpose:
            if self.weight.is_cuda:
                self.weight_transpose = kernels.transpose(self.weight)
            else:
                self.weight_transpose = self.weight.T.contiguous()

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, mode={self.mode!r}, "
                f"forward_dtype={self.forward_dtype!r}, backward_dtype={self.backward_dtype!r}, "
                f"cache_weight_transpose={self.cache_weight_transpose}, "
                f"saved_input_dtype={self.saved_input_dtype!r}, autotune={self.autotune}")
