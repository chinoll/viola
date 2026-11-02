"""Gated MLP with GEMM-fused forward and backward epilogues."""

import torch
from torch import nn

from .linear import Linear, _Linear
from ._grouped_linear import GroupedLinear


class _ProjectionState:
    """Reuse Linear's quantization policy inside one MLP autograd operation."""

    def __init__(self, needs, **options):
        self.needs_input_grad = (*needs, False, False)
        self.saved_tensors = ()
        self.__dict__.update(options)

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class _GateUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, layer, activation, need_backward, encode_output, encode_transpose):
        ctx.gated_forward = activation
        ctx.encode_output, ctx.encode_transpose = encode_output, encode_transpose
        ctx.impl = GroupedLinear if layer._grouped_scaling else _Linear
        ctx.set_materialize_grads(False)
        h = ctx.impl.forward(ctx, x, weight, bias, layer, need_backward)
        z = ctx.gate_aux
        scale, ht, st = ctx.gate_encoded
        del ctx.gate_aux
        del ctx.gate_encoded
        ctx.mark_non_differentiable(h, scale, ht, st)
        return h, scale, ht, st, z

    @staticmethod
    def backward(ctx, unused_dh, unused_ds, unused_dht, unused_dst, dz):
        dx, dw, db, _, _ = ctx.impl.backward(ctx, dz)
        return dx, dw, db, None, None, None, None, None


class _Down(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, scale, ht, st, z, weight, bias, layer, activation, need_backward, need_up):
        state = _ProjectionState((need_up, ctx.needs_input_grad[5], ctx.needs_input_grad[6]),
                                  gated_backward=activation)
        if scale.numel():
            state.encoded_input = (scale, ht, st)
            state.input_dtype = z.dtype
        ctx.impl = GroupedLinear if layer._grouped_scaling else _Linear
        y = ctx.impl.forward(state, h, weight, bias, layer, need_backward)
        if hasattr(state, 'encoded_input'):
            del state.encoded_input
        if need_backward:
            ctx.save_for_backward(*state.saved_tensors, z)
            state.saved_tensors = ()
            ctx.state = state
        return y

    @staticmethod
    def backward(ctx, dy):
        tensors = ctx.saved_tensors
        state = ctx.state
        state.saved_tensors, state.gate_aux = tensors[:-1], tensors[-1]
        dz, dw, db, _, _ = ctx.impl.backward(state, dy)
        if dz is not None:
            dz = dz.reshape(tensors[-1].shape)
        state.saved_tensors = ()
        del state.gate_aux
        # Send the already differentiated gradient through z. h needs no dH
        # allocation. Autograd releases z before the gate/up projection's dW.
        return None, None, None, None, dz, dw, db, None, None, None, None


class MLP(nn.Module):
    """down(activation(gate(x)) * up(x)), with GEMM-fused gating.

    hidden_features is the expanded width of each branch. Linear's storage,
    FP8 formats, scaling, accumulation and caching options pass through unchanged.
    Autotuning infers the flattened token count from the first real input.
    """

    def __init__(self, in_features, hidden_features, out_features=None, *, activation='silu',
                 autotune=False, **linear_options):
        super().__init__()
        assert activation in ('silu', 'gelu', 'relu')
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = in_features if out_features is None else out_features
        self.activation, self.autotune = activation, autotune
        self.gate_up = Linear(in_features, 2 * hidden_features, autotune=autotune, **linear_options)
        self.down = Linear(hidden_features, self.out_features, autotune=autotune, **linear_options)

    def forward(self, x):
        assert x.is_cuda, 'viola.nn.MLP executes on CUDA.'
        if torch.is_autocast_enabled('cuda'):
            x = x.to(torch.get_autocast_dtype('cuda'))
        if self.gate_up._native_scaling:
            assert torch.cuda.get_device_capability(x.device)[0] == 12, 'Native MXFP8 backend targets SM120/SM121.'
        need_backward = torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in self.parameters()))
        need_up = need_backward and (x.requires_grad or any(p.requires_grad for p in self.gate_up.parameters()))
        encode = (self.down._native_scaling and self.down.input_group_shape == (1, 32)
                  and self.down.saved_input_dtype == 'fp8')
        h, scale, ht, st, z = _GateUp.apply(x, self.gate_up.weight, self.gate_up.bias,
                             self.gate_up, self.activation, need_up, encode,
                             encode and need_backward and self.down.weight.requires_grad)
        return _Down.apply(h, scale, ht, st, z, self.down.weight, self.down.bias,
                            self.down, self.activation, need_backward, need_up)

    def extra_repr(self):
        return (f'in_features={self.in_features}, hidden_features={self.hidden_features}, '
                f'out_features={self.out_features}, activation={self.activation!r}, autotune={self.autotune}')
