"""Optimizer ownership and full-matrix master weights, separate from model storage."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from .config import dtype
from .parameters import broadcast_tensor


class OptimizerState:
    def __init__(self, store, optimizer, config):
        assert not optimizer.state and not hasattr(optimizer, "_viola_linear_masters"), \
            "Prepare an unused optimizer; the engine owns master-weight binding."
        self.store, self.optimizer = store, optimizer
        self.shard = config["sharding"]["optimizer"]
        self.master_dtype = dtype(config["precision"]["master_weight_dtype"])
        self.gradient_dtype = dtype(config["precision"]["gradient_dtype"])
        self.precision = config["precision"]
        self.saved_precision = dict(self.precision, master_weight_dtype=self.master_dtype,
                                    gradient_dtype=self.gradient_dtype)
        self.parameters, self.transient = {}, set()
        self.records = [r for r in store.records if not self.shard or r.owner == store.rank]
        for record in store.records:
            p, layer = record.parameter, record.layer
            # Everyone participates in model-storage collectives, but only
            # optimizer owners allocate persistent full master matrices.
            full = store.full(record)
            if layer is not None and layer.weight_dtype == "fp8":
                assert self.master_dtype is not None, "FP8 weight storage requires floating-point master weights."
                full = layer._initial_weight
                if full is None:
                    original = layer.weight
                    layer.weight = torch.nn.Parameter(store.full(record), requires_grad=False)
                    try:
                        full = layer.dequantize_weight()
                    finally:
                        layer.weight = original
            if not self.shard or record.owner == store.rank:
                target_dtype = self.master_dtype or p.dtype
                if not isinstance(p, DTensor) and target_dtype == p.dtype:
                    update = p
                else:
                    update = torch.nn.Parameter(full.to(dtype=target_dtype, copy=True))
                update.grad_dtype = self.gradient_dtype
                self.parameters[record.index] = update
                if self.master_dtype is None and isinstance(p, DTensor):
                    self.transient.add(record.index)
        for group in optimizer.param_groups:
            records = [store.by_parameter[p] for p in group["params"] if p in store.by_parameter]
            records = [r for r in records if r.index in self.parameters]
            group["params"] = [self.parameters[r.index] for r in records]
            group["viola_parameter_indices"] = [r.index for r in records]
            group["master_rounding"] = self.precision["master_rounding"]
            group["master_rounding_seed"] = self.precision["master_rounding_seed"]
        if hasattr(optimizer, "_partition"):
            optimizer._partition = None
        self.release()
        store.release_initializers()

    @torch.no_grad()
    def materialize(self):
        if self.master_dtype is not None:
            return
        for record in self.store.records:
            if not isinstance(record.parameter, DTensor):
                continue
            full = self.store.full(record)
            if record.index in self.parameters:
                p = self.parameters[record.index]
                p.untyped_storage().resize_(p.numel() * p.element_size())
                p.copy_(full)

    @torch.no_grad()
    def attach_gradients(self, gradients, active):
        for index, record in enumerate(self.store.records):
            if not active[index]:
                continue
            value = gradients.local.get(index)
            if gradients.shard and not self.shard:
                if value is None:
                    value = torch.empty(record.parameter.numel(), device=record.parameter.device,
                                        dtype=self.gradient_dtype)
                broadcast_tensor(value, int(self.store.mesh.mesh[record.owner]), self.store.mesh.get_group())
            if record.index in self.parameters:
                self.parameters[record.index].grad = value.view(record.parameter.shape)

    @torch.no_grad()
    def step(self):
        # Native AdamW requires matching gradient/state types. Reuse the same
        # mixed-precision update as the standalone master-weight binding.
        from ..optim.master_weights import step_mixed_adamw
        restored = step_mixed_adamw(self.optimizer, self.precision)
        self.optimizer.step()
        for parameter, gradient in restored:
            parameter.grad = gradient

    @torch.no_grad()
    def writeback(self, active):
        for record, used in zip(self.store.records, active):
            if used:
                # Full replicas use a fixed source as well; this keeps FP8
                # codes/quantizer state identical with stochastic updates.
                self.store.write(record, self.parameters.get(record.index), record.owner)

    @torch.no_grad()
    def full_value(self, record):
        value = self.parameters.get(record.index)
        if not self.shard:
            return value.detach()
        if value is None:
            value = torch.empty(record.parameter.shape, device=record.parameter.device,
                                dtype=self.master_dtype or record.parameter.dtype)
        broadcast_tensor(value, int(self.store.mesh.mesh[record.owner]), self.store.mesh.get_group())
        return value.detach()

    @torch.no_grad()
    def release(self):
        self.optimizer.zero_grad(set_to_none=True)
        for index in self.transient:
            self.parameters[index].untyped_storage().resize_(0)

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(), precision=self.saved_precision,
                    masters={r.name: self.parameters[r.index].detach()
                             for r in self.records if r.index not in self.transient
                             and self.parameters[r.index] is not r.parameter})

    @torch.no_grad()
    def load_state_dict(self, saved):
        assert saved["precision"] == self.saved_precision, "Checkpoint master/gradient precision differs from this run."
        self.optimizer.load_state_dict(saved["optimizer"])
        for record in self.records:
            if record.name in saved["masters"]:
                self.parameters[record.index].copy_(saved["masters"][record.name])
        self.release()
