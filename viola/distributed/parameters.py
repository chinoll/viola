"""Ordinary compute tensors with DTensor metadata and whole-parameter owners."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

from ..nn import Linear
from .partition import MatrixPartition
from .communication import replicate_tensor, write_from_owner


@torch.no_grad()
def broadcast_tensor(tensor, source, group):
    if not tensor.numel():
        return
    value = tensor.contiguous()
    # Transport FP8 codes as bytes; NCCL need not interpret their FP8 format.
    payload = value.view(torch.uint8) if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else value
    dist.broadcast(payload, src=source, group=group)
    tensor.copy_(value)


def extra_states(model):
    return {name: module.get_extra_state() for name, module in model.named_modules()
            if type(module).get_extra_state is not torch.nn.Module.get_extra_state}


def restore_extra_states(model, states):
    for name, value in states.items():
        model.get_submodule(name).set_extra_state(value)


@torch.no_grad()
def synchronize_model(model, mesh, *, initial=False):
    group, source = mesh.get_group(), int(mesh.mesh[0])
    for value in model.state_dict(keep_vars=True).values():
        if isinstance(value, torch.Tensor):
            broadcast_tensor(value, source, group)
    metadata = [extra_states(model) if mesh.get_local_rank() == 0 else None]
    dist.broadcast_object_list(metadata, src=source, group=group)
    restore_extra_states(model, metadata[0])
    for module in model.modules():
        if isinstance(module, Linear):
            if initial and module._initial_weight is not None:
                broadcast_tensor(module._initial_weight, source, group)
            module.rebuild_weight_cache()


@dataclass
class ParameterRecord:
    name: str
    parameter: torch.nn.Parameter
    tensor: DTensor
    index: int
    owner: int
    layer: Linear | None


class ParameterStore:
    def __init__(self, model, optimizer, mesh, gradient_dtype, fsdp_modules):
        self.model, self.mesh = model, mesh
        self.rank = mesh.get_local_rank()
        names = {id(param): name for name, param in model.named_parameters()}
        layers = {id(m.weight): m for m in model.modules() if isinstance(m, Linear)}
        indexed = [(i, p) for i, p in enumerate(p for g in optimizer.param_groups for p in g["params"])
                   if p.requires_grad]
        views = [p if isinstance(p, DTensor) else DTensor.from_local(p.detach(), mesh, [Replicate()])
                 for _, p in indexed]
        self.partition = MatrixPartition(views)
        self.records = [ParameterRecord(names[id(p)], p, view, i, self.partition.owner(view), layers.get(id(p)))
                        for (i, p), view in zip(indexed, views)]
        self.by_parameter = {r.parameter: r for r in self.records}
        self.local = [r for r in self.records if r.owner == self.rank]
        self.gradient_dtype, self.fsdp_modules = gradient_dtype, fsdp_modules
        self.groups = {name: [] for name in fsdp_modules}
        for record in self.records:
            record.parameter.grad_dtype = gradient_dtype
            if fsdp_modules:
                unit = max((name for name in fsdp_modules
                            if not name or record.name.startswith(name + ".")), key=len)
                self.groups[unit].append(record)

    def bind_gradients(self, gradients):
        self.hooks = []
        if not self.fsdp_modules:
            for i, record in enumerate(self.records):
                gradients.bind(i, record.parameter)
            return
        indices = {r.name: i for i, r in enumerate(self.records)}
        for name, module in self.fsdp_modules.items():
            records = self.groups[name]
            def before_forward(module, args, records=records):
                # Registered after FSDP's pre-forward hook: these are the full
                # ordinary Parameters for this invocation, not row DTensors.
                for record in records:
                    parameter = self.model.get_parameter(record.name)
                    parameter.grad_dtype = self.gradient_dtype
                    gradients.bind(indices[record.name], parameter)
                    if record.layer is not None:
                        record.layer.rebuild_weight_cache()
            def after_forward(module, args, output, records=records):
                # Autograd keeps the transpose if fast mode saved it for dX;
                # the module must not retain another full cache between micros.
                for record in records:
                    if record.layer is not None:
                        self.clear_cache(record.layer)
            self.hooks.append(module.register_forward_pre_hook(before_forward))
            self.hooks.append(module.register_forward_hook(after_forward))

    def metadata(self):
        return [dict(name=r.name, index=r.index, shape=tuple(r.parameter.shape),
                     dtype=r.parameter.dtype, owner=r.owner,
                     placement=str(r.tensor.placements[0])) for r in self.records]

    def full(self, record):
        return replicate_tensor(record.tensor)

    @torch.no_grad()
    def write(self, record, value, source):
        """Quantize on the source, then restore the exact compute storage."""
        layer = record.layer
        if layer is not None and layer.weight_dtype == "fp8":
            codes = None
            if self.rank == source:
                original = layer.weight
                codes = torch.nn.Parameter(torch.empty(record.parameter.shape, device=record.parameter.device,
                                                      dtype=record.parameter.dtype), requires_grad=False)
                layer.weight = codes
                try:
                    layer.refresh_weight(value)
                finally:
                    layer.weight = original
            write_from_owner(record.tensor, codes, source)
            for buffer in (layer.weight_scale, layer.weight_quantizer.history):
                broadcast_tensor(buffer, int(self.mesh.mesh[source]), self.mesh.get_group())
            index = torch.tensor(layer.weight_quantizer.index, device=record.parameter.device, dtype=torch.int64)
            dist.broadcast(index, src=int(self.mesh.mesh[source]), group=self.mesh.get_group())
            layer.weight_quantizer.index = int(index.item())
            if isinstance(record.parameter, DTensor):
                self.clear_cache(layer)
            else:
                layer.rebuild_weight_cache()
        else:
            value = value.to(record.parameter.dtype) if value is not None else None
            write_from_owner(record.tensor, value, source)

    @staticmethod
    def clear_cache(layer):
        layer.weight_transpose = layer.weight_transpose.new_empty((0, 0))
        if hasattr(layer, "weight_transpose_scale"):
            layer.weight_transpose_scale = layer.weight_transpose_scale.new_empty((0, 0))

    def release_initializers(self):
        for module in self.model.modules():
            if isinstance(module, Linear):
                module._initial_weight = None
                if isinstance(module.weight, DTensor):
                    self.clear_cache(module)

    @torch.no_grad()
    def snapshot(self):
        return {r.name: r.tensor.to_local().detach().clone() for r in self.records}

    @torch.no_grad()
    def restore(self, values):
        for record in self.records:
            record.tensor.to_local().copy_(values[record.name])
            torch.autograd.graph.increment_version(record.parameter)

    @torch.no_grad()
    def model_state_dict(self):
        result = {}
        for name, value in self.model.state_dict().items():
            full = replicate_tensor(value) if isinstance(value, torch.Tensor) else value
            if self.rank == 0:
                result[name] = full.cpu() if isinstance(full, torch.Tensor) else full
        return result

    @torch.no_grad()
    def load_model_state_dict(self, saved):
        for name, value in self.model.state_dict(keep_vars=True).items():
            if not isinstance(value, torch.Tensor):
                continue
            full = saved[name].to(value.device) if self.rank == 0 else None
            tensor = value if isinstance(value, DTensor) else DTensor.from_local(value.detach(), self.mesh, [Replicate()])
            write_from_owner(tensor, full, 0)
        extras = [dict((name, saved[f"{name}._extra_state" if name else "_extra_state"])
                       for name in extra_states(self.model)) if self.rank == 0 else None]
        dist.broadcast_object_list(extras, src=int(self.mesh.mesh[0]), group=self.mesh.get_group())
        restore_extra_states(self.model, extras[0])
