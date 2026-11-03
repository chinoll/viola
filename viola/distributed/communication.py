"""Owner-directed DTensor transfers for optimizer updates (outside autograd).

One-dimensional data-parallel meshes are supported. Partial gradients retain
their declared sum/average semantics; an already reduced Shard or Replicate
gradient is never divided by world size again.
"""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard



def partial_gradient(local, mesh, *, average=True):
    """Describe unsynchronized gradients of local mean losses, without a copy.

    ``average=True`` assumes equal sample/token counts across ranks. For unequal
    counts, weight local contributions by the global denominator and use sum.
    This function annotates contributions; reduction happens in the optimizer.
    """
    assert mesh.ndim == 1
    return DTensor.from_local(local, mesh, [Partial("avg" if average else "sum")])


def _mesh(value):
    mesh = value.device_mesh
    assert mesh.ndim == 1, "Owner transfers currently support a 1-D DeviceMesh."
    return mesh, mesh.get_group(), mesh.get_local_rank()


def _packed_shape(value, dim):
    shape = list(value.shape)
    width = math.ceil(shape.pop(dim) / value.device_mesh.size())
    return (width, *shape)


def wire_tensor(value):
    """NCCL transports FP8 codes byte-for-byte, without interpreting them."""
    return value.view(torch.uint8) if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else value


@torch.no_grad()
def replicate_tensor(value):
    """Materialize one DTensor as a full tensor, including FP8 storage."""
    if not isinstance(value, DTensor):
        return value.detach()
    mesh, group, _ = _mesh(value)
    placement, = value.placements
    local = value.to_local()
    if isinstance(placement, Replicate):
        return local
    if isinstance(placement, Partial):
        result = local.contiguous().clone()
        dist.all_reduce(result, group=group)
        if placement.reduce_op == "avg":
            result.div_(mesh.size())
        else:
            assert placement.reduce_op == "sum"
        return result
    dim = placement.dim % value.ndim
    shape = _packed_shape(value, dim)
    moved = local.movedim(dim, 0)
    packed = torch.zeros(shape, device=local.device, dtype=local.dtype)
    packed[:moved.shape[0]].copy_(moved)
    result = torch.empty((mesh.size(), *shape), device=local.device, dtype=local.dtype)
    dist.all_gather(list(wire_tensor(result).unbind(0)), wire_tensor(packed), group=group)
    return result.flatten(0, 1)[:value.shape[dim]].movedim(0, dim).contiguous()


@torch.no_grad()
def gather_to_owner(value, owner):
    """Return the full logical tensor on ``owner`` only; others return None.

    ``owner`` is a rank within the mesh. Shards are gathered, Partial values are
    reduced, and replicated values are used locally without communication.
    The returned value may alias the input for Replicate; callers may update it.
    """
    mesh, group, rank = _mesh(value)
    placement, = value.placements
    local = value.to_local()
    dst = int(mesh.mesh[owner])
    if isinstance(placement, Replicate):
        return local if rank == owner else None
    if isinstance(placement, Partial):
        assert placement.reduce_op in ("sum", "avg")
        reduced = local.contiguous().clone()
        dist.reduce(reduced, dst=dst, op=dist.ReduceOp.SUM, group=group)
        if rank == owner:
            if placement.reduce_op == "avg":
                reduced.div_(mesh.size())
            return reduced
        return None

    assert isinstance(placement, Shard)
    dim = placement.dim % value.ndim
    packed_shape = _packed_shape(value, dim)
    moved = local.movedim(dim, 0)
    if moved.shape[0] == packed_shape[0]:
        packed = moved.contiguous()
    else:
        packed = torch.zeros(packed_shape, device=local.device, dtype=local.dtype)
        packed[:moved.shape[0]].copy_(moved)
    received = (torch.empty((mesh.size(), *packed_shape), device=local.device,
                            dtype=local.dtype) if rank == owner else None)
    dist.gather(wire_tensor(packed), list(wire_tensor(received).unbind(0)) if received is not None else None,
                dst=dst, group=group)
    if received is not None:
        return received.flatten(0, 1)[:value.shape[dim]].movedim(0, dim).contiguous()
    return None


@torch.no_grad()
def write_from_owner(target, full, owner):
    """Write an owner's updated tensor back to the existing DTensor layout."""
    mesh, group, rank = _mesh(target)
    placement, = target.placements
    local = target.to_local()
    src = int(mesh.mesh[owner])
    if isinstance(placement, Replicate):
        # Communication uses a contiguous buffer; DTensor local storage may be
        # a view (e.g. column-sharded storage before redistribution).
        buffer = local.contiguous()
        if rank == owner:
            buffer.copy_(full)
        dist.broadcast(wire_tensor(buffer), src=src, group=group)
        local.copy_(buffer)
    else:
        assert isinstance(placement, Shard)
        dim = placement.dim % target.ndim
        packed_shape = _packed_shape(target, dim)
        packed = None
        if rank == owner:
            moved = full.movedim(dim, 0)
            if moved.shape[0] == mesh.size() * packed_shape[0]:
                packed = moved.contiguous()
            else:
                packed = torch.zeros((mesh.size() * packed_shape[0], *packed_shape[1:]),
                                     device=local.device, dtype=local.dtype)
                packed[:target.shape[dim]].copy_(moved)
            packed = packed.view(mesh.size(), *packed_shape)
        received = torch.empty(packed_shape, device=local.device, dtype=local.dtype)
        dist.scatter(wire_tensor(received), list(wire_tensor(packed).unbind(0)) if packed is not None else None,
                     src=src, group=group)
        local.copy_(received[:local.shape[dim]].movedim(0, dim))
    torch.autograd.graph.increment_version(target)
