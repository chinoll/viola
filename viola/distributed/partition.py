"""Assign whole optimizer tensors to owners, independently of storage layout."""

from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


@dataclass(frozen=True)
class MatrixOwner:
    index: int
    rank: int


class MatrixPartition:
    """Greedy size-balanced ownership for complete matrices (and auxiliary vectors).

    DTensor Shard placements continue to describe model storage. Ownership is
    separate metadata, so a Muon update never treats a row shard as a matrix.
    All ranks must supply the same parameter order and meshes.
    """

    def __init__(self, parameters):
        self.owners = {}
        self.meshes = []
        self.batches = []
        by_mesh = {}
        for index, param in enumerate(parameters):
            if not isinstance(param, DTensor):
                continue
            mesh = param.device_mesh
            assert mesh.ndim == 1, "Matrix ownership currently supports a 1-D DeviceMesh."
            assert isinstance(param.placements[0], (Replicate, Shard))
            key = (mesh.device_type, tuple(mesh.mesh.tolist()))
            if key not in by_mesh:
                by_mesh[key] = (mesh, [])
                self.meshes.append(mesh)
            by_mesh[key][1].append((index, param))

        for mesh, entries in by_mesh.values():
            loads = [0] * mesh.size()
            for index, param in sorted(entries, key=lambda item: (-item[1].numel(), item[0])):
                owner = min(range(mesh.size()), key=loads.__getitem__)
                self.owners[param] = MatrixOwner(index, owner)
                loads[owner] += param.numel()
            queues = [[] for _ in range(mesh.size())]
            for _, param in entries:
                queues[self.owner(param)].append(param)
            # One complete tensor per owner in a round: different GPUs execute
            # their local updates concurrently, with one tensor's temporary
            # full weight/gradient per GPU rather than gathering the model.
            self.batches.extend([queue[i] for queue in queues if i < len(queue)]
                                for i in range(max(map(len, queues))))

    def owner(self, param):
        return self.owners[param].rank

    def is_owner(self, param):
        return param.device_mesh.get_local_rank() == self.owner(param)

    def state_dict(self):
        """Placement/ownership metadata for rank-local optimizer checkpoints."""
        if not self.owners:
            return {}
        entries = [dict(index=entry.index, shape=tuple(param.shape),
                        placement=str(param.placements[0]),
                        ranks=param.device_mesh.mesh.tolist(), owner=entry.rank)
                   for param, entry in sorted(self.owners.items(), key=lambda item: item[1].index)]
        return dict(rank=dist.get_rank(), parameters=entries)
