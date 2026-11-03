"""Complete-matrix optimizer updates, independent of distributed storage."""

from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from ..distributed.communication import gather_to_owner, replicate_tensor, write_from_owner
from ..distributed.partition import MatrixPartition


class MatrixOptimizer(torch.optim.Optimizer):
    """Base for optimizers whose update needs a whole logical matrix.

    Implement ``_step_local(parameter, gradient, state, group, index)`` using
    ordinary full tensors. DTensor parameters may be replicated or sharded;
    ``shard_state`` chooses owner-local or replicated optimizer states. The
    engine also uses this interface with owner-local ordinary parameters.
    Neither path orthogonalizes a physical row shard as a standalone matrix.
    """

    def __init__(self, params, defaults, *, shard_state=True, skip_nonfinite=False):
        self.shard_state = shard_state
        self.skip_nonfinite = skip_nonfinite
        self.step_was_skipped = False
        self._partition = None
        super().__init__(params, defaults)
        self.register_state_dict_post_hook(self._save_partition)
        self.register_load_state_dict_pre_hook(self._before_load)
        self.register_load_state_dict_post_hook(self._after_load)

    def add_param_group(self, group):
        assert self._partition is None or not self._partition.owners, \
            "Add DTensor parameter groups before the first step or checkpoint."
        super().add_param_group(group)
        self._partition = None

    @property
    def partition(self):
        if self._partition is None:
            self._partition = MatrixPartition(p for g in self.param_groups for p in g["params"])
        return self._partition

    @abstractmethod
    def _step_local(self, parameter, gradient, state, group, index):
        """Update a full parameter in-place; index is stable across sharding."""
        raise NotImplementedError

    def _has_nonfinite(self):
        bad = any(not torch.isfinite(p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).all()
                  for g in self.param_groups for p in g["params"] if p.grad is not None)
        for mesh in self.partition.meshes:
            flag = torch.tensor(int(bad), device=mesh.device_type, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=mesh.get_group())
            bad = bool(flag.item())
        return bad

    @torch.no_grad()
    def step(self, closure=None):
        self.step_was_skipped = False
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.skip_nonfinite and self._has_nonfinite():
            self.step_was_skipped = True
            return loss
        contexts, index = {}, 0
        for group in self.param_groups:
            indices = group.get("viola_parameter_indices")
            for local_index, param in enumerate(group["params"]):
                parameter_index = indices[local_index] if indices is not None else index
                index += 1
                if param.grad is None:
                    continue
                if isinstance(param, DTensor):
                    contexts[param] = (group, parameter_index)
                else:
                    self._step_local(param, param.grad, self.state[param], group, parameter_index)
        if contexts:
            for batch in self.partition.batches:
                self._step_batch(batch, contexts)
        return loss

    def _step_batch(self, batch, contexts):
        work = []
        for param in batch:
            if param not in contexts:
                continue
            owner = self.partition.owner(param)
            if self.shard_state:
                gradient = gather_to_owner(param.grad, owner)
                value = gather_to_owner(param, owner)
            else:
                gradient = replicate_tensor(param.grad)
                value = replicate_tensor(param)
            if value is not None:
                value = value.contiguous()
            work.append((param, value, gradient, owner))
        for param, value, gradient, _ in work:
            if value is not None:
                group, index = contexts[param]
                self._step_local(value, gradient.contiguous(), self.state[param], group, index)
        for param, value, _, owner in work:
            # All replicas compute the same update when state is replicated;
            # use one source for an exact, identical writeback on every rank.
            write_from_owner(param, value, owner)

    def _save_partition(self, optimizer, state):
        if self.partition.owners:
            state["viola_dtensor"] = dict(shard_state=self.shard_state,
                                         partition=self.partition.state_dict())
        return state

    def _before_load(self, optimizer, saved):
        saved = dict(saved)
        layout = saved.pop("viola_dtensor", {})
        expected = (dict(shard_state=self.shard_state, partition=self.partition.state_dict())
                    if self.partition.owners else {})
        assert layout == expected, "Matrix optimizer checkpoint ownership/layout differs from this run."
        # Optimizer.load_state_dict otherwise casts floating states to the
        # parameter dtype, losing independently configured state precision.
        self._loading = {pid: {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
                         for pid, state in saved["state"].items()}
        self._loading_groups = saved["param_groups"]
        return dict(saved, state={pid: {k: v for k, v in state.items() if not isinstance(v, torch.Tensor)}
                                  for pid, state in saved["state"].items()})

    def _after_load(self, optimizer):
        for old, group in zip(self._loading_groups, self.param_groups):
            for pid, param in zip(old["params"], group["params"]):
                for name, value in self._loading.get(pid, {}).items():
                    self.state[param][name] = value.to(device=param.device)
        del self._loading, self._loading_groups
