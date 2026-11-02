"""Automatically updated, DTensor-sharded exponential moving averages."""

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard


class ShardedEMA:
    """Bind EMA updates to optimizer.step(); decay=0 allocates no EMA state.

    ``parameters`` is a module or an iterable of (name, DTensor parameter).
    Replicated parameters use local dim-0 slices for EMA, without communication.
    Sharded parameters retain their placement. When bind_optimizer is used, EMA
    reads the corresponding master. The initial implementation keeps the model
    layout stable for the lifetime of the EMA.
    """

    def __init__(self, parameters, optimizer, *, decay=0, dtype=torch.float32):
        assert 0 <= decay < 1
        self.decay, self.dtype = decay, dtype
        self.updates = 0
        self.weights = {}
        self._sources = {}
        self._hook = None
        if decay == 0:
            return
        pairs = parameters.named_parameters() if isinstance(parameters, torch.nn.Module) else parameters
        binding = getattr(optimizer, "_viola_linear_masters", None)
        masters = {id(p): master for p, master, _ in binding.pairs} if binding is not None else {}
        for name, param in pairs:
            if not param.requires_grad:
                continue
            source = masters.get(id(param), param)
            assert isinstance(source, DTensor), "ShardedEMA expects DTensor parameters."
            assert source.device_mesh.ndim == 1
            placement, = source.placements
            assert isinstance(placement, (Replicate, Shard))
            placement = Shard(0) if isinstance(placement, Replicate) and source.ndim else placement
            self._sources[name] = source
            self.weights[name] = source.detach().redistribute(placements=[placement]).to(dtype=dtype, copy=True)
        self._hook = optimizer.register_step_post_hook(self._after_step)

    def _after_step(self, optimizer, args, kwargs):
        if not getattr(optimizer, "step_was_skipped", False):
            self.update()

    @torch.no_grad()
    def update(self):
        if self.decay == 0:
            return
        for name, ema in self.weights.items():
            source = self._sources[name].detach().redistribute(placements=ema.placements)
            ema.to_local().lerp_(source.to_local().to(self.dtype), 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        if self.decay == 0:
            return {}
        return dict(decay=self.decay, updates=self.updates, dtype=self.dtype, weights=self.weights)

    @torch.no_grad()
    def load_state_dict(self, state):
        if self.decay == 0:
            return
        assert state["dtype"] == self.dtype
        assert state["weights"].keys() == self.weights.keys()
        self.decay, self.updates = state["decay"], state["updates"]
        for name, saved in state["weights"].items():
            current = self.weights[name]
            current.to_local().copy_(saved.redistribute(placements=current.placements).to_local())

    def close(self):
        """Remove the optimizer hook when replacing an EMA instance."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
