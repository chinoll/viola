"""Compose independent parameter, gradient, optimizer and EMA placements."""

import torch
import torch.distributed as dist

from .config import dtype
from .parameters import ParameterStore
from .gradients import GradientState
from .optimizer import OptimizerState
from .ema import EMAState


class DistributedState:
    def __init__(self, model, optimizer, mesh, config, fsdp_modules):
        self.mesh, self.sharding = mesh, {k: config["sharding"][k]
                                         for k in ("parameters", "gradients", "optimizer", "ema")}
        self.parameters = ParameterStore(model, optimizer, mesh,
                                        dtype(config["precision"]["gradient_dtype"]), fsdp_modules)
        self.optimizer = OptimizerState(self.parameters, optimizer, config)
        self.gradients = GradientState(self.parameters,
            gradient_dtype=dtype(config["precision"]["gradient_dtype"]),
            communication_dtype=config["communication"]["dtype"],
            bucket_mb=config["communication"]["bucket_mb"],
            fp8_format=config["communication"]["fp8_format"],
            fp8_tile_size=config["communication"]["fp8_tile_size"], shard=self.sharding["gradients"])
        self.parameters.bind_gradients(self.gradients)
        self.ema = EMAState(self.parameters, self.optimizer, decay=config["ema"]["decay"],
                            dtype=dtype(config["ema"]["dtype"]), shard=self.sharding["ema"])

    def backward(self, loss):
        self.gradients.begin_backward()
        loss.backward()
        self.gradients.finish_backward()

    @torch.no_grad()
    def step(self, divisor):
        active = self.gradients.reduce(divisor)
        if not any(active):
            self.zero_grad()
            return False
        optimizer = self.optimizer.optimizer
        if getattr(optimizer, "skip_nonfinite", False):
            bad = (~torch.isfinite(self.gradients.accumulator).all()).to(torch.int32)
            dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=self.mesh.get_group())
            if bad.item():
                optimizer.step_was_skipped = True
                self.zero_grad()
                return False
        self.optimizer.materialize()
        self.optimizer.attach_gradients(self.gradients, active)
        self.optimizer.step()
        self.optimizer.writeback(active)
        self.ema.update()
        self.zero_grad()
        return True

    def zero_grad(self):
        self.optimizer.release()
        self.gradients.zero_grad()

    def state_dict(self):
        return dict(rank=dist.get_rank(), mesh=self.mesh.mesh.tolist(), sharding=self.sharding,
                    parameters=self.parameters.metadata(), optimizer=self.optimizer.state_dict(),
                    ema=self.ema.state_dict())

    def load_state_dict(self, state):
        assert (state["rank"] == dist.get_rank() and state["mesh"] == self.mesh.mesh.tolist()
                and state["sharding"] == self.sharding and state["parameters"] == self.parameters.metadata()), \
            "Checkpoint parameter ownership/layout differs from this run."
        self.optimizer.load_state_dict(state["optimizer"])
        self.ema.load_state_dict(state["ema"])

    def close(self):
        for handle in [*self.parameters.hooks, *self.gradients.hooks]:
            handle.remove()
