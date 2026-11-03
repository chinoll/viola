"""Training interface with independent state-sharding policies."""

import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from ..precision import set_gradient_dtype, set_master_weight_dtype
from . import checkpoint
from .config import dtype, read_config
from .data import prepare_dataloader
from .parameters import synchronize_model
from .state import DistributedState
from .fsdp import shard_parameters


class violaEngine:
    def __init__(self, config=None):
        self.config = read_config(config)
        device_type = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_type, int(os.environ.get("LOCAL_RANK", 0))) if device_type == "cuda" else torch.device(device_type)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if not dist.is_initialized():
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo")
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.is_main_process = self.rank == 0
        self.mesh = init_device_mesh(self.device.type, (self.world_size,), mesh_dim_names=("dp",))
        policy = self.config["precision"]
        set_master_weight_dtype(dtype(policy["master_weight_dtype"]), rounding=policy["master_rounding"],
                                seed=policy["master_rounding_seed"])
        set_gradient_dtype(dtype(policy["gradient_dtype"]))
        self.accumulation_steps = self.config["gradient_accumulation_steps"]
        self.global_step = self.micro_step = self._pending_microbatches = 0
        self.sync_gradients = True
        self._in_accumulate = self._in_ema = False
        self._weighted = None
        self._loss_weight = 0.
        self.model = self.state = self.optimizer = self.ema = None

    def prepare_model(self, model):
        # Parameter storage is the model's choice. mixed_precision controls
        # autocast, independently of BF16/FP16/FP8 parameter storage.
        self.model = model.to(self.device)
        synchronize_model(self.model, self.mesh, initial=True)
        self.fsdp_modules = (shard_parameters(self.model, self.mesh, blocks=self.config["sharding"]["blocks"])
                             if self.config["sharding"]["parameters"] else {})
        return self.model

    def prepare_optimizer(self, factory):
        optimizer = factory(self.model.parameters()) if callable(factory) else factory
        self.state = DistributedState(self.model, optimizer, self.mesh, self.config, self.fsdp_modules)
        self.optimizer, self.ema = optimizer, self.state.ema
        return optimizer

    def prepare_dataloader(self, loader):
        return prepare_dataloader(loader, rank=self.rank, world_size=self.world_size)

    def autocast(self):
        precision = self.config["mixed_precision"]
        return (nullcontext() if precision == "no" else
                torch.autocast(self.device.type, dtype=dtype(precision)))

    def _start_microbatch(self):
        self.micro_step += 1
        self._pending_microbatches += 1
        self.sync_gradients = self._pending_microbatches >= self.accumulation_steps

    @contextmanager
    def accumulate(self, model=None):
        assert not self._in_accumulate
        self._in_accumulate = True
        self._start_microbatch()
        try:
            yield
        finally:
            self._in_accumulate = False

    def backward(self, loss, *, weight=None):
        """Backpropagate a local mean loss; weight may be its valid-token count."""
        if not self._in_accumulate:
            self._start_microbatch()
        weighted = weight is not None
        assert self._weighted is None or self._weighted == weighted
        self._weighted = weighted
        if weighted:
            weight = float(weight)
            self._loss_weight += weight
            loss = loss * weight
        else:
            loss = loss / self.accumulation_steps
        self.state.backward(loss)

    def step(self, optimizer=None, *, scheduler=None, force=False):
        """Return True only after a real, synchronized optimizer update."""
        assert not self._in_ema
        assert optimizer is None or optimizer is self.optimizer
        if not self._pending_microbatches or (not self.sync_gradients and not force):
            return False
        if self._weighted:
            total = torch.tensor(self._loss_weight, device=self.device, dtype=torch.float64)
            dist.all_reduce(total, group=self.mesh.get_group())
            divisor = total
        else:
            # Also normalize a final, explicitly flushed partial accumulation.
            divisor = self.world_size * self._pending_microbatches / self.accumulation_steps
        updated = self.state.step(divisor)
        self._pending_microbatches = 0
        self._weighted, self._loss_weight = None, 0.
        self.sync_gradients = True
        if updated:
            self.global_step += 1
            if scheduler is not None:
                scheduler.step()
        return updated

    @torch.no_grad()
    def reduce(self, value, *, reduction="mean"):
        value = torch.as_tensor(value, device=self.device).detach().clone()
        dist.all_reduce(value, group=self.mesh.get_group())
        return value / self.world_size if reduction == "mean" else value

    @contextmanager
    def ema_scope(self):
        """Evaluate with EMA, then restore exact compute weights and buffers."""
        assert not self._pending_microbatches and not self._in_ema
        self._in_ema = True
        flags = [(module, module.training) for module in self.model.modules()]
        buffers = checkpoint.capture_buffers(self.model, clone=True)
        saved = self.state.parameters.snapshot() if self.ema.decay else {}
        try:
            if self.ema.decay:
                self.ema.apply()
            self.model.eval()
            with torch.no_grad():
                yield self.model
        finally:
            with torch.no_grad():
                if self.ema.decay:
                    self.state.parameters.restore(saved)
                checkpoint.restore_buffers(self.model, buffers)
            for module, training in flags:
                module.training = training
            self._in_ema = False

    def save_checkpoint(self, path, *, scheduler=None, extra=None):
        return checkpoint.save(self, path, scheduler=scheduler, extra=extra)

    def load_checkpoint(self, path, *, scheduler=None):
        return checkpoint.load(self, path, scheduler=scheduler)

    def close(self):
        if self.state is not None:
            self.state.close()
