"""One model file plus owner-local optimizer/EMA/RNG files."""

import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from ..nn import Linear
from .parameters import extra_states, restore_extra_states, ParameterStore
from torch.distributed.tensor import DTensor


def capture_buffers(model, *, clone=False):
    parameter_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    buffers = {name: value for name, value in model.state_dict(keep_vars=True).items()
               if isinstance(value, torch.Tensor) and name not in parameter_names}
    if clone:
        buffers = {name: value.detach().clone() for name, value in buffers.items()}
    return dict(buffers=buffers, extra=copy.deepcopy(extra_states(model)))


@torch.no_grad()
def restore_buffers(model, state):
    buffers = dict(model.named_buffers(remove_duplicate=False))
    for name, value in state["buffers"].items():
        buffers[name].copy_(value)
    restore_extra_states(model, state["extra"])
    for module in model.modules():
        if isinstance(module, Linear):
            if isinstance(module.weight, DTensor):
                ParameterStore.clear_cache(module)
            else:
                module.rebuild_weight_cache()


def rng_state(device):
    state = dict(torch=torch.get_rng_state(), python=random.getstate())
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    if "numpy" in sys.modules:
        state["numpy"] = sys.modules["numpy"].random.get_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
    if "numpy" in state:
        import numpy
        numpy.random.set_state(state["numpy"])


def _save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save(engine, path, *, scheduler=None, extra=None):
    assert engine._pending_microbatches == 0 and not engine._in_ema, "Save after an optimizer update."
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = dict(state=engine.state.state_dict(), buffers=capture_buffers(engine.model),
                 rng=rng_state(engine.device), global_step=engine.global_step, micro_step=engine.micro_step,
                 scheduler=scheduler.state_dict() if scheduler is not None else None, extra=extra)
    _save(state, path / f"rank-{engine.rank}.pt")
    model_state = engine.state.parameters.model_state_dict()
    if engine.is_main_process:
        _save(dict(model=model_state, global_step=engine.global_step), path / "model.pt")
    dist.barrier(group=engine.mesh.get_group())
    if engine.is_main_process:
        metadata = dict(sharding=engine.state.sharding, world_size=engine.world_size, global_step=engine.global_step)
        temporary = path / "metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(path / "metadata.json")
    dist.barrier(group=engine.mesh.get_group())


def load(engine, path, *, scheduler=None):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["sharding"] == engine.state.sharding and metadata["world_size"] == engine.world_size
    state = torch.load(path / f"rank-{engine.rank}.pt", map_location="cpu", weights_only=False)
    assert state["global_step"] == metadata["global_step"]
    engine.state.load_state_dict(state["state"])
    # Load compute weights after optimizer hooks. This preserves the exact
    # stochastic writeback / FP8 codes stored at the checkpoint boundary.
    if engine.is_main_process:
        model = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        assert model["global_step"] == metadata["global_step"]
    else:
        model = None
    engine.state.parameters.load_model_state_dict(model["model"] if model is not None else None)
    del model
    restore_buffers(engine.model, state["buffers"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    engine.global_step, engine.micro_step = state["global_step"], state["micro_step"]
    engine._pending_microbatches = 0
    engine._loss_weight = 0.
    engine._weighted = None
    engine.sync_gradients = True
    engine.state.zero_grad()
    restore_rng(state["rng"], engine.device)
    return state["extra"]
