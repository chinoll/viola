"""FSDP2 parameter lifetime; gradient reduction is owned by Viola."""

from fnmatch import fnmatchcase
from types import MethodType

import torch
from torch import nn
from torch.utils._pytree import tree_map
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def _init_storage_dtypes(group):
    # FSDP's uniform-dtype restriction is for its gradient reduction. Viola
    # consumes every full gradient itself, while FSDP's all-gather already
    # packs mixed storage dtypes as bytes. Keep this private-API adapter local
    # to our parameter-only groups, without changing PyTorch globally.
    assert not group.reduce_grads, "Viola owns gradient reduction for this FSDP group."
    for parameter in group.fsdp_params:
        parameter.init_dtype_attrs(MixedPrecisionPolicy())
    group._orig_dtype = group._reduce_dtype = None
    group._fsdp_params_with_wider_grad_dtype = []


def _separate_output(module, args, output):
    # A parent returning a child's Tensor unchanged otherwise shares the same
    # backward-hook node. A rank-dependent optional root operation can then
    # reverse parent/child all-gather order. A view gives each unit its own
    # backward boundary without allocating another activation buffer.
    if torch.is_grad_enabled():
        return tree_map(lambda value: value.view_as(value)
                        if isinstance(value, torch.Tensor) and value.requires_grad else value, output)
    return output


def shard_parameters(model, mesh, *, blocks=None):
    """Shard complete blocks (or the root) without selecting a gradient policy.

    Explicit block paths may contain glob patterns. By default, children of
    ModuleList/Sequential containers are units. Shared parameters crossing units
    stay in the root unit so one logical parameter has one storage lifecycle.
    """
    modules = dict(model.named_modules())
    if blocks is None:
        candidates = [f"{name}.{child}".lstrip(".")
                      for name, module in modules.items()
                      if isinstance(module, (nn.ModuleList, nn.Sequential))
                      for child, _ in module.named_children()]
    else:
        candidates = [name for name in modules if name and any(fnmatchcase(name, p) for p in blocks)]
        assert candidates or not blocks, "No modules match sharding.blocks."
    candidates = sorted(candidates, key=lambda name: (name.count("."), name))
    selected = []
    for name in candidates:
        if not any(name.startswith(parent + ".") for parent in selected):
            selected.append(name)
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        unit = next((p for p in selected if name.startswith(p + ".")), "")
        aliases.setdefault(id(param), set()).add(unit)
    tied_units = set().union(*(v for v in aliases.values() if len(v) > 1)) if aliases else set()
    selected = [name for name in selected if name not in tied_units
                and next(modules[name].parameters(), None) is not None]
    # Preserve model storage dtype. Autocast and parameter-gradient dtype are
    # separate policies. No FSDP reduce-scatter runs: the gradient hooks consume
    # unsharded gradients and clear them before FSDP's post-backward phase.
    for name in [*selected, ""]:
        fully_shard(modules[name], mesh=mesh, reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy())
        group = modules[name]._get_fsdp_state()._fsdp_param_group
        if group is not None and len({p.sharded_param.dtype for p in group.fsdp_params}) > 1:
            group._init_mp_dtypes = MethodType(_init_storage_dtypes, group)
        modules[name].register_forward_hook(_separate_output, prepend=True)
    model.set_requires_gradient_sync(False)
    return {name: modules[name] for name in [*selected, ""]}
