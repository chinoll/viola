"""Independent replicated or whole-matrix-owned gradient accumulation."""

from functools import partial

import torch
import torch.distributed as dist

from .config import dtype
from .fp8 import FP8Communication


class GradientState:
    """Ordered reduce-to-owner buckets, drained as backward produces gradients.

    All ranks execute the same collective order. A rank with unused parameters
    fills their slots with zeros at backward's end. Out-of-order gradients wait
    in `pending`; ordinary sequential blocks drain in reverse parameter order.
    Only owner-local accumulators survive between microbatches.
    """

    def __init__(self, store, *, gradient_dtype, communication_dtype=None, bucket_mb=25,
                 fp8_format="e4m3", fp8_tile_size=256, shard=False):
        self.shard = shard
        self.defer_reduction = bool(store.fsdp_modules)
        self.records, self.mesh = store.records, store.mesh
        self.rank, self.group = self.mesh.get_local_rank(), self.mesh.get_group()
        self.gradient_dtype = gradient_dtype
        self.dtype = torch.float32 if communication_dtype == "fp8" else dtype(communication_dtype) or gradient_dtype
        device = self.records[0].parameter.device
        capacity = max(1, int(bucket_mb * 1024 * 1024) // torch.empty((), dtype=self.dtype).element_size())
        capacity = min(capacity, sum(r.parameter.numel() for r in self.records))
        self.accumulator = torch.zeros(sum(r.parameter.numel() for r in self.records if not shard or r.owner == self.rank),
                                       device=device, dtype=gradient_dtype)
        self.local = {}
        offset = 0
        for index, record in enumerate(self.records):
            if not shard or record.owner == self.rank:
                self.local[index] = self.accumulator[offset:offset + record.parameter.numel()]
                offset += record.parameter.numel()

        self.buckets = []
        self.piece_counts = [0] * len(self.records)
        pieces, filled, owner = [], 0, None
        for index in reversed(range(len(self.records))):
            record = self.records[index]
            if pieces and record.owner != owner:
                self.buckets.append((owner, pieces, filled))
                pieces, filled = [], 0
            owner = record.owner
            offset = 0
            while offset < record.parameter.numel():
                length = min(record.parameter.numel() - offset, capacity - filled)
                pieces.append((index, offset, filled, length))
                self.piece_counts[index] += 1
                offset += length
                filled += length
                if filled == capacity:
                    self.buckets.append((owner, pieces, filled))
                    pieces, filled = [], 0
        if pieces:
            self.buckets.append((owner, pieces, filled))

        capacity = max(length for _, _, length in self.buckets)
        self.buffer = torch.empty(capacity, device=device, dtype=self.dtype)
        self.fp8 = (FP8Communication(capacity, self.mesh, device, mode="reduce" if shard else "all_reduce",
                                    fp8_format=fp8_format, tile_size=fp8_tile_size)
                    if communication_dtype == "fp8" else None)

        self.used = [False] * len(self.records)
        self.in_backward = False
        self.pending = [None] * len(self.records)
        self.hooks = []

    def bind(self, index, parameter):
        if getattr(parameter, "_viola_gradient_state", None) is self:
            return
        parameter._viola_gradient_state = self
        self.hooks.append(parameter.register_post_accumulate_grad_hook(partial(self._ready, index)))

    def begin_backward(self):
        self.in_backward = True
        self.seen = [False] * len(self.records)
        self.remaining = self.piece_counts.copy()
        self.next_bucket = 0

    @torch.no_grad()
    def _ready(self, index, parameter):
        self.seen[index] = self.used[index] = True
        gradient = parameter.grad.contiguous().view(-1)
        parameter.grad = None
        if self.shard:
            self.pending[index] = gradient
            # FSDP may issue parameter all-gathers while backward is running.
            # Drain afterwards to preserve collective order with rank-dependent
            # unused parameters. Accumulators remain owner-local between micros.
            if not self.defer_reduction:
                self._drain()
        else:
            self.local[index].add_(gradient)

    @torch.no_grad()
    def _drain(self, *, final=False):
        while self.next_bucket < len(self.buckets):
            owner, pieces, length = self.buckets[self.next_bucket]
            if not final and any(not self.seen[i] for i, _, _, _ in pieces):
                break
            buffer = self.buffer[:length]
            for index, source, target, size in pieces:
                slot = buffer[target:target + size]
                gradient = self.pending[index]
                if gradient is None:
                    slot.zero_()
                else:
                    slot.copy_(gradient[source:source + size])
                self.remaining[index] -= 1
                if not self.remaining[index]:
                    self.pending[index] = None
            if self.fp8 is not None:
                self.fp8.reduce(buffer, owner)
            else:
                dist.reduce(buffer, dst=int(self.mesh.mesh[owner]), group=self.group)
            if self.rank == owner:
                for index, target, source, size in pieces:
                    self.local[index][target:target + size].add_(buffer[source:source + size])
            self.next_bucket += 1

    def finish_backward(self):
        if self.shard:
            self._drain(final=True)
        self.in_backward = False

    @torch.no_grad()
    def reduce(self, divisor):
        # The payload has already been reduced each microbatch. Exchange only
        # usage flags here, then normalize once over the entire accumulation.
        used = torch.tensor(self.used, device=self.buffer.device, dtype=torch.int32)
        dist.all_reduce(used, op=dist.ReduceOp.MAX, group=self.group)
        active = used.bool().tolist()
        if not self.shard:
            for _, pieces, length in self.buckets:
                buffer = self.buffer[:length]
                for index, source, target, size in pieces:
                    buffer[target:target + size].copy_(self.local[index][source:source + size])
                if self.fp8 is not None:
                    self.fp8.all_reduce(buffer)
                else:
                    dist.all_reduce(buffer, group=self.group)
                for index, target, source, size in pieces:
                    self.local[index][target:target + size].copy_(buffer[source:source + size])
        self.accumulator.div_(divisor)
        return active

    @torch.no_grad()
    def zero_grad(self):
        self.accumulator.zero_()
        self.used = [False] * len(self.records)
