"""Scaled FP8 on the wire, FP32 reduction, with bounded reusable workspaces."""

import math

import torch
import torch.distributed as dist


def packet_bytes(groups, tile_size):
    # Keep both the code rows and scale rows aligned for vectorized CUDA access.
    return (groups * (tile_size + 4) + 15) // 16 * 16


class FP8Communication:
    def __init__(self, capacity, mesh, device, *, mode, fp8_format="e4m3", tile_size=256):
        assert device.type == "cuda", "FP8 communication uses CUDA TileLang codecs."
        assert tile_size >= 32 and tile_size & (tile_size - 1) == 0
        self.fp8_dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fp8_format]
        self.mesh, self.group = mesh, mesh.get_group()
        self.rank, self.world = mesh.get_local_rank(), mesh.size()
        self.tile_size, self.mode = tile_size, mode
        self.device = device
        if self.world == 1:
            self.send = torch.empty(0, device=device, dtype=torch.uint8)
            self.receive = self.partial = None
            return
        if mode == "all_reduce":
            groups = math.ceil(capacity / (self.world * tile_size))
            self.send = torch.empty(self.world * packet_bytes(groups, tile_size), device=device, dtype=torch.uint8)
            self.receive = torch.empty_like(self.send)
            self.partial = torch.empty(groups * tile_size, device=device, dtype=torch.float32)
        else:
            # Cap the gather buffer at roughly one FP32 staging bucket, rather
            # than allocating world_size full buckets. Tiny buckets need at
            # least one tile per rank. There is no full-model FP32 residual.
            groups = max(1, min(math.ceil(capacity / tile_size),
                                capacity * 4 // (self.world * (tile_size + 4))))
            self.chunk_size = groups * tile_size
            self.send = torch.empty(packet_bytes(groups, tile_size), device=device, dtype=torch.uint8)
            self.receive = None
            self.partial = None

    @property
    def workspace_bytes(self):
        return sum(value.numel() * value.element_size() for value in
                   (self.send, self.receive, self.partial) if value is not None)

    @torch.no_grad()
    def all_reduce(self, value):
        if self.world == 1:
            return
        from . import _fp8_kernels as codec
        groups = math.ceil(value.numel() / (self.world * self.tile_size))
        width, packet_size = groups * self.tile_size, packet_bytes(groups, self.tile_size)
        send = self.send[:self.world * packet_size].view(self.world, packet_size)
        receive = self.receive[:self.world * packet_size].view(self.world, packet_size)
        partial = self.partial[:width]
        codec.pack(value, send, self.tile_size, self.fp8_dtype)
        # Each rank receives one logical chunk from every contributor. Bytes
        # are transported, never summed as integers; each sender keeps its scale.
        dist.all_to_all_single(receive, send, group=self.group)
        codec.unpack(receive, partial, self.tile_size, self.fp8_dtype, reduce=True)
        codec.pack(partial, send[:1], self.tile_size, self.fp8_dtype)
        dist.all_gather_into_tensor(receive.view(-1), send[0], group=self.group)
        codec.unpack(receive, value, self.tile_size, self.fp8_dtype)

    @torch.no_grad()
    def reduce(self, value, owner):
        if self.world == 1:
            return
        from . import _fp8_kernels as codec
        if self.rank == owner and self.receive is None:
            self.receive = torch.empty(self.world * self.send.numel(), device=self.device, dtype=torch.uint8)
        for offset in range(0, value.numel(), self.chunk_size):
            chunk = value[offset:offset + self.chunk_size]
            groups = math.ceil(chunk.numel() / self.tile_size)
            packet_size = packet_bytes(groups, self.tile_size)
            send = self.send[:packet_size]
            codec.pack(chunk, send.view(1, -1), self.tile_size, self.fp8_dtype)
            receive = (self.receive[:self.world * packet_size].view(self.world, packet_size)
                       if self.rank == owner else None)
            dist.gather(send, list(receive.unbind(0)) if receive is not None else None,
                        dst=int(self.mesh.mesh[owner]), group=self.group)
            if receive is not None:
                codec.unpack(receive, chunk, self.tile_size, self.fp8_dtype, reduce=True)
