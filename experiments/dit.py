"""Class-conditional DiT-B, latent patch4 = pixel patch32 for an f8 VAE."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def position_embedding(size, width):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    omega = 1 / 10000 ** (torch.arange(width // 4).float() / (width // 4))
    x, y = x.flatten()[:, None] * omega, y.flatten()[:, None] * omega
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)[None]


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, c):
        s1, a1, g1, s2, a2, g2 = self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, a1)
        b, n, d = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + g1[:, None] * self.proj(h)
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))


class DiT(nn.Module):
    def __init__(self, width=768, depth=12, heads=12, patch=4):
        super().__init__()
        self.patch = patch
        self.input = nn.Conv2d(4, width, patch, stride=patch)
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.label = nn.Embedding(1001, width)
        self.register_buffer("position", position_embedding(32 // patch, width))
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, patch * patch * 4)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.input.weight.flatten(1))
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.label.weight, std=.02)
        nn.init.normal_(self.time[0].weight, std=.02)
        nn.init.normal_(self.time[2].weight, std=.02)
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
        for m in (self.final_ada[-1], self.output):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, y):
        freqs = torch.exp(-math.log(10000) * torch.arange(128, device=t.device).float() / 128)
        phase = t.float()[:, None] * 1000 * freqs[None]
        c = self.time(torch.cat((phase.cos(), phase.sin()), -1).to(x.dtype)) + self.label(y)
        x = self.input(x).flatten(2).transpose(1, 2) + self.position.to(x.dtype)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.final_ada(c).chunk(2, -1)
        x = self.output(modulate(self.norm(x), shift, scale))
        b, n, _ = x.shape
        h, p = math.isqrt(n), self.patch
        return x.view(b, h, h, p, p, 4).permute(0, 5, 1, 3, 2, 4).reshape(b, 4, h * p, h * p)
