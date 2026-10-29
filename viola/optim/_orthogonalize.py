"""Quintic Newton–Schulz orthogonalization, with no training-framework dependency.

Standard coefficients: https://kellerjordan.github.io/posts/muon/
Gram recurrence: https://tridao.me/blog/2026/gram-newton-schulz/
"""

import torch


@torch.compile(fullgraph=True)
def standard(matrix, steps=5):
    transposed = matrix.shape[-2] > matrix.shape[-1]
    x = matrix.to(torch.bfloat16)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
        x = 3.4445 * x + polynomial @ x
    return x.mT if transposed else x


@torch.compile(fullgraph=True)
def gram(matrix, steps=5):
    # FP16 Gram arithmetic, rebuilding the Gram matrix after two iterations.
    transposed = matrix.shape[-2] > matrix.shape[-1]
    x = matrix.to(torch.float16)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    if x.shape[-2] == x.shape[-1]:
        for _ in range(steps):
            r = x @ x.mT
            x = 3.4445 * x + (-4.7750 * r + 2.0315 * (r @ r)) @ x
    else:
        # Each segment composes polynomials on the small Gram matrix. The
        # restart limits half-precision error before the remaining iterations.
        for length in (min(steps, 2), max(steps - 2, 0)):
            if length:
                r = x @ x.mT
                q = None
                for i in range(length):
                    z = -4.7750 * r + 2.0315 * (r @ r)
                    if i == 0:
                        q = z.clone()
                        q.diagonal(dim1=-2, dim2=-1).add_(3.4445)
                    else:
                        q = 3.4445 * q + z @ q
                    if i + 1 < length:
                        rz = 3.4445 * r + z @ r
                        r = 3.4445 * rz + z @ rz
                x = q @ x
    return x.mT if transposed else x


METHODS = {"standard": standard, "gram": gram}
