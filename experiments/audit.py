"""Diagnostic-only shadow states. They are not part of Viola's memory footprint."""
import json
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from viola.kernels import dequantize


PAIRS = ("fp8_bf16",)
FIELDS = ["grad_rms", "fp8_rms", "fp8_absmax", "zero_fraction",
          "nonfinite_fraction", "code_unchanged_fraction", "endpoint_fraction",
          "scale_min", "scale_max", "local_quant_relative_l2", "local_quant_bias",
          "bf16_rms", "bf16_zero_fraction", "bf16_nonfinite_fraction"]
FIELDS += [f"{pair}_{metric}" for pair in PAIRS for metric in
           ("error_squared_sum", "reference_squared_sum", "relative_l2", "cosine", "max_abs", "bias")]
FIELDS += ["input_fp8_bf16_relative_l2", "input_fp8_bf16_cosine",
           "fp8_bf16_norm_ratio", "fp8_bf16_scale_aligned_relative_l2"]


def comparison(actual, reference):
    delta = actual - reference
    es, rs, xs = delta.square().sum(), reference.square().sum(), actual.square().sum()
    cosine = (actual * reference).sum() / (rs * xs).sqrt().clamp_min(1e-30)
    cosine = torch.where((rs == 0) & (xs == 0), torch.ones_like(cosine), cosine)
    return (es, rs, (es / rs.clamp_min(1e-30)).sqrt(), cosine.clamp(-1, 1), delta.abs().max(), delta.mean())


@torch.compile(fullgraph=True)
def advance_shadow(g, sbf, beta):
    # Materialize low-precision outputs before measuring. A fused cast-to-low-
    # precision followed by cast-to-float can otherwise be optimized away.
    fresh = beta * sbf.float() + (1 - beta) * g
    # Match the production optimizer: this step's NS input uses fresh momentum,
    # before storage rounding. Only history for the next step is rounded.
    return fresh.bfloat16(), ((1 - beta) * g + beta * fresh).bfloat16()


@torch.compile(fullgraph=True)
def measure(g, q, scale, nq, nscale, actual_input, newbf, reference_input, beta, shape, quant):
    n = g.numel()
    old = dequantize(q, scale, shape, **quant)
    current = dequantize(nq, nscale, shape, **quant)
    fresh = beta * old + (1 - beta) * g
    local = comparison(current, fresh)
    metrics = [g.square().mean().sqrt(), current.square().mean().sqrt(),
               current.abs().max(), (current == 0).float().mean(), (~current.isfinite()).float().mean(),
               (q[:n] == nq[:n]).float().mean(), (nq[:n].view(torch.float8_e4m3fn).float().abs() == 448).float().mean(),
               nscale.min(), nscale.max(), local[2], local[5],
               newbf.float().square().mean().sqrt(), (newbf == 0).float().mean(),
               (~newbf.isfinite()).float().mean()]
    reference = newbf.float()
    metrics.extend(comparison(current, reference))
    pair = comparison(actual_input.flatten().float(), reference_input.float())
    metrics.extend((pair[2], pair[3]))
    rs, xs = reference.square().sum(), current.square().sum()
    fitted_scale = (current * reference).sum() / rs.clamp_min(1e-30)
    metrics.extend(((xs / rs.clamp_min(1e-30)).sqrt(),
                    ((current - fitted_scale * reference).square().sum() / xs.clamp_min(1e-30)).sqrt()))
    return torch.stack(metrics)


class MomentumAudit:
    def __init__(self, model, output, rank, ns_interval=100):
        self.names = {id(p): name for name, p in model.named_parameters()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.ns_interval = ns_interval
        self.state, self.pending, self.rows, self.records = {}, {}, [], []
        self.ns_inputs = {}
        self.step = 0
        self.file_index = 0
        self.beta = torch.tensor(.95, device="cuda", dtype=torch.float32)
        self.max_shadow_bytes = 0

    @torch.no_grad()
    def momentum(self, p, grad, q, scale, nq, nscale, matrix, group, loss_scale, quant):
        assert group["momentum"] == .95 and group.get("nesterov", True)
        name = self.names[id(p)]
        n = p.numel()
        if name not in self.state:
            self.state[name] = torch.zeros(n, device=p.device, dtype=torch.bfloat16)
        g = grad.float().flatten() / loss_scale
        sbf, reference_input = advance_shadow(g, self.state[name], self.beta)
        metrics = measure(g, q, scale, nq, nscale, matrix, sbf, reference_input,
                          self.beta, tuple(matrix.shape), quant)
        self.pending[name] = sbf
        self.rows.append((name, list(p.shape), metrics))
        if self.step % self.ns_interval == 0:
            # Same BF16 NS computation for both paths; only momentum history differs.
            self.ns_inputs[name] = reference_input.view_as(matrix)

    @torch.no_grad()
    def update(self, p, actual, group, ns):
        name = self.names[id(p)]
        if name in self.ns_inputs:
            reference = ns(self.ns_inputs.pop(name), steps=group.get("ns_steps", 5))
            values = torch.stack(comparison(actual.float().flatten(), reference.float().flatten()))
            self.rows.append((name + ":NS", list(p.shape), values))

    def finish(self, overflow=False):
        if overflow:
            self.pending.clear(); self.rows.clear(); self.ns_inputs.clear()
            return {"overflow_skipped": True}
        self.state.update(self.pending)
        self.pending.clear()
        momentum_rows = [r for r in self.rows if not r[0].endswith(":NS")]
        ns_rows = [r for r in self.rows if r[0].endswith(":NS")]
        values = torch.stack([r[2] for r in momentum_rows]).float().cpu().numpy()
        for (name, shape, _), row in zip(momentum_rows, values):
            self.records.append(dict(step=self.step, rank=self.rank, name=name, shape=shape,
                                     **dict(zip(FIELDS, map(float, row)))))
        if ns_rows:
            ns_values = torch.stack([r[2] for r in ns_rows]).float().cpu().tolist()
            with (self.output / f"ns-rank{self.rank}.jsonl").open("a") as f:
                for (name, shape, _), row in zip(ns_rows, ns_values):
                    f.write(json.dumps(dict(step=self.step, name=name, shape=shape, reference_precision="bf16",
                        **dict(zip(("error_squared_sum", "reference_squared_sum", "relative_l2", "cosine", "max_abs", "bias"), row)))) + "\n")
        self.rows.clear()
        shadow_bytes = sum(t.numel() * t.element_size() for t in self.state.values())
        summary = dict(reference_precision="bf16", local_matrices=len(momentum_rows), diagnostic_shadow_bytes=shadow_bytes,
                       diagnostic_shadow_pending_peak_bytes=2 * shadow_bytes)
        for pair in PAIRS:
            i = FIELDS.index(pair + "_error_squared_sum")
            summary[pair + "_local_relative_l2"] = float(np.sqrt(values[:, i].sum() / max(values[:, i + 1].sum(), 1e-30)))
            summary[pair + "_max_matrix_relative_l2"] = float(values[:, i + 2].max())
            summary[pair + "_min_matrix_cosine"] = float(values[:, i + 3].min())
        if len(self.records) >= 20000:
            self.flush()
        return summary

    def flush(self):
        if self.records:
            path = self.output / f"momentum-rank{self.rank}-{self.file_index:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(self.records), path, compression="zstd")
            self.records.clear()
            self.file_index += 1
