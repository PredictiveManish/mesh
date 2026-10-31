"""Trust plane: deciding whether a stranger's update is honest.

Two gates, both cheap:

  1. norm gate      -- reject updates whose magnitude is wildly out of family
  2. direction gate -- reject updates that point somewhere else than the
                       consensus of their peers

Together they stop random-noise and random-direction poisoning at almost no
compute cost. They do NOT stop a careful adversary who aligns with the
consensus and perturbs within tolerance; that needs recompute sampling or
loss-impact scoring, which is the next thing to build (see docs).
"""
from __future__ import annotations

import statistics

import torch

from .types import Verdict


def flatten(delta: dict) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in delta.values()])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / d)


class RobustAggregator:
    def __init__(self, norm_tau: float = 3.0, cos_min: float = 0.10,
                 min_accept: int = 2, clip: bool = True):
        self.norm_tau = norm_tau
        self.cos_min = cos_min
        self.min_accept = min_accept
        self.clip = clip

    def aggregate(self, updates: dict[str, dict]):
        ids = list(updates)
        if len(ids) == 1:
            return updates[ids[0]], [Verdict(ids[0], True, "single worker")]

        vecs = {i: flatten(updates[i]) for i in ids}
        norms = {i: float(vecs[i].norm()) for i in ids}
        med_norm = statistics.median(norms.values()) or 1e-12

        # Reference = the update whose median cosine to all others is highest.
        # Robust to a minority of bad actors without needing a trusted node.
        cos = {(a, b): _cos(vecs[a], vecs[b]) for a in ids for b in ids if a != b}
        ref = max(ids, key=lambda i: statistics.median(
            [cos[(i, j)] for j in ids if j != i]))

        verdicts: list[Verdict] = []
        accepted: list[str] = []
        for i in ids:
            nr = norms[i] / med_norm
            c = 1.0 if i == ref else cos[(i, ref)]
            ok, why = True, "ok"
            if nr > self.norm_tau:
                ok, why = False, f"norm {nr:.1f}x the median (gate {self.norm_tau}x)"
            elif c < self.cos_min:
                ok, why = False, f"direction off consensus (cos {c:+.2f})"
            verdicts.append(Verdict(i, ok, why, norms[i], nr, c))
            if ok:
                accepted.append(i)

        if len(accepted) < self.min_accept:
            # Never let the gate starve the run; fall back to the reference.
            accepted = [ref]

        # Clip survivors to the median norm so no single worker can dominate
        # the average even if it passed both gates.
        scale = {}
        for i in accepted:
            scale[i] = min(1.0, med_norm / max(norms[i], 1e-12)) if self.clip else 1.0

        avg = {}
        for k in updates[accepted[0]]:
            acc = torch.zeros_like(updates[accepted[0]][k], dtype=torch.float32)
            for i in accepted:
                acc += updates[i][k].float() * scale[i]
            avg[k] = acc / len(accepted)

        return avg, verdicts
