# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
flow.py -- the mixture path, its exact few-step jump kernel and the (t, h) features (Chat-VPA Sec. 3.1).

Imported from text-VPA at 6f97346: ``KappaSchedule``, ``mixture_path_sample`` and ``uniform_time_grid`` from
``text_vpa/models/realizers/paths.py``, the whole of ``kernels.py``, and ``timestep_embedding`` from the PR-JEPA
realizer's ``backbone.py`` (docs/provenance.md). The equation numbers below are Chat-VPA's.

Mixture path (Eq. 3). Each position of a block follows

    p_t(x^i | w^i) = (1 - kappa_t) delta_{w_0^i} + kappa_t delta_{w^i},   kappa_0 = 0, kappa_1 = 1, kappa_dot >= 0,

from a source token w_0^i ([MASK] in Chat-VPA) to its token w^i, with schedules  linear kappa_t = t,
sqrt kappa_t = t^(1/2),  cosine kappa_t = 1 - cos(pi t / 2)  (the default).

Exact jump kernel (Eq. 4). Holding the posterior fixed over [t, t + h], a position that has not arrived jumps
with probability

    pi_{t,h} = (kappa_{t+h} - kappa_t) / (1 - kappa_t)  in [0, 1],

and Lemma 1 states that with the oracle posterior the kernel maps p_t to p_{t+h} exactly
(kappa_t + (1 - kappa_t) pi_{t,h} = kappa_{t+h}), that two half steps compose to one step, and that
pi_{t, 1-t} = 1. The Euler jump mass h kappa_dot_t / (1 - kappa_t) is a probability only when it is <= 1, which
the sqrt schedule violates near t = 0.

Energy-tilted, clipped kernel (PR-JEPA Sec. 3.3, Eqs. 15-17 and Prop. 4 there; Chat-VPA has no energies and uses
it with no tilt, where it is Eq. 4, as the independent reference of the samplers' tests). For position i with
current token a, posterior P and energies E^i:

    rho~(j) = min( P(j) exp(-E(j)/T), c_max ),   j != a
    K~(j)   = pi_{t,h} rho~(j) / max(1, sum_{j'!=a} rho~(j'))
    K~(a)   = 1 - sum_{j != a} K~(j)

Rejected jump mass becomes probability of staying. Sampling: with probability
pi sum_j rho~(j) / max(1, sum_j rho~(j)) jump to j ~ rho~ / sum rho~, otherwise keep a. The safety normalisation
max(1, .) is inactive whenever P is a probability vector and the multiplier is at most 1; it is needed for
guided posteriors (1 + w) P_c - w P_u, whose lower clip at 0 is applied here, and for the repair-aware multiplier
exp(-[E(j) - E(a)]/T), which can exceed one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, overload

import torch
from torch import Tensor

__all__ = [
    "JumpKernel",
    "KappaSchedule",
    "MixtureSample",
    "kernel_distribution",
    "mixture_path_sample",
    "repair_aware_multiplier",
    "sample_jump",
    "tilted_jump_kernel",
    "timestep_embedding",
    "uniform_time_grid",
]

SCHEDULES: tuple[str, ...] = ("linear", "sqrt", "cosine")


def uniform_time_grid(num_steps: int) -> list[tuple[float, float]]:
    """``[(t_k, h)]`` for t_k = k/K, h = 1/K, k = 0..K-1 (the grid of Algorithm 1)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    h = 1.0 / num_steps
    return [(k * h, h) for k in range(num_steps)]


class KappaSchedule:
    """Mixture-path schedule kappa_t on [0, 1] with kappa_0 = 0, kappa_1 = 1, kappa_dot >= 0 (Eq. 3).

    Methods accept a Python float or a tensor and return the same kind.
    """

    def __init__(self, kind: str = "linear") -> None:
        if kind not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {kind!r}")
        self.kind: str = kind

    def __repr__(self) -> str:
        return f"KappaSchedule({self.kind!r})"

    @overload
    def kappa(self, t: float) -> float: ...
    @overload
    def kappa(self, t: Tensor) -> Tensor: ...

    def kappa(self, t: float | Tensor) -> float | Tensor:
        """kappa_t."""
        if isinstance(t, Tensor):
            if self.kind == "linear":
                return t.clone()
            if self.kind == "sqrt":
                return t.clamp_min(0.0).sqrt()
            return 1.0 - torch.cos(math.pi * t / 2.0)
        if self.kind == "linear":
            return float(t)
        if self.kind == "sqrt":
            return math.sqrt(max(t, 0.0))
        return 1.0 - math.cos(math.pi * t / 2.0)

    @overload
    def kappa_dot(self, t: float) -> float: ...
    @overload
    def kappa_dot(self, t: Tensor) -> Tensor: ...

    def kappa_dot(self, t: float | Tensor) -> float | Tensor:
        """d kappa_t / dt (``inf`` at t = 0 for the sqrt schedule)."""
        if isinstance(t, Tensor):
            if self.kind == "linear":
                return torch.ones_like(t)
            if self.kind == "sqrt":
                return 0.5 / t.sqrt()
            return (math.pi / 2.0) * torch.sin(math.pi * t / 2.0)
        if self.kind == "linear":
            return 1.0
        if self.kind == "sqrt":
            return math.inf if t <= 0.0 else 0.5 / math.sqrt(t)
        return (math.pi / 2.0) * math.sin(math.pi * t / 2.0)

    @overload
    def exact_jump_mass(self, t: float, h: float) -> float: ...
    @overload
    def exact_jump_mass(self, t: Tensor, h: float | Tensor) -> Tensor: ...

    def exact_jump_mass(self, t: float | Tensor, h: float | Tensor) -> float | Tensor:
        """pi_{t,h} = (kappa_{t+h} - kappa_t) / (1 - kappa_t)  (Eq. 4); t + h is clipped to 1.

        Requires kappa_t < 1, i.e. t < 1: the sampler never evaluates t = 1.
        """
        if isinstance(t, Tensor):
            end = (t + h).clamp_max(1.0)
            k_t = self.kappa(t)
            if bool((t >= 1.0).any()) or bool((k_t >= 1.0).any()):
                raise ValueError("pi_{t,h} is undefined where kappa_t = 1 (t = 1)")
            return (self.kappa(end) - k_t) / (1.0 - k_t)
        k_t = self.kappa(t)
        if t >= 1.0 or k_t >= 1.0:
            raise ValueError("pi_{t,h} is undefined where kappa_t = 1 (t = 1)")
        return (self.kappa(min(t + float(h), 1.0)) - k_t) / (1.0 - k_t)

    @overload
    def euler_jump_mass(self, t: float, h: float) -> float: ...
    @overload
    def euler_jump_mass(self, t: Tensor, h: float | Tensor) -> Tensor: ...

    def euler_jump_mass(self, t: float | Tensor, h: float | Tensor) -> float | Tensor:
        """h kappa_dot_t / (1 - kappa_t): the Euler kernel's jump mass, a probability only if <= 1."""
        if isinstance(t, Tensor):
            return h * self.kappa_dot(t) / (1.0 - self.kappa(t))
        return float(h) * self.kappa_dot(t) / (1.0 - self.kappa(t))

    def step_masses(self, num_steps: int) -> list[float]:
        """pi_{t_k, 1/K} on the uniform grid, k = 0..K-1 (one per realizer pass)."""
        return [self.exact_jump_mass(t, h) for t, h in uniform_time_grid(num_steps)]


# ---------------------------------------------------------------------------------------------
# Mixture-path training states
# ---------------------------------------------------------------------------------------------
class MixtureSample(NamedTuple):
    """A training state on the mixture path."""

    w_t: Tensor     # [B, N] int64
    t: Tensor       # [B] float32 flow time
    h: Tensor       # [B] float32 step size 1/K
    w_0: Tensor     # [B, N] int64 source tokens


def mixture_path_sample(
    w_1: Tensor,
    schedule: KappaSchedule,
    vocab_size: int,
    step_budgets: tuple[int, ...] | list[int],
    source: str = "uniform",
    mask_id: int | None = None,
    time_sampling: str = "grid",
    generator: torch.Generator | None = None,
) -> MixtureSample:
    """Draw (w_t, t, h): w_t^i = w_1^i with probability kappa_t, else w_0^i (Eq. 3).

    ``time_sampling="grid"`` draws K from ``step_budgets`` and t = k/K, k ~ U{0..K-1}: exactly the
    states the sampler evaluates at. ``"uniform"`` draws t ~ U[0, 1) (h still from the budgets).
    """
    if w_1.dim() != 2:
        raise ValueError(f"expected w_1 of shape [B, N], got {tuple(w_1.shape)}")
    b, n = w_1.shape
    device = w_1.device
    budgets = torch.tensor(list(step_budgets), dtype=torch.float32)
    k_steps = budgets[torch.randint(len(budgets), (b,), generator=generator)]
    if time_sampling == "grid":
        k = torch.floor(torch.rand(b, generator=generator) * k_steps)
        t = k / k_steps
    elif time_sampling == "uniform":
        t = torch.rand(b, generator=generator)
    else:
        raise ValueError("time_sampling must be 'grid' or 'uniform'")
    h = 1.0 / k_steps
    if source == "uniform":
        w_0 = torch.randint(vocab_size, (b, n), generator=generator)
    elif source == "mask":
        if mask_id is None:
            raise ValueError("the mask source needs mask_id")
        w_0 = torch.full((b, n), mask_id, dtype=torch.long)
    else:
        raise ValueError("source must be 'uniform' or 'mask'")
    kappa = schedule.kappa(t)                                                  # [B]
    arrived = torch.rand(b, n, generator=generator) < kappa[:, None]
    w_t = torch.where(arrived, w_1.cpu(), w_0)
    return MixtureSample(w_t.to(device), t.to(device), h.to(device), w_0.to(device))


# ---------------------------------------------------------------------------------------------
# The jump kernel of one parallel step
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class JumpKernel:
    """Kernel of one parallel step for every position."""

    rho: Tensor        # [..., S] rho~ with rho~(a) = 0 at the current token
    mass: Tensor       # [...] sum_{j != a} rho~(j)
    jump_prob: Tensor  # [...] pi * mass / max(1, mass)

    @property
    def stay_prob(self) -> Tensor:
        return 1.0 - self.jump_prob


def repair_aware_multiplier(energies: Tensor, current: Tensor, temperature: float) -> Tensor:
    """exp(-[E(j) - E(a)]/T) (PR-JEPA Sec. 3.3), with exp(+inf) for admissible j when E(a) = inf and 0 when both
    are infinite. Shapes: energies ``[..., S]``, current ``[...]`` -> ``[..., S]``."""
    e_cur = energies.gather(-1, current.unsqueeze(-1))
    diff = energies - e_cur
    diff = torch.where(torch.isinf(energies), torch.full_like(diff, float("inf")), diff)
    return torch.exp(-diff / temperature)


def tilted_jump_kernel(
    posterior: Tensor,
    current: Tensor,
    pi: float | Tensor,
    multiplier: Tensor | None = None,
    c_max: float = 1.0,
) -> JumpKernel:
    """PR-JEPA Eq. 17 with its Prop. 4's safety normalisation.

    Args:
        posterior: ``[..., S]`` P (or a guided combination; negative entries are clipped to 0).
        current: ``[...]`` int64 current tokens a.
        pi: exact jump mass pi_{t,h} (scalar or broadcastable to ``[...]``).
        multiplier: ``[..., S]`` exp(-E/T) (None: no tilt); may contain +inf (repair-aware variant).
        c_max: upper clip, in (0, 1].
    """
    if not 0.0 < c_max <= 1.0:
        raise ValueError("c_max must lie in (0, 1] (PR-JEPA Prop. 4(a))")
    p = posterior.clamp_min(0.0)
    if multiplier is not None:
        tilted = p * multiplier
        rho = torch.where(p > 0, tilted, torch.zeros_like(tilted))    # 0 * inf -> 0
        rho = torch.nan_to_num(rho, nan=0.0)
    else:
        rho = p
    rho = rho.clamp_max(c_max)
    rho = rho.scatter(-1, current.unsqueeze(-1), 0.0)                    # j != a
    mass = rho.sum(dim=-1)
    pi_t = torch.as_tensor(pi, dtype=mass.dtype, device=mass.device)
    jump_prob = pi_t * mass / mass.clamp_min(1.0)
    return JumpKernel(rho, mass, jump_prob)


def kernel_distribution(kernel: JumpKernel, current: Tensor) -> Tensor:
    """The full K~ over V, ``[..., S]``: rows sum to one (PR-JEPA Prop. 4(a))."""
    scale = torch.where(kernel.mass > 0, kernel.jump_prob / kernel.mass.clamp_min(1e-30), torch.zeros_like(kernel.mass))
    dist = kernel.rho * scale.unsqueeze(-1)
    return dist.scatter_add(-1, current.unsqueeze(-1), kernel.stay_prob.unsqueeze(-1))


def sample_jump(kernel: JumpKernel, current: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """One position's draw: jump with probability ``jump_prob`` to j ~ rho~ / sum rho~, else stay.

    Uses the Gumbel-max trick on log rho~ (zero entries are never drawn); ``generator`` must live on the
    tensors' device.
    """
    u = torch.rand(kernel.jump_prob.shape, generator=generator, device=current.device, dtype=torch.float32)
    jump = (u < kernel.jump_prob.float()) & (kernel.mass > 0)
    g = torch.rand(kernel.rho.shape, generator=generator, device=current.device, dtype=torch.float32)
    gumbel = -torch.log(-torch.log(g.clamp_min(torch.finfo(torch.float32).tiny)))
    proposal = (torch.log(kernel.rho.float()) + gumbel).argmax(dim=-1)
    return torch.where(jump, proposal, current)


# ---------------------------------------------------------------------------------------------
# (t, h) features
# ---------------------------------------------------------------------------------------------
def timestep_embedding(t: Tensor, dim: int, max_period: float = 10_000.0, scale: float = 1000.0) -> Tensor:
    """Sinusoidal features of ``scale * t``: ``[B]`` -> ``[B, dim]`` (cos half, then sin half)."""
    if t.dim() != 1:
        raise ValueError(f"expected t of shape [B], got {tuple(t.shape)}")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (scale * t.float())[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
