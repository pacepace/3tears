"""Vectorised restatement of the package's mixture bet over lattice-valued paired differences, plus the candidate
recommendation statistics. Differences live on the lattice v = (i - 3) / 3, i = 0..6 (pass/fail uses i in {0, 3, 6}),
so every quantity read on the capital NOW (constant stakes) is a function of the counts per lattice value alone.
Checked against bp-decide's analysis/stats.py by check_core.py before any table is drawn."""
from __future__ import annotations

import math

import numpy as np

VALUES = (np.arange(7) - 3) / 3.0          # the difference on [-1, 1]
Z = (VALUES + 1.0) / 2.0                    # rescaled to [0, 1] as _scaled_to_unit does for (-1, 1)
D = 10
CAP = 0.9
W = np.arange(1, D + 1) / (D * (D + 1) / 2)  # _MIXTURE_WEIGHTS, in proportion to j
LOGW = np.log(W)


def _up_table(m: np.ndarray | float) -> np.ndarray:
    """log(1 + (j/D)(CAP/m)(z_v - m)) for every stake j and lattice value v; m broadcasts in front."""
    m = np.asarray(m, dtype=float)[..., None, None]
    j = np.arange(1, D + 1)[:, None] / D
    return np.log1p(j * CAP / m * (Z[None, :] - m))


def _down_table(m: np.ndarray | float) -> np.ndarray:
    m = np.asarray(m, dtype=float)[..., None, None]
    j = np.arange(1, D + 1)[:, None] / D
    return np.log1p(j * CAP / (1.0 - m) * (m - Z[None, :]))


def _mix(logcaps: np.ndarray) -> np.ndarray:
    """log of the weighted mixture over the last axis (stakes)."""
    x = logcaps + LOGW
    top = x.max(axis=-1)
    return top + np.log(np.exp(x - top[..., None]).sum(axis=-1))


def log_up_down(counts: np.ndarray, m: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """counts (..., 7): the upward and downward mixture log capital against the rescaled null m, now."""
    up = _mix(counts @ _up_table(m).T)
    down = _mix(counts @ _down_table(m).T)
    return up, down


def hedged_log_at(counts: np.ndarray, m: np.ndarray, toward: float = 0.5) -> np.ndarray:
    """The hedged capital now at a per-element null m (shape counts.shape[:-1])."""
    up = _mix(np.einsum("...v,...jv->...j", counts, _up_table(m)))
    down = _mix(np.einsum("...v,...jv->...j", counts, _down_table(m)))
    return np.logaddexp(up + math.log(toward), down + math.log1p(-toward))


def cs_lower(counts: np.ndarray, alpha: float = 0.1, steps: int = 32) -> np.ndarray:
    """The current-capital even-hedge confidence sequence's lower end, on the difference scale [-1, 1]
    (bounded_difference_interval(..., mixture=True, current=True) at confidence 1 - alpha)."""
    thr = math.log(1.0 / alpha)
    n = counts.sum(axis=-1)
    centre = (counts @ Z) / np.maximum(n, 1)
    rejected = np.zeros(counts.shape[:-1])
    inside = centre.copy()
    zero_kept = hedged_log_at(counts, np.full(counts.shape[:-1], 1e-12)) < thr
    for _ in range(steps):
        mid = (rejected + inside) / 2.0
        kept = hedged_log_at(counts, mid) < thr
        inside = np.where(kept, mid, inside)
        rejected = np.where(kept, rejected, mid)
    low = np.where(zero_kept, 0.0, rejected)
    return -1.0 + 2.0 * low


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(x) / math.sqrt(2.0)))


_erf = np.vectorize(math.erf, otypes=[float])


def dp_posterior(counts: np.ndarray, a: float = 1.0, base_m2: float = 1.0 / 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and sd of the mean difference under a Dirichlet-process prior of concentration a whose base is
    uniform on [-1, 1] (mean 0, second moment 1/3): the exact first two moments of the linear functional."""
    n = counts.sum(axis=-1)
    s1 = counts @ VALUES
    s2 = counts @ (VALUES**2)
    A = n + a
    mu = s1 / A
    m2 = (s2 + a * base_m2) / A
    var = np.maximum(m2 - mu**2, 0.0) / (A + 1.0)
    return mu, np.sqrt(var)


def p_better(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), np.sign(mu) * np.inf)
    return norm_cdf(x)


def expected_loss_switch(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """E[max(0, -delta)] under the normal approximation to the posterior."""
    sd = np.maximum(sd, 1e-12)
    x = mu / sd
    pdf = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return sd * pdf - mu * norm_cdf(-x)
