# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The two loops of the inpainting engine on the mirror-extended working image.

Sparsity: FISTA with Nesterov momentum and the notebooks' restart on a rise of the full cost
every ten iterations, the cycle-spun soft threshold as the prox, the paper's correction of
the known pixels after each step; it stops on the relative change inside the regions and
returns the iterate with the lowest held-out error (else the lowest known-pixel variance
error). Cascade: the paper's plain loop (a gradient step on the guide terms when there are
any, the inference, the correction), no momentum, stopping on the variance error.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from dynamix.core.inpaint.correct import correct as _correct
from dynamix.core.inpaint.forward import data_cost, gradient
from dynamix.core.inpaint.readouts import errors

__all__ = ["Controls", "History", "Problem", "cascade_loop", "fista"]


@dataclasses.dataclass(frozen=True)
class Controls:
    n_iter: int = 200
    tol: float = 1e-3
    stop_var_err: float = 0.01
    batch: float = 1.0
    correct: bool = True
    seed: int = 0
    patience: int = 20


@dataclasses.dataclass
class History:
    cost: list
    score: list
    stop: str
    returned: int
    iterations: int


@dataclasses.dataclass
class Problem:
    """The terms (the target's first), the working data ``y``, the known pixels, the
    boundary ring, the region (gaps, veils and held-out blocks), the step bound ``L`` and
    the held-out mask on the working grid."""

    terms: list
    y: np.ndarray
    known: np.ndarray
    ring: np.ndarray
    region: np.ndarray
    L: float
    heldout: "np.ndarray | None" = None

    @property
    def has_heldout(self) -> bool:
        return self.heldout is not None and bool(self.heldout.any())

    def score(self, x) -> float:
        """The held-out RMS of the corrected iterate when blocks are held out, else the
        known-pixel variance error of the uncorrected one."""
        if self.has_heldout:
            return errors(x, self.y, self.heldout)["rms_rel"]
        return errors(x, self.y, self.known)["var_err"]

    def batched(self, rng, fraction: float) -> list:
        """The terms with the target's weights thinned to a random ``fraction`` of the known
        pixels (scaled back up), the others untouched."""
        t = self.terms[0]
        keep = rng.random(t.weights.shape) < float(fraction)
        w = np.where(keep, t.weights / float(fraction), 0.0)
        return [dataclasses.replace(t, weights=w)] + list(self.terms[1:])


def _check(cancel) -> None:
    if cancel is not None and cancel():
        from dynamix.core.wtmm_backend import ComputeCancelled

        raise ComputeCancelled()


def _tell(progress, text, frac) -> None:
    if progress is not None:
        progress(text, float(frac))


def fista(problem: Problem, prior, x0, controls: Controls, *, cancel=None, progress=None):
    """A mini-batch rescales the kept weights by 1/fraction, so the step is the fraction of 1/L."""
    rng = np.random.default_rng(int(controls.seed) + 1)
    x = np.array(x0, dtype=np.float64)
    x_prev = x.copy()
    t = 1.0
    best_score, best_x, best_k, worse = math.inf, x.copy(), 0, 0
    cost_hist, score_hist = [], []
    last_cost = math.inf
    stop, k = "cap", 0
    n = int(controls.n_iter)
    region = problem.region
    step = (controls.batch if controls.batch < 1.0 else 1.0) / problem.L
    for k in range(1, n + 1):
        _check(cancel)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = x + ((t - 1.0) / t_next) * (x - x_prev)
        terms = problem.terms if controls.batch >= 1.0 else problem.batched(rng, controls.batch)
        x_new = prior.prox(z - step * gradient(terms, z), step)
        # the known-pixel variance error reads the iterate before the correction pins the
        # known pixels; the held-out error reads the corrected iterate, which is what is returned
        score = None if problem.has_heldout else problem.score(x_new)
        if controls.correct:
            x_new = x_new - _correct(x_new, problem.y, problem.known, problem.ring, pad=0)
        if score is None:
            score = problem.score(x_new)
        restart = False
        if k % 10 == 0:
            cost = data_cost(problem.terms, x_new) + prior.penalty(x_new)
            cost_hist.append(float(cost))
            restart = cost > last_cost
            last_cost = cost
        score_hist.append(float(score))
        if score < best_score:
            best_score, best_x, best_k, worse = score, x_new.copy(), k, 0
        else:
            worse += 1
        change = (np.linalg.norm((x_new - x)[region])
                  / max(np.linalg.norm(x_new[region]), 1e-12)) if region.any() else 0.0
        if restart:
            x_prev, x, t = x_new.copy(), x_new, 1.0
        else:
            x_prev, x, t = x, x_new, t_next
        _tell(progress, f"sparsity {k}/{n}", k / n)
        if change < controls.tol:
            stop = "converged"
            break
        if (not np.isfinite(score)) or (worse >= controls.patience and score > 2.0 * best_score):
            stop = f"diverged at {k}"
            break
    return best_x, History(cost_hist, score_hist, stop, best_k, k)


def cascade_loop(problem: Problem, prior, x0, controls: Controls, *, extra_terms=(),
                 cancel=None, progress=None):
    x = np.array(x0, dtype=np.float64)
    best_score, best_x, best_k, worse = math.inf, x.copy(), 0, 0
    score_hist, cost_hist = [], []
    stop, k = "cap", 0
    n = int(controls.n_iter)
    extra = list(extra_terms)
    for k in range(1, n + 1):
        _check(cancel)
        if extra:
            x = x - gradient(extra, x) / problem.L
        x = prior.infer(x)
        # The paper's variance error: the inference's own deviation on the known pixels,
        # read before the correction pins them (after it the stop would fire at once).
        var_err = errors(x, problem.y, problem.known)["var_err"]
        score = None if problem.has_heldout else problem.score(x)
        if controls.correct:
            x = x - _correct(x, problem.y, problem.known, problem.ring, pad=0)
        if score is None:
            score = problem.score(x)
        cost_hist.append(float(var_err))
        score_hist.append(float(score))
        if score < best_score:
            best_score, best_x, best_k, worse = score, x.copy(), k, 0
        else:
            worse += 1
        _tell(progress, f"cascade {k}/{n}", k / n)
        if var_err < controls.stop_var_err:
            stop = "converged"
            break
        if (not np.isfinite(score)) or (worse >= controls.patience and score > 2.0 * best_score):
            stop = f"diverged at {k}"
            break
    return best_x, History(cost_hist, score_hist, stop, best_k, k)
