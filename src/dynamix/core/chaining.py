# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Cross-scale chaining as a pluggable interface.

A chainer links the maxima of one scale to the next and hands back V-chains in the one schema the
rest of the app reads (``PythonWTMMBackend.chains2d``: a list of dicts with ``x``, ``y``, ``mod``,
``log2_mod`` and ``log2_scales``, index 0 the finest scale). :class:`XsmurfBench` is today's chaining,
the exact port of xsmurf's ``vchain``, and the benchmark every other chainer is scored against on
synthetic cusps of known Hölder exponent (``tests/test_chaining_benchmark.py``). The WTMM device's
``chainer`` param names one; its default is the benchmark, so the keys of every existing recipe are
what they were.
"""
from __future__ import annotations

from typing import Protocol

#: The chainer a recipe that names none runs: the benchmark.
DEFAULT = "xsmurf_bench"


class Chainer(Protocol):
    name: str

    def link(self, extrema: list, scales, params: dict, backend) -> list:
        """V-chains from the per-scale ``extrema`` (``extrema2d``'s schema) over ``scales``
        (finest first); ``params`` is the resolved WTMM parameter set, ``backend`` the
        :class:`~dynamix.core.wtmm_backend.PythonWTMMBackend` whose single-maxima and chaining
        passes a chainer may reuse."""


class XsmurfBench:
    """The benchmark: ``PythonWTMMBackend.chains2d``, the exact port of xsmurf's ``vchain``
    (single maxima, the Chebyshev box, the similitude band, the squared-distance cap, the
    nearest-claimant competition)."""

    name = "xsmurf_bench"

    def link(self, extrema: list, scales, params: dict, backend) -> list:
        return backend.chains2d(extrema, scales, similitude=params["similitude"],
                                box_ratio=params["box_ratio"], dist2_max=params["dist2_max"],
                                min_len=params["min_chain_len"], smooth=params["smooth"])


#: Every chainer by name.
CHAINERS: dict[str, Chainer] = {c.name: c for c in (XsmurfBench(),)}


def chainer(name: str) -> Chainer:
    """The chainer called ``name``; ``KeyError`` naming the known ones otherwise."""
    try:
        return CHAINERS[name]
    except KeyError:
        raise KeyError(f"no chainer named {name!r}; known: {sorted(CHAINERS)}") from None


def link(extrema: list, scales, params: dict, backend) -> list:
    """V-chains by the chainer ``params["chainer"]`` names (:data:`DEFAULT` when absent)."""
    return chainer(params.get("chainer", DEFAULT)).link(extrema, scales, params, backend)
