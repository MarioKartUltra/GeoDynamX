# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Three devices chosen to stress the param schema. See tests/test_stub_devices.py.

These are deliberately not the real WTMM/Hölder/orientation devices -- those are Phase 3. They
exist to answer one question before phases 3 and 4 commit to it: can a device's knobs be fully
described by a declaration, with no GUI code?
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind


class StubWavelet:
    """Transform. Mixes three param kinds and needs a stable, param-sensitive cache key."""

    name = "stub_wavelet"
    params = (
        Param("n_octaves", ParamKind.INT, default=4, min=1, max=10, label="Octaves"),
        Param("wavelet", ParamKind.CHOICE, default="mexican",
              choices=("mexican", "morlet"), label="Wavelet"),
        Param("normalise", ParamKind.BOOL, default=True, label="Normalise"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        if progress is not None:
            progress("stub_wavelet", 1.0)
        return {"n_octaves": params["n_octaves"], "wavelet": params["wavelet"]}

    def cache_key(self, source_id: str, params: dict) -> str:
        parts = ":".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{self.name}:{source_id}:{parts}"


class StubHolder:
    """Filter. A PAIRED range: no single Param can know that h_min must not exceed h_max.

    The hard bounds (-2..2) are the legal Hölder range; the soft bounds (-0.5..1.5) are the
    default slider span, far narrower, so the two pairs are distinguishable in practice and not
    just in principle.
    """

    name = "stub_holder"
    params = (
        Param("h_min", ParamKind.FLOAT, default=-0.5, min=-2.0, max=2.0,
              soft_min=-0.5, soft_max=1.5, label="Hölder min"),
        Param("h_max", ParamKind.FLOAT, default=1.5, min=-2.0, max=2.0,
              soft_min=-0.5, soft_max=1.5, label="Hölder max"),
    )

    def check(self, params: dict) -> None:
        if params["h_min"] > params["h_max"]:
            raise ValueError(
                f"{self.name}: h_min ({params['h_min']}) must not exceed h_max ({params['h_max']})"
            )

    def apply(self, result: dict, params: dict) -> dict:
        keep = [h for h in result.get("holder", []) if params["h_min"] <= h <= params["h_max"]]
        return dict(result, holder=keep)


class StubWedge:
    """Filter. An orientation wedge: a periodic ANGLE plus the frame it is measured against."""

    name = "stub_wedge"
    params = (
        Param("centre", ParamKind.ANGLE, default=0.0, units="deg", wrap=180.0, label="Strike"),
        Param("half_width", ParamKind.FLOAT, default=15.0, min=0.0, max=90.0,
              units="deg", label="± "),
        Param("north", ParamKind.CHOICE, default="true",
              choices=("grid", "true", "magnetic"), label="North"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        centre, half = params["centre"], params["half_width"]
        keep = [o for o in result.get("orientations", [])
                if _wrapped_delta(o, centre) <= half]
        return dict(result, orientations=keep)


def _wrapped_delta(a: float, b: float, period: float = 180.0) -> float:
    """Smallest separation between two orientations mod ``period``.

    Orientation is axial, not directional: an NE-SW lineament is one orientation regardless of
    gradient polarity, so 175 deg and 10 deg are 15 deg apart, not 165.
    """
    d = abs(a - b) % period
    return min(d, period - d)
