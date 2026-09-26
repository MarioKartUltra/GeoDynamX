# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Filters over WTMM *chains*, wrapping the tested implementations in ``wtmm_ebsd.chain_filters``.

These are the counterpart to :mod:`dynamix.devices.filters`, and the distinction matters:

* ``filters.py`` acts on **extrema** -- the points found at one scale. Filtering there answers
  "which features are present at this scale".
* this module acts on **chains** -- the maxima lines threading *across* scales. Filtering here
  answers "which features behave a certain way as scale changes", which is where the Holder
  exponent lives, because Holder IS the slope of log2|W| against log2 a.

When ``wtmm_ebsd`` is installed, nothing here is reimplemented: it already has these, tested, with
two Holder estimators; this module declares their parameters so the GUI can generate controls, and
calls straight into it.

``wtmm_ebsd`` is a documented local install and is imported lazily, so the core still imports
without it. When it is absent these devices fall back to :mod:`dynamix.core.chain_stats`, a
numpy-only reimplementation of the same estimators (see that module's own docstring for exactly
which reference functions it mirrors, and where it deliberately differs); the ``wtmm_ebsd`` path is
preferred whenever the package is present. The degraded mode is real filtering: a user without
``wtmm_ebsd`` installed sees the SAME cutoff produce the SAME kind of effect (kept/dropped),
computed by a different but equivalent estimator. A silent pass-through would read as "the filter
is broken" rather than "an optional dependency is missing".
"""
from __future__ import annotations

import numpy as np

from dynamix.core import chain_stats
from dynamix.core.chain_product import narrow_selection, selection_of
from dynamix.model.param import Param, ParamKind


def _narrow_v(result: dict, sel: dict, keep: np.ndarray) -> dict:
    """One chain filter's ``keep_v`` narrowing: intersect, re-bind,
    and stamp this device's own honest ``_chains_dropped`` -- the count it removed from what
    reached it, matching the dict path's ``len(chains) - len(kept)`` bookkeeping."""
    out = dict(result)
    prev = sel["keep_v"]
    prev_n = int(prev.size) if prev is not None else int(np.asarray(keep).size)
    narrow_selection(out, sel, v=keep)
    out["_chains_dropped"] = prev_n - int(out["_selection"]["keep_v"].size)
    return out

#: Shared degraded-mode / no-data reading text (EQSelect's own three-tier
#: degradation): a device's own ``reading()`` says this instead of a kept-count sentence
#: when the metric its cutoff depends on cannot be computed for the CURRENT chains (none present,
#: or none carry the needed key -- e.g. chains built without ``log2_mod``).
_UNAVAILABLE = "unavailable — recompute the transform to enable"

#: Shared NaN-sentinel hint (module-private convention, documented on ``DeviceBox.sync_from_result``
#: in ``workflow_zone.py``, the one place that reads it): ``data_hints`` returns this 3-tuple for a
#: param whose metric is unavailable, rather than omitting the key -- a caller can tell "no data to
#: hint from, disable the control" apart from "this param is never hint-driven" (e.g. ``estimator``)
#: only if every metric-driven param is guaranteed to appear in the dict one way or the other.
_NO_HINT = (float("nan"), float("nan"), float("nan"))


def _chain_filters():
    """The backing module, or None when the optional package is not installed."""
    try:
        from dynamix._vendor.wtmm_ebsd import chain_filters
    except ImportError:
        return None
    return chain_filters


class ChainHolderFilter:
    """Keep chains whose Hölder exponent is at or above a cutoff.

    The Hölder exponent is the scaling of |W| with scale along a maxima line: low values mean a
    sharp, near-discontinuous feature, high values a smooth one. Thresholding it separates edges
    from gradients, which is the filter the spec's `[WTMM]→[Hölder > 0.3]` example names.
    """

    name = "chain_holder"
    selection_aware = True
    params = (
        # -3.0 (the HARD min) is unambiguously "off" against the histogram's own documented
        # (-3, 2) range and pass-all by construction. Any default below the data (the kam_64
        # fixture's real Hölder range is -1.28..+1.00) leaves a freshly dropped filter a true
        # no-op that reads as "broken", so data_hints (below) snaps this to the ACTUAL data floor
        # the moment a result lands, and the very next drag-step prunes.
        Param("cutoff", ParamKind.FLOAT, default=-3.0, min=-10.0, max=10.0,
              soft_min=-1.0, soft_max=1.5, label="Hölder ≥"),
        Param("estimator", ParamKind.CHOICE, default="ols", choices=("ols", "max"),
              label="Hölder estimator"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        chains = result.get("chains")
        if not chains:
            return result
        cutoff = float(params["cutoff"])
        estimator = str(params["estimator"])
        sel = selection_of(result)
        if sel is not None:
            # Selection path: the SAME estimator, as a static product column (the builder pins
            # its agreement with chain_stats); NaN -> dropped, kept iff h >= cutoff.
            vals = np.asarray(
                result["chain_product"]["v_holder_ols" if estimator == "ols"
                                        else "v_holder_max"])
            return _narrow_v(result, sel, np.isfinite(vals) & (vals >= cutoff))
        # Vectorized + memoized chain_stats always: cf.filter_by_holder is a per-chain Python
        # loop (~1 s over a 59k-chain DEM); chain_stats.stats_for is the SAME estimator with the
        # SAME NaN semantics (kept iff h >= cutoff), vectorized and memoized on the cached chains,
        # so it is ~30x faster with identical output (tests/test_chain_stats.py). The stamped
        # ``_holder_summary`` lets reading()/data_hints() read the metric range instead of each
        # recomputing the whole per-chain OLS again this same tweak (the other 2/3 of the 2.5 s).
        vals = chain_stats.stats_for(chains, estimator)
        mask = np.isfinite(vals) & (vals >= cutoff)
        kept = [c for c, keep in zip(chains, mask) if keep]
        out = dict(result)
        out["chains"] = kept
        out["_chains_dropped"] = len(chains) - len(kept)
        finite = vals[np.isfinite(vals)]
        out["_holder_summary"] = {
            "estimator": estimator, "kept": len(kept), "total": len(chains),
            "lo": float(finite.min()) if finite.size else None,
            "hi": float(finite.max()) if finite.size else None,
        }
        return out

    def reading(self, result: dict, params: dict) -> str:
        """"kept N/M · h ∈ [lo, hi]" (the pinned reading format, 2 dp) -- N/M reconstructed
        from THIS call's own ``chains``/``_chains_dropped`` (M = N + dropped), which is honest
        exactly when this device's own ``apply`` produced them (main_window's own established
        limitation for chained filters -- see workflow_zone.DeviceBox.sync_from_result). [lo, hi]
        is the range of the CURRENT estimator's Hölder value across the chains THIS call sees,
        always computed via chain_stats (not the wtmm_ebsd backend) so the reading matches
        data_hints -- both need to agree on what "the data range" means, real backend or not.
        """
        # Prefer the stamp apply() left: avoids a THIRD full per-chain OLS pass this
        # tweak. Its lo/hi are the metric range over apply's own INPUT chains (the honest "what
        # this cutoff sees"), and kept/total are apply's own counts.
        s = result.get("_holder_summary")
        if s is not None and s.get("estimator") == str(params.get("estimator", "ols")):
            if s["lo"] is None:
                return _UNAVAILABLE
            return f"kept {s['kept']}/{s['total']} · h ∈ [{s['lo']:.2f}, {s['hi']:.2f}]"
        chains = result.get("chains") or []
        vals = chain_stats.stats_for(chains, str(params.get("estimator", "ols"))) if chains \
            else np.array([])
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return _UNAVAILABLE
        dropped = int(result.get("_chains_dropped", 0))
        kept = len(chains)
        total = kept + dropped
        return f"kept {kept}/{total} · h ∈ [{float(finite.min()):.2f}, {float(finite.max()):.2f}]"

    def data_hints(self, result: dict, params: dict) -> dict:
        """``{"cutoff": (lo, hi, snap)}`` -- ``lo``/``hi`` are the observed Hölder range for the
        CURRENT estimator (matching ``reading``); ``snap`` is ``lo``, so snapping there is always
        pass-all ("identical behavior... but the very next drag-step prunes"). Display
        only -- this dict never reaches ``cache_key`` (this is a Filter; only a Transform ever
        computes one), see ``tests/test_chain_filter_devices.py``'s dedicated assertion.

        **Known limitation.** ``result`` is whatever
        ``resolve()`` handed back for the WHOLE chain -- its terminal, post-every-filter
        population -- not this box's own pre-filter input (``resolve()``'s architecture keeps no
        per-step intermediate results; see ``workflow_zone.DeviceBox.sync_from_result``'s own
        docstring for the same caveat on the reading side). For the TERMINAL chain filter in a
        chain with no OTHER chain filter after it, this is exactly right: its own ``apply`` just
        produced ``result["chains"]``. For a NON-terminal chain filter (one or more further
        ``chain_*`` filters run after it), ``[lo, hi]`` and the ``N/M`` in ``reading`` describe the
        population AFTER those later filters ran too -- narrower than, and not necessarily
        centered the same way as, what THIS filter's own cutoff actually saw. ``sync_from_result``
        deliberately snaps only in the pass-all direction (``current < lo``, never ``current >
        hi``) so this can never silently yank a user's own deliberately-high cutoff down -- but the
        DISPLAYED range/count for a non-terminal box can still misstate that box's own effect.
        Fixing this for real needs ``resolve()`` to expose per-step results.
        """
        s = result.get("_holder_summary")
        if s is not None and s.get("estimator") == str(params.get("estimator", "ols")):
            if s["lo"] is None:
                return {"cutoff": _NO_HINT}
            return {"cutoff": (s["lo"], s["hi"], s["lo"])}
        chains = result.get("chains") or []
        vals = chain_stats.stats_for(chains, str(params.get("estimator", "ols"))) if chains \
            else np.array([])
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {"cutoff": _NO_HINT}
        lo, hi = float(finite.min()), float(finite.max())
        return {"cutoff": (lo, hi, lo)}


class ChainModulusFilter:
    """Keep chains whose maximum log2 modulus is at or above a threshold.

    Chain-level noise rejection: a chain that never gets strong at any scale is usually not a
    feature. Complementary to the per-scale modulus filter, which cannot see across scales.
    """

    name = "chain_modulus"
    selection_aware = True
    params = (
        Param("threshold", ParamKind.FLOAT, default=-99.0, min=-99.0, max=32.0,
              soft_min=-8.0, soft_max=8.0, label="max log₂|W| ≥"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        chains = result.get("chains")
        if not chains:
            return result
        threshold = float(params["threshold"])
        sel = selection_of(result)
        if sel is not None:
            vals = np.asarray(result["chain_product"]["v_max_log2_mod"])
            return _narrow_v(result, sel, np.isfinite(vals) & (vals >= threshold))
        # Vectorized + memoized chain_stats always (same reasoning as chain_holder):
        # matches filter_by_modulus (NaN -> dropped, kept iff max log2|W| >= threshold) but
        # without the per-chain Python loop.
        vals = chain_stats.max_log2_modulus_for(chains)
        mask = np.isfinite(vals) & (vals >= threshold)
        kept = [c for c, keep in zip(chains, mask) if keep]
        out = dict(result)
        out["chains"] = kept
        out["_chains_dropped"] = len(chains) - len(kept)
        return out

    def reading(self, result: dict, params: dict) -> str:
        """"kept N/M · max log₂|W| ∈ [lo, hi]" -- same shape and honesty caveats as
        ``ChainHolderFilter.reading``."""
        chains = result.get("chains") or []
        vals = chain_stats.max_log2_modulus_for(chains) if chains else np.array([])
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return _UNAVAILABLE
        dropped = int(result.get("_chains_dropped", 0))
        kept = len(chains)
        total = kept + dropped
        return (f"kept {kept}/{total} · max log₂|W| ∈ "
               f"[{float(finite.min()):.2f}, {float(finite.max()):.2f}]")

    def data_hints(self, result: dict, params: dict) -> dict:
        """``{"threshold": (lo, hi, snap)}`` -- see ``ChainHolderFilter.data_hints``."""
        chains = result.get("chains") or []
        vals = chain_stats.max_log2_modulus_for(chains) if chains else np.array([])
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {"threshold": _NO_HINT}
        lo, hi = float(finite.min()), float(finite.max())
        return {"threshold": (lo, hi, lo)}


class ChainLengthFilter:
    """Keep chains spanning at least N scales.

    A chain surviving many octaves is a feature with genuine scale range; a two-scale chain is
    usually a coincidence of the extrema linking.
    """

    name = "chain_length"
    selection_aware = True
    params = (
        Param("min_len", ParamKind.INT, default=2, min=1, max=1024,
              soft_min=2, soft_max=16, label="Min scales spanned"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        chains = result.get("chains")
        if not chains:
            return result
        min_len = int(params["min_len"])
        sel = selection_of(result)
        if sel is not None:
            lengths = np.asarray(result["chain_product"]["v_persist"])
            keep = np.ones(lengths.size, dtype=bool) if min_len <= 0 \
                else lengths >= min_len
            return _narrow_v(result, sel, keep)
        # chain_stats always: matches filter_by_length (min_len<=0 keeps all, else
        # kept iff length >= min_len). length_for is a cheap structural read, no per-chain fit.
        if min_len <= 0:
            kept = list(chains)
        else:
            lengths = chain_stats.length_for(chains)
            kept = [c for c, n in zip(chains, lengths) if n >= min_len]
        out = dict(result)
        out["chains"] = kept
        out["_chains_dropped"] = len(chains) - len(kept)
        return out

    def reading(self, result: dict, params: dict) -> str:
        """"kept N/M · scales ∈ [lo, hi]" -- chain length is structural (derived from the chain's
        own arrays, never from ``log2_mod``'s validity), so this is "unavailable" only when there
        are no chains at all, never merely because a Hölder-style metric is missing."""
        chains = result.get("chains") or []
        if not chains:
            return _UNAVAILABLE
        lengths = chain_stats.length_for(chains)
        dropped = int(result.get("_chains_dropped", 0))
        kept = len(chains)
        total = kept + dropped
        return f"kept {kept}/{total} · scales ∈ [{int(lengths.min())}, {int(lengths.max())}]"

    def data_hints(self, result: dict, params: dict) -> dict:
        """``{"min_len": (lo, hi, snap)}`` -- ``snap`` is ``lo`` (the shortest observed chain),
        which keeps ``filter_by_length``'s own pass-all-when-``min_len<=0`` spirit: snapping to the
        data's own floor can only ever be a no-op or MORE permissive than whatever was there. Same
        non-terminal-box known limitation as ``ChainHolderFilter.data_hints`` -- see its docstring."""
        chains = result.get("chains") or []
        if not chains:
            return {"min_len": _NO_HINT}
        lengths = chain_stats.length_for(chains)
        lo, hi = float(lengths.min()), float(lengths.max())
        return {"min_len": (lo, hi, lo)}
