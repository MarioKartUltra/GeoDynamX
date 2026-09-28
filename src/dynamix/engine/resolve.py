# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Resolve a layer's chain to a renderable.

One pass: run the chain's transforms (cached, expensive), then apply its filters (uncached, cheap)
to whatever they produced. The chain's transforms-then-filters invariant is what makes this a
single pass rather than a graph walk.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable

import numpy as np

from dynamix.core.chain_product import materialize_selection
from dynamix.engine.cache import Cache, cache_key
from dynamix.model.device import (Output, apply_view, declared_outputs, get_device,
                                  is_transform, keyed_params, validate_params)


@dataclasses.dataclass
class Renderable:
    """What a resolved layer hands to a view.

    ``result`` is the filtered transform output -- for WTMM that is the extrema/chain dicts. The
    counters exist so a UI can honestly report what it did rather than implying everything was
    recomputed.

    The ``analysis_*`` fields record the chain's LAST transform -- the one whose lazy outputs
    :func:`resolve_output` computes: its device name, its validated params (view-only ones
    included), its cache key, and the input it consumed (the field for a first transform, the
    previous transform's result otherwise; ``None`` for an ROI result, whose input is a region
    window the runner reads). All ``None`` when the chain runs no transform.
    """

    layer_id: int
    result: dict
    field: Any = None
    cache_hits: int = 0
    cache_misses: int = 0
    transforms_run: tuple[str, ...] = ()
    filters_run: tuple[str, ...] = ()
    analysis_device: str | None = None
    analysis_params: dict | None = None
    analysis_key: str | None = None
    analysis_input: Any = None

    @property
    def from_cache(self) -> bool:
        """True when nothing expensive ran -- i.e. this redraw was a filter change."""
        return self.cache_misses == 0


def _mapping_fingerprint(layer) -> str:
    """A short, deterministic fingerprint of a point layer's column mapping (``layer.tags[
    "points.mapping"]``, the CSV column mapping ``dynamix.shell.point_import`` persists per
    layer) -- folded into :func:`resolve`'s own source identity below.

    **Mapping identity.** The mapping is what actually defines a PointSet's content
    (which header became lon/lat/depth/mag), but it rides per-LAYER tags while cache identity was
    source-scoped alone: two layers over the SAME source with DIFFERENT mappings previously shared
    a cache key, so the second was silently served the first's result. Folding a fingerprint of
    the mapping into ``sid`` makes that a genuine miss instead.

    Empty string -- a complete no-op on ``sid`` -- for a layer with no such tag (every raster
    layer, the overwhelming majority of resolves) or one that fails to parse (tolerated the same
    way ``main_window._display_style_of`` tolerates a garbled tag: never raise over view-adjacent
    metadata). The JSON is re-serialised sorted-key/no-whitespace first, so two mappings that are
    the same dict written in a different key order fingerprint identically -- the same "canonical
    JSON" discipline ``engine.cache.cache_key`` already applies to ``params``.
    """
    raw = layer.tags.get("points.mapping") if layer is not None else None
    if not raw:
        return ""
    try:
        mapping = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def source_identity(layer, source_id: str | None = None) -> str:
    """The resolve identity for ``layer``'s own source -- what every ``cache_key`` call for this
    layer's transforms must use as its ``source_id`` argument.

    Defaults to ``layer.source_id``; a caller with a better identity than the layer's own (a
    content hash of the raster, say, rather than a path that may have been relocated) may pass
    ``source_id`` to override that default -- but even then, a point layer's ``points.mapping``
    fingerprint (:func:`_mapping_fingerprint`) is still folded in, since the mapping is about the
    layer, not about whatever identity the caller supplied for the underlying source.

    **The ONE place this derivation lives.** :func:`resolve` and
    ``MainWindow._cache_keys_for`` (a duplicate, ``cache_key``-based probe that predicts
    ``resolve``'s own keys WITHOUT running it, used to decide whether a chain is already cached)
    both call this rather than re-deriving it inline -- otherwise the two could drift onto
    different identities for the identical layer, which is exactly the class of bug this guards against (a derivation duplicated in two places, one of them not updated) and exactly what
    ``_cache_keys_for``'s own docstring already warns a duplicated format string would risk.
    """
    sid = source_id if source_id is not None else layer.source_id
    fingerprint = _mapping_fingerprint(layer)
    if fingerprint:
        sid = f"{sid}|{fingerprint}"
    # ROI child datasets: a windowed-child layer analyzes a CROP of its source, so its window
    # must join the resolve identity -- two different windows on one source (or a window vs the
    # whole) must never share a cache line. Same tags-as-truth fold as the mapping fingerprint.
    window = layer.tags.get("roi.window") if layer is not None else None
    if window:
        sid = f"{sid}|win:{window}"
    # A band removed from a stack dataset: the source keeps its id but not its content, so
    # the band set joins the identity (same tags-as-truth fold) -- a result computed on the
    # full stack must never answer for the reduced one.
    bands = layer.tags.get("data.bands") if layer is not None else None
    if bands:
        sid = f"{sid}|bands:{bands}"
    return sid


def preview_resolve(layer, field):
    """A finest-scale-only preview of ``layer``'s chain (progressive compute, design §3): call the
    leading transform's ``preview(field, params)`` (only ``wtmm2d`` offers one), then apply the
    chain's cheap FILTER steps on top. Transforms AFTER the first are skipped -- ``chain_topology``
    et al. cannot be previewed cheaply, and the filters that need them (``min_vchains``, the
    ``chain_*`` filters) no-op on a preview (no topology, empty ``chains``). Returns a normal
    result dict the canvas can draw, or ``None`` when the chain has no previewable leading
    transform (nothing to show early -- the caller just waits for the full resolve).

    Uncached and side-effect-free: it never touches ``cache`` and produces a throwaway frame the
    full ``resolve`` replaces. Any exception propagates to the worker, which treats a failed
    preview as simply "no early frame".
    """
    steps = layer.chain.steps
    if not steps:
        return None
    # No preview on a display PICTURE or for an ROI
    # layer -- the preview would analyse the resampled picture (or the whole parent) and draw
    # it in the picture's coordinates; the region's own result lands quickly anyway.
    from dynamix.roi.picture import display_stride

    if layer.tags.get("roi.window") or display_stride(field) > 1:
        return None
    first = get_device(steps[0].device)
    preview = getattr(first, "preview", None)
    if preview is None:
        return None
    result = preview(field, validate_params(first, steps[0].params))
    for ref in steps[1:]:
        device = get_device(ref.device)
        if is_transform(device):
            continue                       # can't preview a downstream transform; its filters no-op
        result = device.apply(result, validate_params(device, ref.params))
    return result


def _roi_prefix(layer) -> list:
    """``[(device, params), ...]`` the ROI runner owns for ``layer``: its leading field-stage
    transforms (``field_stage = True``, e.g. ``noise``) plus the first analyzer -- or ``[]``
    when the layer carries no ``roi.window`` or its chain does not open with a transform. A
    device that reads its own pixels off the file (``reads_source``: ``wtmm2d_roi``) is
    already region-aware and runs as an ordinary step."""
    if layer is None or not layer.tags.get("roi.window"):
        return []
    run = []
    for ref in layer.chain.steps:
        device = get_device(ref.device)
        if not is_transform(device) or getattr(device, "reads_source", False):
            break
        run.append((device, validate_params(device, ref.params)))
        if not getattr(device, "field_stage", False):
            return run                      # the analyzer closes the region step
    return []                               # a field stage with no analyzer: nothing to crop


def _refuse_the_picture(layer, field) -> None:
    """No tool runs on a display PICTURE: its samples are a drawing of the
    file, never data -- analysing them would be analysis of resampled pixels. Tools run on a
    saved ROI (``tags["roi.window"]``), which reads native pixels; a device that reads its own
    pixels off ``provenance["source"]`` (``reads_source``, i.e. ``wtmm2d_roi``) is region-aware
    already and passes."""
    from dynamix.roi.picture import display_stride

    if display_stride(field) <= 1:
        return
    if layer.tags.get("roi.window"):
        if _roi_prefix(layer) or not any(is_transform(get_device(r.device))
                                         for r in layer.chain.steps):
            return                          # the region runner reads native pixels
        raise ValueError(
            "a field stage alone (noise) does not analyse -- add an analyzer after it; it "
            "then runs on the ROI and its margin")
    for ref in layer.chain.steps:
        device = get_device(ref.device)
        if is_transform(device):
            if getattr(device, "reads_source", False):
                return
            raise ValueError(
                f"{ref.device}: this dataset is too big to analyse whole — its display is a "
                "picture, not data. Save an ROI and run the tool on it (native pixels).")


def resolve(layer, field, cache: Cache, *, source_id: str | None = None,
            progress: Callable[[str, float], None] | None = None,
            cancel: Callable[[], bool] | None = None) -> Renderable:
    """Run ``layer``'s chain over ``field`` and return the renderable.

    ``source_id`` defaults to the layer's, and is what ties a cached transform to its input; pass
    it explicitly only when the caller has a better identity than the layer does (a content hash of
    the raster, say, rather than a path that may have been relocated). See :func:`source_identity`
    for how this is actually derived, including the point-mapping fold-in.

    ``cancel``: a zero-arg predicate threaded into any
    transform that opts in with ``wants_cancel = True`` (``wtmm2d``/``wtmm2d_roi``), where it
    reaches ``run_wtmm2d``'s per-stage check. A set flag raises ``ComputeCancelled`` out of the
    compute (and thus out of ``resolve``) without caching a partial result. Only long transforms
    opt in, so the common path adds nothing.
    """
    sid = source_identity(layer, source_id)
    _refuse_the_picture(layer, field)
    hits0, misses0 = cache.hits, cache.misses
    ran_t: list[str] = []
    ran_f: list[str] = []

    result: dict = {}
    upstream: str | None = None      # key of the preceding transform; threads the lineage
    analysis: tuple = (None, None, None, None)   # (device, params, key, input) of the last one
    analysis_raw = None                          # the last transform's (viewed) result
    start = 0
    roi_steps = _roi_prefix(layer)
    if roi_steps:
        # The leading field stage + the analyzer run ON THE REGION as one
        # step -- ROI + the tools' declared margins read once off the dataset, cropped back to
        # the ROI, lines re-labelled inside it. One cache line per (window, recipe): the
        # window already rides in ``sid`` (source_identity's ``|win:`` fold); the field-stage
        # keys thread in as the analyzer key's upstream.
        from dynamix.roi.runner import run_on_region

        rect = tuple(int(v) for v in layer.tags["roi.window"].split(","))
        for device, params in roi_steps:
            upstream = cache_key(device.name, sid, keyed_params(device, params),
                                 upstream=upstream)

        def _region(run=roi_steps, r=rect):
            return run_on_region(run, field, r, progress=progress, cancel=cancel)

        result = cache.get_or_compute(upstream, _region)
        analyzer, analyzer_params = roi_steps[-1]
        result = apply_view(analyzer, result, analyzer_params)
        analysis_raw = result
        analysis = (analyzer.name, analyzer_params, upstream, None)
        ran_t.extend(d.name for d, _p in roi_steps)
        start = len(roi_steps)
    for ref in layer.chain.steps[start:]:
        device = get_device(ref.device)
        params = validate_params(device, ref.params)

        if is_transform(device):
            key = cache_key(device.name, sid, keyed_params(device, params), upstream=upstream)

            # A transform after the first consumes the previous result, not the raw field.
            src = field if upstream is None else result

            def _compute(d=device, p=params, s=src):
                if getattr(d, "wants_cancel", False):
                    return d.compute(s, p, progress=progress, cancel=cancel)
                return d.compute(s, p, progress=progress)

            result = apply_view(device, cache.get_or_compute(key, _compute), params)
            analysis_raw = result
            analysis = (device.name, params, key, src)
            upstream = key
            ran_t.append(device.name)
        else:
            # Selection model: aware filters narrow index selections
            # over the transform's chain product instead of copying dicts. A device outside
            # that model must see honest filtered dicts, exactly as the sequential dict path
            # would hand them over -- so the live selection is materialized ONCE before it
            # runs (a no-op when nothing is narrowed or no product is stamped).
            if not getattr(device, "selection_aware", False):
                result = materialize_selection(result)
            result = device.apply(result, params)
            ran_f.append(device.name)

    # The one honest gather per resolve: whatever selection is still live becomes the terminal
    # result's filtered extrema/chains, while the product + selection stay stamped for the
    # views' own index path.
    result = materialize_selection(result)
    # An analysis whose lazy outputs read their own selection per level draws, at the level on
    # show, what they read there (``shown_constraints``); the filters' result otherwise.
    shown = getattr(get_device(analysis[0]), "shown_constraints", None) if analysis[0] else None
    if shown is not None and analysis_raw is not None:
        device, params = get_device(analysis[0]), analysis[1]
        result = shown(result, analysis_raw, params,
                       lambda l: level_keeps(layer, analysis_raw, device, params, l))

    return Renderable(
        layer_id=layer.layer_id,
        result=result,
        field=field,
        cache_hits=cache.hits - hits0,
        cache_misses=cache.misses - misses0,
        transforms_run=tuple(ran_t),
        filters_run=tuple(ran_f),
        analysis_device=analysis[0],
        analysis_params=analysis[1],
        analysis_key=analysis[2],
        analysis_input=analysis[3],
    )


def selection_steps(steps) -> list:
    """``[(key, ref), ...]``: the filter steps of ``steps`` a selecting output reads (every filter
    but ``scale_select``, whose level pick is the display's), keyed by device name, ``#2``, ``#3``
    ... for a device's repeats, the keys per-level filter settings are stored under."""
    seen: dict = {}
    out = []
    for ref in steps:
        if ref.device == "scale_select" or is_transform(get_device(ref.device)):
            continue
        seen[ref.device] = seen.get(ref.device, 0) + 1
        n = seen[ref.device]
        out.append((ref.device if n == 1 else f"{ref.device}#{n}", ref))
    return out


def selection_recipe(layer) -> list:
    """What a selecting output's key folds in: ``[key, keyed params]`` of each of ``layer``'s
    :func:`selection_steps`. The window predicts output keys through this same function."""
    out = []
    for key, ref in selection_steps(layer.chain.steps):
        device = get_device(ref.device)
        out.append([key, keyed_params(device, validate_params(device, ref.params))])
    return out


def level_keeps(layer, raw: dict, device, params: dict, level: int) -> np.ndarray:
    """Which extrema of level ``level`` (1-based) of the analysis result ``raw`` ``layer``'s
    filters keep: the :func:`selection_steps` applied to that level alone, as ``scale_select``
    would hand it over, with the analysis device's per-level settings for the level
    (``device.filter_overrides(params, level)``, ``{key: params}``) over each step's own. A bool
    per extremum, in the level's own order."""
    base = raw["extrema"][level - 1]
    result = {**raw, "extrema": [base], "_scale_idx": level - 1, "_ext_base": base}
    scales = raw.get("scales")
    if scales is not None and len(scales) >= level:
        result["_scale_px"] = float(np.asarray(scales)[level - 1])
    overrides_of = getattr(device, "filter_overrides", None)
    overrides = overrides_of(params, level) if overrides_of is not None else {}
    for key, ref in selection_steps(layer.chain.steps):
        filt = get_device(ref.device)
        names = {p.name for p in filt.params}
        extra = {k: v for k, v in overrides.get(key, {}).items() if k in names}
        p = validate_params(filt, {**ref.params, **extra})
        if not getattr(filt, "selection_aware", False):
            result = materialize_selection(result)
        result = filt.apply(result, p)
    result = materialize_selection(result)
    kept = (result.get("extrema") or [base])[0]
    nx = int(raw["_shape"][1])

    def flat(e):
        return np.asarray(e["y"], np.int64) * nx + np.asarray(e["x"], np.int64)

    return np.isin(flat(base), flat(kept))


def output_key(device_name: str, output: Output, params: dict, analysis_key: str,
               selection: list | None = None) -> str:
    """The cache key of lazy ``output`` of ``device_name``: its own name, the view-only ``params``
    it declares, and the analysis key as its upstream -- so any change to the analysis or
    anything before it re-keys the output, and a knob the output does not read never does. A
    selecting output (``Output.selects``) also folds in ``selection``, the layer's
    :func:`selection_recipe`."""
    own = {p: params[p] for p in output.params}
    if output.selects:
        own["_selection"] = selection or []
    return cache_key(f"{device_name}/{output.name}", "", own, upstream=analysis_key)


def resolve_output(layer, field, cache: Cache, name: str, *, source_id: str | None = None,
                   progress: Callable[[str, float], None] | None = None,
                   cancel: Callable[[], bool] | None = None) -> dict:
    """The lazy output ``name`` of ``layer``'s last transform: ``{"raster": float32 ndarray,
    "diag": dict, **extras}``. Resolves the layer first (cache hits once it has run), then
    computes the output under :func:`output_key` via ``cache.get_or_compute`` -- a cancelled run
    raises ``ComputeCancelled`` and caches nothing. Refuses (ValueError, the reason in the
    message) an ROI layer (``tags["roi.window"]``) and an undeclared or eager output.

    The device's ``compute_output(name, values, result, params, *, fetch, progress, cancel)``
    receives the analysed values, the CACHED analysis result (before its ``view``), the
    validated params, and ``fetch(other)``, which resolves another lazy output of the same layer
    through the cache -- so an output derived from another is keyed under that one's work.
    """
    if layer.tags.get("roi.window"):
        raise ValueError(f"{name}: lazy outputs are not computed on ROI results")
    r = resolve(layer, field, cache, source_id=source_id, progress=progress, cancel=cancel)
    outputs = declared_outputs(get_device(r.analysis_device)) if r.analysis_device else ()
    output = next((o for o in outputs if o.name == name and o.lazy), None)
    if output is None:
        raise ValueError(f"{r.analysis_device}: no lazy output named {name!r}")
    selection = selection_recipe(layer) if output.selects else None
    key = output_key(r.analysis_device, output, r.analysis_params, r.analysis_key, selection)
    raw = cache.get(r.analysis_key)
    values = np.asarray(r.analysis_input.values)

    def fetch(other: str) -> dict:
        return resolve_output(layer, field, cache, other, source_id=source_id,
                              progress=progress, cancel=cancel)

    def _compute():
        device = get_device(r.analysis_device)
        extra = {}
        if output.selects:
            # What the filters keep at every level, each with its own settings.
            extra["keeps"] = [level_keeps(layer, raw, device, r.analysis_params, l)
                              for l in range(1, len(raw.get("extrema") or ()) + 1)]
        return device.compute_output(name, values, raw, r.analysis_params, fetch=fetch,
                                     progress=progress, cancel=cancel, **extra)

    return cache.get_or_compute(key, _compute)
