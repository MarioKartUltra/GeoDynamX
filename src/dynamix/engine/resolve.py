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

from dynamix.core import fft_policy
from dynamix.core.chain_product import materialize_selection
from dynamix.engine.cache import Cache, cache_key
from dynamix.engine import execution
from dynamix.model import capabilities
from dynamix.model.contracts import FIELD, input_contract, produces_of, stamp_fill, step_states
from dynamix.model.device import (Output, apply_view, declared_outputs, get_device,
                                  is_transform, keyed_params, validate_params)
from dynamix.model.plugins import origin_of


def step_key(device, sid: str, params: dict, upstream: str | None) -> str:
    """The cache key of one transform step: the device, the source identity, its keyed params,
    the key of the transform before it and, for a plugin, its origin. The one derivation:
    :func:`resolve` and the shell's own walk of a chain (``MainWindow._cache_keys_for``) both call
    it, so the key the shell looks up is the key the engine stored."""
    return cache_key(device.name, sid, keyed_params(device, params), upstream=upstream,
                     origin=origin_of(device.name))


@dataclasses.dataclass
class Renderable:
    """What a resolved layer hands to a view.

    ``result`` is the filtered transform output -- for WTMM that is the extrema/chain dicts. The
    counters exist so a UI can honestly report what it did rather than implying everything was
    recomputed.

    The ``analysis_*`` fields record the chain's last APPLYING transform -- the one whose lazy
    outputs :func:`resolve_output` computes: its device name, its validated params (view-only
    ones included), its cache key, and the input it consumed (the field for a first transform,
    the previous transform's result otherwise; ``None`` for an ROI result, whose input is a
    region window the runner reads). All ``None`` when the chain runs no transform.

    ``step_states`` holds one :class:`~dynamix.model.contracts.StepState` per chain step, in
    chain order: whether the step applies on what reaches it, and the reason when it does not.
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
    step_states: tuple = ()

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


def preview_resolve(layer, field, level: int | None = None):
    """A finest-scale-only preview of ``layer``'s chain (progressive compute, design §3): call the
    leading transform's ``preview(field, params)`` (only ``wtmm2d`` offers one), then apply the
    chain's cheap FILTER steps on top. Transforms AFTER the first are skipped -- ``chain_topology``
    et al. cannot be previewed cheaply, and the filters that need them (``min_vchains``, the
    ``chain_*`` filters) no-op on a preview (no topology, empty ``chains``). Returns a normal
    result dict the canvas can draw, or ``None`` when the chain has no previewable leading
    transform (nothing to show early -- the caller just waits for the full resolve).

    Steps that do not apply (:func:`~dynamix.model.contracts.step_states`) are skipped as
    :func:`resolve` skips them, so the leading transform is the first applying one; ``level``
    reduces the preview before its filters as :func:`resolve` does (a one-level preview passes
    whole), and with ``level`` given a ``scale_select`` step does not apply.

    Uncached and side-effect-free: it never touches ``cache`` and produces a throwaway frame the
    full ``resolve`` replaces. Any exception propagates to the worker, which treats a failed
    preview as simply "no early frame".
    """
    if not layer.chain.steps:
        return None
    # No preview on a display PICTURE or for an ROI
    # layer -- the preview would analyse the resampled picture (or the whole parent) and draw
    # it in the picture's coordinates; the region's own result lands quickly anyway.
    from dynamix.roi.picture import display_stride

    if layer.tags.get("roi.window") or display_stride(field) > 1:
        return None
    states = step_states(layer.chain.steps, input_contract(field),
                         level_view=level is not None)
    steps = [ref for ref, state in zip(layer.chain.steps, states) if state.applies]
    if not steps:
        return None
    first = get_device(steps[0].device)
    preview = getattr(first, "preview", None)
    if preview is None:
        return None
    first_params = validate_params(first, steps[0].params)
    with fft_policy.scope_for(first, first_params):
        result = preview(field, first_params)
    if level is not None:
        result = _at_level(result, level)
    for ref in steps[1:]:
        device = get_device(ref.device)
        if is_transform(device):
            continue                       # can't preview a downstream transform; its filters no-op
        result = device.apply(result, validate_params(device, ref.params))
    return result


def _roi_prefix(layer, steps) -> list:
    """``[(device, params), ...]`` the ROI runner owns for ``layer``: the leading field-stage
    transforms (``field_stage = True``, e.g. ``noise``) of ``steps`` plus the first analyzer --
    or ``[]`` when the layer carries no ``roi.window`` or ``steps`` do not open with a
    transform. ``steps`` are the chain's applying steps, so a step that does not apply never
    joins the region job. A device that reads its own pixels off the file (``reads_source``:
    ``wtmm2d_roi``) is already region-aware and runs as an ordinary step."""
    if layer is None or not layer.tags.get("roi.window"):
        return []
    run = []
    for ref in steps:
        device = get_device(ref.device)
        if not is_transform(device) or capabilities.has(device, "SourceReader"):
            break
        run.append((device, validate_params(device, ref.params)))
        if not capabilities.has(device, "FieldStage"):
            return run                      # the analyzer closes the region step
    return []                               # a field stage with no analyzer: nothing to crop


def _refuse_the_picture(layer, field, steps) -> None:
    """No tool runs on a display PICTURE: its samples are a drawing of the
    file, never data -- analysing them would be analysis of resampled pixels. Tools run on a
    saved ROI (``tags["roi.window"]``), which reads native pixels; a device that reads its own
    pixels off ``provenance["source"]`` (``reads_source``, i.e. ``wtmm2d_roi``) is region-aware
    already and passes. ``steps`` are the chain's applying steps: a step that does not apply
    never runs, so it is never refused."""
    from dynamix.roi.picture import display_stride

    if display_stride(field) <= 1:
        return
    if layer.tags.get("roi.window"):
        if _roi_prefix(layer, steps) or not any(is_transform(get_device(r.device))
                                                for r in steps):
            return                          # the region runner reads native pixels
        raise ValueError(
            "a field stage alone (noise) does not analyse -- add an analyzer after it; it "
            "then runs on the ROI and its margin")
    for ref in steps:
        device = get_device(ref.device)
        if is_transform(device):
            if capabilities.has(device, "SourceReader"):
                return
            raise ValueError(
                f"{ref.device}: this dataset is too big to analyse whole — its display is a "
                "picture, not data. Save an ROI and run the tool on it (native pixels).")


def _at_level(result, level: int):
    """``result`` reduced to level ``level`` (0-based, a ``scale_select`` index) by
    ``scale_select``'s own apply, so it carries the stamps that step makes (``_scale_idx``,
    ``_ext_base``, ``_ext_base_runs``, ``_ext_base_closed``, ``_scale_px``) and keeps its
    chains. A result of one level or none, and a field stage's field, pass whole."""
    if not isinstance(result, dict) or len(result.get("extrema") or ()) <= 1:
        return result
    # The engine imports no device module at import time; this one is read on demand.
    from dynamix.devices.filters import ScaleSelect

    return ScaleSelect().apply(result, {"scale_idx": int(level)})


def shown_level(layer) -> int | None:
    """The level (0-based, a ``scale_select`` index) the main view shows on ``layer``: its
    ``ui.scale_idx`` tag, or ``None`` when the tag is absent or not an integer."""
    text = layer.tags.get("ui.scale_idx")
    if text is None:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def seed_shown_level(layer) -> bool:
    """Give ``layer`` the shown level its chain was saved with: when it has no ``ui.scale_idx``
    tag and its chain holds a ``scale_select`` step, that step's ``scale_idx`` becomes the tag.
    The chain is not edited; under a shown level the step does not apply
    (:func:`~dynamix.model.contracts.step_states` with ``level_view``). True when the tag was
    written."""
    if layer is None or "ui.scale_idx" in layer.tags:
        return False
    ref = next((r for r in layer.chain.steps if r.device == "scale_select"), None)
    if ref is None:
        return False
    try:
        idx = int(ref.params.get("scale_idx", 0))
    except (TypeError, ValueError):
        return False
    layer.tags["ui.scale_idx"] = str(max(idx, 0))
    return True


def resolve(layer, field, cache: Cache, *, source_id: str | None = None,
            progress: Callable[[str, float], None] | None = None,
            cancel: Callable[[], bool] | None = None,
            level: int | None = None) -> Renderable:
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

    A step that does not apply on what reaches it (:func:`~dynamix.model.contracts.step_states`,
    from ``field``'s contract) is skipped: it is never computed or applied and adds no cache key,
    and a skipped transform neither advances ``upstream`` nor becomes the analysis. Running such a
    step is not equivalent (``min_vchains`` after a chainless analysis deletes every labelled
    H-line), so a skipped step passes everything by construction.

    ``level``: the scale level on show (0-based, a ``scale_select`` index), or ``None``. When
    given, the result of the last applying transform is reduced to that level by
    ``scale_select``'s own apply before the first filter runs (at the end when there is none), so
    the filters see one level carrying the stamps a ``scale_select`` step makes; the chains stay
    whole. With ``level`` given the steps are judged with ``level_view=True``, so a
    ``scale_select`` step the chain still holds does not apply and the stack is reduced once.

    The analysis it lands is what ``layer`` shows (:meth:`Cache.mark_shown`), kept with its
    upstream entries through the cache's bounds; past ``fft_policy.MLX_CACHE_LIMIT`` mlx's
    buffer cache goes back to the system.
    """
    sid = source_identity(layer, source_id)
    layer_input = input_contract(field)
    states = step_states(layer.chain.steps, layer_input, level_view=level is not None)
    steps = [ref for ref, state in zip(layer.chain.steps, states) if state.applies]
    _refuse_the_picture(layer, field, steps)
    hits0, misses0 = cache.hits, cache.misses
    ran_t: list[str] = []
    ran_f: list[str] = []

    result: dict = {}
    upstream: str | None = None      # key of the preceding transform; threads the lineage
    analysis: tuple = (None, None, None, None)   # (device, params, key, input) of the last one
    analysis_raw = None                          # the last transform's (viewed) result
    start = 0
    roi_steps = _roi_prefix(layer, steps)
    if roi_steps:
        # The leading field stage + the analyzer run ON THE REGION as one
        # step -- ROI + the tools' declared margins read once off the dataset, cropped back to
        # the ROI, lines re-labelled inside it. One cache line per (window, recipe): the
        # window already rides in ``sid`` (source_identity's ``|win:`` fold); the field-stage
        # keys thread in as the analyzer key's upstream.
        from dynamix.roi.runner import run_on_region

        rect = tuple(int(v) for v in layer.tags["roi.window"].split(","))
        for device, params in roi_steps:
            upstream = step_key(device, sid, params, upstream)

        def _region(run=roi_steps, r=rect):
            value = run_on_region(run, field, r, progress=progress, cancel=cancel, stage_cache=cache.stages)
            return execution.attach(value, _landed={"analysis_key": upstream, "filters": [], "level": None})

        result = cache.get_or_compute(upstream, _region)
        analyzer, analyzer_params = roi_steps[-1]
        result = apply_view(analyzer, result, analyzer_params)
        analysis_raw = result
        analysis = (analyzer.name, analyzer_params, upstream, None)
        ran_t.extend(d.name for d, _p in roi_steps)
        start = len(roi_steps)
    leveled = level is None          # the shown level is taken once, before the first filter
    for ref in steps[start:]:
        device = get_device(ref.device)
        params = validate_params(device, ref.params)

        if is_transform(device):
            key = step_key(device, sid, params, upstream)

            # A transform after the first consumes the previous result, not the raw field.
            src = field if upstream is None else result

            def _compute(d=device, p=params, s=src):
                if (getattr(s, "provenance", {}).get("grid.requires_fork") and
                        FIELD not in produces_of(d, p, frozenset({FIELD}))):
                    raise ValueError("The interpolated grid has new pixel spacing. Fork derivative dataset "
                                     "from the preparation output before analysing it on its own grid.")
                kw = {}
                if capabilities.has(d, "Cancellable"):
                    kw["cancel"] = cancel
                if getattr(d, "uses_stage_cache", False):
                    kw["stage_cache"] = cache.stages
                value = execution.run(d, p,
                    lambda: stamp_fill(d.compute(s, p, progress=progress, **kw), s),
                    s if upstream is not None else None)
                return execution.attach(value, _landed={"analysis_key": key, "filters": [], "level": None})

            result = apply_view(device, cache.get_or_compute(key, _compute, upstream=upstream),
                                params)
            if (roi_steps and isinstance(result, dict) and not result.get("_roi")
                    and hasattr(src, "values")):
                # A transform after a region step that made a field (Periodic extension on an
                # ROI) works in that field's grid, which starts at the ROI: drawn there.
                rect = tuple(int(v) for v in layer.tags["roi.window"].split(","))
                result = {**result, "_roi": {"source": None, "roi": rect, "boundary": "auto"}}
            analysis_raw = result
            analysis = (device.name, params, key, src)
            upstream = key
            ran_t.append(device.name)
        else:
            if not leveled:
                result, leveled = _at_level(result, level), True
            # Selection model: aware filters narrow index selections
            # over the transform's chain product instead of copying dicts. A device outside
            # that model must see honest filtered dicts, exactly as the sequential dict path
            # would hand them over -- so the live selection is materialized ONCE before it
            # runs (a no-op when nothing is narrowed or no product is stamped).
            if not capabilities.has(device, "SelectionAware"):
                result = materialize_selection(result)
            result = device.apply(result, params)
            ran_f.append(device.name)

    if not leveled:
        result = _at_level(result, level)
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
                       lambda l: level_keeps(layer, analysis_raw, device, params, l,
                                             layer_input))

    # The transform's own chain count rides on a result its filters or level view already
    # copied, so a result whose chains a filter narrowed can rebuild its partition tables over
    # the survivors and say "N of M"; an untouched result keeps its identity (no stamp needed).
    if (isinstance(result, dict) and isinstance(analysis_raw, dict)
            and result is not analysis_raw and analysis_raw.get("chains") is not None):
        result = {**result, "_chains_total": len(analysis_raw["chains"])}
    if analysis[2] is not None:
        result = cache.landed_result(analysis[2], result, execution.recipe_identity(
            layer, layer_input, level=level, source=source_id))
    cache.mark_shown(layer.layer_id, analysis[2])
    fft_policy.trim_mlx_cache()
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
        step_states=tuple(states),
    )


def cached_input(layer, field, cache: Cache, steps) -> Any:
    """The field reaching the step after ``steps`` (the layer's enabled steps before it), from
    ``cache`` alone: ``field`` where no transform comes first, the last transform's cached field
    where every transform there has run, else None. Nothing is computed: the edge footprint
    reads it on the GUI thread."""
    sid = source_identity(layer)
    upstream, reached = None, field
    for ref in steps:
        device = get_device(ref.device)
        if not is_transform(device):
            continue
        upstream = step_key(device, sid, validate_params(device, ref.params), upstream)
        if upstream not in cache:
            return None
        reached = cache.get(upstream)
    return reached if hasattr(reached, "values") else None


def selection_steps(steps, layer_input: str | None = None) -> list:
    """``[(key, ref), ...]``: the filter steps of ``steps`` a selecting output reads (every filter
    but ``scale_select``, whose level pick is the display's), keyed by device name, ``#2``, ``#3``
    ... for a device's repeats, the keys per-level filter settings are stored under. Given the
    chain's ``layer_input`` contract, a filter that does not apply
    (:func:`~dynamix.model.contracts.step_states`) is left out; it still counts toward its
    device's repeat numbering, so the keys stored for the others stay put."""
    states = step_states(steps, layer_input) if layer_input is not None else None
    seen: dict = {}
    out = []
    for i, ref in enumerate(steps):
        if ref.device == "scale_select" or is_transform(get_device(ref.device)):
            continue
        seen[ref.device] = seen.get(ref.device, 0) + 1
        n = seen[ref.device]
        if states is not None and not states[i].applies:
            continue
        out.append((ref.device if n == 1 else f"{ref.device}#{n}", ref))
    return out


def selection_recipe(layer, layer_input: str | None = None) -> list:
    """What a selecting output's key folds in: ``[key, keyed params]`` of each of ``layer``'s
    :func:`selection_steps` (given ``layer_input``, the applying ones). The window predicts
    output keys through this same function."""
    out = []
    for key, ref in selection_steps(layer.chain.steps, layer_input):
        device = get_device(ref.device)
        out.append([key, keyed_params(device, validate_params(device, ref.params))])
    return out


def level_keeps(layer, raw: dict, device, params: dict, level: int,
                layer_input: str | None = None) -> np.ndarray:
    """Which extrema of level ``level`` (1-based) of the analysis result ``raw`` ``layer``'s
    filters keep: the :func:`selection_steps` applied to that level alone, as ``scale_select``
    would hand it over, with the analysis device's per-level settings for the level
    (``device.filter_overrides(params, level)``, ``{key: params}``) over each step's own. A bool
    per extremum, in the level's own order. Given the chain's ``layer_input`` contract, the
    filters that do not apply are left out (:func:`selection_steps`)."""
    base = raw["extrema"][level - 1]
    result = {**raw, "extrema": [base], "_scale_idx": level - 1, "_ext_base": base}
    scales = raw.get("scales")
    if scales is not None and len(scales) >= level:
        result["_scale_px"] = float(np.asarray(scales)[level - 1])
    overrides_of = getattr(device, "filter_overrides", None)
    overrides = overrides_of(params, level) if overrides_of is not None else {}
    for key, ref in selection_steps(layer.chain.steps, layer_input):
        filt = get_device(ref.device)
        names = {p.name for p in filt.params}
        extra = {k: v for k, v in overrides.get(key, {}).items() if k in names}
        p = validate_params(filt, {**ref.params, **extra})
        if not capabilities.has(filt, "SelectionAware"):
            result = materialize_selection(result)
        result = filt.apply(result, p)
    result = materialize_selection(result)
    kept = (result.get("extrema") or [base])[0]
    nx = int(raw["_shape"][1])

    def flat(e):
        return np.asarray(e["y"], np.int64) * nx + np.asarray(e["x"], np.int64)

    return np.isin(flat(base), flat(kept))


def output_key(device_name: str, output: Output, params: dict, analysis_key: str,
               selection: list | None = None, level: int | None = None) -> str:
    """The cache key of lazy ``output`` of ``device_name``: its own name, the view-only ``params``
    it declares, and the analysis key as its upstream -- so any change to the analysis or
    anything before it re-keys the output, and a knob the output does not read never does. A
    selecting output (``Output.selects``) also folds in ``selection``, the layer's
    :func:`selection_recipe`; a per-level output (``Output.per_level``) folds in ``level``
    (``None`` is level 0, the finest), so each level is its own entry. Any other output's key
    is the same whatever ``level`` says."""
    own = {p: params[p] for p in output.params}
    if output.selects:
        own["_selection"] = selection or []
    if output.per_level:
        own["_level"] = 0 if level is None else int(level)
    return cache_key(f"{device_name}/{output.name}", "", own, upstream=analysis_key)


def _resolve_region_output(layer, field, cache: Cache, name: str, *, source_id, progress,
                           cancel, level) -> dict:
    """A lazy output of an ROI layer, computed where the region runner computes the analysis:
    its own processing window (the same read, the same summed margin -- each tool's reach at its
    largest scale -- the same reflected edges and field stages), by the tool's ``compute`` there
    and its ``compute_output`` on that, then cut to the region and placed by the region's own
    axes (``roi_axes``). For a tool without ``compute_roi`` that is exactly the runner's path, so
    the output agrees with the analysis on screen; the window and its outputs are cached under
    keys of their own, and the analysis is untouched. Outputs that read the filtered maxima see
    the window's lines whole, as the analysis does before its cut. Refused, with the reason: a
    tool with its own region path, and an output on its own coarser grid."""
    from dynamix.roi.runner import read_processing_window, roi_margin_of

    r = resolve(layer, field, cache, source_id=source_id, progress=progress, cancel=cancel)
    device = get_device(r.analysis_device) if r.analysis_device else None
    outputs = declared_outputs(device) if device is not None else ()
    output = next((o for o in outputs if o.name == name and o.lazy), None)
    if output is None:
        raise ValueError(f"{r.analysis_device}: no lazy output named {name!r}")
    if getattr(device, "compute_roi", None) is not None:
        raise ValueError(f"{name}: {device.name} has its own region path -- its outputs are "
                         "not computed on ROI results yet")
    if output.grid == "stride":
        raise ValueError(f"{name}: an output on its own coarser grid is not computed on ROI "
                         "results")
    layer_input = input_contract(field)
    states = step_states(layer.chain.steps, layer_input)
    roi_steps = _roi_prefix(layer, [ref for ref, st in zip(layer.chain.steps, states)
                                    if st.applies])
    if not roi_steps or roi_steps[-1][0].name != device.name:
        raise ValueError(f"{name}: the region's analysis is not {device.name}")
    row, col, h, w = (int(v) for v in layer.tags["roi.window"].split(","))
    m = sum(roi_margin_of(d, p)[0] for d, p in roi_steps)
    analyzer, aparams = roi_steps[-1]

    def _window() -> dict:
        window, _info = read_processing_window(field, (row, col, h, w), m)
        for d, p in roi_steps[:-1]:
            kw = {"cancel": cancel} if capabilities.has(d, "Cancellable") else {}
            previous = window
            window = execution.run(d, p, lambda: d.compute(previous, p, progress=progress, **kw), previous)
        kw = {"cancel": cancel} if capabilities.has(analyzer, "Cancellable") else {}
        raw = execution.run(analyzer, aparams,
            lambda: analyzer.compute(window, aparams, progress=progress, **kw), window)
        return {"window": window, "raw": raw}

    window_key = f"{r.analysis_key}|window"

    def window_output(other: str) -> dict:
        out = next(o for o in outputs if o.name == other)
        sel = selection_recipe(layer, layer_input) if out.selects else None
        lvl = (0 if level is None else int(level)) if out.per_level else None
        key = output_key(device.name, out, r.analysis_params, r.analysis_key, sel,
                         level=lvl) + "|window"

        def _compute():
            win = cache.get_or_compute(window_key, _window)
            raw = win["raw"]
            extra = {}
            if out.selects:
                extra["keeps"] = [level_keeps(layer, raw, device, r.analysis_params, l,
                                              layer_input)
                                  for l in range(1, len(raw.get("extrema") or ()) + 1)]
            if out.per_level:
                extra["level"] = lvl
            return execution.run(device, r.analysis_params,
                lambda: device.compute_output(other, np.asarray(win["window"].values), raw,
                                               r.analysis_params, fetch=window_output,
                                               progress=progress, cancel=cancel, **extra))

        return cache.get_or_compute(key, _compute)

    selection = selection_recipe(layer, layer_input) if output.selects else None
    lvl = (0 if level is None else int(level)) if output.per_level else None
    key = output_key(device.name, output, r.analysis_params, r.analysis_key, selection,
                     level=lvl)

    def _cut() -> dict:
        window = cache.get_or_compute(window_key, _window)["window"]
        value = window_output(name)
        cut = dict(value)
        cut["raster"] = np.asarray(value["raster"])[m:m + h, m:m + w]
        cut["roi_axes"] = (np.asarray(window.x_axis)[m:m + w].copy(),
                           np.asarray(window.y_axis)[m:m + h].copy())
        cut["_output_identity"] = {"key": key, "name": name, "level": lvl,
            "params": {k: r.analysis_params[k] for k in output.params}}
        return cut

    return cache.get_or_compute(key, _cut)


def resolve_output(layer, field, cache: Cache, name: str, *, source_id: str | None = None,
                   progress: Callable[[str, float], None] | None = None,
                   cancel: Callable[[], bool] | None = None,
                   level: int | None = None) -> dict:
    """The lazy output ``name`` of ``layer``'s last transform: ``{"raster": float32 ndarray,
    "diag": dict, **extras}``. Resolves the layer first (cache hits once it has run), then
    computes the output under :func:`output_key` via ``cache.get_or_compute`` -- a cancelled run
    raises ``ComputeCancelled`` and caches nothing. Refuses (ValueError, the reason in the
    message) an ROI layer (``tags["roi.window"]``) and an undeclared or eager output.

    The device's ``compute_output(name, values, result, params, *, fetch, progress, cancel)``
    receives the analysed values, the CACHED analysis result (before its ``view``), the
    validated params, and ``fetch(other)``, which resolves another lazy output of the same layer
    through the cache -- so an output derived from another is keyed under that one's work.

    ``level`` (0-based, finest first, the ``ui.scale_idx`` numbering; ``None`` is 0) reaches a
    per-level output (``Output.per_level``) as ``compute_output(..., level=)`` and its key;
    ``fetch`` passes it on, and every other output ignores it. The layer itself resolves without
    a level, so ``compute_output`` reads the whole stack's result.
    """
    if layer.tags.get("roi.window"):
        return _resolve_region_output(layer, field, cache, name, source_id=source_id,
                                      progress=progress, cancel=cancel, level=level)
    r = resolve(layer, field, cache, source_id=source_id, progress=progress, cancel=cancel)
    outputs = declared_outputs(get_device(r.analysis_device)) if r.analysis_device else ()
    output = next((o for o in outputs if o.name == name and o.lazy), None)
    if output is None:
        raise ValueError(f"{r.analysis_device}: no lazy output named {name!r}")
    layer_input = input_contract(field)
    selection = selection_recipe(layer, layer_input) if output.selects else None
    lvl = (0 if level is None else int(level)) if output.per_level else None
    key = output_key(r.analysis_device, output, r.analysis_params, r.analysis_key, selection,
                     level=lvl)
    raw = cache.get(r.analysis_key)
    values = np.asarray(r.analysis_input.values)

    def fetch(other: str) -> dict:
        return resolve_output(layer, field, cache, other, source_id=source_id,
                              progress=progress, cancel=cancel, level=level)

    def _compute():
        device = get_device(r.analysis_device)
        extra = {}
        if output.selects:
            # What the filters keep at every level, each with its own settings.
            extra["keeps"] = [level_keeps(layer, raw, device, r.analysis_params, l,
                                          layer_input)
                              for l in range(1, len(raw.get("extrema") or ()) + 1)]
        if output.per_level:
            extra["level"] = lvl
        value = execution.run(device, r.analysis_params,
            lambda: device.compute_output(name, values, raw, r.analysis_params, fetch=fetch,
                                           progress=progress, cancel=cancel, **extra))
        return execution.attach(value, _output_identity={"key": key, "name": name, "level": lvl,
            "params": {k: r.analysis_params[k] for k in output.params}})

    return cache.get_or_compute(key, _compute)
