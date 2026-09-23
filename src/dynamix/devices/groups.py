# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The commit transaction's engine half: ``group_paint`` + ``group_filter``.

Group *designation* is arrangement-view state -- clicking chains in the vector view, accumulating
a palette of named groups. None of that lives here. What lives here is the other side of the
transaction boundary: the WINDOW writes the designated groups into ``group_paint``'s
``spec_json`` param at commit time, and this module turns that param into painted tags a Filter
can read, purely as a function of ``(result, params)`` -- the same shape every other device in
this package already has.

``spec_json`` is a JSON object, ``{name: {"signature": str, "chains": [int], "color": [r,g,b]}}``:
``chains`` are indices into ``result["chains"]`` (chain identity = position in that result's chain
list), ``color`` is an RGB triple, and ``signature`` is the transform signature the
group was designated against (see :class:`GroupPaint`'s docstring for what this device can and
cannot verify about it).

Params ARE the cache key (``model/device.py``, the whole design's premise): writing a re-commit's
groups into ``spec_json`` is what makes the change visible to everything downstream, without this
module -- or the immutable ``chain_classify``-style result dicts it reads -- ever being mutated in
place.
"""
from __future__ import annotations

import json

from dynamix.model.param import Param, ParamKind


def encode_groups(groups: dict) -> str:
    """Canonicalise a committed-groups dict to JSON: ``chains`` as ints, ``color`` as ints,
    ``signature`` as a string -- so a value built from JSON-loaded ints/floats/tuples round-trips
    to the exact same shape ``decode_groups`` hands back, whatever the caller's own dict held."""
    normalized = {
        str(name): {
            "signature": str(spec.get("signature", "")),
            "chains": [int(i) for i in spec.get("chains", ())],
            "color": [int(c) for c in spec.get("color", (0, 0, 0))],
        }
        for name, spec in groups.items()
    }
    return json.dumps(normalized)


def decode_groups(text: str) -> dict:
    """Inverse of :func:`encode_groups` for text it produced. Empty/falsy text decodes to ``{}``
    rather than raising -- the no-groups-committed-yet state, which ``GroupPaint.apply`` also
    treats as passthrough.

    For anything else this is a thin ``json.loads`` wrapper, and it is NOT total: malformed JSON
    still raises ``json.JSONDecodeError``, and a well-formed-but-wrong-shaped payload (e.g. a JSON
    array where the schema promises a ``{name: {...}}`` object) decodes without error into
    something that is not that mapping. Catching either is not this function's job -- it is a
    plain codec, and round-tripping :func:`encode_groups`'s own output is what it is tested
    against. ``GroupPaint.apply`` is the caller reading a value it does not control (a possibly
    stale or hand-edited project on disk) and is the one that must not raise; it wraps this call
    and validates the decoded shape itself."""
    if not text:
        return {}
    return json.loads(text)


def _coerce_indices(gspec) -> list[int]:
    """One group's member-chain indices, coerced to ``int``.

    Raises -- ``AttributeError`` if ``gspec`` is not a mapping (no ``.get``), ``TypeError``/
    ``ValueError`` if a ``"chains"`` entry cannot become an ``int`` -- rather than defaulting
    anything, because an index is correctness-bearing: silently dropping or renumbering one would
    paint the wrong chain. The caller (:meth:`GroupPaint.apply`) catches all three and treats them
    as one stale group, never a fatal error escaping a resolve."""
    return [int(i) for i in gspec.get("chains", [])]


def _coerce_color(gspec) -> list[int]:
    """One group's RGB triple, defaulting to ``[0, 0, 0]`` on any shape or type mismatch.

    Unlike an index, a color is display-only, never correctness-bearing -- a malformed one paints
    black instead of guessing, and is never a reason to flag the whole group stale."""
    try:
        color = [int(c) for c in gspec.get("color", (0, 0, 0))]
    except (TypeError, ValueError, AttributeError):
        return [0, 0, 0]
    return color if len(color) == 3 else [0, 0, 0]


class GroupPaint:
    """Auto-appended on first commit: reads the ``spec_json`` the WINDOW wrote at
    commit time and stamps ``group:<name>`` tags onto COPIES of member chains -- the same
    copy-on-tag idiom :class:`dynamix.devices.chain_classify.ChainClassify` uses, so cached
    results stay immutable and downstream caches re-key off the (changed) params, never off a
    mutated result.

    **Freshness -- an honest v1 note.** The authoritative guard is a transform-signature
    comparison: a group's member indices are only meaningful against the exact chain list they
    were designated against, and a later param change on the upstream transform can reorder or
    resize ``result["chains"]`` out from under them. That comparison would need a
    ``result["_transform_signature"]`` field to check ``params["signature"]`` against -- and
    ``dynamix/core`` is verbatim law, so no field gets added there to carry it. The
    ``signature`` param still travels with every commit (params are the cache key -- see the
    module docstring), so a re-commit is always visible downstream even when this device makes no
    use of the value itself.

    What this device CAN check, structurally, with no signature at all: a group whose member
    index no longer fits inside ``result["chains"]`` is unambiguously stale, because the transform
    that produced the current result cannot have produced that chain. That is a floor, not the
    fix -- a param change that leaves ``len(chains)`` unchanged (or coincidentally re-lengthens it
    to the same count) is a false negative this device cannot catch. The signature recording
    is the real guard; this is what a Filter can verify on its own, honestly labelled as partial.

    A group failing that check is flagged in ``out["_stale_groups"]`` and its members are left
    unpainted; every other, in-range group in the same commit still paints normally.

    **Identity-preserving no-op.** ``result["chains"]`` is only
    ever COPIED once a chain is actually about to be tagged. A commit with nothing to paint this
    redraw (no groups, every referenced group empty, or every referenced index stale) hands back
    the exact same ``chains`` list object the input carried -- and, when there is additionally
    nothing to REPORT either (no ``_stale_groups``), the exact same ``result`` object too. This
    matters beyond tidiness: ``dynamix.shell.canvas.Canvas``'s geometry cache (the 147x scrub
    fix) keys its trail-polyline memoization on ``id(result["chains"])`` -- an unconditional copy
    here broke that identity on every single redraw of a committed layer, silently paying a full
    rebuild on every scrub tick regardless of whether anything about the chains had changed.
    Painting itself is never affected: a touched chain is still always a copy, never a mutation.

    **Malformed ``spec_json`` is a no-op, not a raise.** A resolve runs on every redraw; a project
    that was hand-edited, or committed by an older/newer schema version, must not turn into a
    permanent error strip. Undecodable JSON or a decoded value that is not the ``{name: {...}}``
    object the schema promises is a wholesale passthrough with ``out["_spec_error"]`` set to a
    short reason -- visible, but not fatal. A per-group shape problem (a non-mapping group entry,
    a ``"chains"`` entry that will not coerce to ``int``) is narrower: only that one group is
    flagged stale in ``out["_stale_groups"]``, exactly like an out-of-range index, so one garbage
    group next to well-formed ones does not block the rest of the commit from painting.
    """

    name = "group_paint"
    params = (
        Param("spec_json", ParamKind.TEXT, default="", label="Groups"),
        Param("signature", ParamKind.TEXT, default="", label="Signature"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        spec_json = params["spec_json"]
        if not spec_json:
            return result

        try:
            groups = decode_groups(spec_json)
            if not isinstance(groups, dict):
                raise ValueError(f"spec_json decoded to a {type(groups).__name__}, not an object")
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
            # A stale/hand-edited/wrong-schema project must no-op honestly here, not raise mid-
            # resolve -- a raise would land the layer in a persistent error strip. Passthrough,
            # with the failure surfaced rather than swallowed silently.
            out = dict(result)
            out["_spec_error"] = f"{type(exc).__name__}: {exc}"
            return out

        chains = result.get("chains") or []
        n = len(chains)

        # `out_chains` stays `None` until the FIRST chain is actually touched: copying `chains` unconditionally here -- as this used to do -- broke
        # `result["chains"]`'s identity on EVERY resolve of a committed layer, even a redraw where
        # nothing was actually painted (every group empty, or all stale). The session canvas's
        # geometry cache (`Canvas._cached_geometry`, the 147x scrub fix) keys on exactly that
        # identity, so a committed layer was silently paying a full trail-geometry rebuild on
        # every scrub tick. Painting itself still copies -- immutability is never negotiable --
        # this only removes the copy from the path that paints nothing.
        out_chains = None
        touched: dict[int, dict] = {}
        stale_groups = []
        for gname, gspec in groups.items():
            try:
                indices = _coerce_indices(gspec)
            except (AttributeError, TypeError, ValueError):
                stale_groups.append(gname)     # one malformed group is stale, not fatal
                continue
            if indices and (min(indices) < 0 or max(indices) >= n):
                stale_groups.append(gname)
                continue
            color = _coerce_color(gspec)
            for idx in indices:
                chain_copy = touched.get(idx)
                if chain_copy is None:
                    if out_chains is None:
                        out_chains = list(chains)      # copy ONLY once painting actually starts
                    chain_copy = dict(out_chains[idx])
                    chain_copy["tags"] = list(chain_copy.get("tags") or [])
                    touched[idx] = chain_copy
                    out_chains[idx] = chain_copy
                chain_copy["tags"].append(f"group:{gname}")
                chain_copy["tag_origin"] = "user:commit"
                chain_copy["group_color"] = color

        if out_chains is None and not stale_groups:
            # Nothing painted, nothing to report: an honest no-op, identity-preserving ALL the way
            # up (not just `chains`) -- exactly like the empty-`spec_json` passthrough above and
            # `GroupFilter`'s own empty-`group` passthrough. Cannot do the same when `stale_groups`
            # is non-empty: that has to land somewhere the caller can read it, and mutating the
            # (possibly shared, cached) input `result` in place to add it would break the
            # immutability the copy-on-tag discipline exists to protect -- so THAT path still
            # returns a new top-level dict, but keeps `chains` itself identity-preserved (see
            # below), which is the part the geometry cache actually keys on.
            return result

        out = dict(result)
        out["chains"] = out_chains if out_chains is not None else chains
        if stale_groups:
            out["_stale_groups"] = stale_groups
        return out


class GroupFilter:
    """Keep or drop chains tagged ``group:<group>`` by :class:`GroupPaint` -- an ordinary chain
    filter, free consequence of the tag schema. Empty ``group`` (nothing selected in
    the palette) is a passthrough, the same convention every other device in this package uses for
    its own "no-op" setting."""

    name = "group_filter"
    params = (
        # editable=True: unlike group_paint's spec_json/signature
        # (write-only from the commit transaction), this is the one TEXT param a user must
        # actually type a group name into -- ParamKind.TEXT's default read-only "label" widget
        # left it a dead control in the browser with no other way to set it.
        Param("group", ParamKind.TEXT, default="", label="Group", editable=True),
        Param("mode", ParamKind.CHOICE, default="keep", choices=("keep", "drop"), label="Mode"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        group = params["group"]
        if not group:
            return result
        chains = result.get("chains") or []
        tag = f"group:{group}"
        keep = params["mode"] == "keep"
        kept = [c for c in chains if (tag in (c.get("tags") or [])) == keep]
        out = dict(result)
        out["chains"] = kept
        out["_chains_dropped"] = len(chains) - len(kept)
        return out
