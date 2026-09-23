# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Zlatanova (2000) 9-intersection relation codes.

A relation between two simple objects is a 9-bit integer: which of the nine
boundary/interior/exterior pairwise intersections are non-empty, read MSB-first in the order
(bb, ii, bi, ib, ee, eb, ei, be, ie) where b/i/e = boundary/interior/exterior and the first
letter is A's part. Normative source for every code and set in this module:
Zlatanova's thesis (relation tables extracted, counts verified three ways).

Thesis convention that decides the point codes: a POINT has empty interior -- it IS its boundary.
"""
from __future__ import annotations

POINT, LINE, SURFACE, BODY = "point", "line", "surface", "body"
KINDS = (POINT, LINE, SURFACE, BODY)

#: MSB-first bit order; index 0 is bit 8.
_ORDER = ("bb", "ii", "bi", "ib", "ee", "eb", "ei", "be", "ie")
_BIT = {pair: 8 - i for i, pair in enumerate(_ORDER)}


def encode(nonempty) -> int:
    """Code from the set of non-empty intersection pairs, e.g. ``{"ii", "ee"}``."""
    code = 0
    for pair in nonempty:
        if pair not in _BIT:
            raise ValueError(f"unknown intersection pair {pair!r}; known: {_ORDER}")
        code |= 1 << _BIT[pair]
    return code


def converse(code: int) -> int:
    """The code of the same relation read B-to-A: a pure bit permutation (6<->5, 3<->1, 2<->0)."""
    fixed = code & ((1 << 8) | (1 << 7) | (1 << 4))
    return (fixed
            | ((code >> 1) & (1 << 5)) | ((code & (1 << 5)) << 1)
            | ((code >> 2) & (1 << 1)) | ((code & (1 << 1)) << 2)
            | ((code >> 2) & (1 << 0)) | ((code & (1 << 0)) << 2))


# --- Per-configuration permitted-relation sets -------------------------------------------------
# Data transcribed from the thesis extraction (section 3, "the enumeration by
# object-pair configuration"); that extraction is normative -- if a value here disagrees with it, this
# file is wrong. Sets shared across configurations (P,X / X,P for every X; L,B == L,S in R^2;
# B,B == S,S in R^2) are shared by construction below because the doc states those identities
# explicitly, not merely coincidentally.

#: A point against any other kind, in any dimension where both kinds are legal (doc: "dimension
#: independent"): disjoint (R030), point in the other's interior (R092), point on its boundary
#: (R284). Erratum honored: the thesis's p.135 prints R031 for the disjoint member; Appendix 3
#: Table 2 and the negative conditions (a point has empty interior) both say R030.
_P_X = frozenset({30, 92, 284})
#: The converse images of _P_X (R027, R051, R275) -- X against a point.
_X_P = frozenset({27, 51, 275})
#: Point-point, any dimension: disjoint (R026), coincide (R272).
_P_P = frozenset({26, 272})

#: Line-line in R^1 (section 3.3): the eight named relations.
_LL_1 = frozenset({31, 179, 220, 255, 287, 400, 435, 476})
#: Line-line in R^2 and R^3 (section 3.4): identical set in both spaces.
_LL_23 = frozenset({
    31, 55, 63, 93, 95, 117, 119, 125, 127, 159, 179,
    183, 191, 220, 221, 223, 245, 247, 253, 255, 277, 287,
    311, 349, 373, 400, 405, 415, 435, 439, 476, 477, 501,
})

#: Line-surface in R^2, A = line (section 3.5).
_LS_2 = frozenset({
    31, 63, 191, 220, 252, 253, 255, 285, 287, 316,
    317, 319, 412, 444, 445, 447, 476, 508, 509,
})
#: Surface-line in R^2, A = surface -- the converse images of _LS_2 (section 3.5).
_SL_2 = frozenset({
    31, 95, 179, 223, 243, 247, 255, 279, 287, 339,
    343, 351, 403, 435, 467, 471, 479, 499, 503,
})

#: Surface-line in R^3, A = surface (section 3.6, Figure 6-4 labelling -- "the most reliable of
#: the two"). Errata honored: R055 belongs here, not to L,S -- a surface's boundary is a closed
#: curve that cannot lie inside a line's interior, so R055 (which forces boundary-in-interior)
#: requires A = surface.
_SL_3 = frozenset({
    31, 55, 63, 95, 119, 127, 159, 179, 183, 191, 223,
    243, 247, 255, 279, 287, 311, 339, 343, 351, 375, 403,
    407, 415, 435, 439, 467, 471, 479, 499, 503,
})
#: Line-surface in R^3, A = line -- the converse images of _SL_3, including R093 (converse of
#: R055) in place of R055 itself (section 3.6 erratum).
_LS_3 = frozenset({
    31, 63, 93, 95, 125, 127, 159, 191, 220, 221, 223,
    252, 253, 255, 285, 287, 316, 317, 319, 349, 381, 412,
    413, 415, 444, 445, 447, 476, 477, 508, 509,
})

#: Line-body / body-line in R^3 (section 3.7): identical sets to line-surface / surface-line in
#: R^2 -- the thesis states this explicitly (p.122), so these are the same objects, not copies.
_LB_3 = _LS_2
_BL_3 = _SL_2

#: Surface-surface in R^2 (section 3.8): the eight named relations, with R511 (not R255) as
#: overlap.
_SS_2 = frozenset({31, 179, 220, 287, 400, 435, 476, 511})
#: Surface-surface in R^3 (section 3.9): the largest single configuration, 38 codes.
_SS_3 = frozenset({
    31, 55, 63, 93, 95, 117, 119, 125, 127, 159, 179,
    183, 191, 220, 221, 223, 247, 253, 255, 277, 287, 311,
    319, 349, 351, 375, 381, 383, 400, 405, 415, 435, 439,
    447, 476, 477, 479, 511,
})

#: Surface-body in R^3, A = surface (section 3.10).
_SB_3 = frozenset({
    31, 63, 191, 220, 252, 253, 285, 287, 316, 317,
    319, 412, 444, 445, 447, 476, 508, 509, 511,
})
#: Body-surface in R^3, A = body -- the converse images of _SB_3 (section 3.10).
_BS_3 = frozenset({
    31, 95, 179, 223, 243, 247, 279, 287, 339, 343,
    351, 403, 435, 467, 471, 479, 499, 503, 511,
})

#: Body-body in R^3 (section 3.11): identical set to surface-surface in R^2 -- the thesis states
#: this explicitly (p.124).
_BB_3 = _SS_2

#: (kind_a, kind_b, space_dim) -> permitted codes. Data transcribed from the thesis extraction; that extraction is normative -- if a value here disagrees
#: with it, this file is wrong. The cross-checks in tests (counts, converse images, union == 69)
#: exist to catch transcription slips.
_PERMITTED: dict[tuple[str, str, int], frozenset[int]] = {
    (POINT, POINT, 1): _P_P,
    (POINT, POINT, 2): _P_P,
    (POINT, POINT, 3): _P_P,

    (POINT, LINE, 1): _P_X,
    (POINT, LINE, 2): _P_X,
    (POINT, LINE, 3): _P_X,
    (LINE, POINT, 1): _X_P,
    (LINE, POINT, 2): _X_P,
    (LINE, POINT, 3): _X_P,

    (POINT, SURFACE, 2): _P_X,
    (POINT, SURFACE, 3): _P_X,
    (SURFACE, POINT, 2): _X_P,
    (SURFACE, POINT, 3): _X_P,

    (POINT, BODY, 3): _P_X,
    (BODY, POINT, 3): _X_P,

    (LINE, LINE, 1): _LL_1,
    (LINE, LINE, 2): _LL_23,
    (LINE, LINE, 3): _LL_23,

    (LINE, SURFACE, 2): _LS_2,
    (SURFACE, LINE, 2): _SL_2,
    (LINE, SURFACE, 3): _LS_3,
    (SURFACE, LINE, 3): _SL_3,

    (LINE, BODY, 3): _LB_3,
    (BODY, LINE, 3): _BL_3,

    (SURFACE, SURFACE, 2): _SS_2,
    (SURFACE, SURFACE, 3): _SS_3,

    (SURFACE, BODY, 3): _SB_3,
    (BODY, SURFACE, 3): _BS_3,

    (BODY, BODY, 3): _BB_3,
}

THE_69: frozenset[int] = frozenset().union(*_PERMITTED.values())


def permitted(kind_a: str, kind_b: str, space_dim: int) -> frozenset[int]:
    """Codes possible between kind_a and kind_b embedded in a space of space_dim dimensions."""
    try:
        return _PERMITTED[(kind_a, kind_b, space_dim)]
    except KeyError:
        raise KeyError(
            f"no permitted-relation set for ({kind_a}, {kind_b}) in dim {space_dim}; "
            f"either an unknown kind or an impossible embedding"
        ) from None


# --- Names ---------------------------------------------------------------------------------
# Only the configurations where the thesis names all eight group-representative codes: line-line
# in R^1 (section 3.3), surface-surface in R^2 (section 3.8), body-body in R^3 (section 3.11).
# "overlap" is R255 for line-line but R511 for surface-surface and body-body -- the doc's own
# headline example (p.122) of why a name is meaningless without its configuration.
_NAMED: dict[tuple[str, str, int], dict[int, str]] = {
    (LINE, LINE, 1): {
        31: "disjoint", 287: "meet", 179: "contains", 220: "inside",
        435: "covers", 476: "covered_by", 400: "equal", 255: "overlap",
    },
    (SURFACE, SURFACE, 2): {
        31: "disjoint", 287: "meet", 179: "contains", 220: "inside",
        435: "covers", 476: "covered_by", 400: "equal", 511: "overlap",
    },
    (BODY, BODY, 3): {
        31: "disjoint", 287: "meet", 179: "contains", 220: "inside",
        435: "covers", 476: "covered_by", 400: "equal", 511: "overlap",
    },
}


def name_for(code: int, kind_a: str, kind_b: str, space_dim: int) -> str | None:
    """The canonical name of ``code`` in THIS configuration, or None. A name without its
    configuration is meaningless (overlap is 255 for lines in R^1 but 511 for bodies in R^3),
    which is why no name-only lookup exists."""
    return _NAMED.get((kind_a, kind_b, space_dim), {}).get(code)


# --- Relation builders -----------------------------------------------------------------------
# Used by the WTMM binding; derived from the thesis conventions, verified against the
# transcribed sets by the tests.

def point_code(*, at: str) -> int:
    """Point A against a line/surface/body B: at="interior" (92), "boundary" (284),
    "disjoint" (30). A point's interior is empty, so only its boundary bit-pairs can be set."""
    if at == "interior":
        return encode({"bi", "ee", "eb", "ei"})          # 92
    if at == "boundary":
        return encode({"bb", "ee", "eb", "ei"})          # 284
    if at == "disjoint":
        return encode({"be", "ee", "eb", "ei"})          # 30
    raise ValueError(f"at must be interior|boundary|disjoint, got {at!r}")


def line_in_line_code(*, equal: bool = False, touches_end: bool = False) -> int:
    """Segment A of a parent line B, evaluated in the 1-D along-line space.
    equal: A is all of B (400). touches_end: one endpoint of A is an endpoint of B (476).
    Neither: A strictly interior (220)."""
    if equal:
        return encode({"bb", "ii", "ee"})                            # 400
    if touches_end:
        return encode({"bb", "ii", "bi", "ee", "eb", "ei"})          # 476
    return encode({"ii", "bi", "ee", "eb", "ei"})                    # 220
