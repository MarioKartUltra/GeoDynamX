# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Zlatanova relation-code mechanics. Normative source: Zlatanova's thesis."""
from dynamix.topology.codes import BODY, KINDS, LINE, POINT, SURFACE, converse, encode


def test_kinds():
    assert KINDS == (POINT, LINE, SURFACE, BODY) == ("point", "line", "surface", "body")


def test_encode_bit_order():
    # MSB-first: bb, ii, bi, ib, ee, eb, ei, be, ie
    assert encode({"bb"}) == 256
    assert encode({"ii"}) == 128
    assert encode({"bi"}) == 64
    assert encode({"ib"}) == 32
    assert encode({"ee"}) == 16
    assert encode({"eb"}) == 8
    assert encode({"ei"}) == 4
    assert encode({"be"}) == 2
    assert encode({"ie"}) == 1
    assert encode(set()) == 0
    assert encode({"bb", "ii", "bi", "ib", "ee", "eb", "ei", "be", "ie"}) == 511


def test_encode_rejects_unknown_pair():
    import pytest
    with pytest.raises(ValueError):
        encode({"xx"})


def test_converse_is_the_bit_swap():
    # swap 6<->5, 3<->1, 2<->0; bits 8, 7, 4 fixed
    assert converse(encode({"bi"})) == encode({"ib"})
    assert converse(encode({"eb"})) == encode({"be"})
    assert converse(encode({"ei"})) == encode({"ie"})
    assert converse(encode({"bb", "ii", "ee"})) == encode({"bb", "ii", "ee"})
    assert converse(255) == 255 and converse(511) == 511


def test_converse_is_an_involution():
    assert all(converse(converse(c)) == c for c in range(512))


from dynamix.topology.codes import THE_69, line_in_line_code, name_for, permitted, point_code


def _cimage(codes):
    return frozenset(converse(c) for c in codes)


def test_the_totals_and_per_config_counts():
    assert len(THE_69) == 69
    assert len(permitted(POINT, POINT, 3)) == 2
    for x in (LINE, SURFACE, BODY):
        assert permitted(POINT, x, 3) == frozenset({30, 92, 284})
        assert permitted(x, POINT, 3) == _cimage(frozenset({30, 92, 284}))
    assert len(permitted(LINE, LINE, 1)) == 8
    assert len(permitted(LINE, LINE, 2)) == len(permitted(LINE, LINE, 3)) == 33
    assert len(permitted(LINE, SURFACE, 2)) == 19
    assert len(permitted(LINE, SURFACE, 3)) == 31
    assert len(permitted(SURFACE, SURFACE, 2)) == 8
    assert len(permitted(SURFACE, SURFACE, 3)) == 38
    assert len(permitted(LINE, BODY, 3)) == len(permitted(SURFACE, BODY, 3)) == 19
    assert len(permitted(BODY, BODY, 3)) == 8


def test_converse_images_and_closure():
    from dynamix.topology import codes as codes_mod
    for (ka, kb, d), codes in codes_mod._PERMITTED.items():
        # the converse configuration must exist and be the exact converse image; for symmetric
        # configurations (ka == kb) this doubles as converse-closure of the set itself
        assert permitted(kb, ka, d) == _cimage(codes)


def test_errata_honored():
    assert 30 in permitted(POINT, BODY, 3)
    # p.135's R(P,X) list prints R031; it must be R030 (a point has empty interior). R031 itself
    # is a legitimate code elsewhere (e.g. line-line disjoint) and remains in THE_69 -- the
    # erratum is that it does not belong to the P,X sets specifically.
    assert 31 not in permitted(POINT, BODY, 3)


def test_impossible_configuration_raises():
    import pytest
    with pytest.raises(KeyError):
        permitted(BODY, BODY, 2)


def test_names_are_configuration_relative():
    assert name_for(255, LINE, LINE, 1) == "overlap"
    assert name_for(511, SURFACE, SURFACE, 2) == "overlap"
    assert name_for(511, BODY, BODY, 3) == "overlap"
    assert name_for(255, BODY, BODY, 3) is None


def test_relation_builders_land_in_permitted_sets():
    assert {point_code(at=a) for a in ("interior", "boundary", "disjoint")} \
        == permitted(POINT, LINE, 1) == frozenset({92, 284, 30})
    lll = permitted(LINE, LINE, 1)
    built = {line_in_line_code(equal=True), line_in_line_code(touches_end=True),
             line_in_line_code()}
    assert built == {400, 476, 220} and built <= lll
    assert name_for(400, LINE, LINE, 1) == "equal"
