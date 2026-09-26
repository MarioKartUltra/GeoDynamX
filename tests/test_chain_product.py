# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.chain_product -- the draw-ready CSR chain bundle.

The transform
stamps, worker-side, EQSelect's draw-ready representation -- ordered CSR point arrays plus a
per-chain metric table -- so every downstream filter is an index selection and every draw is a
bounded concatenation. This module is that representation's builder.

The H side enumerates each scale's ordered H-line runs (``dynamix.core.hlines.hline_runs`` is the
ordering authority); the V side flattens ``result["chains"]``. Metrics must agree with the
existing per-chain estimators (``dynamix.core.chain_stats``) to float tolerance -- a filter that
switches from the dict path to the metric table must keep/drop the SAME chains.
"""
from __future__ import annotations

import numpy as np

from dynamix.core.chain_product import build_chain_product
from dynamix.core.chain_stats import max_log2_modulus_for, stats_for
from dynamix.core.hlines import hline_runs

SHAPE = (8, 64)


def _layer(mods, line_ids, y=0, args=None):
    """Points along x in index order (the walk then orders them the same way)."""
    mod = np.asarray(mods, dtype=np.float64)
    n = mod.size
    arg = np.asarray(args, dtype=np.float64) if args is not None else np.linspace(0.0, 1.0, n)
    return {"x": np.arange(n, dtype=np.int64), "y": np.full(n, y, dtype=np.int64),
            "mod": mod, "arg": arg, "line_id": np.asarray(line_ids, dtype=np.int64)}


def _chain(xs, ys, mods, scales):
    mod = np.asarray(mods, dtype=np.float64)
    k = mod.size
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_mod = np.log2(np.abs(mod))
    return {"x": np.asarray(xs, dtype=np.int64), "y": np.asarray(ys, dtype=np.int64),
            "mod": mod, "log2_mod": log2_mod,
            "log2_scales": np.log2(np.asarray(scales, dtype=np.float64))[:k]}


def _fixture():
    """Two scales: layer 0 has two H-lines + one singleton, layer 1 one H-line.
    Two V chains, both finest-anchored at points that exist in their scale's layer."""
    scales = [1.0, 2.0]
    l0 = _layer([1.0, 0.8, 0.6, 0.5, 0.4, 0.9], [0, 0, 0, 1, 1, -1], y=0)
    l1 = _layer([0.7, 0.3, 0.2], [0, 0, 0], y=2)
    chains = [
        _chain([0, 0], [0, 2], [1.0, 0.7], scales),   # foot at l0 point 0, then l1 point 0
        _chain([3, 1], [0, 2], [0.5, 0.3], scales),   # foot at l0 point 3, then l1 point 1
    ]
    return [l0, l1], chains, np.asarray(scales)


def test_h_side_is_the_ordered_runs_as_csr():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    runs = [(si, run) for si, layer in enumerate(extrema)
            for run in hline_runs(layer, SHAPE)]
    off = np.asarray(p["h_off"])
    assert off[0] == 0 and len(off) == len(runs) + 1
    for i, (si, run) in enumerate(runs):
        a, b = int(off[i]), int(off[i + 1])
        layer = extrema[si]
        np.testing.assert_array_equal(p["h_x"][a:b], np.asarray(layer["x"])[run])
        np.testing.assert_array_equal(p["h_y"][a:b], np.asarray(layer["y"])[run])
        np.testing.assert_allclose(p["h_mod"][a:b], np.asarray(layer["mod"])[run])
        np.testing.assert_allclose(p["h_arg"][a:b], np.asarray(layer["arg"])[run])
        assert p["h_scale"][i] == si
        assert p["h_len"][i] == run.size


def test_h_per_chain_metrics_are_sup_and_mean_along_the_run():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    off = np.asarray(p["h_off"])
    for i in range(len(off) - 1):
        seg = np.asarray(p["h_mod"][off[i]:off[i + 1]])
        assert p["h_mod_sup"][i] == seg.max()
        np.testing.assert_allclose(p["h_mod_mean"][i], seg.mean())


def test_singletons_live_in_the_iso_side_not_the_h_chains():
    """A singleton is not a line (EQSelect's rep has none), but per-point filters and the
    orphan-dots display need it -- so it rides a separate flat ``iso_*`` side-table."""
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    # layer 0's singleton sits at x=5, y=0 -- no H point may carry it
    on = (np.asarray(p["h_x"]) == 5) & (np.asarray(p["h_y"]) == 0)
    assert not on.any()
    l0 = extrema[0]
    np.testing.assert_array_equal(p["iso_x"], [5])
    np.testing.assert_array_equal(p["iso_y"], [0])
    np.testing.assert_allclose(p["iso_mod"], [l0["mod"][5]])
    np.testing.assert_allclose(p["iso_arg"], [l0["arg"][5]])
    np.testing.assert_array_equal(p["iso_scale"], [0])
    np.testing.assert_array_equal(p["iso_src"], [5])   # index into layer 0's own arrays


def test_h_src_maps_every_product_point_back_to_its_layer():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    off = np.asarray(p["h_off"])
    for i in range(len(off) - 1):
        si = int(p["h_scale"][i])
        src = np.asarray(p["h_src"][off[i]:off[i + 1]])
        layer = extrema[si]
        np.testing.assert_array_equal(np.asarray(layer["x"])[src], p["h_x"][off[i]:off[i + 1]])
        np.testing.assert_array_equal(np.asarray(layer["y"])[src], p["h_y"][off[i]:off[i + 1]])


def test_scale_peak_is_the_per_scale_max_over_lines_and_singletons():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    expect = [float(np.nanmax(np.asarray(l["mod"], float))) for l in extrema]
    np.testing.assert_allclose(p["scale_peak"], expect)


def test_v_side_flattens_chains_with_metrics():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    off = np.asarray(p["v_off"])
    assert off[0] == 0 and len(off) == len(chains) + 1
    for i, ch in enumerate(chains):
        a, b = int(off[i]), int(off[i + 1])
        np.testing.assert_array_equal(p["v_x"][a:b], ch["x"])
        np.testing.assert_array_equal(p["v_y"][a:b], ch["y"])
        np.testing.assert_allclose(p["v_mod"][a:b], ch["mod"])
        np.testing.assert_array_equal(p["v_scale"][a:b], np.arange(b - a))
        assert p["v_persist"][i] == len(ch["mod"])
        assert p["v_mod_finest"][i] == ch["mod"][0]
        assert p["v_mod_sup"][i] == np.max(ch["mod"])


def test_v_holder_metrics_agree_with_chain_stats():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    np.testing.assert_allclose(p["v_holder_ols"], stats_for(chains, "ols"), rtol=1e-10)
    np.testing.assert_allclose(p["v_holder_max"], stats_for(chains, "max"), rtol=1e-10)
    np.testing.assert_allclose(p["v_max_log2_mod"], max_log2_modulus_for(chains), rtol=1e-10)


def test_v_arg_is_joined_from_the_extrema_layers():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    # chain 0 foot: l0 point (x=0, y=0) -> arg index 0 of layer 0; its scale-1 point (0, 2)
    # -> l1 point 0. chain 1 foot: l0 point (3, 0) -> arg index 3; scale-1 (1, 2) -> l1 point 1.
    l0, l1 = extrema
    np.testing.assert_allclose(
        p["v_arg"], [l0["arg"][0], l1["arg"][0], l0["arg"][3], l1["arg"][1]])
    np.testing.assert_allclose(p["v_arg_finest"], [l0["arg"][0], l0["arg"][3]])


def test_v_arg_is_nan_where_the_point_is_not_in_its_layer():
    extrema, chains, scales = _fixture()
    stray = [_chain([60, 60], [7, 7], [0.2, 0.1], scales)]   # nowhere in any layer
    p = build_chain_product(extrema, stray, scales, SHAPE)
    assert np.isnan(p["v_arg"]).all()
    assert np.isnan(p["v_arg_finest"]).all()


def test_missing_arg_key_yields_nan_h_arg():
    extrema, chains, scales = _fixture()
    bare = [{k: v for k, v in layer.items() if k != "arg"} for layer in extrema]
    p = build_chain_product(bare, chains, scales, SHAPE)
    assert np.isnan(p["h_arg"]).all()


def test_precomputed_runs_give_the_identical_product():
    extrema, chains, scales = _fixture()
    runs = [hline_runs(layer, SHAPE) for layer in extrema]
    a = build_chain_product(extrema, chains, scales, SHAPE)
    b = build_chain_product(extrema, chains, scales, SHAPE, runs=runs)
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]),
                                      err_msg=f"key {key!r} differs")


def test_empty_inputs_build_an_empty_product():
    p = build_chain_product([], [], np.asarray([]), SHAPE)
    assert list(p["h_off"]) == [0] and list(p["v_off"]) == [0]
    assert len(p["h_x"]) == 0 and len(p["v_x"]) == 0
    assert len(p["h_len"]) == 0 and len(p["v_persist"]) == 0
    assert p["n_scales"] == 0


def test_scales_and_shape_ride_along():
    extrema, chains, scales = _fixture()
    p = build_chain_product(extrema, chains, scales, SHAPE)
    np.testing.assert_allclose(p["scales"], scales)
    assert p["n_scales"] == 2
    assert tuple(p["shape"]) == SHAPE


# ---------------------------------------------------------------------------------------------
# npz persistence: schema v4 (EQSelect extrema-layers spec Phase B -- v3 layout + h_arg/v_arg),
# plus additive pixel columns for an exact DynamiX round-trip. The SAME bundle is
# the on-disk product. Compatibility gate: dynamix.core.wtmm_backend.load_chains_npz is the
# VERBATIM EQSelect loader, so it loading our file IS the interop check.
# ---------------------------------------------------------------------------------------------

def _field():
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    ny, nx = SHAPE
    return RasterField(name="synth", values=np.zeros(SHAPE), frame=LocalFrame(units="px"),
                       x_axis=np.arange(nx, dtype=np.float64) * 2.0,
                       y_axis=np.arange(ny, dtype=np.float64) * 3.0)


def _product():
    extrema, chains, scales = _fixture()
    return build_chain_product(extrema, chains, scales, SHAPE)


def test_export_writes_schema_v4_with_frame_unit_coordinates(tmp_path):
    from dynamix.core.chain_product import export_chain_product_npz
    p = _product()
    field = _field()
    path = tmp_path / "chains.npz"
    export_chain_product_npz(path, p, field=field, params={"n_oct": 1})
    d = np.load(path, allow_pickle=True)
    assert int(d["schema_version"]) == 4
    for key in ("h_xyz", "h_off", "h_scale", "h_len", "h_mod", "h_arg",
                "v_xyz", "v_off", "v_persist", "v_scale", "v_mod", "v_arg",
                "frame_kind", "frame_meta", "params", "scales", "n_scales", "region"):
        assert key in d.files, f"missing {key}"
    assert d["h_xyz"].dtype == np.float32 and d["h_xyz"].shape == (len(p["h_x"]), 3)
    np.testing.assert_allclose(d["h_xyz"][:, 0], field.x_axis[p["h_x"]], rtol=1e-6)
    np.testing.assert_allclose(d["h_xyz"][:, 1], field.y_axis[p["h_y"]], rtol=1e-6)
    assert (d["h_xyz"][:, 2] == 0.0).all()
    np.testing.assert_allclose(d["v_arg"], p["v_arg"], rtol=1e-6)
    assert str(d["region"]) == "synth"


def test_export_is_loadable_by_the_verbatim_eqselect_loader(tmp_path):
    from dynamix.core.chain_product import export_chain_product_npz
    from dynamix.core.wtmm_backend import load_chains_npz
    p = _product()
    path = tmp_path / "chains.npz"
    export_chain_product_npz(path, p, field=_field(), params={"n_oct": 1})
    te = load_chains_npz(path)
    assert len(te["h_segments"]) == len(p["h_len"])
    np.testing.assert_array_equal(te["h_len"], p["h_len"])
    np.testing.assert_array_equal(te["h_scale"], p["h_scale"])
    np.testing.assert_array_equal(te["v_persist"], p["v_persist"])
    assert te["params"] == {"n_oct": 1}
    assert te["frame"].kind == "local"


def test_npz_round_trips_the_product(tmp_path):
    from dynamix.core.chain_product import export_chain_product_npz, load_chain_product_npz
    p = _product()
    path = tmp_path / "chains.npz"
    export_chain_product_npz(path, p, field=_field(), params={})
    p2 = load_chain_product_npz(path)
    for key in ("h_x", "h_y", "h_off", "h_scale", "h_len", "v_x", "v_y", "v_off",
                "v_scale", "v_persist", "iso_x", "iso_y", "iso_scale"):
        np.testing.assert_array_equal(np.asarray(p2[key]), np.asarray(p[key]),
                                      err_msg=f"key {key!r}")
    for key in ("h_mod", "h_arg", "h_mod_sup", "h_mod_mean", "v_mod", "v_arg",
                "v_mod_finest", "v_mod_sup", "v_holder_ols", "v_holder_max",
                "v_max_log2_mod", "v_arg_finest", "iso_mod", "iso_arg", "scale_peak"):
        np.testing.assert_allclose(np.asarray(p2[key]), np.asarray(p[key]), rtol=1e-4,
                                   atol=1e-6, err_msg=f"key {key!r}")
    assert p2["n_scales"] == p["n_scales"]
    assert tuple(p2["shape"]) == tuple(p["shape"])
    # src columns map into layers a loaded product does not have -- sentinel, same lengths
    assert (np.asarray(p2["h_src"]) == -1).all() and len(p2["h_src"]) == len(p["h_src"])
    assert (np.asarray(p2["iso_src"]) == -1).all() and len(p2["iso_src"]) == len(p["iso_src"])


def test_empty_product_round_trips(tmp_path):
    from dynamix.core.chain_product import export_chain_product_npz, load_chain_product_npz
    p = build_chain_product([], [], np.asarray([]), SHAPE)
    path = tmp_path / "empty.npz"
    export_chain_product_npz(path, p, field=_field(), params={})
    p2 = load_chain_product_npz(path)
    assert list(p2["h_off"]) == [0] and list(p2["v_off"]) == [0]
    assert len(p2["h_x"]) == 0 and len(p2["v_x"]) == 0


# ---------------------------------------------------------------------------------------------
# The draw cap (EQSelect _TE_MAX): a view's draw has a ceiling no matter the
# dataset -- keep the LONGEST H chains / MOST PERSISTENT V chains, and report the honest totals
# so the views can say "drawing X of Y" instead of silently truncating.
# ---------------------------------------------------------------------------------------------

def _big_product():
    scales = [1.0]
    l0 = _layer(np.linspace(1.0, 0.1, 12),
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, -1])
    chains = [_chain([0], [0], [0.5], scales),
              _chain([1, 1], [0, 0], [0.5, 0.4], [1.0, 2.0]),
              _chain([2, 2, 2], [0, 0, 0], [0.5, 0.4, 0.3], [1.0, 2.0, 4.0])]
    return build_chain_product([l0], chains, np.asarray(scales), SHAPE)


def test_capped_h_keeps_the_longest_and_reports_the_total():
    from dynamix.core.chain_product import capped_h
    p = _big_product()                     # h_len: [2, 3, 4, 2]
    idx, total = capped_h(p, None, cap=2)
    assert total == 4
    assert set(np.asarray(p["h_len"])[idx]) == {4, 3}

    all_idx, total = capped_h(p, None, cap=100)
    assert total == 4 and len(all_idx) == 4


def test_capped_h_caps_within_the_current_selection():
    from dynamix.core.chain_product import capped_h
    p = _big_product()
    keep = np.asarray([0, 1, 3])           # lengths 2, 3, 2
    idx, total = capped_h(p, keep, cap=1)
    assert total == 3
    assert list(np.asarray(p["h_len"])[idx]) == [3]


def test_capped_v_keeps_the_most_persistent():
    from dynamix.core.chain_product import capped_v
    p = _big_product()                     # v_persist: [1, 2, 3]
    idx, total = capped_v(p, None, cap=2)
    assert total == 3
    assert set(np.asarray(p["v_persist"])[idx]) == {3, 2}


def test_stub_capped_chains_keeps_positions_for_picking():
    """The scene draws ``result["chains"]`` with chain identity = LIST POSITION (its picking
    lookup indexes that list), so its cap must not renumber -- capped-out chains become empty
    stubs that own no points, and the note says what was withheld."""
    from dynamix.core.chain_product import stub_capped_chains
    chains = [{"mod": np.zeros(k), "x": np.zeros(k), "y": np.zeros(k)} for k in (1, 3, 2)]

    kept, note = stub_capped_chains(chains, cap=2)
    assert kept is not chains and len(kept) == 3
    assert kept[1] is chains[1] and kept[2] is chains[2]     # the two most persistent survive
    assert len(kept[0].get("mod", ())) == 0                  # the shortest became a stub
    assert note is not None and "2 of 3" in note

    same, note = stub_capped_chains(chains, cap=10)
    assert same is chains and note is None                   # under the cap: untouched, no copy


# ------------------------------------------------------------------------- subpixel columns


def test_float_display_columns_ride_alongside_the_integer_identity():
    """Extrema stamped with x_sub/y_sub (the interpolate knob) produce parallel h_xf/h_yf and
    iso_xf/iso_yf FLOAT columns in the product, in the same CSR order as h_x/h_y and
    iso_x/iso_y; the integer columns are untouched (they are the mask/pick/projection
    identity). Without the channels, the float columns do not exist."""
    extrema, chains, scales = _fixture()
    plain = build_chain_product(extrema, chains, scales, (4, 8))
    assert "h_xf" not in plain and "iso_xf" not in plain

    stamped = [dict(l, x_sub=np.asarray(l["x"], float) + 0.4,
                    y_sub=np.asarray(l["y"], float) - 0.3) for l in extrema]
    prod = build_chain_product(stamped, chains, scales, (4, 8))
    for f_key, i_key in (("h_xf", "h_x"), ("h_yf", "h_y"),
                         ("iso_xf", "iso_x"), ("iso_yf", "iso_y")):
        assert prod[f_key].shape == prod[i_key].shape
    np.testing.assert_allclose(prod["h_xf"], prod["h_x"] + 0.4)
    np.testing.assert_allclose(prod["h_yf"], prod["h_y"] - 0.3)
    np.testing.assert_allclose(prod["iso_xf"], prod["iso_x"] + 0.4)
    np.testing.assert_allclose(prod["iso_yf"], prod["iso_y"] - 0.3)
    assert prod["h_x"].dtype == np.int64 and prod["iso_x"].dtype == np.int64
