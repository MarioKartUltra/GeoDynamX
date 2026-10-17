# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Spec 22 §4 and test 6: the step's ``edge`` report in the .gprv (each scale's reach, the grids and
their ratio, the halo, real and synthetic px, the cut border), the warning report read as the run
read the parent, and padded values that change no kept maximum under a cut."""
from __future__ import annotations

import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from test_roi_halo import _synthetic_parent, _write_tif  # noqa: E402


def _record(tmp_path, steps, name):
    gprv = pytest.importorskip("gprv")
    import hashlib

    from dynamix.core import fft_policy
    from dynamix.core.chain_product import export_chain_product_npz
    from dynamix.core.ingest import load_grid
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.gprv_export import write_record
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.project import Project
    from test_q_wavelet_widths import _dem

    register_builtin_devices()
    fft_policy.configure("fftw", 64)
    dem = tmp_path / "dem.tif"
    if not dem.exists():
        _dem(dem)
    field = load_grid(dem)
    project = Project()
    src = project.add_source(str(dem), sha256=hashlib.sha256(dem.read_bytes()).hexdigest())
    layer = project.add_layer(name, src.source_id,
                              Chain(tuple(DeviceRef(d, p) for d, p in steps)))
    rendered = resolve(layer, field, Cache())
    out = tmp_path / f"{name}.npz"
    export_chain_product_npz(out, rendered.result["chain_product"], field=field,
                             params=json.loads(json.dumps(rendered.result.get("params") or {},
                                                          default=list)))
    data = write_record(project, layer, out, export="chains_npz",
                        result=rendered.result).read_bytes()
    assert gprv.check(data) == []
    return json.loads(data)["processStep"][0].get("report", [])


def _report(reports, name):
    return next((r["description"] for r in reports if r["name"] == name), None)


ROI_CORNER = {"n_oct": 2, "n_voice": 2, "precision": "f64", "roi_row": 0, "roi_col": 0,
              "roi_h": 24, "roi_w": 24}


def test_the_record_carries_an_edge_report(clean_registry, tmp_path):
    edge = _report(_record(tmp_path, [("wtmm2d_roi", ROI_CORNER)], "corner"), "edge")
    assert edge["mode"] == "reflect" and edge["eps"] == 0.001
    assert edge["budget_px"] == 64_000_000 and edge["parent_shape"] == [64, 64]
    row = edge["scales"][0]
    assert row["grids"] == [1024, 2048] and row["ratio"] > 0 and row["converged"] is True
    assert row["synthetic_px"]["N"] == row["margin"] and row["real_px"]["N"] == 0
    assert row["whole"] is False and "border" not in row


def test_a_cut_record_names_each_scales_border(clean_registry, tmp_path):
    edge = _report(_record(tmp_path, [("wtmm2d", {"n_oct": 2, "n_voice": 2, "precision": "f64",
                                                  "edge": "cut", "cut_scope": "per_scale"})],
                           "cut"), "edge")
    assert edge["mode"] == "cut" and edge["cut_scope"] == "per_scale"
    assert all(isinstance(r["border"], int) and r["border"] > 0 for r in edge["scales"])


def test_the_default_whole_raster_record_has_no_edge_report(clean_registry, tmp_path):
    reports = _record(tmp_path, [("wtmm2d", {"n_oct": 2, "n_voice": 2, "precision": "f64"})],
                      "plain")
    assert _report(reports, "edge") is None


def test_the_warning_report_reads_the_parent_as_the_run_did(clean_registry, tmp_path):
    """At a_min 0.5 the 64 px parent is read whole: the record's warning report carries the
    lattice warning, not the honest-halo one."""
    reports = _record(tmp_path, [("wtmm2d_roi", dict(ROI_CORNER, a_min=0.5))], "fine")
    edge = _report(reports, "edge")
    assert edge["scales"][0]["whole"] is True
    warned = _report(reports, "warning")["a_min"]
    assert any("η = " in w for w in warned)
    assert not any("does not converge" in w for w in warned)


# -------------------------------------------------- test 6: padded values out of every statistic

def _spiked_reflect_pad(spike):
    from dynamix.roi import halo

    plain = halo._reflect_pad

    def padded(arr, top, bottom, left, right):
        out = plain(arr, top, bottom, left, right)
        if top > 0 and out.dtype.kind == "f":
            out = out.copy()
            out[top - 1, out.shape[1] // 2] += spike       # beside the mirror line
        return out
    return padded


@pytest.mark.parametrize("edge", ["reflect", "cut"])
def test_a_spike_in_the_padded_zone_and_what_it_reaches(tmp_path, monkeypatch, edge):
    from dynamix.roi import halo

    tif = _write_tif(tmp_path / "p.tif", _synthetic_parent(384, seed=7))
    # thresh 0: the threshold is a fraction of the window's own modulus max, padding included, so a
    # spike there moves it everywhere; this test isolates what the convolution carries.
    params = {"n_oct": 2, "n_voice": 4, "a_min": 1.0, "fracint_alpha": 0.0, "thresh": 0.0,
              "edge": edge}
    roi = (0, 96, 64, 64)                     # touches the N edge only
    plain = halo.run_wtmm2d_roi(tif, (384, 384), roi, params)
    monkeypatch.setattr(halo, "_reflect_pad", _spiked_reflect_pad(1e4))
    spiked = halo.run_wtmm2d_roi(tif, (384, 384), roi, params)
    worst, moved = 0.0, 0
    for a, b in zip(plain["extrema"], spiked["extrema"]):
        pa = {(int(y), int(x)): float(m) for y, x, m in zip(a["y"], a["x"], a["mod"])}
        pb = {(int(y), int(x)): float(m) for y, x, m in zip(b["y"], b["x"], b["mod"])}
        moved += len(pa.keys() ^ pb.keys())
        scale_max = max(pa.values(), default=1.0)
        worst = max([worst] + [abs(pa[k] - pb[k]) / scale_max for k in pa.keys() & pb.keys()])
    if edge == "cut":
        # nothing kept moves, and no kept modulus by more than the reach's eps (3.2e-4 measured)
        assert moved == 0 and worst < 1e-3
        for key, value in plain["hd_std"].items():
            assert np.allclose(value, spiked["hd_std"][key], rtol=1e-3, atol=1e-6, equal_nan=True)
    else:
        assert moved > 0                       # the spike reaches the maxima under reflect (318)


def test_a_daf_record_names_its_tuning_or_its_fallback(clean_registry, tmp_path):
    """The test DEM is a random walk, not band-limited: the record says the DAF fell back and
    why, with the band limit it measured."""
    edge = _report(_record(tmp_path, [("wtmm2d", {"n_oct": 2, "n_voice": 2, "precision": "f64",
                                                  "edge": "daf_periodic"})], "daf"), "edge")
    assert edge["requested"] == "daf_periodic" and edge["mode"] == "reflect"
    assert "points per wavelength" in edge["fallback"] and edge["f_b"] > 0 and edge["M"] == 12


# ------------------------------------- the threshold reads real pixels only

@pytest.mark.parametrize("edge", ["cut", "daf_periodic", "gaussian_taper"])
def test_a_padded_spike_does_not_move_the_threshold(tmp_path, monkeypatch, edge):
    """thresh is a fraction of the modulus max over the window's REAL pixels: a spike in the padding
    (or the fill) moves no kept maximum anywhere in the ROI at the default thresh."""
    from dynamix.roi import halo
    from test_edge_taper import _labyrinth

    # the DAF and the taper need a band-limited field to take effect on a 64 px ROI
    v = _synthetic_parent(384, seed=7) if edge == "cut" else _labyrinth(384)
    tif = _write_tif(tmp_path / "p.tif", v)
    params = {"n_oct": 2, "n_voice": 4, "a_min": 1.0, "fracint_alpha": 0.0, "edge": edge}
    roi = (0, 96, 128, 128)
    plain = halo.run_wtmm2d_roi(tif, (384, 384), roi, params)
    assert plain["_edge"]["mode"] == edge
    if edge == "cut":
        monkeypatch.setattr(halo, "_reflect_pad", _spiked_reflect_pad(1e4))
    elif edge == "gaussian_taper":
        from dynamix.core import edges

        taper = edges.tapered

        def spiked_band(block, rows, cols, parent_shape, mu, w_t):
            out = taper(block, rows, cols, parent_shape, mu, w_t)
            i = np.flatnonzero(np.asarray(rows) == 1)          # parent row 1: inside the band
            if i.size:
                out[i[0], out.shape[1] // 2] += 1e4
            return out
        monkeypatch.setattr(edges, "tapered", spiked_band)
    else:
        from dynamix.core import daf

        extend = daf.extend_2d

        def spiked(values, tuned):
            out = extend(values, tuned)
            out[values.shape[0] + tuned.gap // 2, out.shape[1] // 2] += 1e4   # in the filled gap
            return out
        monkeypatch.setattr(daf, "extend_2d", spiked)
    spiked_res = halo.run_wtmm2d_roi(tif, (384, 384), roi, params)
    far = 0
    # Compared: maxima beyond the widest halo (59 px) and off the spike's column (ROI x 64, or 64
    # plus half the DAF gap), where the kernel's axis ringing carries a 1e4 spike a long way.
    def _far(level):
        return (level["y"] >= 64) & ((level["x"] < 56) | (level["x"] > 80))

    for a, b in zip(plain["extrema"], spiked_res["extrema"]):
        keep = _far(a)
        pa = {(int(y), int(x)) for y, x in zip(a["y"][keep], a["x"][keep])}
        keepb = _far(b)
        pb = {(int(y), int(x)) for y, x in zip(b["y"][keepb], b["x"][keepb])}
        assert pa == pb
        far += len(pa)
    assert far > 0
