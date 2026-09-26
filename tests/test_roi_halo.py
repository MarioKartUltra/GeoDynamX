# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""THE ORACLE for the ROI halo engine: a halo read must equal a whole-parent read.

The whole slice rests on one claim: analysing ``ROI + m(a)`` per scale and cropping to the ROI
gives the SAME answer as analysing a window big enough to contain every scale's halo and cropping
to the ROI. These tests are that claim, made falsifiable.

The oracle is built in two layers, because the two layers have genuinely different exactness:

1. **The halo claim proper -- EXACT.** Every per-scale window ``read_halo_window`` produces is
   asserted BIT-IDENTICAL to the corresponding sub-rect of the union window (built here with
   plain numpy, independently of ``dynamix.roi``). Nothing about margins, clamping or reflection
   is approximate; if the halo engine reads the wrong pixels this fails outright.

2. **The backend's response to the two windows -- exact in position, float32 in value.** The two
   windows have different EXTENTS, so ``cwt2d``'s float32 FFT rounds differently on each. Measured
   over both oracles (interior and corner ROI, 8 scales each) at the shipped ``2.5 * a`` margin:
   worst ``2.44e-6 * mod_max`` (corner, a=11.73; interior worst ``1.49e-6``) -- a per-array-max
   quantity (~20 float32 ulps of the largest spectral component), NOT a per-element one. It is
   provably not halo leakage:

   - it does not shrink when the margin is enlarged 7x (measured: flat),
   - it is EXACTLY 0.0 at the coarsest scale, where the per-scale window IS the union window,
   - it scales linearly with the field amplitude, and reproduces identically on the numpy engine.

   Modulus VALUES are therefore compared with ``rtol=1e-6`` plus
   ``atol = FFT_EXTENT_ATOL_FRAC * mod_max``. That atol is the honest float32 statement, pinned
   with ~3x headroom over the measured worst case and asserted to be a max-relative floor, not a
   licence: :func:`test_oracle_residual_is_the_fft_extent_floor_not_leakage` holds the measured
   residual DOWN, so a real leak (which would be orders of magnitude larger) still fails.

3. **Extrema positions: equal up to a small budget of NMS knife-edges per scale, each checked for
   isolation.** ``extrema2d``'s suppression is DIRECTIONAL -- each candidate is compared against
   its two bilinearly-interpolated neighbours along the wavelet gradient angle -- so a ridge point
   sitting almost exactly on the suppression boundary can be won by either of two adjacent pixels
   once ``arg`` moves by a float32 ulp. Measured over a margin sweep from ``1.0*a`` to ``6.28*a``
   and four fixture seeds, the mismatch count is 0 or 1 per scale and does NOT decrease with margin
   -- one seed still ties at ``6.28*a``, 28 sigma of halo. It is not leakage and no margin buys it
   away, so asserting bare equality would be pinning fixture luck.

   The oracle instead allows at most :data:`MAX_NMS_TIES_PER_SCALE` mismatched position per scale
   AND verifies each mismatch is not isolated: the other side must hold an extremum within one
   pixel (same ridge, different sub-pixel winner). That shared-partner check is necessary, not
   sufficient -- it does not by itself prove directional NMS caused the mismatch -- so the BUDGET
   is what actually does the discriminating; see :data:`MAX_NMS_TIES_PER_SCALE` for the measurement
   behind it. A genuine leak moves many points, and moves them off the ridge entirely, so it still
   fails -- verified by mutation at ``0.25/0.5/0.75 * a``.

One fixture subtlety, deliberate and documented: ``thresh`` is a FRACTION of the analysed array's
modulus max, so a window's extent sets its own threshold. The synthetic parent therefore carries
its loudest structure INSIDE the ROI, which makes both sides' thresholds identical and isolates
the halo claim from the (separate, inherent) relative-threshold semantics of the ROI path.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
from rasterio.transform import from_origin

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.core.wtmm_backend import compute_scales2d, get_backend, run_wtmm2d
from dynamix.roi.halo import (MARGIN_PER_SCALE, MIN_A_MIN, MIN_ROI_SIDE, SIGMA_PER_A_MIN,
                              SIGMA_PER_SCALE, _reflect_pad, halo_margin, read_halo_window,
                              run_wtmm2d_roi, sigma_px_amin, sigma_px_scale)

_CRS = "EPSG:32615"
_NODATA = -9999.0

#: 8 scales (2 octaves x 4 voices), a_min = 1.0 -> scales 6.98 .. 23.47 px, margins 18 .. 59 px.
#: Two octaves, not the plan's three: three would need a much larger parent for the coarsest halo
#: to sit inside it, for no extra proof -- the margins already span 3.3x here.
# fracint_alpha=0 pinned, since the lift applies on the scalar path: these are window-assembly
# plumbing oracles (ROI path == single-window reference under IDENTICAL params), and at alpha=1
# with n_voice=4 the lifted cross-scale modulus ratios push most links outside the similitude
# band -- both sides chain to zero and the chain oracle goes vacuous (its own `> 0` guard catches
# exactly this). The lift is orthogonal to what these tests prove
# (the halo path applies the identical `_apply_fracint2d`, covered by tests/test_fracint_scalar.py),
# and several reference sides here call backend.cwt2d directly, which never lifts.
PARAMS = {"n_oct": 2, "n_voice": 4, "a_min": 1.0, "fracint_alpha": 0.0}
SCALES = compute_scales2d(PARAMS["n_oct"], PARAMS["n_voice"], PARAMS["a_min"])
MARGINS = [halo_margin(a) for a in SCALES]

N_PARENT = 384
ROI_INTERIOR = (160, 160, 64, 64)          # every margin real
ROI_CORNER = (0, 0, 64, 64)                # N and W reflected at every scale
ROI_NEAR_EDGE = (24, 24, 64, 64)           # real at the two finest scales, clipped beyond

#: float32-FFT-extent noise floor, as a fraction of that scale's modulus max. Measured worst over
#: both oracles: 2.44e-6. See the module docstring for why this is max-relative, not per-element.
FFT_EXTENT_ATOL_FRAC = 1e-5

#: Directional-NMS knife-edges tolerated per scale, each checked for a same-ridge partner (an
#: extremum within a pixel on the other side) rather than proven to be a suppression tie -- that
#: shared-partner check is necessary, not sufficient, so this budget is what actually does the
#: discriminating, not the check. Measured across >=35 independent fields: a budget of 1
#: spuriously reds ~17% of fields on directional-NMS knife-edge ties alone, while a genuine leak
#: injected at 0.8*a still fails hard at a budget of 2 (the off-ridge displacement and modulus
#: checks carry that discrimination).
MAX_NMS_TIES_PER_SCALE = 2

#: Deviations measured by the oracles, printed at teardown so the report can quote real numbers.
MEASURED: dict = {}


# --------------------------------------------------------------------------------- the fixtures


def _synthetic_parent(n=N_PARENT, seed=7, loud_roi=None):
    """Smooth field + gaussian bumps, deterministic. ``loud_roi`` gets the tallest bumps.

    The loud bumps are what put the modulus max inside the ROI at every scale (module docstring).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    f = 3.0 * np.sin(2 * np.pi * xx / 97.0) * np.cos(2 * np.pi * yy / 71.0)
    for _ in range(20):
        cy, cx = rng.uniform(8, n - 8, 2)
        f += rng.uniform(-8.0, 8.0) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(4.0, 14.0) ** 2))
    if loud_roi is not None:
        r0, c0, h, w = loud_roi
        for _ in range(6):
            cy = rng.uniform(r0 + 10, r0 + h - 10)
            cx = rng.uniform(c0 + 10, c0 + w - 10)
            sign = 1.0 if rng.uniform() > 0.5 else -1.0
            f += sign * rng.uniform(40.0, 80.0) * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(3.0, 8.0) ** 2))
    return (f - f.mean()).astype(np.float32)


def _write_tif(path, values, nodata=None):
    kw = {} if nodata is None else {"nodata": nodata}
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs=_CRS,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0), **kw) as dst:
        dst.write(values, 1)
    return str(path)


def _reference_reflect_pad(arr, top, bottom, left, right):
    """Independent numpy reference for the reflect padding (chunked so pad >= dim works).

    Deliberately a SEPARATE implementation from ``halo._reflect_pad`` -- the oracle's whole point
    is that the halo windows are checked against something that is not themselves. It is only ever
    called on non-degenerate arrays; the size-1 axis is the engine's own test, above.
    """
    out = arr
    rem = [[top, bottom], [left, right]]
    while any(v > 0 for ax in rem for v in ax):
        step = [[min(v, out.shape[i] - 1) for v in rem[i]] for i in (0, 1)]
        out = np.pad(out, ((step[0][0], step[0][1]), (step[1][0], step[1][1])), mode="reflect")
        rem = [[rem[i][j] - step[i][j] for j in (0, 1)] for i in (0, 1)]
    return out


def _window_reference(parent, roi, margin):
    """``roi + margin`` clamped to ``parent`` then reflect-padded -- plain numpy, no dynamix."""
    r0, c0, h, w = roi
    ph, pw = parent.shape
    want_r0, want_c0 = r0 - margin, c0 - margin
    want_h, want_w = h + 2 * margin, w + 2 * margin
    rr0, cc0 = max(0, want_r0), max(0, want_c0)
    rr1, cc1 = min(ph, want_r0 + want_h), min(pw, want_c0 + want_w)
    return _reference_reflect_pad(parent[rr0:rr1, cc0:cc1].astype(np.float64),
                                  rr0 - want_r0, (want_r0 + want_h) - rr1,
                                  cc0 - want_c0, (want_c0 + want_w) - cc1)


def _field(values, name="union"):
    ny, nx = values.shape
    return RasterField(name=name, values=np.asarray(values, dtype=np.float64),
                       frame=LocalFrame(dx=2.0, dy=2.0, units="metre"),
                       x_axis=2.0 * np.arange(nx, dtype=np.float64),
                       y_axis=2.0 * np.arange(ny, dtype=np.float64))


def _crop_extrema(e, off, h, w):
    """Keep extrema inside the ``off``-shifted ROI sub-rect; shift positions onto the ROI grid."""
    keep = ((e["x"] >= off) & (e["x"] < off + w) & (e["y"] >= off) & (e["y"] < off + h))
    return {"x": e["x"][keep] - off, "y": e["y"][keep] - off, "mod": e["mod"][keep],
            "arg": e["arg"][keep], "line_id": e["line_id"][keep]}


@pytest.fixture(scope="module")
def interior(tmp_path_factory):
    """Parent GeoTIFF + the union-window ``run_wtmm2d`` reference, for the interior ROI."""
    return _oracle_case(tmp_path_factory, "interior", ROI_INTERIOR, seed=7)


@pytest.fixture(scope="module")
def corner(tmp_path_factory):
    """Same, for a ROI clipped at the parent's NW corner (mixed real/reflective margins)."""
    return _oracle_case(tmp_path_factory, "corner", ROI_CORNER, seed=11)


def _oracle_case(tmp_path_factory, tag, roi, seed):
    parent = _synthetic_parent(seed=seed, loud_roi=roi)
    src = _write_tif(tmp_path_factory.mktemp(tag) / f"{tag}.tif", parent)
    m_max = max(MARGINS)
    union = _window_reference(parent, roi, m_max)
    return {"parent": parent, "source": src, "roi": roi, "m_max": m_max, "union": union,
            "reference": run_wtmm2d(_field(union, tag), dict(PARAMS)),
            "roi_result": run_wtmm2d_roi(src, parent.shape, roi, dict(PARAMS))}


# ------------------------------------------------------------------- margins: hand-pinned values


def test_sigma_helpers_are_named_for_the_unit_they_take():
    """The 6.977x trap, pinned. ``a_min`` is a multiplier; a ``scale`` is already normalized, and
    the two conversions must NOT be interchangeable without noticing."""
    assert SIGMA_PER_A_MIN == 1.57
    assert SIGMA_PER_SCALE == 0.225
    assert sigma_px_amin(1.0) == pytest.approx(1.57)
    assert sigma_px_amin(2.0) == pytest.approx(3.14)         # the device's derived reading
    assert sigma_px_scale(SCALES[0]) == pytest.approx(1.57, rel=1e-3)  # same sigma, other unit
    # a_min = 1 IS scale 6.977: both routes must agree on the physical width.
    assert sigma_px_scale(SCALES[0]) == pytest.approx(sigma_px_amin(PARAMS["a_min"]), rel=1e-3)


def test_halo_margin_values_pinned_by_hand():
    """``halo_margin`` takes a NORMALIZED scale and uses its own budget constant, not a sigma."""
    assert MARGIN_PER_SCALE == 2.5
    assert halo_margin(1) == 3                # ceil(2.5 * 1)   = ceil(2.5)
    assert halo_margin(8) == 20               # ceil(2.5 * 8)   = 20
    assert halo_margin(6.977) == 18           # ceil(17.4425)  -- the a_min=1 finest scale
    assert halo_margin(23.467) == 59          # ceil(58.6675)  -- its coarsest at 2 octaves
    assert halo_margin(4, margin_per_scale=1.0) == 4
    assert isinstance(halo_margin(3.3), int)
    assert MARGINS == [18, 21, 25, 30, 35, 42, 50, 59]


def test_halo_margin_is_about_eleven_true_sigmas():
    """The budget is empirical, but it should stay recognisably far past the kernel's tail -- if
    someone re-derives it from sigma this catches the units slipping back."""
    for a in SCALES:
        assert halo_margin(a) / sigma_px_scale(a) == pytest.approx(11.1, abs=0.4)


# ------------------------------------------------------------------------ read_halo_window alone


def test_read_halo_window_interior_is_exactly_the_parent_subrect(tmp_path):
    parent = _synthetic_parent(seed=3)
    src = _write_tif(tmp_path / "p.tif", parent)
    roi, margin = ROI_INTERIOR, 40
    r0, c0, h, w = roi

    win, info = read_halo_window(src, parent.shape, roi, margin)

    assert win.shape == (h + 2 * margin, w + 2 * margin)
    assert win.dtype == np.float64
    np.testing.assert_array_equal(
        win, parent[r0 - margin:r0 + h + margin, c0 - margin:c0 + w + margin].astype(np.float64))
    assert info["real_frac"] == 1.0
    assert info["reflected_edges"] == ()
    assert info["offset_in_window"] == (margin, margin)


def test_read_halo_window_corner_reflects_only_the_deficit_edges(tmp_path):
    parent = _synthetic_parent(seed=3)
    src = _write_tif(tmp_path / "p.tif", parent)
    margin = 44

    win, info = read_halo_window(src, parent.shape, ROI_CORNER, margin)

    np.testing.assert_array_equal(win, _window_reference(parent, ROI_CORNER, margin))
    assert set(info["reflected_edges"]) == {"N", "W"}
    real = (64 + margin) ** 2
    assert info["real_frac"] == pytest.approx(real / (64 + 2 * margin) ** 2)


def test_reflect_pad_replicates_a_size_one_axis_instead_of_spinning(tmp_path):
    """A size-1 axis has nothing to mirror, so the capped-pass loop added dim-1 == 0 pixels per
    pass and spun forever (reproduced: it hung the thread). Reflecting one row about itself is
    replicating it -- that is what must come out, promptly."""
    row = np.arange(8.0)[None, :]

    out = _reflect_pad(row, 3, 2, 0, 0)

    assert out.shape == (6, 8)
    for i in range(6):
        np.testing.assert_array_equal(out[i], row[0])

    col = np.arange(5.0)[:, None]
    out = _reflect_pad(col, 0, 0, 4, 1)
    assert out.shape == (5, 6)
    np.testing.assert_array_equal(out, np.repeat(col, 6, axis=1))

    # both axes degenerate at once, and a mixed case (one axis reflects, one replicates)
    assert _reflect_pad(np.array([[7.0]]), 2, 2, 3, 3).shape == (5, 7)
    mixed = _reflect_pad(np.arange(4.0)[None, :], 2, 0, 2, 0)
    assert mixed.shape == (3, 6)
    np.testing.assert_array_equal(mixed[0], np.array([2.0, 1.0, 0.0, 1.0, 2.0, 3.0]))


def test_read_halo_window_margin_wider_than_the_parent_still_reflects(tmp_path):
    """A coarse scale on a small parent asks for more padding than numpy's reflect allows in one
    go. It must still come back full-size rather than raising."""
    parent = _synthetic_parent(n=48, seed=5)
    src = _write_tif(tmp_path / "small.tif", parent)
    roi, margin = (8, 8, 16, 16), 120

    win, info = read_halo_window(src, parent.shape, roi, margin)

    assert win.shape == (16 + 240, 16 + 240)
    np.testing.assert_array_equal(win, _window_reference(parent, roi, margin))
    assert set(info["reflected_edges"]) == {"N", "S", "E", "W"}
    assert info["real_frac"] < 0.05


# ------------------------------------------------------------------------------------ THE ORACLE


def _compare_extrema(got, want, mod_max, where):
    """Positions equal bar a budget of NMS knife-edges, each verified not isolated; moduli equal
    at the float32 floor.

    Returns ``(n_ties, max_abs_dev)``. Every mismatched position must have a partner within one
    pixel on the other side -- both sides found the same ridge, they disagree only on which of two
    touching pixels wins the directional suppression. That check is necessary, not sufficient (it
    verifies the mismatch is not isolated; it does not itself prove directional NMS caused it), so
    :data:`MAX_NMS_TIES_PER_SCALE` is what actually does the discriminating. See the module
    docstring, point 3.
    """
    pos_got = {(int(y), int(x)) for y, x in zip(got["y"], got["x"])}
    pos_want = {(int(y), int(x)) for y, x in zip(want["y"], want["x"])}
    ties = pos_got ^ pos_want
    assert len(ties) <= MAX_NMS_TIES_PER_SCALE, (
        f"{where}: {len(ties)} extrema differ, more than the {MAX_NMS_TIES_PER_SCALE} "
        f"directional-NMS knife-edge the float32 floor can explain: {sorted(ties)[:8]}")
    for (y, x) in ties:
        other = pos_want if (y, x) in pos_got else pos_got
        assert any((y + dy, x + dx) in other for dy in (-1, 0, 1) for dx in (-1, 0, 1)), (
            f"{where}: extremum at {(y, x)} has no counterpart within a pixel on the other side "
            "-- that is a displaced ridge, not a suppression tie")

    shared = sorted(pos_got & pos_want)
    mod_got = {(int(y), int(x)): m for y, x, m in zip(got["y"], got["x"], got["mod"])}
    mod_want = {(int(y), int(x)): m for y, x, m in zip(want["y"], want["x"], want["mod"])}
    a_vals = np.array([mod_got[p] for p in shared])
    b_vals = np.array([mod_want[p] for p in shared])
    np.testing.assert_allclose(a_vals, b_vals, rtol=1e-6, atol=FFT_EXTENT_ATOL_FRAC * mod_max,
                               err_msg=f"{where}: modulus")
    dev = float(np.abs(a_vals - b_vals).max()) if a_vals.size else 0.0
    return len(ties), dev


def _assert_oracle(case, tag):
    """Halo windows bit-identical to the union slice; extrema equal in position (bar a budget of
    knife-edges, each verified not isolated) and to the float32 floor in value."""
    parent, roi, m_max = case["parent"], case["roi"], case["m_max"]
    r0, c0, h, w = roi
    ref_ext = case["reference"]["extrema"]
    roi_ext = case["roi_result"]["extrema"]
    assert len(roi_ext) == len(ref_ext) == len(SCALES)

    rows = []
    for i, (a, m) in enumerate(zip(SCALES, MARGINS)):
        # --- layer 1: the halo claim proper, EXACT ------------------------------------------
        win, _ = read_halo_window(case["source"], parent.shape, roi, m)
        slice_of_union = case["union"][m_max - m:m_max - m + h + 2 * m,
                                       m_max - m:m_max - m + w + 2 * m]
        np.testing.assert_array_equal(
            win, slice_of_union,
            err_msg=f"{tag}: halo window at a={a:.3f} is not the union window's sub-rect")

        # --- layer 2: the backend's response ------------------------------------------------
        got = roi_ext[i]
        want = _crop_extrema(ref_ext[i], m_max, h, w)
        mod_max = float(np.abs(want["mod"]).max())
        n_ties, dev = _compare_extrema(got, want, mod_max, f"{tag} a={a:.3f}")
        rows.append({"a": float(a), "margin": m, "n_extrema": int(want["x"].size),
                     "n_ties": n_ties, "max_abs_dev": dev, "frac_of_mod_max": dev / mod_max})
    MEASURED[tag] = rows
    return rows


def test_oracle_interior_roi_equals_the_union_window_crop(interior):
    rows = _assert_oracle(interior, "interior")
    assert max(r["frac_of_mod_max"] for r in rows) < FFT_EXTENT_ATOL_FRAC
    assert rows[-1]["max_abs_dev"] == 0.0      # coarsest halo IS the union window: bit-identical


def test_oracle_corner_roi_equals_the_reflect_padded_union_window(corner):
    rows = _assert_oracle(corner, "corner")
    assert max(r["frac_of_mod_max"] for r in rows) < FFT_EXTENT_ATOL_FRAC
    assert rows[-1]["max_abs_dev"] == 0.0


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_oracle_corner_holds_on_fields_the_fixtures_did_not_choose(seed, tmp_path):
    """The two headline oracles use one seed each. Exactness that only holds for a chosen field is
    not exactness -- so run the harder (corner, mixed real/reflective) case on fields nobody tuned.

    Seed 43 is here on purpose: it produces one directional-NMS knife-edge, and it produces it at
    EVERY margin from 1.0*a to 6.28*a. That is part of the measurement behind
    :data:`MAX_NMS_TIES_PER_SCALE` -- the tie is not something a wider halo buys away, so the
    oracle verifies it is not isolated instead of pretending it is absent.
    """
    parent = _synthetic_parent(seed=seed, loud_roi=ROI_CORNER)
    src = _write_tif(tmp_path / f"corner{seed}.tif", parent)
    r0, c0, h, w = ROI_CORNER
    m_max = max(MARGINS)
    union = _window_reference(parent, ROI_CORNER, m_max)
    backend = get_backend("python")
    cwt = backend.cwt2d(union, SCALES, wavelet="gaussian", pad=32)
    ref = backend.extrema2d(cwt, SCALES, thresh=1e-3, field=union)

    result = run_wtmm2d_roi(src, parent.shape, ROI_CORNER, dict(PARAMS))

    total_ties = 0
    for i, a in enumerate(SCALES):
        want = _crop_extrema(ref[i], m_max, h, w)
        mod_max = float(np.abs(want["mod"]).max())
        n_ties, dev = _compare_extrema(result["extrema"][i], want, mod_max,
                                       f"seed{seed} a={a:.3f}")
        assert dev < FFT_EXTENT_ATOL_FRAC * mod_max
        total_ties += n_ties
    assert total_ties <= MAX_NMS_TIES_PER_SCALE * len(SCALES)


@pytest.mark.parametrize("case_name", ["interior", "corner"])
def test_oracle_chains_match_chain_for_chain(case_name, request):
    """Same cropped extrema on both sides must chain to the same chains. This is the plumbing
    proof: the ROI path hands ``chains2d`` a per-scale list assembled from many windows, the
    reference hands it one assembled from a single window, and the chains come out equal.

    Chain-for-chain equality is asserted EXACTLY, so unlike the extrema comparison it inherits no
    tie allowance: both committed fixtures have zero knife-edges at the shipped margin. If a
    future change introduces one, this is where it will surface first -- read the extrema oracle's
    ``n_ties`` before assuming a plumbing bug.
    """
    case = request.getfixturevalue(case_name)
    r0, c0, h, w = case["roi"]
    backend = get_backend("python")
    resolved = case["roi_result"]["params"]
    kw = dict(similitude=resolved["similitude"], box_ratio=resolved["box_ratio"],
              dist2_max=resolved["dist2_max"], min_len=resolved["min_chain_len"],
              smooth=resolved["smooth"])
    ref_cropped = [_crop_extrema(e, case["m_max"], h, w) for e in case["reference"]["extrema"]]

    got = backend.chains2d(case["roi_result"]["extrema"], SCALES, **kw)
    want = backend.chains2d(ref_cropped, SCALES, **kw)

    assert len(got) == len(want) > 0
    for k, (g, wch) in enumerate(zip(got, want)):
        np.testing.assert_array_equal(g["y"], wch["y"], err_msg=f"{case_name}: chain {k} y")
        np.testing.assert_array_equal(g["x"], wch["x"], err_msg=f"{case_name}: chain {k} x")
    # and the engine's own chains, built inside run_wtmm2d_roi, are those chains
    assert len(case["roi_result"]["chains"]) == len(got)


def test_oracle_residual_is_the_fft_extent_floor_not_leakage(interior):
    """Hold the residual DOWN. Halo leakage -- reading too few real pixels -- shows up as a
    deviation orders of magnitude above the float32 floor, so this bound is what makes the
    ``atol`` in the oracle a measurement rather than a licence."""
    rows = MEASURED.get("interior") or _assert_oracle(interior, "interior")
    for r in rows:
        assert r["frac_of_mod_max"] < 5e-6, r


# ------------------------------------------------------------------------- margin bookkeeping


def test_interior_roi_reports_every_margin_real(interior):
    margins = interior["roi_result"]["_roi_margins"]
    assert len(margins) == len(SCALES)
    for rec, a, m in zip(margins, SCALES, MARGINS):
        assert rec["a"] == pytest.approx(float(a))
        assert rec["margin"] == m
        assert rec["real_frac"] == 1.0
        assert rec["reflected_edges"] == ()
    assert interior["roi_result"]["_coi_radii"] == MARGINS


def test_corner_roi_names_the_reflected_edges_and_loses_real_fraction(corner):
    margins = corner["roi_result"]["_roi_margins"]
    for rec in margins:
        assert rec["real_frac"] < 1.0
        assert set(rec["reflected_edges"]) == {"N", "W"}
    # coarser scale -> more of the halo falls outside the parent -> less real data
    fracs = [rec["real_frac"] for rec in margins]
    assert fracs == sorted(fracs, reverse=True)
    assert fracs[-1] < fracs[0]


def test_near_edge_roi_is_all_real_at_fine_scales_and_clipped_at_coarse_ones(tmp_path):
    parent = _synthetic_parent(seed=13, loud_roi=ROI_NEAR_EDGE)
    src = _write_tif(tmp_path / "edge.tif", parent)

    result = run_wtmm2d_roi(src, parent.shape, ROI_NEAR_EDGE, dict(PARAMS))

    recs = result["_roi_margins"]
    # margins 18 and 21 fit above/left of row/col 24; 25 onwards do not.
    assert [r["margin"] for r in recs] == MARGINS
    assert recs[0]["real_frac"] == 1.0 and recs[0]["reflected_edges"] == ()
    assert recs[1]["real_frac"] == 1.0
    assert recs[2]["real_frac"] < 1.0
    assert set(recs[2]["reflected_edges"]) == {"N", "W"}
    assert recs[-1]["real_frac"] < recs[2]["real_frac"]


# --------------------------------------------------------------------------------- missing data


def test_nodata_hole_is_zero_filled_before_the_transform_and_recorded(tmp_path):
    parent = _synthetic_parent(seed=17, loud_roi=ROI_INTERIOR)
    r0, c0, h, w = ROI_INTERIOR
    hole = (slice(r0 + 10, r0 + 22), slice(c0 + 30, c0 + 47))
    parent[hole] = _NODATA
    src = _write_tif(tmp_path / "holed.tif", parent, nodata=_NODATA)

    win, info = read_halo_window(src, parent.shape, ROI_INTERIOR, MARGINS[0])

    assert np.isfinite(win).all(), "nodata must be zero-filled, never left as NaN"
    assert not (win == _NODATA).any(), "the sentinel itself must not survive into the transform"
    m = MARGINS[0]
    filled = win[m + 10:m + 22, m + 30:m + 47]
    np.testing.assert_array_equal(filled, np.zeros_like(filled))

    result = run_wtmm2d_roi(src, parent.shape, ROI_INTERIOR, dict(PARAMS))

    mask = result["_missing_mask"]
    assert mask.shape == (h, w) and mask.dtype == np.bool_
    expected = np.zeros((h, w), dtype=bool)
    expected[10:22, 30:47] = True
    np.testing.assert_array_equal(mask, expected)
    assert result["_shape"] == (h, w)
    assert len(result["chains"]) > 0                 # the compute completed, holes and all


def test_missing_mask_is_empty_when_the_source_declares_no_nodata(interior):
    assert not interior["roi_result"]["_missing_mask"].any()


def test_real_frac_does_not_count_zero_filled_nodata_as_real(tmp_path):
    """``real_frac`` is what the strip's honesty reading renders. A zero-filled hole is as
    fabricated as a reflection, so it must not be counted as genuine parent data -- even when the
    hole sits in the HALO and the ROI itself is clean."""
    parent = _synthetic_parent(seed=29, loud_roi=ROI_INTERIOR)
    r0, c0, h, w = ROI_INTERIOR
    margin = MARGINS[-1]
    n_hole_rows, n_hole_cols = 20, 25
    parent[r0 - margin + 2:r0 - margin + 2 + n_hole_rows,
           c0 - margin + 2:c0 - margin + 2 + n_hole_cols] = _NODATA   # entirely in the halo
    src = _write_tif(tmp_path / "halo_hole.tif", parent, nodata=_NODATA)

    win, info = read_halo_window(src, parent.shape, ROI_INTERIOR, margin)

    assert info["reflected_edges"] == ()            # nothing reflected: the loss is all nodata
    assert info["real_frac"] < 1.0
    window_cells = (h + 2 * margin) * (w + 2 * margin)
    assert info["real_frac"] == pytest.approx(
        (window_cells - n_hole_rows * n_hole_cols) / window_cells)
    assert np.isfinite(win).all()

    result = run_wtmm2d_roi(src, parent.shape, ROI_INTERIOR, dict(PARAMS))
    assert not result["_missing_mask"].any()        # the hole never entered the ROI itself
    assert result["_roi_margins"][-1]["real_frac"] < 1.0


def test_nodata_override_is_honoured_over_the_files_own_declaration(tmp_path):
    """A raster whose sentinel is undeclared (or wrong) still has to be maskable by the caller."""
    parent = _synthetic_parent(seed=31, loud_roi=ROI_INTERIOR)
    r0, c0, h, w = ROI_INTERIOR
    parent[r0 + 5:r0 + 9, c0 + 5:c0 + 11] = _NODATA
    src = _write_tif(tmp_path / "undeclared.tif", parent)        # NO nodata written to the file
    margin = MARGINS[0]

    bare, bare_info = read_halo_window(src, parent.shape, ROI_INTERIOR, margin)
    assert bare_info["real_frac"] == 1.0                          # nothing known to be missing
    assert (bare == _NODATA).any()                                # the sentinel survives

    win, info = read_halo_window(src, parent.shape, ROI_INTERIOR, margin, nodata=_NODATA)

    assert not (win == _NODATA).any()
    assert info["real_frac"] < 1.0
    np.testing.assert_array_equal(win[margin + 5:margin + 9, margin + 5:margin + 11],
                                  np.zeros((4, 6)))

    result = run_wtmm2d_roi(src, parent.shape, ROI_INTERIOR, dict(PARAMS), nodata=_NODATA)
    expected = np.zeros((h, w), dtype=bool)
    expected[5:9, 5:11] = True
    np.testing.assert_array_equal(result["_missing_mask"], expected)


# ------------------------------------------------------------------------- the result contract


def test_result_carries_the_canonical_keys_plus_the_roi_extras(interior):
    result = interior["roi_result"]
    for key in ("chains", "extrema", "scales", "hd_std", "hd_cmax", "npz_path", "params",
                "cache_hits"):
        assert key in result, key
    assert result["npz_path"] is None
    assert result["cache_hits"] == set()
    np.testing.assert_allclose(result["scales"], SCALES)
    assert set(result["params"]) == set(interior["reference"]["params"])
    assert result["_roi"] == {"source": interior["source"], "roi": ROI_INTERIOR,
                              "boundary": "auto"}
    assert result["_shape"] == ROI_INTERIOR[2:]
    assert result["_coi_radii"] == MARGINS


def test_unknown_param_key_is_rejected_by_the_shared_typo_guard(interior):
    with pytest.raises(ValueError, match="n_octaves"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), ROI_INTERIOR, {"n_octaves": 2})


def test_sliver_roi_is_refused_at_entry_naming_its_sides(interior):
    """The drag gesture will produce slivers by accident. A 1-px-tall ROI in reflective mode used
    to reach ``_reflect_pad``'s degenerate axis and HANG; it must be refused before any read."""
    assert MIN_ROI_SIDE == 8
    for roi in ((160, 160, 1, 64), (160, 160, 64, 3)):
        with pytest.raises(ValueError, match=r"at least 8 px"):
            run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), roi, dict(PARAMS),
                           boundary="reflective")
    with pytest.raises(ValueError, match=r"1x64"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), (160, 160, 1, 64), dict(PARAMS))
    # the guard is at entry, so it does not depend on the boundary mode or reach the backend
    with pytest.raises(ValueError, match=r"at least 8 px"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), (0, 0, 8, 7), dict(PARAMS))


def test_roi_outside_the_parent_is_refused_at_entry_naming_the_raster_dims(interior):
    """A too-large row/col from the strip knobs (whose Params allow up to 1_000_000, far past
    any real raster -- ``dynamix/devices/wtmm_roi.py``'s ``roi_row``/``roi_col`` Params) would
    otherwise reach ``read_halo_window``'s ``rasterio`` window read and die on ITS internal
    message ("Number of columns or rows must be non-negative") once the clamped read region goes
    empty -- confusing for a user who never sees rasterio directly. Refused here, at
    entry, naming the roi and the raster it does not fit, before any read happens."""
    with pytest.raises(ValueError, match=str(N_PARENT)):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), (500_000, 500_000, 512, 512),
                       dict(PARAMS))
    # a roi that starts in-bounds but overruns the parent along one side must be caught too, not
    # just the "entirely outside" case that happens to crash rasterio.
    with pytest.raises(ValueError, match=r"does not fit"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), (N_PARENT - 10, 0, 64, 64),
                       dict(PARAMS))
    # the guard is at entry, so it does not depend on the boundary mode or reach rasterio
    with pytest.raises(ValueError, match=r"does not fit"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), (500_000, 500_000, 512, 512),
                       dict(PARAMS), boundary="reflective")


def test_a_min_below_one_is_refused_for_non_locality(interior):
    """Below a_min = 1 the sampled kernel is near all-pass at Nyquist, so its influence reaches
    hundreds of pixels and no affordable halo makes the ROI honest. Refuse rather than return
    contaminated numbers quietly."""
    assert MIN_A_MIN == 1.0
    with pytest.raises(ValueError, match=r"a_min"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), ROI_INTERIOR,
                       dict(PARAMS, a_min=0.5))
    with pytest.raises(ValueError, match=r"all-pass|Nyquist"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), ROI_INTERIOR,
                       dict(PARAMS, a_min=0.25))


def test_unknown_boundary_mode_is_rejected(interior):
    with pytest.raises(ValueError, match="boundary"):
        run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), ROI_INTERIOR, dict(PARAMS),
                       boundary="wrap")


def test_progress_reports_once_per_scale_at_roi_granularity(interior):
    seen = []
    run_wtmm2d_roi(interior["source"], (N_PARENT, N_PARENT), ROI_INTERIOR, dict(PARAMS),
                   progress=lambda stage, frac: seen.append((stage, frac)))

    n = len(SCALES)
    assert seen[0] == (f"roi scale 1/{n}", 0.0)
    assert seen[1] == (f"roi scale 1/{n}", 1.0)
    assert seen[-1] == (f"roi scale {n}/{n}", 1.0)
    assert len([s for s, _ in seen if s == f"roi scale 1/{n}"]) == 2


# ------------------------------------------------------------------------------ reflective mode


def test_reflective_boundary_ignores_the_parent_outside_the_roi(tmp_path):
    """``boundary="reflective"`` must give the SAME answer whatever surrounds the ROI -- it reads
    the bare ROI and reflects it. Two parents that share the ROI but differ everywhere else are
    the test."""
    roi = ROI_INTERIOR
    r0, c0, h, w = roi
    a = _synthetic_parent(seed=21, loud_roi=roi)
    b = _synthetic_parent(seed=22, loud_roi=roi)
    b[r0:r0 + h, c0:c0 + w] = a[r0:r0 + h, c0:c0 + w]     # identical ROI, different surroundings
    src_a = _write_tif(tmp_path / "a.tif", a)
    src_b = _write_tif(tmp_path / "b.tif", b)

    ra = run_wtmm2d_roi(src_a, a.shape, roi, dict(PARAMS), boundary="reflective")
    rb = run_wtmm2d_roi(src_b, b.shape, roi, dict(PARAMS), boundary="reflective")

    for i in range(len(SCALES)):
        np.testing.assert_array_equal(ra["extrema"][i]["x"], rb["extrema"][i]["x"])
        np.testing.assert_array_equal(ra["extrema"][i]["y"], rb["extrema"][i]["y"])
        np.testing.assert_array_equal(ra["extrema"][i]["mod"], rb["extrema"][i]["mod"])
    assert all(rec["real_frac"] < 1.0 for rec in ra["_roi_margins"])
    assert ra["_roi"]["boundary"] == "reflective"
    # and it is NOT the auto answer -- reflective throws real halo data away
    auto = run_wtmm2d_roi(src_a, a.shape, roi, dict(PARAMS), boundary="auto")
    assert any(not np.array_equal(auto["extrema"][i]["x"], ra["extrema"][i]["x"])
               for i in range(len(SCALES)))


def test_reflective_window_equals_reflect_padding_the_bare_roi(tmp_path):
    parent = _synthetic_parent(seed=23)
    src = _write_tif(tmp_path / "p.tif", parent)
    r0, c0, h, w = ROI_INTERIOR
    margin = 30

    # the same function against a parent view restricted to the ROI
    win, info = read_halo_window(src, (h, w), (0, 0, h, w), margin)
    bare = parent[r0:r0 + h, c0:c0 + w].astype(np.float64)

    assert win.shape == (h + 2 * margin, w + 2 * margin)
    # (that view starts at the file's origin; the ROI-positioned read is what run_wtmm2d_roi does)
    np.testing.assert_array_equal(
        win, _reference_reflect_pad(parent[0:h, 0:w].astype(np.float64),
                                    margin, margin, margin, margin))
    assert set(info["reflected_edges"]) == {"N", "S", "E", "W"}
    assert bare.shape == (h, w)


# ------------------------------------------------------------------------------------- reporting


def test_zzz_report_measured_deviations(interior, corner):
    """Not an assertion -- the oracle's measured numbers, printed for the task report."""
    if "interior" not in MEASURED:
        _assert_oracle(interior, "interior")
    if "corner" not in MEASURED:
        _assert_oracle(corner, "corner")
    print("\n--- oracle measured max |mod| deviation, ROI extrema, per scale ---")
    for tag, rows in MEASURED.items():
        print(f"  [{tag}]")
        for r in rows:
            print(f"    a={r['a']:7.3f} margin={r['margin']:4d} n={r['n_extrema']:4d} "
                  f"ties={r['n_ties']} max_abs={r['max_abs_dev']:.3e} "
                  f"frac_of_mod_max={r['frac_of_mod_max']:.3e}")


# ------------------------------------------------------------------------------ the subpixel knob


def test_roi_interpolate_knob_carries_the_float_channels(interior):
    """The subpixel refinement runs on the halo window BEFORE the crop (the parabola's probes
    need halo pixels), the float channels shift with the crop, and the integer support is
    untouched -- membership stays integer-decided, so a kept point's float position may sit up
    to half a pixel outside the ROI edge and no further."""
    on = run_wtmm2d_roi(interior["source"], interior["parent"].shape, interior["roi"],
                        dict(PARAMS, interpolate=True))
    off = interior["roi_result"]
    r0, c0, h, w = interior["roi"]
    for e_on, e_off in zip(on["extrema"], off["extrema"]):
        np.testing.assert_array_equal(e_on["x"], e_off["x"])
        np.testing.assert_array_equal(e_on["y"], e_off["y"])
        assert np.all(e_on["mod"] >= e_off["mod"] - 1e-12)
        assert e_on["x_sub"].shape == e_on["x"].shape
        assert np.all(e_on["x_sub"] >= -0.5) and np.all(e_on["x_sub"] <= w - 0.5)
        assert np.all(e_on["y_sub"] >= -0.5) and np.all(e_on["y_sub"] <= h - 0.5)


def test_roi_follow_detector_produces_the_native_channels(interior):
    """detector='follow' on the ROI path: detection on the halo window, crop shifts the float
    channels, integer support inside the ROI grid."""
    res = run_wtmm2d_roi(interior["source"], interior["parent"].shape, interior["roi"],
                         dict(PARAMS, detector="follow"))
    r0, c0, h, w = interior["roi"]
    assert any(e["x"].size for e in res["extrema"])
    for e in res["extrema"]:
        assert "x_sub" in e and e["x_sub"].shape == e["x"].shape
        if e["x"].size:
            assert e["x"].min() >= 0 and e["x"].max() < w
            assert np.all(np.abs(e["x_sub"] - e["x"]) <= 0.5 + 1e-9)


def test_boem_style_undeclared_sentinel_is_missing_not_data(tmp_path):
    """The BOEM tifs DECLARE nodata = 0.0 but FILL with float32-lowest. The two other read
    paths NaN |v| >= 3e38 at read; a halo reader that treats only the declared value as missing
    lets -3.4e38 into the wavelet transform near the coast. It is missing data: zero-filled
    before the transform and reported, like any nodata."""
    import rasterio
    from rasterio.transform import from_origin

    vals = np.full((40, 40), -25.0, dtype=np.float32)
    vals[:6, :] = np.float32(-3.4028235e38)
    path = tmp_path / "boem_like.tif"
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=1,
                       dtype="float32", crs="EPSG:32615", nodata=0.0,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(vals, 1)
    win, info = read_halo_window(str(path), (40, 40), (8, 8, 16, 16), 4)
    assert np.abs(win).max() < 1e30                   # no sentinel reaches the transform
    assert info["missing"][:2].all()                  # rows 4..5 of the file: the fill
    assert not info["missing"][2:].any()
