# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Make divergence from EQSelect a decision, not a discovery.

The analysis core is a COPY. Two copies of a
2,323-line WTMM backend will drift; the mitigation is not to re-couple them but to make drift
visible. Modules copied verbatim are hashed against their originals after reversing the import
rewrite. Modules changed on purpose are listed with their reason and asserted to STILL differ, so
a stale entry fails loudly instead of quietly weakening the check.

Skipped when the EQSelect checkout is absent, so DynamiX still tests standalone.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CORE = REPO / "src" / "dynamix" / "core"
EQSELECT = pathlib.Path("~/projects/slab_earthquake_filter/EQSelect/eqselect").expanduser()

pytestmark = pytest.mark.skipif(
    not EQSELECT.is_dir(), reason=f"EQSelect checkout not present at {EQSELECT}"
)

#: Copied with no edit beyond the import rewrite. These must stay byte-identical once normalised.
VERBATIM = ["selection", "partition", "modality"]

#: Changed on purpose, with the commit that did it. Each must still differ from its original.
INTENTIONALLY_DIVERGED = {
    "projection": "cleanup 2/3 — depth_km became a signed height_km, positive up",
    "frames": "cleanup 2/3 — GeographicFrame.to_scene's z follows projection to height",
    "extrema_sets": "cleanup 1/3 — slab/group became a generic tags dict",
    "rasterfield": "_from_geotiff now stamps provenance[\"crs\"] to match "
                   "from_geotiff_window (CRS gap fix); 2026-08-19: _from_geotiff also frames by "
                   "CRS kind (projected -> LocalFrame in linear units, geographic -> "
                   "GeographicFrame), mirroring from_geotiff_window -- fixes the Vector-tab "
                   "360-unit longitude-fold smear on projected sources (b8cbdfd gap); "
                   "2026-09-22: from_geotiff_window also NaNs |v| >= 3e38 undeclared fill "
                   "(the BOEM headers declare nodata=0 while empty areas hold f32-lowest)",
    "wtmm_backend": "cleanup 3/3 — the reflayers import follows load_topo_extrema to extrema_io; "
                    "unknown-wavelet fall-through now raises ValueError; "
                    "2026-09-14 progressive compute — ComputeCancelled + a cancel= stage-boundary "
                    "check in run_wtmm2d, and run_wtmm2d_preview (finest-scale-only), added; "
                    "2026-09-15 fracint_alpha now applies on the SCALAR path too "
                    "(_apply_fracint2d after the cached cwt stage) — the lift is "
                    "a per-scale positive scalar, identical on |∇W| as on the tensor derivs; "
                    "2026-09-20 interpolate knob (for xsmurf parity) — "
                    "optional subpixel/value refinement of NMS maxima via dynamix.core.subpixel "
                    "inside the extrema stage (key carries it), x_sub/y_sub round-trip through "
                    "the extrema (de)serializers, preview refines identically; default False is "
                    "byte-identical to before",
}

#: Extracted from a module DynamiX does not copy: reflayers.py lines 302-382. ``edits`` lists the
#: ONLY differences permitted, as (ours, theirs) pairs — reversing them must reproduce the original
#: exactly, so a recorded change never becomes a blanket exemption for the whole module.
EXTRACTED = {
    "extrema_io": {
        "origin": ("reflayers", 302, 382),
        "edits": [
            (
                "            segs.append(np.column_stack([s[:, 0], s[:, 1], s[:, 2] / 1000.0]))",
                "            segs.append(np.column_stack([s[:, 0], s[:, 1], -s[:, 2] / 1000.0]))",
            ),
            (
                "            node_pts=np.column_stack([nx[:, 0], nx[:, 1], nx[:, 2] / 1000.0]),  # elev_m -> height_km",
                "            node_pts=np.column_stack([nx[:, 0], nx[:, 1], -nx[:, 2] / 1000.0]),  # elev_m -> depth_km",
            ),
        ],
    }
}

#: New modules created for DynamiX (not copied from EQSelect). Not subject to drift detection.
#: mzlib/fftbackend: verbatim port from research/reconstruction/,
#: not EQSelect -- their own drift guard is tests/test_mzlib_port.py, a sibling of this file.
#: mz_edges: the product-bundle builder over mzlib -- app-facing
#: code written for DynamiX, not a port of anything, so no separate drift guard applies.
#: chain_pick: shell picking math over chain point sets.
#: chain_stats: per-chain Hölder/modulus/length statistics --
#: a REIMPLEMENTATION against documented reference semantics (never wtmm_ebsd's own source),
#: not a copy of anything, so no separate drift guard applies -- see its own module docstring.
#: transect: transect v1 sampling/smoothing/swath-select math
#: -- a REIMPLEMENTATION of EQSelect's own eqselect/transect.py DOCUMENTED algorithm (bilinear
#: sample, NaN-bridge-then-filter-then-restore smoothing), never a copy or import of it (the project rules forbid modifying/importing EQSelect at all) -- same "no separate drift guard" reasoning as
#: chain_stats above; see its own module docstring.
#: hlines: ordered H-line point runs for the 3-D scene (2026-08-28) -- a thin wrapper over the
#: copied wtmm_backend._order_lines walk (the same private-symbol coupling canvas.py and
#: topology/wtmm.py carry), written for DynamiX, not copied from anything.
#: hillshade: shaded-relief display product --
#: written for DynamiX, numpy only, not a copy of anything.
#: stretch: display contrast stretches (2026-08-29) -- numpy only, written for DynamiX.
#: spectra: multifractal spectrum scale-window fitter -- a
#: REIMPLEMENTATION of wtmm_ebsd.partition.fit_hq_Dq_weighted's documented semantics, never a
#: copy or import of it (same reasoning as chain_stats); oracle-pinned in tests/test_spectra.py.
#: frac_bspline: Unser-Blu fractional B-spline evaluators, copied VERBATIM from Creep
#: wtmm/wavelets.py Parts A/A2 -- their own drift guard is
#: tests/test_frac_bspline_port.py (the mzlib pattern: Creep provenance, not EQSelect, so this
#: file's EQSelect machinery does not apply).
#: wavelet_skeleton: the Tang-You constructed wavelet + modulus-minima thinning -- a
#: REIMPLEMENTATION against the corpus papers (tang_you_2003_ribbon_skeleton_wavelet,
#: you_etal_2006_thinning_modulus_minima), never a copy of anything (the chain_stats/spectra
#: license); theorem-pinned in tests/test_wavelet_skeleton.py.
#: cdf: complex cross-diffusion filtering (linear LCDF + nonlinear NCDF), lifted VERBATIM (recorded
#: parameter edits only) from research/reconstruction scripts 24/25/27 (rescued probes
#: 17-20, 2026-08-17) -- its own guard is tests/test_cdf_port.py (the mzlib pattern:
#: research provenance + script 28's functional gates).
#: chain_groups: Boltzmann-weight chain grouping over the partition function's tilted measure (the EBSD-workbook cell-52 port) -- numpy classification written fresh; the partition-table math itself stays in
#: wtmm_ebsd via lazy delegation (the partition2d law), oracle-pinned in tests/test_chain_groups.py.
#: pca / tucker_havok: the common-decomposition tools (PCA + the ASTER Tucker-HAVOK) -- ported in SPIRIT
#: from the author's aster_him.ipynb, rebuilt numpy/BLAS-efficient (covariance-trick PCA; stride-view
#: Hankel + blockwise-Gram HOSVD -- the notebook's materialized-tensor crash class removed);
#: not copies of anything, pinned in tests/test_pca_tucker.py.
#: ingest: multi-sensor grid ingestion (netCDF/HDF5 via rasterio's GDAL, HDF4 via pyhdf,
#: multiband GeoTIFF; 2026-09-21) -- convention-MIRRORS of the frozen rasterfield (never a
#: copy of it), pinned in tests/test_ingest.py.
#: follow2d: the xsmurf follow (kappa zero-crossing) detector ('i still
#: dont see the option to switch between follow and nms') -- gkapa/gkapap formulas ported
#: from xsmurf interpreter/wt2d_cmds.c:2948/3022 (verified byte-identical to upstream
#: pkestene/xsmurf), detection discretization written fresh; pinned in tests/test_follow2d.py.
#: subpixel: parabolic refinement of NMS maxima along the gradient (the
#: "interpolation for xsmurf parity" ask) -- a REIMPLEMENTATION against the documented reference
#: semantics of LastWave-1D's ext_compute.c parabola and xsmurf's follow modulus channel,
#: never a copy of anything (the
#: chain_stats/spectra license); pinned in tests/test_subpixel.py.
#: ssa2d: 2D singular-spectrum analysis -- a REIMPLEMENTATION of Golyandina & Usevich 2010
#: (the corpus paper golyandina_usevich_2010_2d_ssa: Hankel-block-Hankel SVD, elementary
#: reconstructed components, w-correlations), never a copy of anything; pinned against an
#: explicit SVD of that matrix and the paper's rank results in tests/test_ssa2d.py.
#: derivative: derivative datasets -- a result's rasters and/or extrema and maxima lines written
#: once as a RasterField npz (vectors under vec_* keys, packed by wtmm_backend's own stage-cache
#: codecs); the app's own module, pinned in tests/test_derivative.py.
#: frac_bspline_exact: the symmetric fractional B-spline and its derivative in the time domain
#: (Unser & Blu's series, Richardson-closed) -- the app's own module beside the verbatim Creep
#: evaluators, exact at every order; pinned in tests/test_frac_bspline_exact.py.
#: q_fourier: the 2-D Fourier transform of the Tsallis q-Gaussian in closed form (Matern for
#: q > 1, Jahnke-Emde lambda for q < 1), evaluated stably at every order -- log space, the
#: small-argument series and Debye's uniform expansions; the app's own module, pinned against
#: 40-digit mpmath and a numerical Hankel transform in tests/test_q_fourier.py.
#: bus: band buses -- raw bands routed by reference (file + grid id / band index) into one
#: (ny, nx, n) stack on a host grid, same-grid law enforced, windows aligned geometrically
#: with the ROI runner's reflect padding; the app's own module, pinned in tests/test_bus.py.
NEW_MODULES = ["scale_units", "pointset", "mzlib", "fftbackend", "mz_edges", "chain_pick",
              "chain_stats", "transect", "hlines", "hillshade", "stretch", "chain_product",
              "spectra", "microcanonical", "sieve", "frac_bspline", "wavelet_skeleton",
              "cdf", "subpixel", "chain_groups", "follow2d", "pca", "tucker_havok", "ingest", "pm", "xsmurf_follow",
              "fft_policy", "ssa2d", "derivative", "frac_bspline_exact", "q_fourier",
              "bus"]


def _normalised(path: pathlib.Path) -> str:
    """File text with the import rewrite reversed, so a verbatim copy hashes to its original."""
    return path.read_text().replace("dynamix.core.", "eqselect.")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _diff_summary(a: str, b: str, name: str) -> str:
    import difflib

    lines = list(difflib.unified_diff(
        b.splitlines(), a.splitlines(),
        fromfile=f"eqselect/{name}.py", tofile=f"dynamix/core/{name}.py", lineterm="", n=1,
    ))
    head = "\n".join(lines[:40])
    more = f"\n... ({len(lines) - 40} more diff lines)" if len(lines) > 40 else ""
    return head + more


@pytest.mark.parametrize("name", VERBATIM)
def test_verbatim_modules_have_not_drifted(name):
    ours, theirs = CORE / f"{name}.py", EQSELECT / f"{name}.py"
    assert theirs.is_file(), f"missing EQSelect original for {name}"
    a, b = _normalised(ours), theirs.read_text()
    assert _sha(a) == _sha(b), (
        f"{name}.py has drifted from its EQSelect original.\n"
        f"If the change is deliberate, move it into INTENTIONALLY_DIVERGED with a reason.\n\n"
        + _diff_summary(a, b, name)
    )


@pytest.mark.parametrize("name,reason", sorted(INTENTIONALLY_DIVERGED.items()))
def test_intentional_divergence_is_still_present(name, reason):
    """A stale entry is worse than no entry — it exempts a module from drift detection for a change
    that no longer exists."""
    ours, theirs = CORE / f"{name}.py", EQSELECT / f"{name}.py"
    assert theirs.is_file(), f"missing EQSelect original for {name}"
    assert _sha(_normalised(ours)) != _sha(theirs.read_text()), (
        f"{name}.py is listed as intentionally diverged ({reason}) but now matches its EQSelect "
        f"original. Either the change was reverted, or the entry is stale and should be moved "
        f"to VERBATIM."
    )


@pytest.mark.parametrize("name,spec", sorted(EXTRACTED.items()))
def test_extracted_bodies_differ_only_by_recorded_edits(name, spec):
    """extrema_io holds function bodies lifted from a module DynamiX does not copy. Reversing the
    recorded edits must reproduce the original byte-for-byte; anything else is undeclared drift."""
    src_name, first, last = spec["origin"]
    original = (EQSELECT / f"{src_name}.py").read_text().splitlines()[first - 1:last]
    ours = (CORE / f"{name}.py").read_text().splitlines()
    start = next((i for i, ln in enumerate(ours) if ln.startswith("def ")), None)
    assert start is not None, f"no function definitions found in {name}.py"

    reverted = list(ours[start:])
    for mine, theirs in spec["edits"]:
        assert mine in reverted, (
            f"recorded edit is stale — {name}.py no longer contains:\n  {mine}\n"
            f"Either the change was reverted, or the entry needs updating."
        )
        reverted[reverted.index(mine)] = theirs

    assert reverted == original, (
        f"{name}.py differs from {src_name}.py lines {first}-{last} beyond its recorded edits.\n"
        f"Add the new difference to EXTRACTED['{name}']['edits'] if deliberate.\n"
        + _diff_summary("\n".join(reverted), "\n".join(original), name)
    )


def test_every_copied_module_is_accounted_for():
    """No module may sit in core/ unclassified — that is how a copy escapes drift detection."""
    on_disk = {p.stem for p in CORE.glob("*.py") if p.stem != "__init__"}
    classified = set(VERBATIM) | set(INTENTIONALLY_DIVERGED) | set(EXTRACTED) | set(NEW_MODULES)
    assert on_disk == classified, (
        f"unclassified: {sorted(on_disk - classified)}; "
        f"listed but absent: {sorted(classified - on_disk)}"
    )
