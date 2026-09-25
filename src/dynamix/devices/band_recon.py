# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""band_recon — reconstruction from an arbitrary h band, as a Transform.

Runs the SAME h(x) engine as ``holder_map`` (``holder_arrays`` -- estimator, wavelet family,
Turiel scale grid), then takes the singularity set ``h ∈ [h_lo, h_hi)``
(:func:`dynamix.core.microcanonical.band_mask` -- the MSC generalized: the informative
manifold need not be the most singular component, demonstrated on the M-Z house where the
best single band was h ∈ [-0.76, -0.17), NOT the MSC) and reconstructs the field from the
gradient restricted to it (:func:`~dynamix.core.microcanonical.reconstruct_from_msc`, with
the 2026-09-16 symmetric-extension border fix). ``h_lo``/``h_hi`` are ordinary scrubbable
knobs, so sweeping the band and watching the reconstruction IS the workflow.

Display contract: the result's ``"raster_out"`` key (the generalization of ``holder_map``'s
``"h_map"`` -- ``main_window._display_raster_of``) shows the RECONSTRUCTION on the canvas in
place of the field; ``"h_map"``/``"band_mask"`` ride along for the Slice dialog and any
downstream consumer, and ``psnr_db``/``rel_err`` are surfaced as a status reading when the
raster lands. Standalone transform (heads its own chain): it needs the FIELD, and a Transform
placed after another receives only that transform's result -- so it recomputes its own h-map;
at these sizes that is ~a second, and the transform cache keys it all.

Frame note: the band knobs are in whatever frame the chosen estimator produces (multiaffine
gamma, or measure h = gamma - 1) -- the same numbers the holder_map display shows for the
same estimator settings, so read the band off that display or the Slice histogram.
"""
from __future__ import annotations

from dynamix.devices.holder_map import HolderMap, field_values_2d, holder_arrays
from dynamix.devices.holder_methods import HolderMeasure, HolderMultiaffine
from dynamix.model.param import Param, ParamKind


class BandRecon:
    name = "band_recon"
    #: holder_map's engine params verbatim (same names, same defaults -- one mental model),
    #: plus the band. Soft bounds cover the gamma ranges real fields showed
    #: (house/DEMs: roughly -1.5 .. 2.5).
    params = HolderMap.params + (
        Param("h_lo", ParamKind.FLOAT, default=-2.0, min=-6.0, max=6.0,
              soft_min=-1.5, soft_max=1.0, units="", label="h ≥"),
        Param("h_hi", ParamKind.FLOAT, default=0.0, min=-6.0, max=6.0,
              soft_min=-0.5, soft_max=2.0, units="", label="h <"),
        # Island sieve: drop mask components below min_island px (and,
        # when max_island > 0, above it) BEFORE inversion -- reconstruct from coherent
        # structures (dithered fault ribbons), not speck noise. 8-connectivity default keeps
        # diagonal ribbons whole; 4 chops them at every diagonal step.
        Param("min_island", ParamKind.INT, default=0, min=0, max=100000,
              soft_min=0, soft_max=500, units="px", label="Min island"),
        Param("max_island", ParamKind.INT, default=0, min=0, max=10000000,
              soft_min=0, soft_max=100000, units="px", label="Max island"),
        Param("connectivity", ParamKind.CHOICE, default="8", choices=("8", "4"),
              label="Connectivity"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        from dynamix.core import microcanonical as mc

        vals = field_values_2d(field, self.name)
        h_map, r2_map, scales = holder_arrays(vals, params, progress)
        mask = mc.band_mask(h_map, params["h_lo"], params["h_hi"])
        if params["min_island"] or params["max_island"]:
            from dynamix.core.sieve import sieve_mask
            mask = sieve_mask(mask, int(params["min_island"]), int(params["max_island"]),
                              connectivity=int(params["connectivity"]))
        if progress is not None:
            progress("band reconstruction", 0.9)
        recon, psnr, rel_err = mc.reconstruct_from_msc(vals, mask)
        if progress is not None:
            progress("band reconstruction", 1.0)
        return {
            "raster_out": recon, "h_map": h_map, "r2_map": r2_map, "band_mask": mask,
            "psnr_db": psnr, "rel_err": rel_err, "band_density": float(mask.mean()),
            "chains": [], "extrema": [], "scales": scales, "params": dict(params),
            "_frame": getattr(field, "frame", None), "_shape": vals.shape,
        }

    def roi_margin(self, params: dict) -> int:
        from dynamix.devices.holder_methods import holder_roi_margin

        method = "multiaffine" if params["estimator"] == "multiaffine" else "measure"
        return holder_roi_margin(method, params)

    def compute_roi(self, window_field, core, params: dict, *, info=None,
                    progress=None) -> dict:
        return _band_roi(self, holder_arrays, window_field, core, params, progress)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)


# ----------------------------------------------------------------------------------------
# The source split: one band-reconstruction variant per h-map method, each carrying its
# holder sibling's engine params VERBATIM (shared by identity -- the h-map a band is picked
# from is the h-map the reconstruction uses) plus BandRecon's own band knobs. ``BandRecon``
# above is the conflated predecessor: untouched, still registered for saved projects,
# routed to the browser's Dev category alongside holder_map.

#: BandRecon's own knobs -- the band plus the island sieve -- by identity, so the tail can
#: never drift from the superseded device's.
_BAND_KNOBS = BandRecon.params[len(HolderMap.params):]


def _split_band_compute(device, method: str, field, params: dict, progress=None) -> dict:
    """BandRecon.compute over the SPLIT engine (:func:`~dynamix.devices.holder_methods.
    method_arrays` -- method-true wavelet menu, regression/punctual estimator) -- same
    result contract, key for key."""
    from dynamix.core import microcanonical as mc
    from dynamix.devices.holder_methods import method_arrays

    vals = field_values_2d(field, device.name)
    h_map, r2_map, scales = method_arrays(vals, method, params, progress)
    mask = mc.band_mask(h_map, params["h_lo"], params["h_hi"])
    if params["min_island"] or params["max_island"]:
        from dynamix.core.sieve import sieve_mask
        mask = sieve_mask(mask, int(params["min_island"]), int(params["max_island"]),
                          connectivity=int(params["connectivity"]))
    if progress is not None:
        progress("band reconstruction", 0.9)
    recon, psnr, rel_err = mc.reconstruct_from_msc(vals, mask)
    if progress is not None:
        progress("band reconstruction", 1.0)
    return {
        "raster_out": recon, "h_map": h_map, "r2_map": r2_map, "band_mask": mask,
        "psnr_db": psnr, "rel_err": rel_err, "band_density": float(mask.mean()),
        "chains": [], "extrema": [], "scales": scales, "params": dict(params),
        "_frame": getattr(field, "frame", None), "_shape": vals.shape,
    }


def _band_roi(device, arrays, window_field, core, params: dict, progress=None) -> dict:
    """The band reconstruction on a REGION: h(x) over the WINDOW (ROI + the Hölder margin), cut to the
    ROI, then the band mask, the sieve and the reconstruction from the ROI's OWN values."""
    from dynamix.core import microcanonical as mc

    vals = field_values_2d(window_field, device.name)
    h_map, r2_map, scales = arrays(vals, params, progress)
    r0, c0, h, w = (int(v) for v in core)
    sl = (slice(r0, r0 + h), slice(c0, c0 + w))
    h_map, r2_map, roi_vals = h_map[sl], r2_map[sl], vals[sl]
    mask = mc.band_mask(h_map, params["h_lo"], params["h_hi"])
    if params["min_island"] or params["max_island"]:
        from dynamix.core.sieve import sieve_mask
        mask = sieve_mask(mask, int(params["min_island"]), int(params["max_island"]),
                          connectivity=int(params["connectivity"]))
    if progress is not None:
        progress("band reconstruction", 0.9)
    recon, psnr, rel_err = mc.reconstruct_from_msc(roi_vals, mask)
    if progress is not None:
        progress("band reconstruction", 1.0)
    return {
        "raster_out": recon, "h_map": h_map, "r2_map": r2_map, "band_mask": mask,
        "psnr_db": psnr, "rel_err": rel_err, "band_density": float(mask.mean()),
        "chains": [], "extrema": [], "scales": scales, "params": dict(params),
        "_frame": getattr(window_field, "frame", None), "_shape": (h, w),
    }


def _split_arrays(method: str):
    from dynamix.devices.holder_methods import method_arrays

    return lambda vals, params, progress: method_arrays(vals, method, params, progress)


class BandReconMeasure:
    name = "band_recon_measure"
    params = HolderMeasure.params + _BAND_KNOBS

    def roi_margin(self, params: dict) -> int:
        from dynamix.devices.holder_methods import holder_roi_margin

        return holder_roi_margin("measure", params)

    def compute_roi(self, window_field, core, params: dict, *, info=None,
                    progress=None) -> dict:
        return _band_roi(self, _split_arrays("measure"), window_field, core, params, progress)

    def compute(self, field, params: dict, *, progress=None) -> dict:
        return _split_band_compute(self, "measure", field, params, progress)

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        from dynamix.devices.holder_methods import knob_warning

        return knob_warning("measure", name, value, params)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)


class BandReconMultiaffine:
    name = "band_recon_multiaffine"
    params = HolderMultiaffine.params + _BAND_KNOBS

    def roi_margin(self, params: dict) -> int:
        from dynamix.devices.holder_methods import holder_roi_margin

        return holder_roi_margin("multiaffine", params)

    def compute_roi(self, window_field, core, params: dict, *, info=None,
                    progress=None) -> dict:
        return _band_roi(self, _split_arrays("multiaffine"), window_field, core, params,
                         progress)

    def compute(self, field, params: dict, *, progress=None) -> dict:
        return _split_band_compute(self, "multiaffine", field, params, progress)

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        from dynamix.devices.holder_methods import knob_warning

        return knob_warning("multiaffine", name, value, params)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
