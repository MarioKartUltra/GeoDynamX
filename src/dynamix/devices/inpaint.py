# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""inpaint and fill -- the inpainting engine as a Reconstruction tool and as a field stage.

``inpaint`` fills the declared regions of the host band (or of ``target_band`` on a stack) from
the band's own cascade, from cycle-spun wavelet sparsity, or both, with co-registered guides
sent to it each on its own grid (``dynamix.core.inpaint``), and reports the error on held-out
blocks and whether the fill kept the field's cascade. ``fill`` is the same engine as a field
stage ahead of an analyzer: the clone carries the fill's stamps so a downstream reading says
that analysis inside the region reads a reconstruction.
"""
from __future__ import annotations

from dynamix.core.dwt import BANKS
from dynamix.model.contracts import COMPONENTS, FIELD, RECEIVED, STACK
from dynamix.model.device import Output
from dynamix.model.param import Param, ParamKind

_SPARSE = ("route", ("sparsity", "both"))
_CASCADE = ("route", ("cascade", "both"))
_THRESHOLD = ("mask_source", ("threshold", "all"))

PARAMS = (
    Param("route", ParamKind.CHOICE, default="sparsity", choices=("sparsity", "cascade", "both"),
          label="Route"),
    Param("wavelet", ParamKind.CHOICE, default="sym7", choices=BANKS, label="Wavelet"),
    Param("level", ParamKind.INT, default=0, min=0, max=10, soft_min=0, soft_max=8,
          label="Levels (0 = the route's depth)"),
    Param("lam_rel", ParamKind.FLOAT, default=0.2, min=0.0, max=10.0, soft_min=0.0, soft_max=1.0,
          label="Lambda (x sigma sqrt 2 ln N)", active_when=_SPARSE),
    Param("n_iter", ParamKind.INT, default=0, min=0, max=5000, soft_min=0, soft_max=500,
          label="Iterations (0 = the route's default)"),
    Param("tol", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0, soft_min=0.0, soft_max=0.01,
          label="Stop: relative change", active_when=_SPARSE),
    Param("stop_var_err", ParamKind.FLOAT, default=0.01, min=0.0, max=1.0, soft_min=0.0,
          soft_max=0.1, label="Stop: variance error", active_when=_CASCADE),
    Param("seed", ParamKind.INT, default=0, min=0, max=2 ** 31 - 1, soft_min=0, soft_max=64,
          label="Seed"),
    Param("batch", ParamKind.FLOAT, default=1.0, min=0.05, max=1.0, soft_min=0.05, soft_max=1.0,
          label="Mini-batch", active_when=_SPARSE),
    Param("cascade_synthesis", ParamKind.BOOL, default=False,
          label="Synthesis pass (beyond the paper)", active_when=_CASCADE),
    Param("target_band", ParamKind.TEXT, default="", label="Target band", editable=True),
    Param("mask_source", ParamKind.CHOICE, default="nodata",
          choices=("nodata", "threshold", "region", "send", "all"), label="Mask"),
    Param("mask_band", ParamKind.TEXT, default="", label="Threshold band", editable=True,
          active_when=_THRESHOLD),
    Param("mask_op", ParamKind.CHOICE, default=">", choices=(">", "<"), label="Threshold",
          active_when=_THRESHOLD),
    Param("mask_value", ParamKind.FLOAT, default=0.0, min=-1e30, max=1e30, soft_min=0.0,
          soft_max=1.0, label="Threshold value", active_when=_THRESHOLD),
    Param("mask_min_px", ParamKind.INT, default=16, min=1, max=10 ** 6, soft_min=1,
          soft_max=1024, label="Smallest region (px)", active_when=_THRESHOLD),
    Param("threshold_kind", ParamKind.CHOICE, default="gap", choices=("gap", "veil"),
          label="Threshold regions are", active_when=_THRESHOLD),
    Param("region_kind", ParamKind.CHOICE, default="gap", choices=("gap", "veil"),
          label="Sent mask regions are", active_when=("mask_source", ("send", "all"))),
    Param("veil_model", ParamKind.CHOICE, default="gain", choices=("gain", "offset"),
          label="Veil model"),
    Param("veil_levels", ParamKind.INT, default=3, min=1, max=8, soft_min=1, soft_max=6,
          label="Veil levels"),
    Param("veil_passes", ParamKind.INT, default=2, min=1, max=8, soft_min=1, soft_max=4,
          label="Veil passes"),
    Param("sun_azimuth", ParamKind.ANGLE, default=0.0, wrap=360.0, label="Sun azimuth"),
    Param("sun_elevation", ParamKind.FLOAT, default=45.0, min=0.0, max=90.0, soft_min=0.0,
          soft_max=90.0, label="Sun elevation"),
    Param("prior_weight", ParamKind.FLOAT, default=0.3, min=0.0, max=10.0, soft_min=0.0,
          soft_max=1.0, label="Value prior weight"),
    Param("structure_weight", ParamKind.FLOAT, default=0.1, min=0.0, max=10.0, soft_min=0.0,
          soft_max=1.0, label="Structure weight"),
    Param("structure_levels", ParamKind.INT, default=3, min=1, max=8, soft_min=1, soft_max=6,
          label="Structure levels"),
    Param("abacus", ParamKind.CHOICE, default="self", choices=("self", "file"), label="Abacus",
          active_when=_CASCADE),
    Param("abacus_file", ParamKind.TEXT, default="", label="Abacus file", editable=True,
          active_when=(_CASCADE, ("abacus", ("file",)))),
    Param("abacus_min_pairs", ParamKind.INT, default=1000, min=100, max=10 ** 6, soft_min=100,
          soft_max=10000, label="Abacus: fewest pairs",
          active_when=(_CASCADE, ("abacus", ("self",)))),
    Param("correct", ParamKind.BOOL, default=True, label="Correct known pixels"),
    Param("heldout_blocks", ParamKind.INT, default=8, min=0, max=256, soft_min=0, soft_max=32,
          label="Held-out blocks"),
    Param("heldout_side", ParamKind.INT, default=16, min=4, max=256, soft_min=4, soft_max=64,
          label="Held-out side (px)"),
    Param("show", ParamKind.CHOICE, default="filled", choices=("filled", "residual", "heldout"),
          label="Show", view=True),
    # Written by the shell: the sends (role, source, weights) and the drawn regions.
    Param("_sends", ParamKind.TEXT, default="[]", label="Sends"),
    Param("_mask_polygons", ParamKind.TEXT, default="[]", label="Drawn regions"),
)
