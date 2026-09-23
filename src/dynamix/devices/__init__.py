# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Device implementations.

A device is the extension point: one pure Python class declaring its params, plus ``compute`` (a
Transform) or ``apply`` (a Filter). Nothing else. That is deliberately the whole contract, because
"any Python code can be a processing tool" is the point -- there is no plugin SDK to learn and no
licensed toolbox to buy.

Registration is EXPLICIT and never an import side effect. Importing a module must not mutate global
state: it makes test isolation a matter of import order, and it means a device can be registered
twice by two importers and never at all by an application that happens to import neither.
Applications call :func:`register_builtin_devices` once at start-up.
"""
from __future__ import annotations

from dynamix.devices.backproject import Backproject
from dynamix.devices.band_recon import BandRecon, BandReconMeasure, BandReconMultiaffine
from dynamix.devices.decompose import PCADevice, TuckerHavok
from dynamix.devices.chain_classify import ChainClassify
from dynamix.devices.chain_filters import (ChainHolderFilter, ChainLengthFilter,
                                           ChainModulusFilter)
from dynamix.devices.filters import (HLineLength, HLineModulus, ModulusThreshold, OrientationWedge,
                                     ScaleSelect)
from dynamix.devices.groups import GroupFilter, GroupPaint
from dynamix.devices.holder_map import HolderMap
from dynamix.devices.cdf_edges import CDFEdges
from dynamix.devices.holder_methods import HolderMeasure, HolderMultiaffine
from dynamix.devices.noise import Noise
from dynamix.devices.mz_edges import MZEdges
from dynamix.devices.pm_edges import PMEdges
from dynamix.devices.stubs import StubHolder, StubWavelet, StubWedge
from dynamix.devices.topology import ChainTopology, MinVChains
from dynamix.devices.wavelet_skeleton import WaveletSkeleton
from dynamix.devices.wtmm import WTMM2D
from dynamix.devices.wtmm_roi import WTMM2DROI
from dynamix.model.device import DEVICES, register_device

#: Every device this build ships, in registration order.
#:
#: Two filter families, and the split is not arbitrary: `filters` act on EXTREMA (points at one
#: scale), `chain_filters` on CHAINS (maxima lines across scales, where the Hölder exponent lives).
#: `topology` devices act on the incidence graph -- ChainTopology builds it, MinVChains queries it.
#: `groups` devices are the arrangement-view commit transaction's engine half: GroupPaint
#: stamps committed groups' tags onto chain copies, GroupFilter selects by them.
#: The Stub* three are the Phase 1 param-schema gate, retained because tests pin the schema
#: through them.
#: `Backproject` is the point-layer registration transform -- it acts
#: on a `PointSet`, not a raster, but is otherwise an ordinary Transform in this same registry.
#: `MZEdges` is the Mallat-Zhong dyadic transform -- a peer of the
#: wtmm pair, never a mode of it: its own per-scale extrema bundle, dyadic (not geometric) scales,
#: and `chains: []` stamped deliberately since cross-scale M-Z chaining is reserved script 11.
#: `CDFEdges` (2026-09-20) is complex cross-diffusion filtering as an analyzer (linear
#: LCDF + nonlinear NCDF behind the Variant knob -- named for the family) -- one
#: LCDF/NCDF evolution, a dyadic M-Z-convention edge pyramid out (the Ricker-CWT identity
#: makes the evolution a scale space); lifted verbatim from the research scripts.
#: `WaveletSkeleton` (2026-09-19) is the Tang-You/You-2006 medial-axis extractor -- the CWT
#: gradient-modulus machinery with the method's OWN constructed wavelet, modulus MINIMA as
#: intrinsic skeletons; a peer of wtmm2d/mz_edges, never a mode of either.
#: `PMEdges` (2026-09-21) is Perona-Malik 1990 anisotropic diffusion proper (the paper's
#: scheme, real gradient-driven conduction, both g's) -- a peer of cdf_edges, never a mode
#: of it: edge-sharpening backward diffusion under the discrete max principle, so edges
#: stay sharp and in place across the stack (immediate localization, pinned as behavior).
#: `HolderMeasure`/`HolderMultiaffine` (2026-09-19 split) are the per-method microcanonical
#: tools -- one device per method, each with only the wavelet class its method admits.
#: `HolderMap` (the conflated predecessor) STAYS registered: saved projects name it, and the
#: never-delete rule holds -- the browser routes it to the Dev category instead.
BUILTIN_DEVICES = (
    Noise,
    WTMM2D,
    WTMM2DROI,
    MZEdges,
    WaveletSkeleton,
    CDFEdges,
    PMEdges,
    HolderMap,
    HolderMeasure,
    HolderMultiaffine,
    BandRecon,
    BandReconMeasure,
    BandReconMultiaffine,
    PCADevice,
    TuckerHavok,
    ChainTopology,
    MinVChains,
    ScaleSelect,
    OrientationWedge,
    ModulusThreshold,
    HLineLength,
    HLineModulus,
    ChainHolderFilter,
    ChainModulusFilter,
    ChainLengthFilter,
    ChainClassify,
    GroupPaint,
    GroupFilter,
    Backproject,
    StubWavelet,
    StubHolder,
    StubWedge,
)


def register_builtin_devices() -> list[str]:
    """Register every device this build ships. Returns the names newly registered.

    Nothing else under ``src/`` calls ``register_device``, so without this an ordinary
    ``open_project`` raises ``KeyError: no device named ...`` for any project that names a real
    device -- the registry is empty unless the caller populated it by hand.

    Idempotent: a name already present is skipped rather than raising, so a second call (a
    re-opened window, a plugin that also registers) is safe.
    """
    added = []
    for factory in BUILTIN_DEVICES:
        device = factory()
        if device.name in DEVICES:
            continue
        register_device(device)
        added.append(device.name)
    return added
