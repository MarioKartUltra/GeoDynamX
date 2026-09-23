# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.units -- CRS-linear-unit conversion, derived readings, and the
``aₘᵢₙ`` relabel ("Model and
interaction").

The bug this closes: ``MainWindow._px_to_unit`` used to read a pixel's size straight off the
field's x-axis with no unit conversion -- correct for a metre CRS, silently wrong for BOEM's
NAD27 grid, whose cells are 40 US-survey-FEET, not metres ("Readings and display").

Real WKT text below is not invented -- it is what ``rasterio.crs.CRS`` actually produces (checked
while writing this module): ``str(crs)`` is the FULL WKT whenever the CRS has no exact EPSG match
(BOEM's custom NAD27 grids fall in this bucket) and a bare ``"EPSG:<code>"`` when it does; the
latter is exercised too (``test_a_bare_short_epsg_code_...``), since it carries no unit text to
parse at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import GeographicFrame, LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.shell.units import linear_unit_to_m, px_to_metres

# rasterio.crs.CRS.from_epsg(32040).to_wkt(version="WKT1_GDAL") -- NAD27 / Texas South Central,
# US survey foot. A real BOEM-style custom grid falls in the same "no exact EPSG match" bucket, so
# `str(crs)` for one of those is exactly this shape.
_SURVEY_FOOT_WKT = (
    'PROJCS["NAD27 / Texas South Central",GEOGCS["NAD27",DATUM["North_American_Datum_1927",'
    'SPHEROID["Clarke 1866",6378206.4,294.978698213898,AUTHORITY["EPSG","7008"]],'
    'AUTHORITY["EPSG","6267"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4267"]],'
    'PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",27.8333333333333],'
    'PARAMETER["central_meridian",-99],PARAMETER["standard_parallel_1",28.3833333333333],'
    'PARAMETER["standard_parallel_2",30.2833333333333],PARAMETER["false_easting",2000000],'
    'PARAMETER["false_northing",0],UNIT["US survey foot",0.304800609601219,'
    'AUTHORITY["EPSG","9003"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'AUTHORITY["EPSG","32040"]]'
)

# rasterio.crs.CRS.from_epsg(32615).to_wkt(version="WKT1_GDAL") -- WGS 84 / UTM zone 15N, metre.
_METRE_WKT = (
    'PROJCS["WGS 84 / UTM zone 15N",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",-93],PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'AUTHORITY["EPSG","32615"]]'
)

# rasterio.crs.CRS.from_epsg(4326).to_wkt(version="WKT1_GDAL") -- a bare geographic CRS.
_GEOGCS_WKT1 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
)

# rasterio.crs.CRS.from_epsg(4326).to_wkt(version="WKT2_2019") -- same CRS, WKT2 dialect. Its
# ELLIPSOID carries its own LENGTHUNIT["metre",1] for the datum's radius: a naive "the last
# UNIT[]/LENGTHUNIT[] wins" parse would mistake that for a 1.0 m/m linear CRS unit. It must not.
_GEOGCRS_WKT2 = (
    'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",'
    'MEMBER["World Geodetic System 1984 (Transit)"],'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],'
    'AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],'
    'AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],'
    'ID["EPSG",4326]]'
)


def _field(dx, *, units, provenance=None):
    """A minimal ``RasterField`` with a ``dx``-pixel ``LocalFrame`` -- enough for
    ``px_to_metres``, which only ever reads ``.frame.units``, ``.x_axis`` and ``.provenance``."""
    n = 4
    frame = LocalFrame(x0=0.0, y0=0.0, dx=dx, dy=dx, units=units)
    axis = np.arange(n, dtype=float) * dx
    return RasterField(name="f", values=np.zeros((n, n)), frame=frame, x_axis=axis, y_axis=axis,
                       provenance=provenance or {})


# --------------------------------------------------------------------------- linear_unit_to_m


def test_survey_foot_factor_is_pinned_to_12_decimals():
    factor, unit = linear_unit_to_m(_SURVEY_FOOT_WKT)
    assert factor == 0.30480060960121924
    assert round(factor, 12) == 0.304800609601
    assert unit == "m"


def test_metre_passes_through_unconverted():
    assert linear_unit_to_m(_METRE_WKT) == (1.0, "m")


def test_plain_foot_is_point_3048():
    factor, unit = linear_unit_to_m('UNIT["foot",0.3048,AUTHORITY["EPSG","9002"]]')
    assert factor == 0.3048
    assert unit == "m"


def test_bare_geographic_crs_stays_native_degrees_wkt1():
    assert linear_unit_to_m(_GEOGCS_WKT1) == (None, "deg")


def test_bare_geographic_crs_stays_native_degrees_wkt2_with_its_ellipsoid_lengthunit_trap():
    assert linear_unit_to_m(_GEOGCRS_WKT2) == (None, "deg")


def test_unknown_unit_never_returns_a_silent_one():
    """A real unit name this app does not recognise. The point of ``None`` instead of a guessed
    ``1.0``: a caller that multiplied by ``1.0`` here would print a false metre reading for a
    link-based grid, which is exactly the bug class this module exists to close."""
    factor, label = linear_unit_to_m('UNIT["Clarke\'s link",0.201166195164]')
    assert factor is None
    assert factor != 1.0
    assert "link" in label.lower()


def test_a_bare_short_epsg_code_has_no_unit_text_to_parse_and_is_still_never_1_0():
    """``str(rasterio's CRS)`` is a bare ``"EPSG:<code>"`` for any CRS with an exact EPSG match
    (a metre UTM zone, WGS84 itself) -- there is no unit text left in that short string to parse
    at all. Still never a silent 1.0."""
    factor, label = linear_unit_to_m("EPSG:32615")
    assert factor is None
    assert label == "EPSG:32615"


def test_provenance_dict_input_reads_its_crs_key():
    factor, unit = linear_unit_to_m({"crs": _SURVEY_FOOT_WKT, "source": "irrelevant.tif"})
    assert factor == pytest.approx(0.30480060960121924)
    assert unit == "m"


def test_no_crs_at_all_is_unknown_not_1_0():
    assert linear_unit_to_m({"crs": None}) == (None, "unknown")
    assert linear_unit_to_m(None) == (None, "unknown")
    assert linear_unit_to_m("") == (None, "unknown")


# --------------------------------------------------------------------------- px_to_metres


def test_forty_survey_foot_cell_converts_to_12_19202_metres():
    """The BOEM number from the spec: a 40-ft cell is 12.19202 m, not 40 (assumed-metre) and not
    40 x 0.3048 = 12.192 (international-foot, the wrong constant for a US survey-foot grid)."""
    field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})
    px, unit = px_to_metres(field)
    assert unit == "m"
    assert px == pytest.approx(12.192024384048768)
    assert f"{px:.5f}" == "12.19202"


def test_metre_crs_passes_through_unconverted():
    field = _field(2.0, units="metre", provenance={"crs": _METRE_WKT})
    assert px_to_metres(field) == (pytest.approx(2.0), "m")


def test_no_crs_key_uses_the_frames_own_unit_directly():
    """An EBSD map or a plain image: no CRS at all, so the axis is already in the frame's own
    physical unit (microns here) and there is nothing to convert -- the exact case every
    pre-existing window test relies on (``LocalFrame(units="px")`` fields stay unconverted)."""
    field = _field(0.07, units="µm")
    assert px_to_metres(field) == (pytest.approx(0.07), "µm")


def test_geographic_frame_stays_in_native_degrees():
    frame = GeographicFrame()
    axis = np.linspace(-95.0, -94.0, 5)
    field = RasterField(name="f", values=np.zeros((5, 5)), frame=frame, x_axis=axis, y_axis=axis,
                        provenance={"crs": "EPSG:4326"})
    px, unit = px_to_metres(field)
    assert unit == "deg"
    assert px == pytest.approx(0.25)


def test_unrecognised_linear_unit_stays_native_not_1_0():
    field = _field(3.0, units="Clarke's link",
                   provenance={"crs": 'UNIT["Clarke\'s link",0.201166195164]'})
    px, unit = px_to_metres(field)
    assert px == pytest.approx(3.0)          # unconverted -- honest, not a false metre claim
    assert unit != "m"


def test_degenerate_axis_is_none_not_a_fabricated_one():
    field = RasterField(name="f", values=np.zeros((1, 3)), frame=LocalFrame(units="px"),
                        x_axis=np.array([5.0]), y_axis=np.array([0.0, 1.0, 2.0]))
    assert px_to_metres(field) == (None, "px")


def test_a_bare_array_field_is_handled_like_the_old_getattr_fallback():
    """``MainWindow.load_field`` accepts a bare ndarray too (every stub fixture in
    tests/test_shell_window.py does this) -- no ``.frame``, ``.provenance`` or ``.x_axis``."""
    assert px_to_metres(np.zeros((4, 4))) == (None, "px")
    assert px_to_metres(None) == (None, "px")


# --------------------------------------------------------------------------- WTMM2D.derived_reading


def test_derived_reading_is_none_for_every_param_but_a_min():
    from dynamix.devices.wtmm import WTMM2D

    dev = WTMM2D()
    field = _field(1.0, units="px")
    for p in dev.params:
        if p.name == "a_min":
            continue
        assert dev.derived_reading(p.name, p.default, field) is None


def test_derived_reading_a_min_shows_sigma_in_px_and_metres():
    """The design's worked example: a_min=2.0 on a 40-ft-cell field -> sigma = 1.57*2 = 3.14 px
    -> 3.14 * 12.192024... = 38.3 m; lambda (mexican, the device's own default wavelet) =
    11.4 px -> 139 m."""
    from dynamix.devices.wtmm import WTMM2D

    dev = WTMM2D()
    field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})
    text = dev.derived_reading("a_min", 2.0, field)
    assert "3.1 px" in text
    assert "38" in text
    assert "λ 11.4 px" in text
    assert "139" in text
    assert "m" in text


def test_derived_reading_a_min_omits_the_physical_half_with_no_physical_unit():
    from dynamix.devices.wtmm import WTMM2D

    dev = WTMM2D()
    field = _field(1.0, units="px")
    assert dev.derived_reading("a_min", 1.0, field) == "σ ≈ 1.6 px · λ 5.7 px"


def test_derived_reading_a_min_handles_a_missing_field():
    from dynamix.devices.wtmm import WTMM2D

    dev = WTMM2D()
    assert dev.derived_reading("a_min", 1.0, None) == "σ ≈ 1.6 px · λ 5.7 px"


@pytest.mark.parametrize("wavelet", ["gaussian", "mexican"])
def test_derived_reading_lambda_is_the_params_wavelets_own_bandpass_peak(wavelet):
    """``params={"wavelet": ...}`` drives lambda directly off ``scale_units.lambda_peak_px`` at
    the same finest scale the sigma half uses -- not a value this test hardcodes."""
    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.devices.wtmm import WTMM2D

    dev = WTMM2D()
    field = _field(1.0, units="px")
    text = dev.derived_reading("a_min", 1.0, field, params={"wavelet": wavelet})
    expected = lambda_peak_px(1.0 * 6.0 / 0.86, wavelet, 1)
    assert f"λ {expected:.1f} px" in text


def test_derived_reading_lambda_differs_between_wavelets_mexican_smaller():
    """The scale_units doctrine this module inherits: mexican probes finer structure than
    gaussian at equal scale, so its bandpass lambda is SMALLER, not larger."""
    from dynamix.devices.wtmm import WTMM2D

    dev = WTMM2D()
    field = _field(1.0, units="px")
    gaussian_text = dev.derived_reading("a_min", 1.0, field, params={"wavelet": "gaussian"})
    mexican_text = dev.derived_reading("a_min", 1.0, field, params={"wavelet": "mexican"})
    assert gaussian_text != mexican_text
    assert "λ 9.9 px" in gaussian_text
    assert "λ 5.7 px" in mexican_text


def test_derived_reading_params_none_falls_back_to_the_devices_own_default_wavelet():
    """No ``params`` at all -- the pre-Task-5 call shape -- falls back to whatever ``WTMM2D``
    itself declares as ``wavelet``'s default in its param table, not a hardcoded guess."""
    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.model.device import defaults_for

    dev = WTMM2D()
    default_wavelet = defaults_for(dev)["wavelet"]
    field = _field(1.0, units="px")
    text = dev.derived_reading("a_min", 1.0, field, params=None)
    expected = lambda_peak_px(1.0 * 6.0 / 0.86, default_wavelet, 1)
    assert f"λ {expected:.1f} px" in text


def test_a_min_param_is_relabeled_unitless():
    from dynamix.devices.wtmm import WTMM2D

    a_min = next(p for p in WTMM2D().params if p.name == "a_min")
    assert a_min.label == "aₘᵢₙ"
    assert a_min.units == ""


# --------------------------------------------------------------------------- chain_strip rendering


@pytest.fixture
def registered_builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_strip_renders_the_derived_reading_label_beside_its_control(qtbot, registered_builtins):
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.chain_strip import DeviceStrip

    dev = get_device("wtmm2d")
    field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})
    strip = DeviceStrip(0, dev, defaults_for(dev), field=field)
    qtbot.addWidget(strip)

    assert "a_min" in strip._derived_labels
    label = strip._derived_labels["a_min"]
    assert label.property("reading") == "true"
    assert label.property("muted") == "true"
    # a_min defaults to 1.0: sigma = 1.57 px -> 1.57 * 12.192... = 19.1... m; lambda (mexican,
    # wtmm2d's own default wavelet, threaded through via defaults_for -> the strip's params) =
    # 5.7 px
    assert "1.6 px" in label.text()
    assert "λ 5.7 px" in label.text()
    assert "19" in label.text()
    assert "n_oct" not in strip._derived_labels


def test_strip_derived_label_updates_on_param_change(qtbot, registered_builtins):
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.chain_strip import DeviceStrip

    dev = get_device("wtmm2d")
    field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})
    strip = DeviceStrip(0, dev, defaults_for(dev), field=field)
    qtbot.addWidget(strip)

    strip.controls["a_min"].valueChanged.emit(4.0)     # as the widget itself would propose

    assert "6.3 px" in strip._derived_labels["a_min"].text()      # sigma_px_amin(4.0) = 6.28


def test_strip_derived_label_refreshes_when_a_sibling_param_changes(qtbot, registered_builtins):
    """``self._params`` used to be a construction-time snapshot, so
    flipping ``wavelet`` never touched it and the aₘᵢₙ line's λ stayed pinned to whichever wavelet
    was selected when the strip was built. ``_on_control_changed`` now writes every edit into
    ``self._params`` first and recomputes ALL derived-reading labels off the refreshed dict, so a
    wavelet flip -- not just an a_min edit -- must update the aₘᵢₙ label's λ immediately."""
    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.chain_strip import DeviceStrip

    dev = get_device("wtmm2d")
    strip = DeviceStrip(0, dev, defaults_for(dev))     # a_min=1.0, wavelet="mexican" (the default)
    qtbot.addWidget(strip)

    mexican_lambda = lambda_peak_px(1.0 * 6.0 / 0.86, "mexican", 1)
    gaussian_lambda = lambda_peak_px(1.0 * 6.0 / 0.86, "gaussian", 1)
    assert f"λ {mexican_lambda:.1f} px" in strip._derived_labels["a_min"].text()

    strip.controls["wavelet"].valueChanged.emit("gaussian")   # as the widget itself would propose

    label_text = strip._derived_labels["a_min"].text()
    assert f"λ {gaussian_lambda:.1f} px" in label_text
    assert f"λ {mexican_lambda:.1f} px" not in label_text


def test_strip_has_no_derived_label_when_the_device_lacks_the_method(qtbot, registered_builtins):
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.chain_strip import DeviceStrip

    dev = get_device("scale_select")
    strip = DeviceStrip(0, dev, defaults_for(dev))
    qtbot.addWidget(strip)
    assert strip._derived_labels == {}


def test_zone_threads_field_through_to_every_strip(qtbot, registered_builtins):
    from dynamix.model.chain import DeviceRef
    from dynamix.shell.chain_strip import ChainStripZone

    field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})
    zone = ChainStripZone((DeviceRef("wtmm2d", {}),), field=field)
    qtbot.addWidget(zone)
    assert "3.1 px" not in zone.strip(0)._derived_labels["a_min"].text()   # default a_min=1.0
    assert "19" in zone.strip(0)._derived_labels["a_min"].text()


# --------------------------------------------------------------------------- MainWindow routing


def test_main_window_units_and_px_to_unit_route_through_units_py(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})

    assert win._units() == "m"
    assert win._px_to_unit() == pytest.approx(12.192024384048768)


def test_transport_reading_agrees_with_units_py(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow
    from dynamix.core.scale_units import sigma_px

    win = MainWindow()
    qtbot.addWidget(win)
    win.field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})
    win._scales = (2.0,)

    px, unit = px_to_metres(win.field)
    sigma = sigma_px(2.0)
    assert win._scale_reading(0) == f"a = 2.0 · σ {sigma:.1f} px ≈ {sigma * px:.3g} {unit}"


def test_scale_bar_reading_agrees_with_units_py(qtbot, clean_registry):
    """The canvas re-derives the scale bar on every camera move (``viewChanged`` -> the ``_update_scale_bar``); wired in ``MainWindow.__init__`` before any ``load_field`` call, so
    setting ``.field`` directly and moving the view exercises the real production path."""
    from dynamix.shell.canvas import nice_round_scalebar
    from dynamix.shell.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.field = _field(40.0, units="US survey foot", provenance={"crs": _SURVEY_FOOT_WKT})

    win.canvas.view.setXRange(0, 16, padding=0)
    text = win.canvas.scale_bar_label.text()

    (x0, x1), _ = win.canvas.view.viewRange()
    _, bar_px = nice_round_scalebar(x1 - x0)
    px, unit = px_to_metres(win.field)
    assert unit == "m"
    assert text == f"{bar_px * px:.4g} {unit}"


def test_a_plain_px_field_still_reads_in_bare_pixels(qtbot, clean_registry):
    """The one existing behaviour that must survive byte-for-byte: a non-georeferenced field
    (every current window test's fixture) shows a bare pixel count, no unit conversion applied
    and no stray unit string appended."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.core.scale_units import sigma_px

    win = MainWindow()
    qtbot.addWidget(win)
    win._scales = (2.0,)
    assert win._units() == "px"
    sigma = sigma_px(2.0)
    assert win._scale_reading(0) == f"a = 2.0 · σ {sigma:.1f} px"


def test_scale_reading_with_physical_unit_example_from_brief(qtbot, clean_registry):
    """Test the specific example from the design: 40 m pixels, scale = 6.9767
    (compute_scales2d(1,1)[0]), expecting 'a = 7.0 · σ 1.6 px ≈ 62.8 m'."""
    from dynamix.shell.main_window import MainWindow
    from dynamix.core.scale_units import sigma_px

    win = MainWindow()
    qtbot.addWidget(win)
    # Use the value from compute_scales2d(1, 1)[0] as specified in the design
    scale = 6.9767  # compute_scales2d(1, 1)[0]
    win.field = _field(40.0, units="metre", provenance={"crs": _METRE_WKT})
    win._scales = (scale,)

    sigma = sigma_px(scale)
    px, unit = px_to_metres(win.field)
    phys = sigma * px
    expected = f"a = {scale:.1f} · σ {sigma:.1f} px ≈ {phys:.3g} {unit}"
    assert win._scale_reading(0) == expected
    # Verify the specific numbers from the design
    assert expected == "a = 7.0 · σ 1.6 px ≈ 62.8 m"
