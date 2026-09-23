# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.pointset -- CSV point-catalogue import, headless.

Pure numpy/stdlib, no Qt anywhere in this file (mirrors ``test_rasterfield.py``'s own headless
discipline for the raster side).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.pointset import PointSet, read_csv_points, resolve_columns

# lon/lat/depth/mag, one junk row (garbage text, same column count) in the middle.
_CSV_WITH_JUNK = (
    "Lon,Lat,Depth_km,Mag\n"
    "1.0,2.0,3.0,4.0\n"
    "JUNK,ROW,HERE,BAD\n"
    "5.0,6.0,7.0,8.0\n"
)


def _write(tmp_path, text: str, name: str = "points.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_reads_lon_lat_depth_mag_and_drops_the_junk_row(tmp_path):
    path = _write(tmp_path, _CSV_WITH_JUNK)
    pset = read_csv_points(path)

    assert isinstance(pset, PointSet)
    assert pset.lon.shape == (2,)
    assert pset.lat.shape == (2,)
    assert pset.depth.shape == (2,)
    np.testing.assert_allclose(pset.lon, [1.0, 5.0])
    np.testing.assert_allclose(pset.lat, [2.0, 6.0])
    np.testing.assert_allclose(pset.depth, [3.0, 7.0])
    # mag has no dedicated dataclass field -- it lands in attrs.
    np.testing.assert_allclose(pset.attrs["mag"], [4.0, 8.0])


def test_dropped_rows_counted_in_provenance(tmp_path):
    path = _write(tmp_path, _CSV_WITH_JUNK)
    pset = read_csv_points(path)
    assert pset.provenance["dropped_rows"] == "1"
    assert pset.provenance["source"] == str(path)


# --------------------------------------------------------------------------- Quote-
# aware data parsing -- genfromtxt is not quote-aware, so a
# quoted field containing a comma (USGS catalogues' own `place` column) used to split into extra
# columns and get silently dropped by genfromtxt WITHOUT entering dropped_rows.

_CSV_WITH_QUOTED_COMMA = (
    "lon,lat,place\n"
    '1.0,2.0,"10km ENE of Somewhere, CA"\n'
    "3.0,4.0,Elsewhere\n"
)


def test_a_quoted_field_containing_a_comma_survives_with_correct_values(tmp_path):
    """The header is already read with stdlib ``csv`` (quote-aware); the DATA read must be too.

    The first data row's ``place`` value has a comma INSIDE quotes -- naive comma-splitting
    (``np.genfromtxt``'s own delimiter parsing) reads it as a 4th field, disagreeing with the
    header's 3 columns and the second (unquoted) row's genuine 3. ``genfromtxt(invalid_raise=
    False)`` resolves that disagreement by silently dropping whichever row doesn't match its
    OWN inferred column count -- reproduced directly against this exact fixture in the finding's
    own investigation: it keeps the quote-corrupted row and drops the second, perfectly
    well-formed one, uncounted. Quote-aware parsing must keep BOTH rows; neither is malformed."""
    path = _write(tmp_path, _CSV_WITH_QUOTED_COMMA)
    pset = read_csv_points(path)

    assert pset.lon.shape == (2,)
    np.testing.assert_allclose(pset.lon, [1.0, 3.0])
    np.testing.assert_allclose(pset.lat, [2.0, 4.0])
    assert pset.provenance["dropped_rows"] == "0"


_CSV_WITH_SHORT_ROW = (
    "lon,lat,depth\n"
    "1.0,2.0,3.0\n"
    "4.0,5.0\n"           # short row -- wrong column count
    "6.0,7.0,8.0\n"
)


def test_a_row_with_the_wrong_column_count_is_dropped_and_honestly_counted(tmp_path):
    """The mechanism I3 names directly: ``np.genfromtxt(..., invalid_raise=False)`` drops a
    column-count mismatch SILENTLY, without it ever reaching ``dropped_rows`` -- the same
    undercount a quoted-comma field triggers. Every dropped row must be counted, honestly."""
    path = _write(tmp_path, _CSV_WITH_SHORT_ROW)
    pset = read_csv_points(path)

    assert pset.lon.shape == (2,)
    np.testing.assert_allclose(pset.lon, [1.0, 6.0])
    assert pset.provenance["dropped_rows"] == "1"


_CSV_WITH_BAD_LON = (
    "lon,lat\n"
    "1.0,2.0\n"
    "notanumber,4.0\n"
    "5.0,6.0\n"
)


def test_a_row_with_an_unparsable_lon_is_dropped_and_counted(tmp_path):
    path = _write(tmp_path, _CSV_WITH_BAD_LON)
    pset = read_csv_points(path)

    assert pset.lon.shape == (2,)
    np.testing.assert_allclose(pset.lon, [1.0, 5.0])
    assert pset.provenance["dropped_rows"] == "1"


_CSV_MIXED_MALFORMED = (
    "lon,lat\n"
    "1.0,2.0\n"
    "just_one_field\n"     # short row
    "notanumber,4.0\n"     # unparsable lon
    "5.0,6.0\n"
)


def test_each_kind_of_malformed_row_is_counted_independently(tmp_path):
    """I3's own instruction: a short row AND an unparsable float are DIFFERENT failure kinds, and
    a file with one of each must count both -- not stop at the first, not conflate them."""
    path = _write(tmp_path, _CSV_MIXED_MALFORMED)
    pset = read_csv_points(path)

    assert pset.lon.shape == (2,)
    np.testing.assert_allclose(pset.lon, [1.0, 5.0])
    assert pset.provenance["dropped_rows"] == "2"


def test_an_unparsable_depth_does_not_drop_the_row(tmp_path):
    """Only lon/lat unparsability makes a row unusable (the module's own pre-existing contract,
    unchanged by the quote-aware rewrite) -- an optional axis (depth here) losing one value must
    not cost the whole point."""
    path = _write(tmp_path, "lon,lat,depth\n1.0,2.0,notanumber\n3.0,4.0,5.0\n")
    pset = read_csv_points(path)

    assert pset.lon.shape == (2,)
    assert pset.provenance["dropped_rows"] == "0"
    assert np.isnan(pset.depth[0])
    np.testing.assert_allclose(pset.depth[1], 5.0)


def test_detection_is_case_insensitive():
    """Header capitalisation (``Lon``/``Lat``/...) must not defeat the lowercase detection table."""
    resolved = resolve_columns(["Lon", "Lat", "Depth_km", "Mag"])
    assert resolved == {"lon": "Lon", "lat": "Lat", "depth": "Depth_km", "mag": "Mag"}


def test_detection_recognises_alternate_spellings():
    resolved = resolve_columns(["longitude", "latitude", "z", "magnitude"])
    assert resolved == {
        "lon": "longitude", "lat": "latitude", "depth": "z", "mag": "magnitude",
    }


def test_mapping_overrides_detection(tmp_path):
    """A CSV whose headers don't match the detection table at all still loads once a mapping
    names the columns -- the shell's mapping-dialog contract."""
    path = _write(tmp_path, "X,Y,Z\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    pset = read_csv_points(path, mapping={"lon": "X", "lat": "Y", "depth": "Z"})
    np.testing.assert_allclose(pset.lon, [1.0, 4.0])
    np.testing.assert_allclose(pset.lat, [2.0, 5.0])
    np.testing.assert_allclose(pset.depth, [3.0, 6.0])


def test_partial_mapping_still_auto_detects_the_rest(tmp_path):
    """A mapping only needs to name the axis auto-detection actually got wrong/missed -- any key
    it omits still falls through to :data:`~dynamix.core.pointset._DETECT`."""
    path = _write(tmp_path, "X,lat,depth\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    pset = read_csv_points(path, mapping={"lon": "X"})
    np.testing.assert_allclose(pset.lon, [1.0, 4.0])
    np.testing.assert_allclose(pset.lat, [2.0, 5.0])
    np.testing.assert_allclose(pset.depth, [3.0, 6.0])


def test_unmappable_lon_lat_raises_listing_the_actual_header(tmp_path):
    path = _write(tmp_path, "X,Y,Z\n1.0,2.0,3.0\n")
    with pytest.raises(ValueError) as exc:
        read_csv_points(path)
    message = str(exc.value)
    assert "X" in message and "Y" in message and "Z" in message


def test_missing_depth_and_mag_are_simply_absent():
    """depth/mag are optional axes -- a file without them loads fine, ``depth`` stays ``None``
    and ``attrs`` stays empty rather than either being required."""
    resolved = resolve_columns(["lon", "lat"])
    assert resolved == {"lon": "lon", "lat": "lat"}


def test_missing_depth_and_mag_columns_end_to_end(tmp_path):
    path = _write(tmp_path, "lon,lat\n1.0,2.0\n3.0,4.0\n")
    pset = read_csv_points(path)
    assert pset.depth is None
    assert pset.attrs == {}


def test_single_data_row_still_shapes_as_one_point(tmp_path):
    """``np.genfromtxt`` collapses a lone data row to 1D -- a numpy quirk this module must not
    leak: one row must still read back as shape ``(1,)`` arrays, not a scalar."""
    path = _write(tmp_path, "lon,lat\n1.0,2.0\n")
    pset = read_csv_points(path)
    assert pset.lon.shape == (1,)
    assert pset.lat.shape == (1,)
    np.testing.assert_allclose(pset.lon, [1.0])
    np.testing.assert_allclose(pset.lat, [2.0])
