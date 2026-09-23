# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_rasterfield.py"""
import json
import numpy as np
import pytest
from dynamix.core.frames import GeographicFrame, LocalFrame
from dynamix.core.rasterfield import RASTERFIELD_NPZ_SCHEMA, RasterField


def _field():
    v = np.arange(12.0).reshape(3, 4); v[1, 2] = np.nan
    return RasterField(name="toy", values=v, frame=LocalFrame(dx=0.5, dy=0.5, units="µm"),
                       x_axis=0.5 * np.arange(4), y_axis=0.5 * np.arange(3),
                       units="deg", provenance={"src": "unit-test"})

def test_npz_roundtrip_exact(tmp_path):
    rf = _field(); p = tmp_path / "toy.npz"; rf.save_npz(p)
    back = RasterField.load_npz(p)
    assert back.name == "toy" and back.units == "deg" and back.frame == rf.frame
    np.testing.assert_array_equal(back.values, rf.values)   # NaN-preserving
    np.testing.assert_array_equal(back.x_axis, rf.x_axis)
    assert back.provenance == {"src": "unit-test"}

def test_from_file_dispatch(tmp_path):
    rf = _field(); rf.save_npz(tmp_path / "toy.npz")
    assert RasterField.from_file(tmp_path / "toy.npz").name == "toy"
    np.save(tmp_path / "bare.npy", rf.values)
    bare = RasterField.from_file(tmp_path / "bare.npy")
    assert bare.frame == LocalFrame(units="px") and bare.nx == 4
    np.testing.assert_array_equal(bare.x_axis, np.arange(4, dtype=float))
    np.savetxt(tmp_path / "bare.txt", np.nan_to_num(rf.values))
    assert RasterField.from_file(tmp_path / "bare.txt").ny == 3
    with pytest.raises(ValueError):
        RasterField.from_file(tmp_path / "nope.xyz")

def test_geotiff_missing_rasterio_msg(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__
    def block(name, *a, **k):
        if name == "rasterio": raise ImportError("nope")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", block)
    (tmp_path / "x.tif").write_bytes(b"II*\x00")
    with pytest.raises(ValueError, match="rasterio"):
        RasterField.from_file(tmp_path / "x.tif")

def test_from_ebsd_fields():
    fields = {"kam": np.ones((3, 4), np.float32), "gb_mask": np.zeros((3, 4), bool),
              "_norm_p99": {"kam": 2.5}}
    rf = RasterField.from_ebsd_fields(fields, "kam", px_um=0.25)
    assert rf.frame == LocalFrame(dx=0.25, dy=0.25, units="µm")
    assert rf.values.dtype == np.float64
    np.testing.assert_allclose(rf.x_axis, 0.25 * np.arange(4))
    assert rf.provenance["norm_p99"] == 2.5
    assert RasterField.from_ebsd_fields(fields, "gb_mask", px_um=0.25).values.dtype == np.float64

def test_to_scene_mesh_layout():
    rf = _field(); m = rf.to_scene_mesh()
    assert m["dims"] == (4, 3, 1) and m["points"].shape == (12, 3)
    # C-order: index iy*nx+ix; check pixel (iy=1, ix=2) lands at (x_axis[2], y_axis[1])
    np.testing.assert_allclose(m["points"][1 * 4 + 2, :2], [rf.x_axis[2], rf.y_axis[1]])
    assert np.isnan(m["values"][1 * 4 + 2])

def test_kam_fixture_loads():
    from pathlib import Path
    from dynamix.core.rasterfield import RasterField
    p = Path(__file__).parent / "fixtures" / "kam_64.npz"
    rf = RasterField.load_npz(p)
    assert rf.frame.kind == "local" and rf.frame.units == "µm"
    assert rf.values.shape[0] <= 64 and np.isfinite(rf.values).any()


def test_multicomponent_field(tmp_path):
    v = np.ones((3, 4, 3)); v[0, 0, 1] = np.nan
    rf = RasterField(name="log_ori", values=v, frame=LocalFrame(units="µm"),
                     x_axis=np.arange(4.0), y_axis=np.arange(3.0))
    assert rf.n_components == 3 and not rf.is_scalar
    m = rf.to_scene_mesh()
    np.testing.assert_allclose(m["values"][1], np.sqrt(3.0))     # L2 norm display
    assert np.isnan(m["values"][0])                              # any-NaN -> NaN
    rf.save_npz(tmp_path / "v3.npz")
    back = RasterField.load_npz(tmp_path / "v3.npz")
    assert back.values.shape == (3, 4, 3)
    assert RasterField(name="s", values=np.ones((2, 2)), frame=LocalFrame(),
                       x_axis=np.arange(2.0), y_axis=np.arange(2.0)).is_scalar


# ------------------------------------------------------- plain (non-georeferenced) rasters

def _png(path, arr):
    """Write `arr` (uint8, 2-D or (h,w,3)) as a PNG using Pillow."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    Image.fromarray(arr).save(path)


def test_from_file_reads_a_grayscale_png(tmp_path):
    """A plain image loads as a pixel-indexed field -- the non-georeferenced testing path."""
    a = (np.arange(64 * 96, dtype=np.uint8) % 251).reshape(64, 96)
    p = tmp_path / "gray.png"
    _png(p, a)
    rf = RasterField.from_file(p)
    assert rf.values.shape == (64, 96) and rf.is_scalar
    assert isinstance(rf.frame, LocalFrame) and rf.frame.units == "px"
    assert rf.x_axis.tolist() == list(range(96)) and rf.y_axis.tolist() == list(range(64))
    assert np.allclose(rf.values, a.astype(float))
    assert rf.name == "gray"


def test_rgb_png_collapses_to_luminance_by_default(tmp_path):
    """RGB -> a single luminance channel, because scalar WTMM is what a plain image is for."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255                                   # pure red
    p = tmp_path / "red.png"
    _png(p, rgb)
    rf = RasterField.from_file(p)
    assert rf.values.shape == (8, 8) and rf.is_scalar
    assert 50.0 < rf.values.mean() < 90.0               # Rec.601 luma of pure red ~76, not 255 or 85


def test_rgb_png_can_load_as_three_components(tmp_path):
    """`components=True` keeps RGB as an (ny, nx, 3) field -- the shape the tensor WTMM path wants."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    p = tmp_path / "green.png"
    _png(p, rgb)
    rf = RasterField.from_file(p, components=True)
    assert rf.values.shape == (8, 8, 3) and rf.n_components == 3 and not rf.is_scalar


def test_npz_holding_a_bare_array_loads_as_a_plain_raster(tmp_path):
    """`np.savez(f, arr)` is a reasonable thing to hand the app; it must not demand our own schema."""
    a = np.random.default_rng(0).random((16, 24))
    p = tmp_path / "bare.npz"
    np.savez(p, a)
    rf = RasterField.from_file(p)
    assert rf.values.shape == (16, 24)
    assert isinstance(rf.frame, LocalFrame) and rf.frame.units == "px"
    assert np.allclose(rf.values, a)


def test_unrecognisable_npz_names_its_keys(tmp_path):
    """A WTMM extrema export (several 2-D arrays, no schema) is ambiguous -- say what was found.
    The old failure was a raw `KeyError: 'schema is not a file in the archive'`, which named an
    npz internal rather than the problem."""
    p = tmp_path / "topo_extrema.npz"
    np.savez(p, h_xyz=np.zeros((5, 3)), v_xyz=np.zeros((7, 3)), v_persist=np.zeros(7))
    with pytest.raises(ValueError) as ei:
        RasterField.from_file(p)
    msg = str(ei.value)
    assert "h_xyz" in msg and "v_xyz" in msg
    assert "schema is not a file in the archive" not in msg


def test_tif_falls_back_to_pillow_without_rasterio(tmp_path, monkeypatch):
    """No rasterio -> a .tif still loads, as a PLAIN pixel-indexed field rather than a dead end."""
    import builtins
    a = (np.arange(32 * 48, dtype=np.uint8) % 251).reshape(32, 48)
    p = tmp_path / "plain.tif"
    _png(p, a)                                          # Pillow writes TIFF from the suffix
    real_import = builtins.__import__

    def no_rasterio(name, *args, **kw):
        if name == "rasterio":
            raise ImportError("no rasterio")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_rasterio)
    rf = RasterField.from_file(p)
    assert rf.values.shape == (32, 48)
    assert isinstance(rf.frame, LocalFrame)             # georeferencing dropped, not fabricated


# --------------------------------------------------- windowed GeoTIFF reads + tiling (big grids)

def _utm_tif(path, ny=300, nx=400, px=12.192, x0=400000.0, y0=3100000.0):
    """A small PROJECTED (UTM 15N, metres) GeoTIFF -- the BOEM bathymetry layout in miniature."""
    rio = pytest.importorskip("rasterio", reason="rasterio not installed")
    from rasterio.transform import from_origin
    vals = (np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) % 1000) - 3000.0
    with rio.open(path, "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
                  crs="EPSG:32615", transform=from_origin(x0, y0, px, px)) as dst:
        dst.write(vals, 1)
    return vals


def test_geotiff_windows_tile_the_grid_without_gaps(tmp_path):
    """Pure tiling maths: every cell covered exactly once, edge tiles truncated not padded."""
    from dynamix.core.rasterfield import geotiff_windows
    wins = list(geotiff_windows(300, 400, tile=128))
    assert len(wins) == 3 * 4                                  # ceil(300/128)=3, ceil(400/128)=4
    covered = np.zeros((300, 400), int)
    for w in wins:
        covered[w["row_off"]:w["row_off"] + w["height"], w["col_off"]:w["col_off"] + w["width"]] += 1
    assert covered.min() == 1 and covered.max() == 1           # exact partition
    edge = [w for w in wins if w["row_off"] == 256][0]
    assert edge["height"] == 300 - 256                          # truncated, not padded


def test_geotiff_info_reports_shape_and_pixel_size(tmp_path):
    from dynamix.core.rasterfield import geotiff_info
    p = tmp_path / "utm.tif"; _utm_tif(p)
    info = geotiff_info(p)
    assert (info["height"], info["width"]) == (300, 400)
    assert info["projected"] is True
    assert abs(info["dx"] - 12.192) < 1e-6 and abs(info["dy"] - 12.192) < 1e-6
    assert info["units"] in ("metre", "m", "meter")


def test_windowed_read_returns_only_that_block(tmp_path):
    """The whole point: read a tile without materialising the full grid."""
    p = tmp_path / "utm.tif"; full = _utm_tif(p)
    rf = RasterField.from_geotiff_window(p, row_off=100, col_off=50, height=64, width=96)
    assert rf.values.shape == (64, 96)
    assert np.allclose(rf.values, full[100:164, 50:146])


def test_projected_geotiff_yields_a_metre_local_frame(tmp_path):
    """A UTM grid must NOT become lon/lat degrees -- WTMM scales have to be metric and equal-area."""
    p = tmp_path / "utm.tif"; _utm_tif(p)
    rf = RasterField.from_geotiff_window(p, row_off=0, col_off=0, height=32, width=32)
    assert isinstance(rf.frame, LocalFrame)
    assert abs(rf.frame.dx - 12.192) < 1e-6 and abs(rf.frame.dy - 12.192) < 1e-6
    assert rf.frame.units in ("metre", "m", "meter")
    assert abs((rf.x_axis[1] - rf.x_axis[0]) - 12.192) < 1e-6   # axes carry real ground distance


def test_full_projected_geotiff_load_frames_by_crs_kind_too(tmp_path):
    """The FULL-file loader (`from_file` -> `_from_geotiff`) must frame by CRS kind exactly like
    the windowed loader above -- the 2026-08-19 fix. Before it, a projected GeoTIFF got a
    ``GeographicFrame`` whose ``to_scene`` fed UTM-scale coordinates through projection.py's
    ``np.mod(x, 360)`` longitude canonicalisation: a 400-column, 12.192 m grid folded every ~30
    columns, smearing the Vector-tab raster into horizontal streaks (measured on the real BOEM
    crop: 400 columns -> 9 distinct x values). The assertions pin both the frame kind and the
    fold-impossibility (scene x strictly monotonic and spanning the FULL metric width)."""
    p = tmp_path / "utm.tif"; _utm_tif(p)
    rf = RasterField.from_file(p)
    assert isinstance(rf.frame, LocalFrame)
    assert abs(rf.frame.dx - 12.192) < 1e-6 and abs(rf.frame.dy - 12.192) < 1e-6
    assert rf.frame.units in ("metre", "m", "meter")
    assert rf.units in ("metre", "m", "meter")                  # loader parity: window sets it too
    pts = rf.frame.to_scene(rf.x_axis, np.full_like(rf.x_axis, rf.y_axis[0]))
    xs = pts[:, 0]
    assert np.all(np.diff(xs) > 0)                              # no 360-unit fold, ever
    assert xs[-1] - xs[0] > 300 * 12.192                        # spans the real ~4.9 km width


def test_full_geographic_geotiff_still_gets_a_geographic_frame(tmp_path):
    """Control for the fix: a genuinely geographic (EPSG:4326) GeoTIFF keeps GeographicFrame --
    the branch keys on ``crs.is_projected``, not on being a GeoTIFF."""
    rio = pytest.importorskip("rasterio", reason="rasterio not installed")
    from rasterio.transform import from_origin
    p = tmp_path / "geo.tif"
    vals = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    with rio.open(p, "w", driver="GTiff", height=20, width=30, count=1, dtype="float32",
                  crs="EPSG:4326", transform=from_origin(-91.0, 28.0, 0.01, 0.01)) as dst:
        dst.write(vals, 1)
    rf = RasterField.from_file(p)
    assert isinstance(rf.frame, GeographicFrame)
    assert rf.units == ""                                        # unchanged for geographic sources


def test_window_axes_are_offset_to_the_tiles_own_position(tmp_path):
    """Tile axes must be absolute projected coordinates, so adjacent tiles abut correctly."""
    p = tmp_path / "utm.tif"; _utm_tif(p)
    a = RasterField.from_geotiff_window(p, row_off=0, col_off=0, height=32, width=32)
    b = RasterField.from_geotiff_window(p, row_off=0, col_off=32, height=32, width=32)
    assert abs((b.x_axis[0] - a.x_axis[0]) - 32 * 12.192) < 1e-4


# --------------------------------------------------- _from_bare_array axis convention
#
# LocalFrame.to_scene is a pure pass-through (frames.py's own class
# docstring: "the affine pixel-index -> frame-unit mapping ... is applied by RasterField when it
# builds its axis arrays") -- every OTHER construction path already bakes a frame's own x0/y0/dx/
# dy into its axes (from_geotiff_window, from_ebsd_fields); _from_bare_array was the one holdout,
# building plain arange() axes regardless of what frame= a caller passed in. Fixed to match the
# documented convention; the default-frame case (x0=0, dx=1, every existing caller/test) is
# unaffected -- x0 + dx*arange(n) == arange(n) exactly.


def test_from_bare_array_bakes_a_non_default_local_frames_affine_into_its_axes():
    """The B4 regression fixture: an asymmetric shape (never square -- a square grid can hide a
    dims/axis mixup) with a non-uniform, negative-dy LocalFrame, matching the triage's own bug3
    probe parameters exactly (dx=2.0, dy=-3.0, x0=100.0, y0=-50.0)."""
    frame = LocalFrame(dx=2.0, dy=-3.0, x0=100.0, y0=-50.0, units="px")
    rf = RasterField._from_bare_array(np.zeros((32, 64)), "asym", frame=frame, name="asym")

    assert rf.nx == 64 and rf.ny == 32
    expected_x = 100.0 + 2.0 * np.arange(64, dtype=np.float64)
    expected_y = -50.0 + -3.0 * np.arange(32, dtype=np.float64)
    np.testing.assert_allclose(rf.x_axis, expected_x)
    np.testing.assert_allclose(rf.y_axis, expected_y)
    # LocalFrame.to_scene is a pure pass-through -- composed with the now-correctly-baked axes,
    # the scene placement matches the frame's own declared affine exactly, both corners.
    scene_pts = rf.frame.to_scene(rf.x_axis[[0, -1]], rf.y_axis[[0, -1]])
    np.testing.assert_allclose(scene_pts[:, 0], [100.0, 100.0 + 2.0 * 63])
    np.testing.assert_allclose(scene_pts[:, 1], [-50.0, -50.0 + -3.0 * 31])


def test_from_bare_array_default_frame_axes_stay_plain_pixel_indices():
    """The fix must be a no-op under the default LocalFrame (x0=0, dx=1) -- the overwhelmingly
    common case, and the one every existing caller/test in this suite already relies on."""
    rf = RasterField._from_bare_array(np.zeros((5, 9)), "bare", name="bare")
    assert rf.frame == LocalFrame(units="px")
    np.testing.assert_array_equal(rf.x_axis, np.arange(9, dtype=np.float64))
    np.testing.assert_array_equal(rf.y_axis, np.arange(5, dtype=np.float64))


def test_from_bare_array_with_a_geographic_frame_falls_back_to_pixel_indices():
    """A bare array paired with a GeographicFrame is not a semantically meaningful combination
    (no x0/dx of its own to bake) -- getattr's default keeps the pre-fix arange() axes rather
    than crashing on a missing attribute."""
    rf = RasterField._from_bare_array(np.zeros((4, 6)), "geo-bare", frame=GeographicFrame(),
                                       name="geo-bare")
    np.testing.assert_array_equal(rf.x_axis, np.arange(6, dtype=np.float64))
    np.testing.assert_array_equal(rf.y_axis, np.arange(4, dtype=np.float64))


def test_geographic_geotiff_still_yields_a_geographic_frame(tmp_path):
    """A lon/lat GeoTIFF keeps the geographic path -- the metric frame is for PROJECTED grids."""
    rio = pytest.importorskip("rasterio", reason="rasterio not installed")
    from rasterio.transform import from_origin
    p = tmp_path / "geo.tif"
    with rio.open(p, "w", driver="GTiff", height=20, width=30, count=1, dtype="float32",
                  crs="EPSG:4326", transform=from_origin(-95.0, 28.0, 0.01, 0.01)) as dst:
        dst.write(np.zeros((20, 30), np.float32), 1)
    rf = RasterField.from_geotiff_window(p, row_off=0, col_off=0, height=10, width=10)
    assert isinstance(rf.frame, GeographicFrame)
