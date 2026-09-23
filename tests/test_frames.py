# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_frames.py"""
import numpy as np
import pytest
from dynamix.core import projection
from dynamix.core.frames import GeographicFrame, LocalFrame, frame_from_meta, frame_to_meta


@pytest.mark.parametrize("mode", projection.MODES)
def test_geographic_matches_projection(mode):
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 50); lat = rng.uniform(-85, 85, 50)
    dep = rng.uniform(0, 600, 50)
    f = GeographicFrame(mode=mode, vexag=2.0)
    np.testing.assert_allclose(f.to_scene(lon, lat, dep),
                               projection.project(lon, lat, dep, mode, 2.0))

def test_geographic_z_none_is_surface():
    f = GeographicFrame(mode="pacific")
    out = f.to_scene(np.array([10.0]), np.array([20.0]))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out, projection.project([10.0], [20.0], [0.0], "pacific", 1.0))

def test_local_passthrough_and_scalars():
    f = LocalFrame(units="µm")
    out = f.to_scene(3.0, 4.0)                       # scalars broadcast to (1, 3)
    np.testing.assert_allclose(out, [[3.0, 4.0, 0.0]])
    x = np.arange(4.0); y = np.arange(4.0) * 2
    np.testing.assert_allclose(f.to_scene(x, y, x)[:, 2], x)

def test_extent_and_labels():
    f = LocalFrame(units="µm")
    assert f.extent(np.array([0.0, 1.5]), np.array([-2.0, 2.0])) == (0.0, 1.5, -2.0, 2.0)
    assert f.axis_labels() == ("x (µm)", "y (µm)")
    g = GeographicFrame()
    assert g.axis_labels() == ("lon (deg)", "lat (deg)")
    assert g.kind == "geographic" and f.kind == "local"

@pytest.mark.parametrize("frame", [GeographicFrame(mode="globe", vexag=3.0),
                                   LocalFrame(x0=1.0, y0=-2.0, dx=0.5, dy=0.5, units="µm")])
def test_meta_roundtrip_exact(frame):
    assert frame_from_meta(frame_to_meta(frame)) == frame

def test_meta_unknown_kind_raises():
    with pytest.raises(ValueError):
        frame_from_meta({"kind": "martian"})

def test_frames_compatible():
    from dynamix.core.frames import frames_compatible
    a = LocalFrame(x0=0.0, y0=0.0, dx=1.0, dy=1.0, units="px")
    b = LocalFrame(x0=0.0, y0=0.0, dx=1.0, dy=1.0, units="px")
    c = LocalFrame(x0=0.0, y0=0.0, dx=2.0, dy=1.0, units="px")
    d = LocalFrame(x0=0.0, y0=0.0, dx=1.0, dy=1.0, units="um")
    e = GeographicFrame(mode="pacific", vexag=1.0)
    assert frames_compatible(a, b)
    assert not frames_compatible(a, c)
    assert not frames_compatible(a, d)
    assert not frames_compatible(a, e)
