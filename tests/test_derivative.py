# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Derivative datasets: a result's rasters and/or vectors written once to an npz that opens as an
ordinary dataset (RasterField's own schema, vectors under ``vec_*``), on the result's own grid."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _field(ny=12, nx=15):
    rng = np.random.default_rng(0)
    return RasterField(name="f", values=rng.standard_normal((ny, nx)), frame=LocalFrame(dx=2.0),
                       x_axis=np.arange(nx) * 2.0, y_axis=np.arange(ny) * 2.0)


def _extrema():
    return [{"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6]), "mod": np.array([.5, .6, .7]),
             "arg": np.array([0., .1, .2]), "line_id": np.array([0, 0, -1])},
            {"x": np.array([7]), "y": np.array([8]), "mod": np.array([.9]),
             "arg": np.array([.3]), "line_id": np.array([-1])}]


def _chains():
    return [{"x": np.array([1, 1]), "y": np.array([4, 4]), "mod": np.array([.5, .8])},
            {"x": np.array([7, 6, 6]), "y": np.array([8, 8, 9]), "mod": np.array([.9, 1., 2.])}]


def test_raster_choices_list_the_shown_raster_first_then_every_stack():
    from dynamix.core.derivative import raster_choices

    comps = np.arange(3 * 12 * 15, dtype=float).reshape(3, 12, 15)
    result = {"_shape": (12, 15), "raster_out": comps[0] * 2, "ssa_components": comps,
              "ssa_residual": np.ones((12, 15)), "filtered": np.zeros((5, 5)),
              "params": {}}
    labels = [label for label, _a in raster_choices(result, shown_label="data − [1]")]
    assert labels == ["data − [1]", "C1", "C2", "C3", "residual"]    # wrong-grid "filtered" out


def test_tucker_choices_follow_the_orientation_and_the_input_bands():
    from dynamix.core.derivative import raster_choices

    sep, comb = np.zeros((2, 4, 5)), np.ones((3, 4, 5))
    base = {"_shape": (4, 5), "raster_out": np.zeros((4, 5)), "tucker_components": sep,
            "tucker_combined_components": comb, "tucker_recon": np.zeros((4, 5)),
            "tucker_recon_bands": np.zeros((4, 5, 2))}
    labels = [lbl for lbl, _a in raster_choices({**base, "params": {"pairs": "combined"}})]
    assert labels[:4] == ["as shown", "C1", "C2", "C3"]
    assert "recon band 1" in labels and "recon band 2" in labels
    labels = [lbl for lbl, _a in raster_choices({**base, "params": {"pairs": "separate"}})]
    assert labels[:3] == ["as shown", "C1", "C2"] and "C3" not in labels


def test_the_grid_is_the_roi_window_for_an_roi_result_else_the_field():
    from dynamix.core.derivative import result_grid

    field = _field()
    frame, x, y = result_grid({"raster_out": np.zeros((12, 15))}, field)
    assert frame is field.frame and np.array_equal(x, field.x_axis)
    roi_frame = LocalFrame(x0=4.0, y0=6.0, dx=2.0)
    frame, x, y = result_grid({"_roi_axes": (np.array([4., 6.]), np.array([6., 8., 10.])),
                               "_frame": roi_frame}, field)
    assert frame is roi_frame and x.tolist() == [4., 6.] and y.tolist() == [6., 8., 10.]


def test_one_band_opens_as_an_ordinary_2d_dataset(tmp_path):
    from dynamix.core.derivative import write_derivative

    field = _field()
    band = field.values * 3.0
    band[2, 3] = np.nan
    path = write_derivative(tmp_path / "d.npz", bands=[("data − [1]", band)], frame=field.frame,
                            x_axis=field.x_axis, y_axis=field.y_axis, name="dem · ssa2d",
                            units="m", provenance={"derived": {"from": "dem"}})
    back = RasterField.from_file(path)
    np.testing.assert_array_equal(back.values, band)
    assert back.values.ndim == 2 and back.name == "dem · ssa2d" and back.units == "m"
    np.testing.assert_array_equal(back.x_axis, field.x_axis)
    assert back.frame == field.frame
    assert back.provenance["bands"] == ["data − [1]"]
    assert back.provenance["derived"] == {"from": "dem"}


def test_several_bands_stack_into_a_multiband_dataset(tmp_path):
    from dynamix.core.derivative import write_derivative

    field = _field()
    a, b = field.values, -field.values
    path = write_derivative(tmp_path / "d.npz", bands=[("C1", a), ("C2", b)], frame=field.frame,
                            x_axis=field.x_axis, y_axis=field.y_axis, name="stack")
    back = RasterField.from_file(path)
    assert back.values.shape == (12, 15, 2)
    np.testing.assert_array_equal(back.values[..., 1], b)
    assert back.provenance["bands"] == ["C1", "C2"]


def test_vectors_round_trip_and_a_vector_only_derivative_has_an_empty_raster(tmp_path):
    from dynamix.core.derivative import read_vectors, vector_counts, write_derivative

    field = _field()
    vectors = {"extrema": _extrema(), "chains": _chains(), "scales": np.array([1.0, 2.0, 4.0])}
    assert vector_counts(vectors) == (4, 2)
    path = write_derivative(tmp_path / "v.npz", bands=[], vectors=vectors, frame=field.frame,
                            x_axis=field.x_axis, y_axis=field.y_axis, name="edges")
    back = RasterField.from_file(path)
    assert back.values.shape == (12, 15) and np.isnan(back.values).all()
    got = read_vectors(path)
    for want, have in zip(_extrema(), got["extrema"]):
        for key in ("x", "y", "mod", "arg", "line_id"):
            np.testing.assert_array_equal(have[key], want[key])
    for want, have in zip(_chains(), got["chains"]):
        for key in ("x", "y", "mod"):
            np.testing.assert_array_equal(have[key], want[key])
    np.testing.assert_array_equal(got["scales"], [1.0, 2.0, 4.0])


def test_a_raster_only_derivative_has_no_vectors(tmp_path):
    from dynamix.core.derivative import read_vectors, write_derivative

    field = _field()
    path = write_derivative(tmp_path / "r.npz", bands=[("x", field.values)], frame=field.frame,
                            x_axis=field.x_axis, y_axis=field.y_axis, name="r")
    assert read_vectors(path) is None


def test_extrema_without_arg_or_line_id_still_pack(tmp_path):
    from dynamix.core.derivative import read_vectors, write_derivative

    field = _field()
    bare = [{"x": np.array([1, 2]), "y": np.array([3, 4]), "mod": np.array([1., 2.])}]
    path = write_derivative(tmp_path / "b.npz", bands=[], vectors={"extrema": bare, "chains": []},
                            frame=field.frame, x_axis=field.x_axis, y_axis=field.y_axis, name="b")
    got = read_vectors(path)
    assert got["extrema"][0]["line_id"].tolist() == [-1, -1]
    assert got["scales"].tolist() == [1.0]                 # no scales: one per level, marked
    assert RasterField.from_file(path).provenance["vector_scales"] == "level index (none given)"


def test_nothing_to_fork_and_a_wrong_grid_are_refused(tmp_path):
    from dynamix.core.derivative import write_derivative

    field = _field()
    with pytest.raises(ValueError, match="nothing"):
        write_derivative(tmp_path / "n.npz", bands=[], frame=field.frame, x_axis=field.x_axis,
                         y_axis=field.y_axis, name="n")
    with pytest.raises(ValueError, match="grid"):
        write_derivative(tmp_path / "g.npz", bands=[("x", np.zeros((3, 3)))], frame=field.frame,
                         x_axis=field.x_axis, y_axis=field.y_axis, name="g")


def test_the_derived_vectors_device_emits_the_frozen_vectors_and_filters_narrow_them(
        tmp_path, clean_registry):
    from dynamix.core.derivative import write_derivative
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer

    register_builtin_devices()
    field = _field()
    path = str(write_derivative(
        tmp_path / "v.npz", bands=[], frame=field.frame, x_axis=field.x_axis,
        y_axis=field.y_axis, name="edges",
        vectors={"extrema": _extrema(), "chains": _chains(), "scales": np.array([1., 2.])}))
    opened = RasterField.from_file(path)

    def run(*steps):
        layer = Layer(layer_id=1, name="L", source_id="s",
                      chain=Chain(tuple(DeviceRef(d, p) for d, p in steps)).materialized())
        return resolve(layer, opened, Cache()).result

    whole = run(("derived_vectors", {"_path": path}))
    assert [len(e["x"]) for e in whole["extrema"]] == [3, 1]
    assert len(whole["chains"]) == 2 and whole["_shape"] == (12, 15)
    one = run(("derived_vectors", {"_path": path}), ("scale_select", {"scale_idx": 1}))
    assert [e["x"].tolist() for e in one["extrema"]] == [[7]]


def test_a_derived_vectors_layer_with_no_file_is_empty_not_an_error(clean_registry):
    from dynamix.devices.derived_vectors import DerivedVectors

    out = DerivedVectors().compute(_field(), {"_path": ""})
    assert out["extrema"] == [] and out["chains"] == []
