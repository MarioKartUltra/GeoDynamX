# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The export UI cannot relabel a held raster or recite a new recipe over old bytes."""
import dataclasses

from test_map_integration import geo_window  # noqa: F401
from test_roi_shell_flow import registered_builtins  # noqa: F401

from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve, shown_level
from dynamix.model.chain import Chain, DeviceRef


def test_lazy_derived_export_requires_the_actual_output_identity(geo_window, qtbot, monkeypatch):
    win = geo_window
    win.layer.chain = Chain((DeviceRef('interpolating_wavelets', {'show': 'horizontal'}),))
    win._derived_fields[win.layer.layer_id] = win.field
    monkeypatch.setattr(win, '_displayed_output_key', lambda layer: 'current')
    assert win._derived_export_field(win.layer) is None
    old = dataclasses.replace(win.field, provenance={'_output_identity': {
        'key': 'old', 'name': 'horizontal', 'level': 2}})
    win._derived_fields[win.layer.layer_id] = old
    assert win._derived_export_field(win.layer) is None
    actual = dataclasses.replace(old, provenance={'_output_identity': {
        'key': 'current', 'name': 'horizontal', 'level': 2}})
    win._derived_fields[win.layer.layer_id] = actual
    exported = win._derived_export_field(win.layer)
    assert exported.provenance['level'] == 2
    assert exported.provenance['output'] == 'interpolating_wavelets/horizontal'


def test_repeated_record_refuses_changed_export_bytes(geo_window, tmp_path):
    win = geo_window
    win.layer.chain = Chain((DeviceRef('wtmm2d', {'n_oct': 1, 'n_voice': 2, 'precision': 'f64'}),))
    win._active_result = resolve(win.layer, win.field, Cache(), level=shown_level(win.layer)).result
    path = tmp_path/'chains.npz'
    path.write_bytes(b'first export')
    assert ' · provenance ' in win._write_provenance(path, 'chains_npz')
    sidecar = path.with_name(path.name+'.gprv')
    original = sidecar.read_bytes()
    path.write_bytes(b'changed file')
    status = win._write_provenance(path, 'chains_npz', repeat=True)
    assert 'no provenance record' in status and 'changed' in status
    assert sidecar.read_bytes() == original


def test_preparation_field_export_preserves_method_but_not_runtime_proof(geo_window, tmp_path):
    from dynamix.core.rasterfield import RasterField
    win = geo_window
    win.layer.chain = Chain((DeviceRef('fractional_integration', {'integration_order': .4}),))
    field = resolve(win.layer, win.field, Cache(), level=shown_level(win.layer)).result
    win._land_field_result(field)
    exported = win._derived_export_field(win.layer)
    assert exported is field
    path = tmp_path/'integrated.npz'
    exported.save_npz(path)
    assert ' · provenance ' in win._write_provenance(path, 'derived_npz')
    loaded = RasterField.load_npz(path)
    assert '_landed' not in loaded.provenance and '_execution' not in loaded.provenance
    assert loaded.provenance['fractional_integration'][0]['order'] == .4


def test_unsupported_record_version_does_not_destroy_export_or_existing_sidecar(
        geo_window, tmp_path, monkeypatch):
    import gprv
    import pytest
    from dynamix.engine import gprv_export
    if gprv.FORMAT_VERSION != '0.2':
        pytest.skip('exercises the installed 0.2 library failure boundary')
    win = geo_window
    win.layer.chain = Chain((DeviceRef('wtmm2d', {'n_oct': 1, 'n_voice': 2, 'precision': 'f64'}),))
    win._active_result = resolve(win.layer, win.field, Cache(), level=shown_level(win.layer)).result
    path = tmp_path/'chains.npz'
    path.write_bytes(b'exported data')
    sidecar = path.with_name(path.name+'.gprv')
    sidecar.write_bytes(b'earlier immutable record')
    record_for = gprv_export.record_for
    def future_record(*args, **kwargs):
        return {**record_for(*args, **kwargs), 'gprv': '0.3'}
    monkeypatch.setattr(gprv_export, 'record_for', future_record)
    status = win._write_provenance(path, 'chains_npz')
    assert 'no provenance record' in status and "unknown version '0.3'" in status
    assert path.read_bytes() == b'exported data'
    assert sidecar.read_bytes() == b'earlier immutable record'
