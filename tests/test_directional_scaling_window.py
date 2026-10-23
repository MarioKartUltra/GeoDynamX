# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Directional moment viewing preserves the estimator and previews two independent axes."""
import json

import numpy as np
import pytest

from dynamix.core import directional as dr
from dynamix.shell.directional_scaling_window import DirectionalScalingWindow


@pytest.fixture
def result():
    J = 8
    mods = {(x,y): np.full((2,2), 2.**(.4*((x or J)+(y or J)))) for x in range(J+1) for y in range(J+1)}
    coef = {'levels': J, 'mods': mods, 'valid': {k: np.ones((2,2), bool) for k in mods},
            'border_cells': 0, 'complete': True, 'seam': False, 'origin': (0,0), 'data_shape': (512,512)}
    data = dr.analyse(np.zeros((32,32)), 'db2', J, j_min=1, p_list=[1.,2.,3.,4.],
                      estimator='gamma', coefficient_cache=coef)
    return {'_directional': data, 'params': {'wavelet': 'db2', 'scale_max': 256, 'estimator': 'gamma'}, 'record': {'method': 'directional_scaling'}}


def test_moment_and_envelope_views_use_retained_populations(qtbot, result, tmp_path):
    win = DirectionalScalingWindow(result, source='test band')
    qtbot.addWidget(win)
    win.show()
    qtbot.wait(30)
    win.grab()
    d = result['_directional']
    assert 'No finite' in win.scaling_plots[2].plotItem.titleLabel.text
    fitted = int(np.isfinite(d['eta_alpha']).any(axis=0).sum())
    assert f"{fitted} of {len(d['alpha'])} anisotropies fitted" in win.scaling_plots[0].plotItem.titleLabel.text
    assert ('needs two fitted' in win.scaling_plots[1].plotItem.titleLabel.text) == (fitted < 2)
    ai = win.alpha.currentIndex()
    fit = d['diagnostic_fits'][ai]
    plotted = win.envelope_plot.listDataItems()[0]
    np.testing.assert_array_equal(plotted.getData()[0], fit['x'])
    np.testing.assert_array_equal(plotted.getData()[1], fit['envelopes'][0,0])
    win.alpha.setCurrentIndex(0)  # unresolved anisotropy must stay blank rather than invent a fit
    assert 'no exponent fit' in win.moment_note.text()
    path = tmp_path/'moments.npz'
    win.export_data(path)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['log2_moments'], d['log2_moments'])
        assert saved['membership'].shape[1] == 5
        context = json.loads(str(saved['context']))
    assert context['source'] == 'test band'
    assert context['estimator'] == d['estimator'] and context['resolution_origin_J'] == 8
    assert context['families'] == ['Γ union'] and context['x'] == d['diagnostic_fits'][0]['xlabel']
    assert 'upper bound' in context['L_e1, L_e2'] and 'not a measured D(h)' in context['L_e1, L_e2']


def test_legacy_result_names_its_three_families(qtbot):
    J = 8
    mods = {(x,y): np.full((2,2), 2.**(.4*((x or J)+(y or J)))) for x in range(J+1) for y in range(J+1)}
    coef = {'levels': J, 'mods': mods, 'valid': {k: np.ones((2,2), bool) for k in mods},
            'border_cells': 0, 'complete': True, 'seam': False, 'origin': (0,0), 'data_shape': (512,512)}
    data = dr.analyse(np.zeros((32,32)), 'db2', J, j_min=1, p_list=[1.,2.], estimator='legacy', coefficient_cache=coef)
    win = DirectionalScalingWindow({'_directional': data, 'params': {'wavelet': 'db2', 'scale_max': 256},
                                    'record': {'method': 'directional_scaling'}}, source='legacy band')
    qtbot.addWidget(win)
    assert [win.family.itemText(i) for i in range(win.family.count())] == ['near diagonal', 'x-side', 'y-side']
    assert data['diagnostic_fits'][0]['xlabel'].startswith('Γ index j')


def test_basis_preview_can_be_opened_without_computing_and_keeps_phi_distinct(qtbot):
    win = DirectionalScalingWindow(params={'wavelet': 'db2', 'scale_max': 64})
    qtbot.addWidget(win)
    assert win.tabs.currentIndex() == 2 and not win.tabs.isTabEnabled(0)
    win.jx.setCurrentIndex(0)
    win.jy.setCurrentIndex(3)
    x,y = win.preview
    assert x['branch'] == 'φ' and np.isnan(x['wavelength'])
    assert y['branch'] == 'ψ' and y['cell_px'] == 8
    assert 'no bandpass' in win.basis_note.text()
    with qtbot.waitSignal(win.bankRequested) as signal:
        win.bank.setCurrentText('db3')
        win.use_bank.click()
    assert signal.args == ['db3']
    win.show()
    qtbot.wait(20)
    win.grab()


def test_preview_settings_do_not_relabel_the_computed_bank(qtbot, result):
    win = DirectionalScalingWindow(result, params={'wavelet': 'sym4', 'scale_max': 64})
    qtbot.addWidget(win)
    assert 'db2' in win.caption.text()
    assert win.bank.currentText() == 'sym4'
    assert win.params['wavelet'] == 'db2'


def test_directional_result_opens_its_own_partition_window_and_bank_edit_respects_owner(qtbot, clean_registry, result, monkeypatch):
    from conftest import make_field
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import get_device, validate_params
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(make_field(shape=(32,32)), '/nonexistent/directional.npz', inert=True)
    params = validate_params(get_device('directional_scaling'), {'scale_max': 16})
    win._set_recipe(['directional_scaling'], [params])
    win.layer.chain = win._chain()
    win._build_strips()
    win._active_result = result
    win._refresh_spectrum_button()
    assert win._spectrum_button.isEnabled()
    assert win._spectrum_button.text() == 'Directional partition functions…'
    win._on_spectrum_button_clicked()
    view = win._directional_window
    assert view.data is result['_directional']
    dispatch = []
    monkeypatch.setattr(win, '_start_worker', lambda: dispatch.append(True))
    win._on_directional_bank_requested(-1, 'db3')
    assert win._params[0]['wavelet'] == 'sym4'
    win.layer.tags["ui.lock"] = "1"
    win._on_directional_bank_requested(win.layer.layer_id, 'db3')
    assert win._params[0]['wavelet'] == 'sym4'
    win.layer.tags.pop("ui.lock")
    win._on_directional_bank_requested(win.layer.layer_id, 'db3')
    assert win._params[0]['wavelet'] == 'db3'
    assert view.params['wavelet'] == 'db2'
    win._forget_layer_tools(win.layer.layer_id)
    assert win._directional_window is None



def test_basis_labels_keep_independent_native_axis_spacings(qtbot):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    field = RasterField(name='rectangular grid', values=np.zeros((32,32)), frame=LocalFrame(units='m'),
                        x_axis=2*np.arange(32.), y_axis=3*np.arange(32.))
    win = DirectionalScalingWindow(params={'wavelet': 'db2', 'scale_max': 16}, field=field)
    qtbot.addWidget(win)
    assert 'Native-axis cell spans 8 × 12 m' in win.basis_note.text()


def test_the_tools_warnings_head_the_window_and_gamma_names_the_integrated_field(qtbot, result):
    warned = dict(result, warnings=["η stays below d = 2 at every fitted anisotropy"],
                  params={**result["params"], "gamma": 1.0})
    win = DirectionalScalingWindow(warned, source='test band')
    qtbot.addWidget(win)
    assert not win.warning.isHidden() and "η stays below" in win.warning.text()
    assert "γ = 1" in win.scaling_plots[2].getAxis('bottom').labelText
    plain = DirectionalScalingWindow(result, source='test band')
    qtbot.addWidget(plain)
    assert plain.warning.isHidden()
    assert plain.scaling_plots[2].getAxis('bottom').labelText == 'H'
