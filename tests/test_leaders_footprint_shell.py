# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Leader footprint overlays use existing selection, scale and view commands."""
import numpy as np
from PySide6 import QtWidgets

from dynamix.shell.canvas import Canvas
from dynamix.devices import register_builtin_devices
from dynamix.model.device import get_device,defaults_for
from tests.test_roi_outputs import _field


def test_leader_overlay_draws_and_clears_without_transform(qtbot,clean_registry,monkeypatch,tmp_path):
    register_builtin_devices();dev=get_device('wavelet_leaders')
    def no_compute(*args,**kwargs):raise AssertionError('preview ran the transform')
    monkeypatch.setattr(type(dev),'compute',no_compute)
    field=_field(512,512)
    params={**defaults_for(dev),'scale_min':2,'scale_max':16,'with_wse':True}
    c=Canvas();qtbot.addWidget(c);c.resize(900,700);c.set_field(field)
    c.view.setRange(xRange=(0,512),yRange=(0,512),padding=0)
    for i,fp in enumerate([dev.footprint(field,params,2,stride=4),
               dev.roi_footprint(field,params,(192,192,128,128),2,stride=4),
               dev.roi_footprint(field,params,(0,0,128,128),2,stride=4)]):
        c.set_footprint(fp)
        assert c.footprint_cells_item.getData()[0].size>0
        assert c.footprint_grid_item.getData()[0].size>0
        assert c.footprint_label.toPlainText()
        c.grab().save(str(tmp_path/f'leaders-footprint-{i}.png'))
    c.clear_footprint()
    assert c.footprint_cells_item.getData()[0] is None or len(c.footprint_cells_item.getData()[0])==0
    assert c.footprint_grid_item.getData()[0] is None or len(c.footprint_grid_item.getData()[0])==0
    assert c.footprint_wse_item.path().isEmpty()
