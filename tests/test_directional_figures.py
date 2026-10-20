# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest
from PySide6 import QtWidgets, QtCore

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.shell.directional_figures import DirectionalFigure, TOOLS
from dynamix.shell.slope_plot import SlopePlot


def source():
    rng = np.random.default_rng(5)
    field = RasterField(name="rough",values=rng.normal(size=(64,64)),frame=LocalFrame(),
                        x_axis=np.arange(64.), y_axis=np.arange(64.))
    extrema=[]
    for a in (1,2,4,8):
        x = np.tile(np.arange(1,32),4)
        y = np.repeat([5,15,25,35],31)
        extrema.append(dict(x=x,y=y,mod=(1+np.sin(x)**2)*a**.4,arg=.3*np.sin(x),line_id=np.repeat(np.arange(4),31)))
    return field,dict(extrema=extrema,scales=np.array([1.,2,4,8]),_shape=(64,64))


@pytest.mark.parametrize("kind",list(TOOLS))
def test_each_tool_opens_and_renders_only_its_own_plots(qtbot,kind):
    field,result=source()
    w=DirectionalFigure(kind)
    qtbot.addWidget(w)
    w.set_result(result,field)
    w.show()
    assert w.data.scales.size==4
    assert len(w.plots)<=6
    assert w.context.text()
    if kind in ("net","distributions","diagnostics"):
        w.population.setCurrentIndex(1)
        assert "requires" not in w.readout.text().lower()
    if kind=="net":
        assert len(w.plots[0].listDataItems())>4
        w.projection.setCurrentText("Wulff")
        assert w.render_state()["controls"]["Projection"]=="Wulff"


def test_two_click_manual_segment_and_drag_keep_data_coordinates(qtbot):
    w=SlopePlot("Power",quantity="β",factor=-1)
    qtbot.addWidget(w)
    w.set_data(np.arange(5.),-2*np.arange(5.))
    w.set_segment([[0,0],[4,-8]])
    assert w.measurement["value"]==2
    assert "se" not in w.measurement
    w.plot.setXRange(-10,10)
    assert w.measurement["value"]==2
    w.clear_manual()
    assert not w.measurement


def test_scalar_compute_is_explicit_and_keeps_signed_values(qtbot):
    f,r=source()
    w=DirectionalFigure("scalar")
    qtbot.addWidget(w)
    w.set_result(r,f)
    assert w._computed is None
    w.compute.click()
    qtbot.waitUntil(lambda:not w._busy,timeout=15000)
    assert w._computed is not None,w.readout.text()
    v=w._computed["signed"]
    assert v.min()<0<v.max()


@pytest.mark.parametrize("kind",["scalar","correlations"])
def test_dense_statistics_reject_missing_source_support(qtbot,kind):
    f,r=source()
    f.values[10,10]=np.nan
    w=DirectionalFigure(kind)
    qtbot.addWidget(w)
    w.set_result(r,f)
    if kind=="correlations":
        w.route.setCurrentIndex(1)
    with pytest.raises(ValueError,match="finite source pixels"):
        w._calculation()


def test_source_pin_and_scale_visibility_are_independent(qtbot):
    f,r=source()
    w=DirectionalFigure("net")
    qtbot.addWidget(w)
    w.set_result(r,f)
    w.pin.setChecked(True)
    other=dict(r,scales=np.array([2.,4,8,16]))
    w.set_result(other,f)
    assert w.data.result is r
    w.scales.item(1).setCheckState(QtCore.Qt.Unchecked)
    assert w.selected_scales()==[0,2,3]


def test_manual_is_two_scene_clicks_and_numeric_edits(qtbot):
    w=SlopePlot("Power",quantity="β",factor=-1)
    qtbot.addWidget(w)
    w.resize(700,500)
    w.set_data(np.arange(5.),-2*np.arange(5.))
    w.mode.setCurrentText("Manual")
    w.show()
    qtbot.wait(30)
    view=w.plot.getViewBox()
    def click(x,y):
        scene=view.mapViewToScene(QtCore.QPointF(x,y))
        pos=w.plot.mapFromScene(scene)
        qtbot.mouseClick(w.plot.viewport(),QtCore.Qt.LeftButton,pos=pos)
    click(1,-2)
    assert w._anchor is not None and not w.measurement
    click(3,-6)
    assert w.measurement["value"]==pytest.approx(2,abs=.04)
    w.endpoints[3].setValue(-4)
    assert w.measurement["value"]==pytest.approx(1,abs=.05)


@pytest.mark.parametrize("kind",["kernel","correlations"])
def test_statistics_compute_and_export_state_are_explicit(qtbot,kind):
    from dynamix.shell.directional_figures import _json_ready
    import json
    f,r=source()
    w=DirectionalFigure(kind)
    qtbot.addWidget(w)
    w.set_result(r,f)
    w.compute.click()
    qtbot.waitUntil(lambda:not w._busy,timeout=15000)
    assert w._computed is not None,w.readout.text()
    assert w.plots[0].listDataItems()
    json.dumps(_json_ready(w._computed),allow_nan=False)


def test_structural_lines_do_not_require_inventing_a_map_reference(qtbot):
    f,r=source()
    w=DirectionalFigure("net")
    qtbot.addWidget(w)
    w.set_result(r,f)
    w.add_orientation(name="line",kind="line",azimuth=90,inclination=45,frame="true")
    assert w.frame_combo.currentData()=="true"
    assert "No geographic" not in w.readout.text()


from tests.test_shell_window import stub_devices, window  # noqa: E402,F401


def test_main_window_menu_opens_focused_figures_and_highlights_points(window,qtbot):
    f,r=source()
    win=window
    win._start_worker=lambda:None
    win.load_field(f,"mem:focused-figures")
    win._active_result=r
    win._open_directional_figure("net")
    net=win._figure_windows["net"]
    assert net.isVisible()
    net.brush_hi.setValue(180)
    net.brush_button.click()
    assert len(win._figure_selection_item.getData()[0])>0
