# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A minimal window that demonstrates the core gesture: turn a control, watch the fabric change.

One layer, one chain: WTMM -> scale select -> orientation wedge -> modulus threshold. The transform
runs once on load; every control after that is a filter, so moving one is a sub-millisecond redraw
off the cached scale stack rather than a recomputation.

Controls are generated from each device's declared `Param` tuple -- no widget is hand-written per
device, which is the bet Phase 1's gate was built to settle.

    python -m dynamix.shell.demo                     # launch
    python -m dynamix.shell.demo --render out.png    # headless render, for verification
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core.rasterfield import RasterField
from dynamix.devices import register_builtin_devices
from dynamix.engine import Cache, resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import get_device
from dynamix.model.param import ParamKind
from dynamix.model.project import Project

#: The chain the demo drives. Transforms then filters, as the invariant requires.
DEMO_CHAIN = (
    ("wtmm2d", {"n_oct": 3, "n_voice": 4}),
    ("scale_select", {"scale_idx": 0}),
    ("orientation_wedge", {"centre": 0.0, "half_width": 90.0}),
    ("modulus_threshold", {"frac": 0.0}),
)

_STEPS = 1000          # integer resolution of a float slider


class ParamSlider(QtWidgets.QWidget):
    """A slider plus a numeric box, generated from a Param declaration.

    Continuous params get a slider whose span is the param's SOFT range (its useful range) while
    the box accepts anything inside the HARD range -- typing a value outside the slider's span
    widens the view rather than being an error. That distinction is the whole reason Param carries
    two pairs of bounds.
    """

    changed = QtCore.Signal(str, object)

    def __init__(self, param, value=None, parent=None):
        super().__init__(parent)
        self.param = param
        self._emitting = False
        value = param.default if value is None else value

        lo, hi = self._span()
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        label = param.label or param.name
        if param.units:
            label = f"{label} ({param.units})"
        lab = QtWidgets.QLabel(label)
        lab.setMinimumWidth(150)
        row.addWidget(lab)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, _STEPS)
        self.slider.setValue(self._to_slider(value, lo, hi))
        self.slider.valueChanged.connect(self._from_slider)
        row.addWidget(self.slider, 1)

        if param.kind is ParamKind.INT:
            self.box = QtWidgets.QSpinBox()
            self.box.setRange(int(param.min if param.min is not None else -10**6),
                              int(param.max if param.max is not None else 10**6))
        else:
            self.box = QtWidgets.QDoubleSpinBox()
            self.box.setDecimals(2)
            self.box.setRange(float(param.min if param.min is not None else -1e9),
                              float(param.max if param.max is not None else 1e9))
        self.box.setValue(value)
        self.box.setMinimumWidth(80)
        self.box.valueChanged.connect(self._from_box)
        row.addWidget(self.box)

    def _span(self):
        """Slider span: the soft range where declared, else the hard range."""
        lo = self.param.soft_min if self.param.soft_min is not None else self.param.min
        hi = self.param.soft_max if self.param.soft_max is not None else self.param.max
        if lo is None:
            lo = 0.0
        if hi is None:
            hi = lo + (self.param.wrap or 1.0)
        return float(lo), float(hi)

    def _to_slider(self, v, lo, hi):
        if hi <= lo:
            return 0
        return int(round((float(v) - lo) / (hi - lo) * _STEPS))

    def _from_slider(self, raw):
        if self._emitting:
            return
        lo, hi = self._span()
        v = lo + (raw / _STEPS) * (hi - lo)
        if self.param.kind is ParamKind.INT:
            v = int(round(v))
        self._emitting = True
        self.box.setValue(v)
        self._emitting = False
        self.changed.emit(self.param.name, v)

    def _from_box(self, v):
        if self._emitting:
            return
        lo, hi = self._span()
        self._emitting = True
        self.slider.setValue(self._to_slider(v, lo, hi))
        self._emitting = False
        self.changed.emit(self.param.name, v)

    def set_max_from(self, n: int) -> None:
        """Retarget an index param once the stack size is known (scale_select)."""
        self.box.setRange(0, max(0, n - 1))
        self.param = type(self.param)(**{**self.param.to_payload(),
                                         "kind": self.param.kind,
                                         "choices": tuple(self.param.choices),
                                         "max": float(max(0, n - 1)),
                                         "soft_max": float(max(0, n - 1))})


class DemoWindow(QtWidgets.QMainWindow):
    def __init__(self, npz_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DynamiX — multiscale structure")
        register_builtin_devices()

        self.field = RasterField.load_npz(npz_path)
        self.cache = Cache()
        self.project = Project(title=Path(npz_path).stem)
        src = self.project.add_source(npz_path)
        self.layer = self.project.add_layer(
            Path(npz_path).stem, src.source_id,
            Chain(tuple(DeviceRef(n, dict(p)) for n, p in DEMO_CHAIN)),
        )
        self._params = {name: dict(p) for name, p in DEMO_CHAIN}

        central = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(central)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setBackground("w")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.scatter = pg.ScatterPlotItem(size=3, pen=None)
        self.plot.addItem(self.scatter)
        outer.addWidget(self.plot, 1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(430)
        self.panel_layout = QtWidgets.QVBoxLayout(panel)
        outer.addWidget(panel)
        self.setCentralWidget(central)

        self.widgets: dict[tuple[str, str], ParamSlider] = {}
        self._build_controls()

        self.readout = QtWidgets.QLabel("")
        self.readout.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.readout.setWordWrap(True)
        self.panel_layout.addWidget(self.readout)
        self.panel_layout.addStretch(1)

        self.image.setImage(np.asarray(self.field.values, float).T)
        self._first_resolve()

    def _build_controls(self):
        """One control group per device, generated from its declared params."""
        for dev_name, _ in DEMO_CHAIN:
            device = get_device(dev_name)
            box = QtWidgets.QGroupBox(dev_name)
            v = QtWidgets.QVBoxLayout(box)
            for p in device.params:
                if p.kind is ParamKind.CHOICE:
                    continue                      # a combo box; not needed for this demo
                w = ParamSlider(p, self._params[dev_name].get(p.name, p.default))
                w.changed.connect(lambda n, val, d=dev_name: self._on_param(d, n, val))
                v.addWidget(w)
                self.widgets[(dev_name, p.name)] = w
            self.panel_layout.addWidget(box)

    def _chain(self) -> Chain:
        return Chain(tuple(DeviceRef(n, dict(self._params[n])) for n, _ in DEMO_CHAIN))

    def _on_param(self, device: str, name: str, value):
        self._params[device][name] = value
        self.layer.chain = self._chain().materialized()
        self._redraw()

    def set_params(self, device: str, params: dict) -> None:
        """Set params programmatically, keeping the widgets in step.

        Used by the headless render script. Without the widget sync a screenshot would show stale
        control positions beside a correctly-filtered image, which is worse than no screenshot.
        """
        for name, value in params.items():
            self._params[device][name] = value
            w = self.widgets.get((device, name))
            if w is not None:
                w.blockSignals(True)
                w._emitting = True
                w.box.setValue(value)
                lo, hi = w._span()
                w.slider.setValue(w._to_slider(value, lo, hi))
                w._emitting = False
                w.blockSignals(False)
        self.layer.chain = self._chain().materialized()
        self._redraw()

    def reset_params(self) -> None:
        """Restore every control to its device's declared default."""
        for dev_name, _ in DEMO_CHAIN:
            device = get_device(dev_name)
            self.set_params(dev_name, {p.name: p.default for p in device.params
                                       if p.kind is not ParamKind.CHOICE})

    def _first_resolve(self):
        self.layer.chain = self._chain().materialized()
        t = time.perf_counter()
        r = resolve(self.layer, self.field, self.cache)
        self._transform_ms = (time.perf_counter() - t) * 1000
        n = len(r.result.get("scales", []))
        w = self.widgets.get(("scale_select", "scale_idx"))
        if w is not None and n:
            w.set_max_from(n)
            w.slider.setRange(0, max(0, n - 1))
            w.slider.setValue(0)
        self._draw(r, self._transform_ms)

    def _redraw(self):
        t = time.perf_counter()
        r = resolve(self.layer, self.field, self.cache)
        self._draw(r, (time.perf_counter() - t) * 1000)

    def _draw(self, r, ms: float):
        layers = r.result.get("extrema") or []
        if layers:
            lay = layers[0]
            x = np.asarray(lay.get("x", []), float)
            y = np.asarray(lay.get("y", []), float)
            mod = np.asarray(lay.get("mod", []), float)
            if len(mod) and np.nanmax(mod) > 0:
                t = np.clip(mod / np.nanmax(mod), 0, 1)
                brushes = [pg.mkBrush(int(255 * v), int(60 + 120 * (1 - v)),
                                      int(40 + 180 * (1 - v)), 220) for v in t]
            else:
                brushes = None
            self.scatter.setData(x=x, y=y, brush=brushes)
            n = len(x)
        else:
            self.scatter.setData(x=[], y=[])
            n = 0

        scale_px = r.result.get("_scale_px")
        phys = self._physical(scale_px) if scale_px else "—"
        self.readout.setText(
            f"extrema shown : {n}\n"
            f"scale         : {('%.1f px = %s' % (scale_px, phys)) if scale_px else '—'}\n"
            f"px size       : {self._px_to_unit():.4g} {self._units()}/px\n"
            f"redraw        : {ms:.1f} ms\n"
            f"cache         : {r.cache_hits} hit / {r.cache_misses} miss"
            f"{'  (filters only)' if r.from_cache else '  (TRANSFORM RAN)'}\n"
            f"transforms    : {', '.join(r.transforms_run) or '—'}\n"
            f"filters       : {', '.join(r.filters_run) or '—'}"
        )

    def _units(self) -> str:
        return str(getattr(getattr(self.field, "frame", None), "units", "px") or "px")

    def _physical(self, scale_px: float) -> str:
        """A scale in physical units, plus an approximate km for a geographic frame.

        Degrees are the honest unit for a `GeographicFrame`; the km figure is explicitly marked
        approximate because a degree of longitude shrinks with latitude, and the spec's rule is
        that pixels are canonical and conversion happens only for display.
        """
        u = self._units()
        v = scale_px * self._px_to_unit()
        if u == "deg":
            return f"{v:.3g} deg (~{v * 111.195:.0f} km at the equator)"
        return f"{v:.4g} {u}"

    def _px_to_unit(self) -> float:
        """Physical size of one pixel, derived from the field's own axis.

        NOT from a frame attribute: ``LocalFrame`` carries ``dx`` but ``GeographicFrame`` does not,
        and a ``getattr(frame, "dx", 1.0)`` fallback silently reported degrees-per-pixel as 1.0 —
        a scale of 11.7 px printed as "11.7 deg" when it is 1.95. The axis is the one source that
        exists for every frame kind.
        """
        ax = np.asarray(getattr(self.field, "x_axis", []), float)
        if ax.size < 2:
            return 1.0
        return float(abs(ax[-1] - ax[0]) / (ax.size - 1))


def _default_npz() -> str:
    here = Path(__file__).resolve().parents[3]
    return str(here / "tests" / "fixtures" / "kam_64.npz")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    render_to = None
    if "--render" in argv:
        i = argv.index("--render")
        render_to = argv[i + 1]
        del argv[i:i + 2]
    npz = argv[0] if argv else _default_npz()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = DemoWindow(npz)
    win.resize(1280, 760)

    if render_to:
        win.show()
        app.processEvents()
        for _ in range(3):
            app.processEvents()
        win.grab().save(render_to)
        print(f"rendered -> {render_to}")
        return 0

    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
