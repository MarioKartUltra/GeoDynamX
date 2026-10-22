# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Focused scale-statistics figures over shared, spatially identified analysis samples."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import anisotropy as an
from dynamix.core.figure_fits import range_fit
from dynamix.core.orientation import Orientation, Frame
from dynamix.shell.figure_data import FigureData
from dynamix.shell.slope_plot import SlopePlot
from dynamix.shell.theme import current
from dynamix.shell.plot_zoom import zoom_plot
from dynamix.shell.window_fit import TOOL_WINDOW, show_while_inactive

TOOLS = {
    "distributions": "WTMM distributions",
    "collapse": "Directional Hurst collapse",
    "sensitivity": "Collapse sensitivity",
    "moments": "Moment convergence",
    "fourier": "Fourier spectra",
    "diagnostics": "Anisotropy diagnostics",
    "net": "Orientation net",
    "kernel": "Self-similarity kernel",
    "correlations": "Space–scale correlations",
    "scalar": "Signed scalar wavelet distributions",
    "spread": "Component spread fitting",
}
# Stable data identities, independent of which scales are visible.
_SCALE_COLORS = ((220, 220, 220), (132, 180, 220), (220, 180, 125), (165, 205, 160),
                 (200, 155, 200), (210, 165, 155))
_SYMBOLS = ("d", "t", "s", "o", "+", "x")
#: Bytes of working set (sixteen per sample per scale: the complex spectrum, the filtered
#: spectrum and the signed values) above which the signed scalar CWT stack is written scale by
#: scale into a memory-mapped file in the scratch folder beside the settings file.
SIGNED_STACK_BUDGET = 512 * 1024 ** 2


def _release(result) -> None:
    """Remove a result's memory-mapped stack file, when it has one. Windows refuses while the
    map is open; the start-up sweep takes it then."""
    name = getattr((result or {}).get("signed"), "filename", None)
    if name:
        try:
            Path(name).unlink()
        except OSError:
            pass


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object, object)


class _Calculation(QtCore.QRunnable):
    def __init__(self, function, cancel):
        super().__init__()
        self.function, self.cancel = function, cancel
        self.signals = _Signals()

    def run(self):
        try:
            from dynamix.core.fft_policy import observing
            with observing() as contexts:
                result, error = self.function(self.cancel), None
            result["fft_contexts"] = contexts
        except Exception as exc:
            result, error = None, str(exc)
        self.signals.done.emit(result, error)


def _spin(lo, hi, value, step=.01):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(3)
    s.setSingleStep(step)
    s.setValue(value)
    return s


def _json_ready(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real":_json_ready(value.real), "imag":_json_ready(value.imag)}
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {str(k):_json_ready(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_json_ready(v) for v in value]
    return value


class _ControlRows(QtWidgets.QGridLayout):
    """Wrap control groups so the figure cannot acquire an oversized minimum width."""
    def __init__(self):
        super().__init__()
        self._next = 0

    def addWidget(self, widget):
        super().addWidget(widget, self._next//4, self._next%4)
        self._next += 1


class DirectionalFigure(QtWidgets.QWidget):
    """A single question in two/four readable plots, with explicit scale and source state."""
    samplesSelected = QtCore.Signal(object)
    figureExported = QtCore.Signal(str, object)

    def __init__(self, kind, parent=None):
        super().__init__(parent, TOOL_WINDOW)
        show_while_inactive(self)
        if kind not in TOOLS:
            raise ValueError(kind)
        self.kind = kind
        self.data = None
        self._identity = None
        self._busy = False
        self._cancel = threading.Event()
        self._job = None
        self._computed = None
        self._computed_state = None
        self._spectra = None
        self._comparison_spectra = {}
        self._seed = True
        self._structural = []
        self._comparisons = []
        self.provenance_context = None
        self.setWindowTitle(TOOLS[kind])
        from dynamix.shell.window_fit import fit_to_screen
        fit_to_screen(self, 1100, 800)
        outer = QtWidgets.QVBoxLayout(self)
        self.context = QtWidgets.QLabel()
        self.context.setWordWrap(True)
        outer.addWidget(self.context)
        bar = QtWidgets.QHBoxLayout()
        self.pin = QtWidgets.QCheckBox("Pin source")
        bar.addWidget(self.pin)
        self.population = QtWidgets.QComboBox()
        self.population.addItem("WTMMM peaks", "wtmmm")
        self.population.addItem("All WTMM chain points", "wtmm")
        bar.addWidget(self.population)
        self.frame_combo = QtWidgets.QComboBox()
        self.frame_combo.addItem("image axes", "pixel")
        bar.addWidget(self.frame_combo)
        self.bins = QtWidgets.QSpinBox()
        self.bins.setRange(8, 256)
        self.bins.setValue(60)
        bar.addWidget(QtWidgets.QLabel("Bins"))
        bar.addWidget(self.bins)
        self.paper = QtWidgets.QCheckBox("Paper background")
        bar.addWidget(self.paper)
        self.export = QtWidgets.QPushButton("Export figure…")
        bar.addWidget(self.export)
        outer.addLayout(bar)
        body = QtWidgets.QHBoxLayout()
        sidebar = QtWidgets.QVBoxLayout()
        sidebar.addWidget(QtWidgets.QLabel("Displayed scales · px"))
        self.scales = QtWidgets.QListWidget()
        self.scales.setMaximumWidth(170)
        sidebar.addWidget(self.scales, 1)
        self.all_scales = QtWidgets.QPushButton("Select all")
        sidebar.addWidget(self.all_scales)
        self.compare = QtWidgets.QPushButton("Keep comparison")
        sidebar.addWidget(self.compare)
        self.clear_compare = QtWidgets.QPushButton("Clear comparisons")
        sidebar.addWidget(self.clear_compare)
        body.addLayout(sidebar)
        self.figure = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.figure)
        body.addWidget(self.figure, 1)
        outer.addLayout(body, 1)
        self.controls = _ControlRows()
        outer.addLayout(self.controls)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        outer.addWidget(self.readout)
        self.plots = []
        self.fitplots = []
        self.controls_widgets = {}
        self._build()
        direction_kinds = ("distributions", "diagnostics", "net")
        self.population.setVisible(kind in direction_kinds or kind in ("kernel", "correlations"))
        self.frame_combo.setVisible(kind in direction_kinds or kind in ("collapse", "sensitivity", "spread"))
        self.compare.setVisible(kind in ("distributions", "collapse", "fourier", "net"))
        self.clear_compare.setVisible(kind in ("distributions", "collapse", "fourier", "net"))
        self.population.currentIndexChanged.connect(self.refresh)
        self.frame_combo.currentIndexChanged.connect(self.refresh)
        self.bins.valueChanged.connect(self.refresh)
        self.scales.itemChanged.connect(self.refresh)
        self.all_scales.clicked.connect(self._select_all)
        self.paper.toggled.connect(self.refresh)
        self.export.clicked.connect(self._export)
        self.compare.clicked.connect(self._keep_comparison)
        self.clear_compare.clicked.connect(self._clear_comparisons)

    def _plot(self, title, row, col, *, fit=False, quantity="slope", factor=1.):
        if fit:
            w = SlopePlot(title, quantity=quantity, factor=factor)
            self.fitplots.append(w)
            p = w.plot
        else:
            p = w = zoom_plot(title=title)
            p.setMinimumSize(220, 190)
        p.setBackground(current().ground)
        p.addLegend(offset=(6, 6))
        for side in ("bottom", "left"):
            p.getAxis(side).enableAutoSIPrefix(False)
        self.grid.addWidget(w, row, col)
        self.plots.append(p)
        return p

    def _control(self, name, widget, label=None):
        group = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(group)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel(label or name))
        row.addWidget(widget)
        self.controls.addWidget(group)
        self.controls_widgets[name] = widget
        if isinstance(widget, QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(self.refresh)
        elif isinstance(widget, QtWidgets.QAbstractButton):
            widget.clicked.connect(self.refresh)
        elif isinstance(widget, QtWidgets.QAbstractSpinBox):
            widget.valueChanged.connect(self.refresh)
        return widget

    def _choice(self, name, values):
        combo = QtWidgets.QComboBox()
        combo.addItems(values)
        return self._control(name, combo)

    def _build(self):
        k = self.kind
        if k == "fourier":
            self._plot("2-D Fourier power", 0, 0).setAspectLocked(True)
            for title, row, col in [("Radial power", 0, 1), ("Native x cuts", 1, 0), ("Native y cuts", 1, 1)]:
                self._plot(title, row, col, fit=True, quantity="β", factor=-1)
            self.windowing = self._choice("Window", ["Hann cuts", "Rectangular cuts"])
            self.contours = self._control("Contours", QtWidgets.QCheckBox())
            self.link = self._control("Link k ranges", QtWidgets.QCheckBox())
            for i, fit in enumerate(self.fitplots):
                fit.region.sigRegionChangeFinished.connect(lambda i=i: self._link_ranges(i))
            self.compute = QtWidgets.QPushButton("Compute spectra")
            self.controls.addWidget(self.compute)
            self.compute.clicked.connect(self._calculate)
        elif k in ("collapse", "moments", "kernel"):
            titles = {"collapse": ["Raw x component", "Raw y component", "Rescaled x component", "Rescaled y component"],
                      "moments": ["q = −1", "q = 1", "q = 2", "q = 3"],
                      "kernel": ["ln |Ĝ|", "Phase Ĝ", "ln |Ĝ| / ln(a′/a)", "Phase / ln(a′/a)"]}[k]
            for i, title in enumerate(titles):
                self._plot(title, i//2, i%2)
        elif k == "sensitivity":
            for i, title in enumerate(["Overview", "Tail zoom", "H − 2δ", "H − δ", "H + δ", "H + 2δ"]):
                self._plot(title, i//2, i%2)
            self.component = self._choice("Component", ["x", "y"])
            self.delta = self._control("δ", _spin(.001, .5, .01, .01))
            self.zoom = pg.LinearRegionItem([-.5, .5])
            self.zoom.sigRegionChangeFinished.connect(self._sync_sensitivity_zoom)
        elif k == "net":
            self._plot("Map directions · dip/plunge unavailable", 0, 0).setAspectLocked(True)
            self.projection = self._choice("Projection", ["Schmidt", "Wulff"])
            self.hemisphere = self._choice("Hemisphere", ["lower", "upper"])
            self.representation = self._choice("Direction", ["gradient", "normal + 90°", "chain tangent"])
            self.axial = self._control("Axial", QtWidgets.QCheckBox())
            self.axial.setChecked(True)
            self.ring = self._control("Scale tracks", QtWidgets.QCheckBox())
            self.ring.setChecked(True)
            self.density = self._control("Angular density", QtWidgets.QCheckBox())
            self.import_button = QtWidgets.QPushButton("Add lines / planes…")
            self.controls.addWidget(self.import_button)
            self.import_button.clicked.connect(self._import_structural)
            self.brush_lo = self._control("Select azimuth min", _spin(0,360,0,5))
            self.brush_hi = self._control("max (wrap allowed)", _spin(0,360,30,5))
            self.brush_button = QtWidgets.QPushButton("Highlight angular sector")
            self.controls.addWidget(self.brush_button)
            self.brush_button.clicked.connect(self._select_sector)
        elif k == "diagnostics":
            self.diagnostic = self._choice("View", ["Peak versus scale", "Gradient and sectors"])
            self._plot("Peak strength", 0, 0)
            self._plot("Peak angle", 0, 1)
        elif k == "spread":
            for i, axis in enumerate(("x", "y")):
                self._plot(f"Component {axis} spread", 0, i, fit=True, quantity="H")
            self.spread = self._choice("Spread", ["median |T|", "standard deviation"])
        else:
            self._plot("Modulus density" if k == "distributions" else "Raw signed coefficient" if k == "scalar" else "C(r; a₁,a₂)",
                       0, 0, fit=k=="correlations", quantity="covariance slope")
            if k in ("distributions", "scalar"):
                self._plot("Angle density" if k == "distributions" else "Rescaled signed coefficient", 0, 1)
        if k in ("collapse", "sensitivity", "moments", "scalar"):
            self.hx = self._control("Hₓ" if k in ("collapse", "sensitivity") else "H", _spin(-2, 3, .5))
            if k == "collapse":
                self.hy = self._control("Hᵧ", _spin(-2, 3, .5))
                self.estimator = self._choice("Seed", ["median", "std", "collapse"])
            if k != "scalar":
                self.seed_button = QtWidgets.QPushButton("Seed H")
                self.controls.addWidget(self.seed_button)
                self.seed_button.clicked.connect(self._seed_h)
            if k != "scalar":
                self.lo = self._control("Fit log₂a min", _spin(-10, 20, 0, .25))
                self.hi = self._control("max", _spin(-10, 20, 10, .25))
        if k == "moments":
            self.q_values = QtWidgets.QLineEdit("-1, 1, 2, 3")
            self.q_values.setMaximumWidth(140)
            self.controls.addWidget(self.q_values)
            self.q_values.editingFinished.connect(self.refresh)
            self.tolerance = self._control("Area tolerance", _spin(0, 1, .1))
        if k in ("kernel", "correlations"):
            self.pairs = QtWidgets.QLineEdit("0:1, 0:2, 1:2")
            self.pairs.setMaximumWidth(160)
            self.pairs.setToolTip("Zero-based fine:coarse scale indices; equal indices allowed for covariance")
            self.controls.addWidget(QtWidgets.QLabel("Pairs (indices)"))
            self.controls.addWidget(self.pairs)
            self.compute = QtWidgets.QPushButton("Compute")
            self.controls.addWidget(self.compute)
            self.compute.clicked.connect(self._calculate)
            self.cancel_button = QtWidgets.QPushButton("Cancel")
            self.controls.addWidget(self.cancel_button)
            self.cancel_button.clicked.connect(self._cancel.set)
        if k == "kernel":
            self.pmax = self._control("|p| max", _spin(.1, 50, 5, .5))
            self.threshold = self._control("|Φ coarse| min", _spin(.001, .99, .05))
        if k == "correlations":
            self.route = self._choice("Magnitude", ["Skeleton ln M", "Continuous WT energy"])
            self.rmin = self._control("r min (px)", _spin(.1, 1e5, 1, 1))
            self.rmax = self._control("r max", _spin(.2, 1e5, 64, 1))
            self.direction = self._choice("Lag direction", ["radial", "native x ±15°", "native y ±15°"])
            self.bump = self._control("Bump width / a", _spin(.1, 5, 1, .1))
            self.pair_target = QtWidgets.QSpinBox()
            self.pair_target.setRange(0,11)
            self._control("Fit pair index", self.pair_target)
            self.model_check = self._control("Cascade model", QtWidgets.QCheckBox())
            self.variance = self._control("σ²", _spin(0,100,.03))
            self.integral = self._control("L (px)", _spin(.1,1e6,128,1))
        if k == "scalar":
            self.wavelet = self._choice("Scalar kernel", ["mexican"])
            self.compute = QtWidgets.QPushButton("Compute signed CWT")
            self.controls.addWidget(self.compute)
            self.compute.clicked.connect(self._calculate)

    def set_result(self, result, field, *, offset=(0, 0), scale_idx=0, fit_window=None, directional=None, data=None):
        if self.pin.isChecked() and self.data is not None:
            return
        identity = (id(result), id(getattr(field, "values", None)))
        if identity == self._identity:
            return
        self._cancel.set()
        self._identity = identity
        self.data = data or FigureData(result, field, offset)
        _release(self._computed)
        self._spectra = self._computed = self._computed_state = None
        self._seed = True
        for fit in self.fitplots:
            fit._seeded = False
            fit.clear_manual()
        self.scales.blockSignals(True)
        self.scales.clear()
        n = len(self.data.scales)
        selected = set(np.linspace(0, n-1, min(4, n)).astype(int)) if n else set()
        for i, a in enumerate(self.data.scales):
            item = QtWidgets.QListWidgetItem(f"{i}: {self.data.scale_label(i)}")
            item.setData(QtCore.Qt.UserRole, i)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if i in selected else QtCore.Qt.Unchecked)
            self.scales.addItem(item)
        self.scales.blockSignals(False)
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        self.frame_combo.addItem("image axes", "pixel")
        if self.data.north is not None:
            if self.data.north[2] is Frame.GRID:
                self.frame_combo.addItem("grid north", "grid")
            self.frame_combo.addItem("true north", "true")
        self.frame_combo.blockSignals(False)
        if hasattr(self, "lo") and n:
            bounds = fit_window or (np.log2(self.data.scales.min()), np.log2(self.data.scales.max()))
            for spin, v in zip((self.lo, self.hi), bounds):
                spin.blockSignals(True)
                spin.setValue(float(v))
                spin.blockSignals(False)
        self.refresh()

    def selected_scales(self):
        return [i for i in range(self.scales.count()) if self.scales.item(i).checkState() == QtCore.Qt.Checked]

    def _select_all(self):
        self.scales.blockSignals(True)
        for i in range(self.scales.count()):
            self.scales.item(i).setCheckState(QtCore.Qt.Checked)
        self.scales.blockSignals(False)
        self.refresh()

    def _keep_comparison(self):
        if self.data is not None:
            self._comparisons.append((self.data, self.selected_scales(), self.frame_combo.currentData()))
            if self.kind=="fourier" and self._spectra is not None:
                self._comparison_spectra[self._spectra[0]] = self._spectra[1]
            self.readout.setText("Comparison retained; select another source, or change the scale selection.")

    def _clear_comparisons(self):
        self._comparisons.clear()
        self._comparison_spectra.clear()
        self.refresh()

    def _ink(self, index):
        if self.paper.isChecked():
            return current().ground
        return _SCALE_COLORS[index % len(_SCALE_COLORS)]

    def _series(self, plot, x, y, index, label, *, density=False, reference=False):
        ink = self._ink(index)
        item = plot.plot(x, y, name=label,
            pen=pg.mkPen(ink, width=1.2) if density or reference else None,
            symbol=None if density or reference else _SYMBOLS[index % len(_SYMBOLS)],
            symbolSize=5, symbolBrush=pg.mkBrush(current().ink if self.paper.isChecked() else current().ground),
            symbolPen=pg.mkPen(ink), connect="finite")
        return item

    def refresh(self, *_):
        if self.data is None:
            return
        try:
            self._draw()
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            self.readout.setText(str(exc))

    def _draw(self):
        for p in self.plots:
            if p not in [f.plot for f in self.fitplots]:
                p.clear()
            p.setBackground(current().ink if self.paper.isChecked() else current().ground)
            for side in ("bottom", "left"):
                p.getAxis(side).setTextPen(current().ground if self.paper.isChecked() else current().ink)
        for fit in self.fitplots:
            fit.paper_mode = self.paper.isChecked()
        d = self.data
        indices = self.selected_scales()
        frame = self.frame_combo.currentData() or "pixel"
        population = self.population.currentData() or "wtmmm"
        self.context.setText(f"{getattr(d.field, 'name', '')} · {self.kind} · {population} · {frame} · "
                             f"{d.result.get('params', {}).get('wavelet', '')} · native pixels")
        self.readout.setText("")
        if self.kind == "fourier":
            self._draw_fourier()
            return
        if self.kind in ("kernel", "correlations", "scalar"):
            self._draw_calculation()
            return
        if self.kind=="net" and self._structural:
            self._draw_net(indices, frame, population)
            return
        if not indices:
            self.readout.setText("Select one or more scales")
            return
        sample_population = population if self.kind in ("distributions", "diagnostics", "net") else "wtmmm"
        samples = d.samples(sample_population)
        if not samples:
            self.readout.setText("This view needs a WTMM result with horizontal chains")
            return
        if self.kind == "distributions":
            self._distributions(d, indices, frame, population)
            for other, picked, other_frame in self._comparisons:
                self._distributions(other, picked, other_frame, population, prefix=getattr(other.field, "name", "")+": ")
        elif self.kind in ("collapse", "sensitivity", "moments", "spread"):
            self._draw_components(indices, frame)
        elif self.kind == "diagnostics":
            self._draw_diagnostics(indices, frame, population)
        elif self.kind == "net":
            self._draw_net(indices, frame, population)

    def _distributions(self, data, indices, frame, population, prefix=""):
        samples = data.samples(population)
        if not samples:
            return
        table = an.modulus_pdfs([samples[i]["mod"] for i in indices], data.scales[indices], bins=self.bins.value())
        for j, i in enumerate(indices):
            label = f"{prefix}a={data.scales[i]:g} · n={len(samples[i]['mod'])}"
            self._series(self.plots[0], table["centres"], table["pdf"][j], i, label, reference=j==0)
            o = data.orientations(i, population, frame)
            x, y = an.angle_pdf(o, self.bins.value())
            self._series(self.plots[1], x, y, i, label, reference=j==0)
        self.plots[0].setLabel("bottom", "M")
        self.plots[0].setLabel("left", "density / modulus unit")
        self.plots[1].setLabel("bottom", f"gradient angle ({frame}, degrees)")
        self.plots[1].setLabel("left", "density / degree")
        self.plots[1].addLine(y=1/360, pen=pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine))

    def _seed_h(self):
        self._seed = True
        self.refresh()

    def _draw_components(self, indices, frame):
        d = self.data
        samples = d.samples()
        components = [d.components(i, frame) for i in range(len(samples))]
        mods = [s["mod"] for s in samples]
        if self.kind == "spread":
            for axis, fit in enumerate(self.fitplots):
                vals = [np.asarray(c[axis], float) for c in components]
                spread = [np.median(np.abs(v)) if self.spread.currentIndex()==0 else np.std(v) for v in vals]
                with np.errstate(divide="ignore", invalid="ignore"):
                    fit.set_data(np.log2(d.scales), np.log2(spread))
                fit.plot.setLabel("bottom", "log₂ a (px)")
                fit.plot.setLabel("left", "log₂ component spread")
            return
        lo, hi = sorted((self.lo.value(), self.hi.value()))
        estimate = an.directional_hurst(components, d.scales, lo-.0005, hi+.0005)
        key = self.estimator.currentText() if hasattr(self, "estimator") else "median"
        h1, h2 = estimate[key][0][0], estimate[key][1][0]
        if self._seed:
            H, _ = an.modulus_hurst(mods, d.scales, lo, hi)
            values = [(self.hx, H if self.kind=="moments" else h1)]
            if hasattr(self, "hy"):
                values.append((self.hy, h2))
            for spin, value in values:
                spin.blockSignals(True)
                if np.isfinite(value):
                    spin.setValue(float(value))
                spin.blockSignals(False)
            self._seed = False
        kept = [components[i] for i in indices]
        scales = d.scales[indices]
        if self.kind == "moments":
            qs = [float(v.strip()) for v in self.q_values.text().split(",") if v.strip()]
            if not 1 <= len(qs) <= 4:
                raise ValueError("Choose one to four q values")
            table = an.modulus_collapse(mods, d.scales, self.hx.value(), qs, lo, hi,
                                        bins=self.bins.value(), tolerance=self.tolerance.value())
            for qj, q in enumerate(qs):
                self.plots[qj].setTitle(f"q={q:g} · area spread {table['area_spread'][qj]:.1%}")
                for j, a in enumerate(table["scales"]):
                    i = int(np.argmin(np.abs(d.scales-a)))
                    if i in indices:
                        self._series(self.plots[qj], table["x"], table["curves"][qj, j], i, f"a={a:g}", density=True)
                self.plots[qj].setLabel("bottom", "x = M / aᴴ")
                self.plots[qj].setLabel("left", "x^q pₓ(x), normalized rescaled density")
            self.readout.setText("Accepted q (max−min area)/mean ≤ tolerance: "+str(np.asarray(qs)[table["usable"]].tolist()))
            return
        if self.kind == "sensitivity":
            axis = self.component.currentIndex()
            H, delta = self.hx.value(), self.delta.value()
            for k, shift in enumerate((0, 0, -2, -1, 1, 2)):
                table = an.component_pdfs(kept, scales, bins=self.bins.value(), rescale=(H+shift*delta, H+shift*delta))
                for j, i in enumerate(indices):
                    self._series(self.plots[k], table["centres"][axis], table["ln_pdf"][axis,j], i,
                                 f"a={d.scales[i]:g}", density=True)
                self.plots[k].setTitle(f"H={H+shift*delta:.3f}" + (" · overview" if k==0 else " · zoom"))
                self.plots[k].setLabel("bottom", "T / aᴴ")
            self.plots[0].addItem(self.zoom)
            self._sync_sensitivity_zoom()
            self.readout.setText("δ is a sensitivity step, not a regression confidence interval")
            return
        raw = an.component_pdfs(kept, scales, bins=self.bins.value())
        collapsed = an.component_pdfs(kept, scales, bins=self.bins.value(), rescale=(self.hx.value(), self.hy.value()))
        for axis in range(2):
            for j, i in enumerate(indices):
                self._series(self.plots[axis], raw["centres"][axis], raw["ln_pdf"][axis,j], i, f"a={d.scales[i]:g}", density=True)
                self._series(self.plots[axis+2], collapsed["centres"][axis], collapsed["ln_pdf"][axis,j], i,
                             f"a={d.scales[i]:g}", density=True)
            self.plots[axis].setLabel("bottom", ("Tₓ" if axis==0 else "Tᵧ") if frame=="pixel" else ("T east" if axis==0 else "T north"))
            self.plots[axis+2].setLabel("bottom", "T / aᴴ")
            self.plots[axis].setLabel("left", "ln p(T)")
            self.plots[axis+2].setLabel("left", "ln p(T/aᴴ)")
        self.readout.setText(f"{key}: Hₓ={h1:.3f}, Hᵧ={h2:.3f} · estimate uses all scales in range; displayed scales are separate")
        for other, picked, other_frame in self._comparisons:
            c = [other.components(i, other_frame) for i in picked]
            raw = an.component_pdfs(c, other.scales[picked], bins=self.bins.value())
            col = an.component_pdfs(c, other.scales[picked], bins=self.bins.value(), rescale=(self.hx.value(), self.hy.value()))
            for axis in range(2):
                for j,i in enumerate(picked):
                    label = f"{other.field.name}: a={other.scales[i]:g}"
                    self._series(self.plots[axis],raw["centres"][axis],raw["ln_pdf"][axis,j],i,label,density=True)
                    self._series(self.plots[axis+2],col["centres"][axis],col["ln_pdf"][axis,j],i,label,density=True)

    def _sync_sensitivity_zoom(self):
        if self.kind != "sensitivity":
            return
        lo, hi = self.zoom.getRegion()
        for p in self.plots[1:]:
            p.setXRange(lo, hi, padding=0)
        for p in self.plots[2:]:
            p.setYLink(self.plots[1])

    def _draw_diagnostics(self, indices, frame, population):
        d = self.data
        samples = d.samples(population)
        orientations = [d.orientations(i, population, frame) for i in indices]
        if self.diagnostic.currentIndex()==0:
            table = an.anisotropy_vs_scale(orientations, d.scales[indices], bins=self.bins.value())
            for p, key, ylabel in zip(self.plots, ("relative_height", "angle_deg"), ("peak / isotropic", "axial peak (degrees)")):
                self._series(p, np.log2(d.scales[indices]), table[key], 0, population)
                p.setLabel("bottom", "log₂ a (px)")
                p.setLabel("left", ylabel)
            self.plots[0].addLine(y=1, pen=pg.mkPen(current().ink_muted))
        else:
            i = indices[0]
            points, o = samples[i], orientations[0]
            a = np.radians(o.degrees)
            self._series(self.plots[0], points["mod"]*np.cos(a), points["mod"]*np.sin(a), i, f"a={d.scales[i]:g}")
            self.plots[0].setAspectLocked(True)
            for sector, x, y, count in an.sector_modulus_pdfs(points["mod"], o, bins=self.bins.value()):
                self._series(self.plots[1], x, y, sector, f"sector {sector} · n={count}", density=True)
            self.plots[1].setLabel("bottom", "log₂ M")
            self.plots[1].setLabel("left", "conditional density")

    def _draw_net(self, indices, frame, population):
        from dynamix.core.orientation_net import project, plane_pole, plane_trace, graticule
        p = self.plots[0]
        t = np.linspace(0, 2*np.pi, 361)
        p.plot(np.sin(t), np.cos(t), pen=pg.mkPen(current().ink_muted))
        projection, hemisphere = self.projection.currentText().lower(), self.hemisphere.currentText()
        for x,y in graticule(projection=projection, hemisphere=hemisphere):
            p.plot(x, y, pen=pg.mkPen(current().border))
        label_radius = 1.25+.07*len(indices) if self.ring.isChecked() else 1.12
        labels = [("N",0,label_radius),("E",label_radius,0),("S",0,-label_radius),("W",-label_radius,0)] if frame!="pixel" else [("−row",0,label_radius),("+col",label_radius,0),("+row",0,-label_radius),("−col",-label_radius,0)]
        for text, x, y in labels:
            label = pg.TextItem(text, color=current().ink_muted, anchor=(.5,.5))
            label.setPos(x,y)
            p.addItem(label)
        counts = []
        track = 0
        layers = [(self.data, indices, frame), *self._comparisons]
        for data, picked, layer_frame in layers:
            if layer_frame != frame:
                continue
            if frame != "pixel" and data.north is None:
                continue
            for i in picked:
                points = data.samples(population)[i]
                o = data.orientations(i, population, frame, self.representation.currentText())
                degrees = np.asarray(o.degrees)
                good = np.isfinite(degrees)
                ids = np.flatnonzero(good)
                # Thinning changes marks only. Counts and angular densities retain every point.
                drawn = ids[::max(1, int(np.ceil(len(ids)/4000)))]
                angle = np.radians(degrees[drawn])
                if frame == "pixel":
                    angle = np.pi/2-angle
                if self.axial.isChecked():
                    angle = np.r_[angle, angle+np.pi]
                radius = 1.04+.07*track if self.ring.isChecked() else 1.
                item = self._series(p, radius*np.sin(angle), radius*np.cos(angle), i,
                                    f"{data.field.name}: {data.scale_label(i)} · n={len(ids)}")
                item.setCurveClickable(False)
                scatter = item.scatter
                point_ids = np.tile(drawn, 2) if self.axial.isChecked() else drawn
                scatter.setData(x=radius*np.sin(angle), y=radius*np.cos(angle), data=point_ids,
                                symbol=_SYMBOLS[i%len(_SYMBOLS)], size=5,
                                pen=pg.mkPen(self._ink(i)), brush=None, hoverable=True,
                                tip=lambda x,y,point_id,i=i,points=points,data=data: f"a={data.scales[i]:g}px · point {points['point_id'][int(point_id)]} · chain {points['line_id'][int(point_id)]} · M={points['mod'][int(point_id)]:.5g}")
                scatter.sigClicked.connect(lambda _s, spots, _e, i=i, points=points, data=data: self.samplesSelected.emit(
                    {"source_identity": id(data.result), "scale_index": i, "point_ids": [int(points['point_id'][int(s.data())]) for s in spots],
                     "x": [float(points['x'][int(s.data())]) for s in spots],
                     "y": [float(points['y'][int(s.data())]) for s in spots], "offset": data.offset}))
                if self.density.isChecked():
                    vals = degrees[good]
                    period = 180 if self.axial.isChecked() else 360
                    hist, edges = np.histogram(vals%period, bins=self.bins.value(), range=(0,period))
                    pdf = hist/(max(1,len(vals))*np.diff(edges))
                    a = np.radians((edges[:-1]+edges[1:])/2)
                    if period==180:
                        a, pdf = np.r_[a,a+np.pi], np.tile(pdf,2)
                    if frame=="pixel":
                        a = np.pi/2-a
                    r = radius + .05*pdf/(pdf.max() if pdf.size and pdf.max()>0 else 1)
                    self._series(p, r*np.sin(a), r*np.cos(a), i, f"density / degree · a={data.scales[i]:g}", density=True)
                counts.append(f"{data.scales[i]:g}px: {len(ids)}")
                track += 1
        for layer in self._structural:
            if layer["frame"] != frame:
                continue
            o = Orientation(layer["azimuth"], Frame(frame))
            if layer["kind"]=="line":
                x,y = project(o, layer["inclination"], projection=projection, hemisphere=hemisphere)
                self._series(p, np.atleast_1d(x), np.atleast_1d(y), 0, layer["name"])
            else:
                pole, plunge = plane_pole(o, layer["inclination"])
                x,y = project(pole, plunge, projection=projection, hemisphere=hemisphere)
                self._series(p, np.atleast_1d(x), np.atleast_1d(y), 0, layer["name"]+" pole")
                x,y = plane_trace(o, layer["inclination"], projection=projection, hemisphere=hemisphere)
                self._series(p,x,y,0,layer["name"]+" plane",density=True)
        extent = label_radius+.12
        p.setXRange(-extent, extent, padding=0)
        p.setYRange(-extent, extent, padding=0)
        p.hideAxis("bottom")
        p.hideAxis("left")
        self.readout.setText(f"{population} · {self.representation.currentText()} · {'axial' if self.axial.isChecked() else 'directed'} · {frame}\n"
            + "Map directions: dip/plunge unavailable. Outer tracks identify scales, not inclination. " + "; ".join(counts))

    def add_orientation(self, *, name, kind, azimuth, inclination, frame):
        if kind not in ("line", "plane") or frame not in ("grid", "true", "magnetic"):
            raise ValueError("line/plane with an explicit geographic reference required")
        self._structural.append(dict(name=name, kind=kind, azimuth=float(azimuth), inclination=float(inclination), frame=frame))
        if self.frame_combo.findData(frame)<0:
            self.frame_combo.addItem(frame+" north · measured layer only",frame)
        self.frame_combo.setCurrentIndex(self.frame_combo.findData(frame))
        self.refresh()

    def _select_sector(self):
        if self.data is None:
            return
        frame = self.frame_combo.currentData()
        if frame!="pixel" and self.data.north is None:
            return
        x,y,ids = [],[],[]
        period = 180 if self.axial.isChecked() else 360
        lo,hi = self.brush_lo.value()%period,self.brush_hi.value()%period
        count = total = 0
        for i in self.selected_scales():
            points = self.data.samples(self.population.currentData())[i]
            o = self.data.orientations(i,self.population.currentData(),frame,self.representation.currentText())
            angle = np.asarray(o.degrees)%period
            if abs(self.brush_hi.value()-self.brush_lo.value())>=period:
                keep = np.ones(angle.shape,bool)
            else:
                keep = ((angle>=lo)&(angle<=hi)) if lo<=hi else ((angle>=lo)|(angle<=hi))
            keep &= np.isfinite(angle)
            x.extend(np.asarray(points["x"])[keep].tolist())
            y.extend(np.asarray(points["y"])[keep].tolist())
            ids.extend(np.asarray(points["point_id"])[keep].tolist())
            count += int(keep.sum())
            total += int(np.isfinite(angle).sum())
        self.samplesSelected.emit({"source_identity":id(self.data.result),"x":x,"y":y,
                                  "point_ids":ids,"offset":self.data.offset})
        self.readout.setText(f"Selected {count}/{total} per-scale observations ({count/max(total,1):.1%}); repeated scales remain separate")

    def _import_structural(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Measured orientation")
        layout = QtWidgets.QFormLayout(dialog)
        kind = QtWidgets.QComboBox(); kind.addItems(["line", "plane"])
        frame = QtWidgets.QComboBox(); frame.addItems(["true", "grid", "magnetic"])
        azimuth, inclination = _spin(0,360,0,1), _spin(0,90,0,1)
        name = QtWidgets.QLineEdit("Measured orientation")
        for label,w in [("Name",name),("Kind",kind),("Frame",frame),("Trend / RHR strike",azimuth),("Plunge / dip",inclination)]:
            layout.addRow(label,w)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec()==QtWidgets.QDialog.Accepted:
            self.add_orientation(name=name.text(), kind=kind.currentText(), frame=frame.currentText(),
                                 azimuth=azimuth.value(), inclination=inclination.value())

    def _draw_fourier(self):
        values = getattr(self.data.field, "values", None)
        if values is None or np.ndim(values)!=2 or min(np.shape(values))<16:
            raise ValueError("Select a native 2-D raster/band for Fourier analysis")
        # The field is named explicitly; no inference that it is the WTMM's prepared input.
        self.context.setText(f"Fourier source: {self.data.field.name} · native raster · cuts follow native rows/columns")
        identity = (id(values), self.windowing.currentIndex())
        if self._spectra is None or self._spectra[0]!=identity:
            self.readout.setText("Press Compute spectra for this native source/window; opening a figure does not run a transform")
            return
        s = self._spectra[1]
        image = pg.ImageItem(s["log2_power"].T)
        ny,nx = values.shape
        image.setRect(QtCore.QRectF(-.5,-.5,1,1))
        self.plots[0].addItem(image)
        self.plots[0].setLabel("bottom", "kₓ (cycles/px)")
        self.plots[0].setLabel("left", "kᵧ (cycles/px)")
        if self.contours.isChecked():
            levels = np.linspace(np.percentile(s["log2_power"], 65), np.percentile(s["log2_power"], 98), 5)
            for level in levels:
                contour = pg.IsocurveItem(s["log2_power"].T, level=level, pen=pg.mkPen(current().ink_muted))
                contour.setScale(1/max(nx,ny))
                contour.setPos(-.5,-.5)
                self.plots[0].addItem(contour)
        for fit,key in zip(self.fitplots,("r","x","y")):
            initial = not fit._seeded
            with np.errstate(divide="ignore"):
                fit.set_data(np.log2(s['k_'+key]), np.log2(s['S_'+key]))
            if initial:
                fit.region.setRegion(np.log2(s["window_"+key]))
            fit.plot.setLabel("bottom", "log₂ k (cycles/px)")
            fit.plot.setLabel("left", "log₂ power")
        for other,picked,frame in self._comparisons:
            v = getattr(other.field, "values", None)
            if v is not None and np.ndim(v)==2:
                key_cache = (id(v), self.windowing.currentIndex())
                table = self._comparison_spectra.get(key_cache)
                if table is None:
                    continue
                for fit,key in zip(self.fitplots,("r","x","y")):
                    self._series(fit.plot, np.log2(table['k_'+key]), np.log2(table['S_'+key]), 1, other.field.name)

    def _link_ranges(self, origin):
        if self.link.isChecked():
            bounds = self.fitplots[origin].region.getRegion()
            for i,fit in enumerate(self.fitplots):
                if i!=origin:
                    fit.region.blockSignals(True)
                    fit.region.setRegion(bounds)
                    fit.region.blockSignals(False)
                    fit._fit()

    def _parse_pairs(self):
        pairs = [tuple(int(v) for v in pair.strip().split(":")) for pair in self.pairs.text().split(",") if pair.strip()]
        if not pairs or len(pairs)>12 or any(len(pair)!=2 for pair in pairs):
            raise ValueError("Choose 1–12 scale pairs, written fine:coarse")
        n = len(self.data.scales)
        if any(min(pair)<0 or max(pair)>=n for pair in pairs):
            raise ValueError("Scale-pair index outside the available scale ladder")
        return pairs

    def _calculate(self):
        if self._busy or self.data is None:
            return
        try:
            function = self._calculation()
        except (ValueError, IndexError, KeyError) as exc:
            self.readout.setText(str(exc)); return
        self._busy = True
        self._cancel = threading.Event()
        self.compute.setEnabled(False)
        self.readout.setText("Computing…")
        self._computed_state = self.render_state()
        self._job = _Calculation(function, self._cancel)
        self._job.signals.done.connect(self._calculated)
        QtCore.QThreadPool.globalInstance().start(self._job)

    def _calculation(self):
        from dynamix.core.space_scale import self_similarity, skeleton_covariance, dense_covariance, local_energy
        d = self.data
        k = self.kind
        if k=="fourier":
            values = np.asarray(d.field.values)
            if values.ndim!=2 or min(values.shape)<16:
                raise ValueError("Fourier spectra need a native 2-D raster/band at least 16 pixels wide")
            index = self.windowing.currentIndex()
            return lambda cancel: {"kind":"fourier", "identity":(id(d.field.values),index),
                "spectra":an.cut_spectra(values,window="hann" if index==0 else "rectangular")}
        if k=="scalar":
            values = np.asarray(d.field.values)
            scales = d.scales.copy()
            if not scales.size:
                scales = np.array([2.,4.,8.,16.])
            if values.ndim!=2:
                raise ValueError("Select a single native raster band for the scalar CWT")
            if not np.isfinite(values).all():
                raise ValueError("Scalar CWT PDFs need finite source pixels; select a complete ROI or inpaint first")
            wavelet = self.wavelet.currentText()
            def scalar(cancel):
                from dynamix.core.wtmm_backend import cwt2d_scalar
                if values.size*len(scales)*16 <= SIGNED_STACK_BUDGET:
                    signed = cwt2d_scalar(values,scales,wavelet=wavelet,signed=True)["signed"]
                else:
                    import uuid
                    from dynamix.shell.settings import scratch_dir
                    path = scratch_dir()/f"signed-{uuid.uuid4().hex}.npy"
                    signed = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                                       shape=(len(scales), *values.shape))
                    try:
                        for i,a in enumerate(scales):
                            if cancel.is_set(): raise InterruptedError("Cancelled")
                            signed[i] = cwt2d_scalar(values,[a],wavelet=wavelet,signed=True)["signed"][0]
                        signed.flush()
                    except BaseException:
                        del signed
                        path.unlink(missing_ok=True)
                        raise
                return {"kind":"scalar", "scales":scales, "signed":signed}
            return scalar
        pairs = self._parse_pairs()
        samples = d.samples(self.population.currentData())
        if k=="kernel":
            p = np.linspace(-self.pmax.value(),self.pmax.value(),201)
            threshold = self.threshold.value()
            return lambda cancel: {"kind":"kernel", **self_similarity([s["mod"] for s in samples],d.scales,pairs,p,denominator_min=threshold)}
        lo,hi = self.rmin.value(),self.rmax.value()
        if lo>=hi:
            raise ValueError("Minimum separation must be smaller than maximum")
        edges = np.geomspace(lo,hi,25)
        direction = None if self.direction.currentIndex()==0 else Orientation(0 if self.direction.currentIndex()==1 else 90,Frame.PIXEL)
        route, width = self.route.currentIndex(), self.bump.value()
        if route and not np.isfinite(d.field.values).all():
            raise ValueError("Continuous energy needs finite source pixels; select a complete ROI or inpaint first")
        params = dict(d.result.get("params", {}))
        field = d.field
        def correlate(cancel):
            rows = []
            dense = {}
            if route:
                from dynamix.core.wtmm_backend import run_cwt_stage
                if not params:
                    raise ValueError("Continuous WT energy needs the source WTMM analysis parameters")
                if d.result.get("_stage") or d.result.get("_roi") or tuple(d.offset)!=(0,0):
                    raise ValueError("Continuous energy: fork the analysed ROI/prepared stage as a dataset first, so the native source is explicit")
                cwt = run_cwt_stage(field,params)
                for i in sorted(set(v for pair in pairs for v in pair)):
                    if cancel.is_set(): raise InterruptedError("Cancelled")
                    dense[i] = local_energy(cwt["mod"][i],d.scales[i],width=width)
            for i,j in pairs:
                if cancel.is_set(): raise InterruptedError("Cancelled")
                row = dense_covariance(dense[i],dense[j],edges,direction=direction) if route else skeleton_covariance(
                    samples[i],samples[j],edges,direction=direction,cancelled=cancel.is_set)
                rows.append({"pair":[i,j],**row})
            return {"kind":"correlations", "rows":rows}
        return correlate

    @QtCore.Slot(object,object)
    def _calculated(self, result, error):
        self._busy = False
        self.compute.setEnabled(True)
        if self._cancel.is_set():
            self.readout.setText("Calculation cancelled or source changed")
            return
        if error:
            self.readout.setText(error)
            return
        _release(self._computed)
        self._computed = result
        if result["kind"]=="fourier":
            self._spectra = result["identity"],result["spectra"]
        self.refresh()

    def _draw_calculation(self):
        result = self._computed
        if result is None:
            self.readout.setText("Select parameters and press Compute; transforms are never started by opening a view")
            return
        if self.kind=="kernel":
            for i,row in enumerate(result["rows"]):
                label = str(row["pair"])
                for p, y in zip(self.plots,(row["logabs"],row["phase"],row["logabs"]/row["depth"],row["phase"]/row["depth"])):
                    self._series(p,result["p"],y,i,label)
                    p.setLabel("bottom","p conjugate to ln M")
            cumulants = "; ".join(f"{r['pair']}: Δκ₁={r['cumulant_difference'][0]:.3g}, Δκ₂={r['cumulant_difference'][1]:.3g}" for r in result["rows"])
            self.readout.setText("Marginal kernel, not a spatial-cascade test; unstable ratios masked. " + cumulants)
        elif self.kind=="correlations":
            target = min(self.pair_target.value(), len(result["rows"])-1)
            for i,row in enumerate(result["rows"]):
                if i==target:
                    self.fitplots[0].set_data(np.log2(row["r"]),row["covariance"])
            for i,row in enumerate(result["rows"]):
                if i!=target:
                    self._series(self.plots[0],np.log2(row["r"]),row["covariance"],i,str(row["pair"]))
            if self.model_check.isChecked():
                from dynamix.core.space_scale import cascade_curve
                row = result["rows"][target]
                i,j = row["pair"]
                valid = row["r"] >= max(self.data.scales[i], self.data.scales[j])
                y = cascade_curve(row["r"], self.variance.value(), self.integral.value())
                self._series(self.plots[0],np.log2(row["r"]),np.where(valid,y,np.nan),0,"W-cascade model",density=True)
            self.plots[0].setLabel("bottom","log₂ r (native px)")
            self.plots[0].setLabel("left","covariance of ln magnitude")
            self.plots[0].addLine(y=0,pen=pg.mkPen(current().ink_muted))
            self.readout.setText("Recorded-position pairs · per-scale centering · covariance, not Pearson · pair counts retained in export")
        else:
            indices = self.selected_scales() or list(range(len(result["scales"])))
            vals = [result["signed"][i].ravel() for i in indices]
            a = result["scales"][indices]
            for axis,p in enumerate(self.plots):
                rescaled = [v/(scale**self.hx.value() if axis else 1) for v,scale in zip(vals,a)]
                allv = np.concatenate(rescaled)
                edges = np.linspace(np.nanmin(allv),np.nanmax(allv),self.bins.value()+1)
                for i,v,scale in zip(indices,rescaled,a):
                    hist,_ = np.histogram(v[np.isfinite(v)],edges,density=True)
                    with np.errstate(divide="ignore"):
                        y = np.where(hist>0,np.log(hist),np.nan)
                    self._series(p,(edges[:-1]+edges[1:])/2,y,i,f"a={scale:g}",density=True)
                p.setLabel("bottom","W / aᴴ" if axis else "signed W")
                p.setLabel("left","ln normalized density")
            where = " · stack memory-mapped in the scratch folder" if isinstance(result["signed"], np.memmap) else ""
            self.readout.setText("Signed scalar coefficients · dense CWT samples · no gradient direction inferred"+where)

    def render_state(self):
        state = {"tool":self.kind,"context":self.data.context() if self.data else {},
                 "population":self.population.currentData(),"frame":self.frame_combo.currentData(),
                 "displayed_scale_indices":self.selected_scales(),"bins":self.bins.value(),
                 "fits":[f.measurement for f in self.fitplots],"controls":{},
                 "comparisons":[{"context":d.context(), "scales":p, "frame":f} for d,p,f in self._comparisons],"structural":self._structural}
        if self.kind=="sensitivity":
            state["zoom"] = list(self.zoom.getRegion())
        state["paper_background"] = self.paper.isChecked()
        for name,w in self.controls_widgets.items():
            state["controls"][name] = w.currentText() if isinstance(w,QtWidgets.QComboBox) else w.isChecked() if isinstance(w,QtWidgets.QAbstractButton) else w.value()
        if hasattr(self,"pairs"): state["pairs"] = self.pairs.text()
        if hasattr(self,"q_values"): state["q_values"] = self.q_values.text()
        return state

    def _export(self):
        filename,_ = QtWidgets.QFileDialog.getSaveFileName(self,"Export figure","figure.png","PNG (*.png)")
        if not filename: return
        if not self.figure.grab().save(filename):
            self.readout.setText("Could not save the figure image")
            return
        state = self.render_state()
        state["computed_with"] = self._computed_state
        state["outcomes"] = self._export_outcomes()
        state = _json_ready(state)
        Path(filename).with_suffix(".json").write_text(json.dumps(state,indent=2,allow_nan=False),encoding="utf-8")
        self.figureExported.emit(filename, state)

    def _export_outcomes(self):
        if self._computed is None:
            out = {}
        else:
            out = {k:v for k,v in self._computed.items() if k not in ("signed","spectra")}
            if "signed" in self._computed:
                out["coefficient_counts"] = [int(np.isfinite(v).sum()) for v in self._computed["signed"]]
        out["plotted_series"] = [[{"name":item.name(), "x":item.getData()[0], "y":item.getData()[1]}
                                  for item in p.listDataItems()] for p in self.plots]
        return out

    def closeEvent(self,event):
        self._cancel.set()
        _release(self._computed)
        super().closeEvent(event)
