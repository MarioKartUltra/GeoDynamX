# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Directional roughness window: WTMMM statistics after Arnéodo, Decoster & Roux 2000 (§6,
Figs 24–26) and Khalil, Joncas, Nekka, Kestener & Arnéodo 2006 (§§4.2.4, 5.2.4, Figs 7–9, 12–14),
in the pixel frame (the paper's A) or as bearings.

Row 1, on the active result's WTMMM (xsmurf ``ssm``): P_a(A) at several scales over one flat
reference (FLAT = isotropic); the WTMMM of one scale in the gradient plane (radial symmetry =
isotropy); the pdf of log2 M in the four axial sectors of the angle (shape invariance = M and A
independent).

Row 2, the two directional Hurst exponents: ln P_a of each component of the WTMMM across the
scales of the fit window, raw and rescaled by a^{H} (one curve when T ~ a^{H}, Khalil et al.
eqs. 33–36); H_x and H_y read by the slope of the components' spread against log2 a or by the
collapse, the rescaling spins seeded from the estimate so a drag is the paper's Fig. 14
sensitivity sweep live, with ± σ companions on request.

Row 3: P_a(M) across the scales; the q-rescaled modulus collapse a^{−qH} M^q P_a(M) against
M / a^{H} with the q whose statistics hold (the 10 % area rule); the anisotropy against scale
(the peak of P_a(A), its height over the isotropic level and its angle, eq. 38); the power
spectra of the one-dimensional cuts and the radial one with their β's; the 2-D power.

Bearings exist only for a georeferenced grid (:func:`dynamix.geo.mapping.field_north`): grid
north for a projected grid (plus true north through the meridian convergence), true north for
lon/lat; the components are then east–west and north–south. A swath or pixel grid offers the
pixel frame alone, and says so.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core.anisotropy import (SECTOR_LABELS, angle_pdf, anisotropy_vs_scale, component_pdfs,
                                     cut_spectra, directional_hurst, gradient_plane, modulus_collapse,
                                     modulus_hurst, modulus_pdfs, sector_modulus_pdfs, wtmmm_by_scale)
from dynamix.core.orientation import Frame, from_arg, grid_to_true, to_azimuth
from dynamix.shell.theme import RESTRAINED_DARK
from dynamix.shell.plot_zoom import zoom_plot
from dynamix.shell.window_fit import TOOL_WINDOW, show_while_inactive

__all__ = ["AnisotropyWindow", "ESTIMATORS"]

#: Series palettes: which sector or scale a curve IS, so data identity, exempt from the Theme Rule as
#: RGB tuples (the canvas convention), never chrome.
_SECTOR_COLORS = ((225, 87, 89), (242, 193, 78), (89, 161, 79), (78, 121, 167))
_SCALE_COLORS = ((78, 121, 167), (89, 161, 79), (242, 193, 78), (225, 87, 89), (176, 122, 161), (156, 117, 95))
#: The directional Hurst estimators of ``core.anisotropy.directional_hurst``, by label.
ESTIMATORS = {"median slope": "median", "std slope": "std", "collapse": "collapse"}
#: The q grid the usable-q readout walks (the WTMM's default moment grid).
_Q_GRID = np.arange(-3.0, 6.01, 0.5)


class AnisotropyWindow(QtWidgets.QWidget):
    """A floating window; :meth:`set_result` feeds it the active result and its field."""

    def __init__(self, parent=None):
        super().__init__(parent, TOOL_WINDOW)
        show_while_inactive(self)
        self.setWindowTitle("Directional roughness — WTMMM statistics")
        from dynamix.shell.window_fit import fit_to_screen

        fit_to_screen(self, 1280, 960)
        self._result = None
        self._field = None
        self._offset = (0.0, 0.0)
        self._wtmmm = None                  # (result, per-scale WTMMM)
        self._north = None                  # field_north(field): axis directions + frame
        self._spectra = None                # (field values id, cut_spectra)
        self._directional = []              # [(layer, η(2, e₁)/2, η(2, e₂)/2)] from the leaders
        self._seeded = False                # the H spins follow the estimate until touched
        self._hurst = None                  # the latest directional_hurst()
        outer = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Angle frame"))
        self.frame_combo = QtWidgets.QComboBox()
        bar.addWidget(self.frame_combo)
        bar.addWidget(QtWidgets.QLabel("Scale"))
        self.scale_spin = QtWidgets.QSpinBox()
        bar.addWidget(self.scale_spin)
        bar.addWidget(QtWidgets.QLabel("Bins"))
        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(8, 180)
        self.bins_spin.setValue(36)
        bar.addWidget(self.bins_spin)
        self.note = QtWidgets.QLabel("")
        bar.addWidget(self.note, 1)
        outer.addLayout(bar)

        # The fit window, the estimator and the collapse's q on one line; the rescaling
        # exponents, their seed and the sweep on the next (two short rows keep the window's
        # minimum width under a laptop screen's).
        fit = QtWidgets.QHBoxLayout()
        fit.addWidget(QtWidgets.QLabel("fit log₂a"))
        self.lo_spin = QtWidgets.QDoubleSpinBox()
        self.hi_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.lo_spin, self.hi_spin):
            spin.setRange(-2.0, 16.0)
            spin.setSingleStep(0.25)
            spin.setDecimals(2)
            fit.addWidget(spin)
        fit.addWidget(QtWidgets.QLabel("estimator"))
        self.estimator_combo = QtWidgets.QComboBox()
        self.estimator_combo.addItems(list(ESTIMATORS))
        fit.addWidget(self.estimator_combo)
        fit.addWidget(QtWidgets.QLabel("q"))
        self.q_spin = QtWidgets.QDoubleSpinBox()
        self.q_spin.setRange(-10.0, 10.0)
        self.q_spin.setSingleStep(0.5)
        self.q_spin.setValue(2.0)
        fit.addWidget(self.q_spin)
        fit.addStretch(1)
        outer.addLayout(fit)
        rescale = QtWidgets.QHBoxLayout()
        self.hx_label = QtWidgets.QLabel("rescale with H_x")
        rescale.addWidget(self.hx_label)
        self.hx_spin = QtWidgets.QDoubleSpinBox()
        self.hy_label = QtWidgets.QLabel("H_y")
        self.hy_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.hx_spin, self.hy_spin):
            spin.setRange(-2.0, 3.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
        rescale.addWidget(self.hx_spin)
        rescale.addWidget(self.hy_label)
        rescale.addWidget(self.hy_spin)
        self.seed_button = QtWidgets.QPushButton("seed from estimate")
        rescale.addWidget(self.seed_button)
        self.sweep_check = QtWidgets.QCheckBox("± σ, ± 2σ")
        rescale.addWidget(self.sweep_check)
        rescale.addStretch(1)
        outer.addLayout(rescale)

        self.readout = QtWidgets.QLabel("")
        self.readout.setWordWrap(True)
        self.readout.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        outer.addWidget(self.readout)

        row1 = QtWidgets.QHBoxLayout()
        self.pdf_plot = zoom_plot(title="P_a(A) across scales")
        self.pdf_plot.addLegend(offset=(-5, 5))
        self.plane_plot = pg.PlotWidget(title="WTMMM in the gradient plane")          # both-axes zoom: aspect locked
        self.plane_plot.setAspectLocked(True)
        self.sector_plot = zoom_plot(title="pdf of log2 M per angle sector")
        self.sector_plot.addLegend(offset=(-5, 5))
        for w in (self.pdf_plot, self.plane_plot, self.sector_plot):
            row1.addWidget(w, 1)
        outer.addLayout(row1, 1)

        row2 = QtWidgets.QHBoxLayout()
        self.comp_plots = [zoom_plot(title="ln P_a(T_ψ1)"), zoom_plot(title="ln P_a(T_ψ2)")]
        self.collapse_plots = [zoom_plot(title="ln P_a(T_ψ1) rescaled by a^H_x"),
                               zoom_plot(title="ln P_a(T_ψ2) rescaled by a^H_y")]
        for w in self.comp_plots + self.collapse_plots:
            w.addLegend(offset=(-5, 5))
            row2.addWidget(w, 1)
        outer.addLayout(row2, 1)

        row3 = QtWidgets.QHBoxLayout()
        self.mod_plot = zoom_plot(title="P_a(M) across scales")
        self.mod_plot.addLegend(offset=(-5, 5))
        self.qcollapse_plot = zoom_plot(title="a^−qH M^q P_a(M)")
        self.qcollapse_plot.addLegend(offset=(-5, 5))
        self.height_plot = zoom_plot(title="P_a(A) peak / isotropic")
        self.angle_plot = zoom_plot(title="P_a(A) peak angle")
        self.spectra_plot = zoom_plot(title="cut spectra")
        self.spectra_plot.addLegend(offset=(-5, 5))
        self.power_plot = pg.PlotWidget(title="2-D power (log2)")          # both-axes zoom: aspect locked
        self.power_plot.setAspectLocked(True)
        self.power_plot.hideAxis("bottom")
        self.power_plot.hideAxis("left")
        self._power_item = pg.ImageItem()
        self.power_plot.addItem(self._power_item)
        for w in (self.mod_plot, self.qcollapse_plot, self.height_plot, self.angle_plot,
                  self.spectra_plot, self.power_plot):
            row3.addWidget(w, 1)
        outer.addLayout(row3, 1)

        caption = QtWidgets.QLabel(
            "Row 1, Arnéodo, Decoster & Roux 2000 §6: a FLAT P_a(A) means isotropic scaling; the gradient-plane "
            "cloud is round when isotropic; identical sector pdfs mean M and A are independent. Rows 2–3, Khalil et "
            "al. 2006: each component of the WTMMM scales on its own, T_ψ1 ~ a^H_x and T_ψ2 ~ a^H_y, so its pdfs "
            "collapse once rescaled (drag H to see the sensitivity); the peaks of P_a(A) sharpen towards fine scales "
            "and turn as a^(H_y − H_x); the cuts' spectra read β = 2H + 1; the q whose rescaled modulus pdfs keep "
            "their area within 10 % are the q the partition functions can carry.")
        caption.setWordWrap(True)
        caption.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        outer.addWidget(caption)

        self.frame_combo.currentIndexChanged.connect(self.refresh)
        self.scale_spin.valueChanged.connect(self.refresh)
        self.bins_spin.valueChanged.connect(self.refresh)
        for spin in (self.lo_spin, self.hi_spin):
            spin.valueChanged.connect(self._on_window_changed)
        self.estimator_combo.currentIndexChanged.connect(self._on_window_changed)
        for spin in (self.hx_spin, self.hy_spin):
            spin.valueChanged.connect(self._on_h_touched)
        self.seed_button.clicked.connect(self._seed)
        self.sweep_check.toggled.connect(lambda _c: self._draw_rows_2_3())
        self.q_spin.valueChanged.connect(lambda _v: self._draw_rows_2_3())

    # -- data ---------------------------------------------------------------------------------
    def set_result(self, result, field, *, offset=(0.0, 0.0), scale_idx: int = 0,
                   fit_window=None, directional=None) -> None:
        """The active result (needs per-scale ``extrema``), its field (for bearings and the cut
        spectra), the result's pixel offset inside that field (an ROI result's origin), the
        fit window in log2 a (default: the scales' span), and the wavelet-leaders' directional
        readings ``[(layer name, η(2, e₁)/2, η(2, e₂)/2)]`` for the readout."""
        self._result, self._field, self._offset = result, field, tuple(offset)
        self._directional = list(directional or [])
        n = len((result or {}).get("extrema") or [])
        self.scale_spin.blockSignals(True)
        self.scale_spin.setRange(0, max(n - 1, 0))
        self.scale_spin.setValue(min(max(int(scale_idx), 0), max(n - 1, 0)))
        self.scale_spin.blockSignals(False)
        scales = self._scales(n)
        lo, hi = ((float(np.log2(scales.min())), float(np.log2(scales.max()))) if fit_window is None
                  else (float(fit_window[0]), float(fit_window[1])))
        for spin, value in ((self.lo_spin, lo), (self.hi_spin, hi)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._seeded = False
        self._rebuild_frames()
        self.refresh()

    def set_scale_window(self, lo: float, hi: float) -> None:
        """The fitter's region (log2 a) drives the fit window: the coupling channel the
        construction window also listens to."""
        for spin, value in ((self.lo_spin, lo), (self.hi_spin, hi)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._on_window_changed()

    def _scales(self, n: int) -> np.ndarray:
        s = (self._result or {}).get("scales") if self._result else None
        return np.asarray(s if s is not None else np.arange(1, n + 1), dtype=float) if n else np.array([1.0])

    def _rebuild_frames(self) -> None:
        from dynamix.geo.mapping import field_north

        current = self.frame_combo.currentData()
        north = None
        try:
            north = field_north(self._field) if self._field is not None else None
        except Exception:
            north = None
        self._north = north
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        self.frame_combo.addItem("pixel angle A", "pixel")
        if north is not None:
            if north[2] is Frame.GRID:
                self.frame_combo.addItem("azimuth — grid north", "grid")
            self.frame_combo.addItem("azimuth — true north", "true")
        idx = self.frame_combo.findData(current)
        self.frame_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.frame_combo.blockSignals(False)
        self.note.setText("" if north is not None else
                          "no georeference (a swath or pixel grid): pixel frame only")

    def _per_scale(self) -> list:
        if self._result is None:
            return []
        if self._wtmmm is None or self._wtmmm[0] is not self._result:
            self._wtmmm = (self._result, wtmmm_by_scale(self._result.get("extrema") or []))
        return self._wtmmm[1]

    def _pixel(self) -> bool:
        return (self.frame_combo.currentData() or "pixel") == "pixel"

    def _orientation(self, w: dict):
        """This scale's WTMMM angles in the chosen frame."""
        from dynamix.geo.mapping import grid_north_bearing

        o = from_arg(w["arg"])
        kind = self.frame_combo.currentData() or "pixel"
        if kind == "pixel" or self._north is None:
            return o
        col_east, row_north, north = self._north
        az = to_azimuth(o, col_east=col_east, row_north=row_north, north=north)
        if kind == "true" and north is Frame.GRID:
            cols = np.asarray(w["x"], float) + self._offset[1]
            rows = np.asarray(w["y"], float) + self._offset[0]
            az = grid_to_true(az, grid_north_bearing(self._field, cols, rows))
        return az

    def _components(self, w: dict, o) -> tuple:
        """The WTMMM gradient's two components in the chosen frame: (T_ψ1, T_ψ2) along the
        pixel axes, or (east, north) for a bearing."""
        mod = np.asarray(w["mod"], float)
        if self._pixel():
            return gradient_plane(mod, w["arg"])
        az = np.radians(np.asarray(o.degrees, float))
        return mod * np.sin(az), mod * np.cos(az)

    def _labels(self) -> tuple:
        """The component and exponent names in the chosen frame."""
        if self._pixel():
            return ("T_ψ1", "T_ψ2", "H_x", "H_y")
        return ("T E–W", "T N–S", "H_E–W", "H_N–S")

    def _window(self) -> tuple:
        """The fit window in log2 a, widened by half the spins' resolution so a scale the spins
        rounded past stays inside."""
        lo, hi = float(self.lo_spin.value()), float(self.hi_spin.value())
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        return lo - 0.005, hi + 0.005

    def _estimate(self) -> tuple:
        """``(H_1, H_2, σ_1, σ_2)`` of the chosen estimator from the latest fit, NaN without one."""
        if self._hurst is None:
            return (np.nan,) * 4
        key = ESTIMATORS[self.estimator_combo.currentText()]
        (h1, s1), (h2, s2) = self._hurst[key]
        return h1, h2, s1, s2

    # -- controls -----------------------------------------------------------------------------
    def _on_window_changed(self, *_a) -> None:
        self._seeded = False
        self.refresh()

    def _on_h_touched(self, *_a) -> None:
        self._seeded = True
        self._draw_rows_2_3()

    def _seed(self, *_a) -> None:
        self._seeded = False
        self._draw_rows_2_3()

    # -- drawing ------------------------------------------------------------------------------
    def refresh(self, *_a) -> None:
        for plot in (self.pdf_plot, self.plane_plot, self.sector_plot):
            plot.clear()
        per_scale = self._per_scale()
        if not per_scale:
            for plot in self.comp_plots + self.collapse_plots + [self.mod_plot, self.qcollapse_plot,
                                                                 self.height_plot, self.angle_plot,
                                                                 self.spectra_plot]:
                plot.clear()
            self._power_item.clear()
            self.readout.setText("the WTMM's maxima are needed: run a wtmm2d analysis")
            return
        scales = self._scales(len(per_scale))
        bins = int(self.bins_spin.value())
        pick = np.unique(np.linspace(0, len(per_scale) - 1,
                                     min(len(per_scale), len(_SCALE_COLORS))).astype(int))
        for colour, si in zip(_SCALE_COLORS, pick):
            o = self._orientation(per_scale[si])
            if np.size(o.degrees) == 0:
                continue
            c, pdf = angle_pdf(o, bins=bins)
            label = f"a = {scales[si]:g} ({np.size(o.degrees)})"
            self.pdf_plot.plot(c, pdf, pen=pg.mkPen(colour, width=1.6), name=label)
        self.pdf_plot.addLine(y=1.0 / 360.0, pen=pg.mkPen(RESTRAINED_DARK.ink_muted, style=QtCore.Qt.DashLine))
        pixel = self._pixel()
        self.pdf_plot.setLabel("bottom", "A (deg, pixel frame)" if pixel else "azimuth (deg)")
        si = int(self.scale_spin.value())
        w = per_scale[si]
        o = self._orientation(w)
        mod = np.asarray(w["mod"], float)
        t1, t2 = self._components(w, o)
        self.plane_plot.setLabel("bottom", "T_ψ1" if pixel else "east")
        self.plane_plot.setLabel("left", "T_ψ2" if pixel else "north")
        self.plane_plot.plot(t1, t2, pen=None, symbol="o", symbolSize=3,
                             symbolBrush=pg.mkBrush(78, 121, 167, 140), symbolPen=None)
        self.plane_plot.setTitle(f"WTMMM in the gradient plane, a = {scales[si]:g}")
        labels = SECTOR_LABELS["pixel" if pixel else "azimuth"]
        for (k, c, pdf, count), colour in zip(sector_modulus_pdfs(mod, o), _SECTOR_COLORS):
            self.sector_plot.plot(c, pdf, pen=pg.mkPen(colour, width=1.6),
                                  name=f"{labels[k]} ({count})")
        self.sector_plot.setLabel("bottom", "log2 M")
        self.sector_plot.setTitle(f"pdf of log2 M per angle sector, a = {scales[si]:g}")
        self._draw_rows_2_3()

    def _draw_rows_2_3(self) -> None:
        per_scale = self._per_scale()
        if not per_scale:
            return
        scales = self._scales(len(per_scale))
        lo, hi = self._window()
        keep = [k for k in range(len(per_scale)) if lo - 1e-9 <= np.log2(scales[k]) <= hi + 1e-9
                and np.size(per_scale[k]["mod"]) > 0]
        orientations = [self._orientation(w) for w in per_scale]
        components = [self._components(w, o) for w, o in zip(per_scale, orientations)]
        mods = [np.asarray(w["mod"], float) for w in per_scale]
        t1_name, t2_name, h1_name, h2_name = self._labels()
        self._hurst = directional_hurst(components, scales, lo, hi)
        h1, h2, s1, s2 = self._estimate()
        if not self._seeded:
            for spin, value in ((self.hx_spin, h1), (self.hy_spin, h2)):
                spin.blockSignals(True)
                spin.setValue(float(value) if np.isfinite(value) else 0.0)
                spin.blockSignals(False)
        self.hx_label.setText(f"rescale with {h1_name}")
        self.hy_label.setText(h2_name)
        rescale = (float(self.hx_spin.value()), float(self.hy_spin.value()))

        # Row 2: the component pdfs raw and collapsed over the window's scales.
        kept_components = [components[k] for k in keep]
        kept_scales = scales[keep]
        raw = component_pdfs(kept_components, kept_scales, bins=int(self.bins_spin.value()))
        col = component_pdfs(kept_components, kept_scales, bins=int(self.bins_spin.value()), rescale=rescale)
        for i, (name, hname) in enumerate(((t1_name, h1_name), (t2_name, h2_name))):
            for plot, table, title in ((self.comp_plots[i], raw, f"ln P_a({name})"),
                                       (self.collapse_plots[i], col, f"ln P_a({name}) rescaled by a^{hname} = {rescale[i]:.3f}")):
                plot.clear()
                plot.setTitle(title)
                for j, (k, colour) in enumerate(zip(keep, _SCALE_COLORS * (len(keep) // len(_SCALE_COLORS) + 1))):
                    plot.plot(table["centres"][i], table["ln_pdf"][i, j], pen=pg.mkPen(colour, width=1.4),
                              connect="finite", name=f"a = {scales[k]:g} ({table['counts'][i, j]})")
            plot = self.collapse_plots[i]
            plot.setLabel("bottom", f"{name} / a^{hname}")
            self.comp_plots[i].setLabel("bottom", name)
            if self.sweep_check.isChecked() and np.isfinite((s1, s2)[i]) and keep:
                sigma = (s1, s2)[i]
                for step in (-2.0, -1.0, 1.0, 2.0):
                    shifted = list(rescale)
                    shifted[i] = rescale[i] + step * sigma
                    sweep = component_pdfs(kept_components, kept_scales, bins=int(self.bins_spin.value()),
                                           rescale=tuple(shifted))
                    for j in (0, len(keep) - 1):                        # the finest and the coarsest
                        plot.plot(sweep["centres"][i], sweep["ln_pdf"][i, j],
                                  pen=pg.mkPen(RESTRAINED_DARK.ink_muted, width=0.8, style=QtCore.Qt.DotLine),
                                  connect="finite", name=f"{hname} {step:+g}σ" if j == 0 else None)

        # Row 3: the modulus pdfs, the q-collapse, the anisotropy against scale, the spectra.
        self.mod_plot.clear()
        mp = modulus_pdfs([mods[k] for k in keep], kept_scales, bins=int(self.bins_spin.value()))
        for j, (k, colour) in enumerate(zip(keep, _SCALE_COLORS * (len(keep) // len(_SCALE_COLORS) + 1))):
            self.mod_plot.plot(mp["centres"], mp["pdf"][j], pen=pg.mkPen(colour, width=1.4),
                               name=f"a = {scales[k]:g}")
        self.mod_plot.setLabel("bottom", "M")
        H, H_se = modulus_hurst(mods, scales, lo, hi)
        q = float(self.q_spin.value())
        self.qcollapse_plot.clear()
        usable_text = ""
        if np.isfinite(H) and keep:
            grid = np.unique(np.concatenate([_Q_GRID, [q]]))
            qc = modulus_collapse(mods, scales, H, grid, lo, hi, bins=int(self.bins_spin.value()))
            qi = int(np.argmin(np.abs(grid - q)))
            for j, (k, colour) in enumerate(zip(keep, _SCALE_COLORS * (len(keep) // len(_Q_GRID) + 2))):
                self.qcollapse_plot.plot(qc["x"], qc["curves"][qi, j], pen=pg.mkPen(colour, width=1.4),
                                         connect="finite", name=f"a = {scales[k]:g}")
            self.qcollapse_plot.setTitle(f"a^−qH M^q P_a(M), q = {q:g}, H = {H:.2f}, "
                                         f"area spread {qc['area_spread'][qi]:.0%}")
            usable = grid[qc["usable"] & np.isin(grid, _Q_GRID)]
            usable_text = (f"q within 10 %: {usable.min():g} … {usable.max():g}" if usable.size
                           else "no q keeps its area within 10 %")
        self.qcollapse_plot.setLabel("bottom", "M / a^H")

        av = anisotropy_vs_scale(orientations, scales, bins=int(self.bins_spin.value()))
        x = np.log2(scales)
        self.height_plot.clear()
        self.height_plot.plot(x, av["relative_height"], pen=pg.mkPen(_SCALE_COLORS[0], width=1.6),
                              symbol="o", symbolSize=5, symbolBrush=pg.mkBrush(*_SCALE_COLORS[0]), symbolPen=None)
        self.height_plot.addLine(y=1.0, pen=pg.mkPen(RESTRAINED_DARK.ink_muted, style=QtCore.Qt.DashLine))
        self.height_plot.setLabel("bottom", "log2 a")
        self.height_plot.setLabel("left", "peak / isotropic")
        self.angle_plot.clear()
        self.angle_plot.plot(x, av["angle_deg"], pen=pg.mkPen(_SCALE_COLORS[3], width=1.6),
                             symbol="o", symbolSize=5, symbolBrush=pg.mkBrush(*_SCALE_COLORS[3]), symbolPen=None)
        self.angle_plot.setLabel("bottom", "log2 a")
        self.angle_plot.setLabel("left", "A mod 180° (pixel, deg)" if self._pixel() else "bearing mod 180° (deg)")
        self.angle_plot.setYRange(-90.0, 90.0)
        for plot in (self.height_plot, self.angle_plot):
            for k in keep[:1] + keep[-1:]:
                plot.addLine(x=float(np.log2(scales[k])), pen=pg.mkPen(RESTRAINED_DARK.ink_muted, style=QtCore.Qt.DotLine))

        spectra = self._cut_spectra()
        self.spectra_plot.clear()
        beta_text = ""
        if spectra is not None:
            for key_k, key_s, key_b, colour, name in (("k_x", "S_x", "beta_x", _SCALE_COLORS[0], "x-cuts" if self._pixel() else "E–W cuts"),
                                                      ("k_y", "S_y", "beta_y", _SCALE_COLORS[3], "y-cuts" if self._pixel() else "N–S cuts"),
                                                      ("k_r", "S_r", "beta_r", _SCALE_COLORS[2], "radial")):
                with np.errstate(divide="ignore"):
                    self.spectra_plot.plot(np.log2(spectra[key_k]), np.log2(spectra[key_s]),
                                           pen=pg.mkPen(colour, width=1.4), connect="finite",
                                           name=f"{name}: β = {spectra[key_b]:.2f}")
            self.spectra_plot.setLabel("bottom", "log2 k (cycles/px)")
            self.spectra_plot.setLabel("left", "log2 S")
            self._power_item.setImage(spectra["log2_power"].T)
            beta_text = (f"β_x {spectra['beta_x']:.2f} ± {spectra['beta_x_se']:.2f} · "
                         f"β_y {spectra['beta_y']:.2f} ± {spectra['beta_y_se']:.2f} · β {spectra['beta_r']:.2f}")
        else:
            self._power_item.clear()

        parts = []
        key = self.estimator_combo.currentText()
        if np.isfinite(h1) and np.isfinite(h2):
            others = {k: self._hurst[v] for k, v in ESTIMATORS.items() if k != key}
            tail = "; ".join(f"{k} {o[0][0]:.2f} / {o[1][0]:.2f}" for k, o in others.items()
                             if np.isfinite(o[0][0]) and np.isfinite(o[1][0]))
            parts.append(f"{h1_name} {h1:.2f} ± {s1:.2f} · {h2_name} {h2:.2f} ± {s2:.2f} ({key}; {tail})")
            parts.append(f"M follows {(t1_name, t2_name)[self._hurst['dominant']]} ({(h1_name, h2_name)[self._hurst['dominant']]})"
                         + (f", H {H:.2f} ± {H_se:.2f}" if np.isfinite(H) else ""))
        else:
            parts.append("fewer than three scales in the fit window: no estimate")
        if beta_text:
            parts.append(beta_text)
        if usable_text:
            parts.append(usable_text)
        for layer, e1, e2 in self._directional:
            parts.append(f"DWT ({layer}): η(2, e₁)/2 {e1:.2f}, η(2, e₂)/2 {e2:.2f}")
        self.readout.setText(" · ".join(parts))

    def _cut_spectra(self):
        """The field's cut spectra, computed once per field."""
        values = getattr(self._field, "values", None) if self._field is not None else None
        if values is None or np.ndim(values) != 2 or min(np.shape(values)) < 16:
            return None
        if self._spectra is None or self._spectra[0] is not values:
            self._spectra = (values, cut_spectra(np.asarray(values, dtype=np.float64)))
        return self._spectra[1]
