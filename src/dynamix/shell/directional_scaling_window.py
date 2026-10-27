# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The hyperbolic moment populations, retained scale fits and independent-axis basis."""
from __future__ import annotations

import json

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import directional as dr
from dynamix.core import dwt
from dynamix.core.hyperbolic_preview import axis_preview, tensor_preview
from dynamix.shell.multifractal_window import Q_COLD, Q_WARM, _q_color
from dynamix.shell.plot_zoom import zoom_plot
from dynamix.shell.settings import start_path
from dynamix.shell.theme import current
from dynamix.shell.window_fit import TOOL_WINDOW, fit_to_screen, show_while_inactive


def _translucent(color, alpha: int):
    c = pg.mkColor(color)
    c.setAlpha(int(alpha))
    return pg.mkBrush(c)


class DirectionalScalingWindow(QtWidgets.QWidget):
    bankRequested = QtCore.Signal(str)

    def __init__(self, result=None, *, params=None, source='', field=None, parent=None):
        super().__init__(parent, TOOL_WINDOW)
        show_while_inactive(self)
        self.result = result or {}
        self.data = self.result.get('_directional')
        # The window's own refits: η_α per anisotropy over its chosen fit range (level line).
        self._ranges = {}
        if self.data is not None:
            self._eta = np.array(self.data['eta_alpha'], dtype=np.float64)
            self._se = np.array(self.data.get('eta_alpha_se', np.zeros_like(self._eta)), dtype=np.float64)
            self._view = self.data
        self.params = dict(self.result.get('params') or params or {})
        self.preview_params = dict(params or self.params)
        self.field = field
        self.source = source
        self.setWindowTitle(f'Directional partition functions and basis — {source}')
        fit_to_screen(self, 1060, 760)
        layout = QtWidgets.QVBoxLayout(self)
        self.caption = QtWidgets.QLabel()
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        # What kept the fit or the bound from existing, as the tool diagnosed it.
        warnings = list((result or {}).get('warnings') or [])
        self.warning = QtWidgets.QLabel('\n'.join(warnings))
        self.warning.setWordWrap(True)
        self.warning.setProperty('reading', 'true')
        self.warning.setVisible(bool(warnings))
        layout.addWidget(self.warning)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._moments_tab()
        self._scaling_tab()
        self._basis_tab()
        self._leaders_tab()
        footer = QtWidgets.QHBoxLayout()
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        footer.addWidget(self.readout, 1)
        self.export = QtWidgets.QPushButton('Save diagnostic data…')
        self.export.clicked.connect(self._save_data)
        self.export.setEnabled(self.data is not None)
        footer.addWidget(self.export)
        layout.addLayout(footer)
        if self.data is None:
            self.tabs.setTabEnabled(0, False)
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabEnabled(3, False)
            self.tabs.setCurrentIndex(2)
            self.caption.setText('Basis preview · choose a bank and independent x/y levels. Run the directional tool for moment tables and scaling fits.')
        else:
            d = self.data
            refit = ('Drag the band on the envelope plot to refit the chosen anisotropy from the stored frontiers.'
                     if self._level_line() else "Fits use the computed scale range; change the tool’s cell limits and Run to refit.")
            self.caption.setText(f"{source} · {self.params.get('wavelet', '')} · "
                                 f"{d.get('estimator_description', d['estimator'])} · {d['moment_normalization']}. {refit}")
            self._draw_moments()
            self._draw_scaling()
        self._draw_basis()

    def _page(self, title):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.tabs.addTab(page, title)
        return layout

    def _plot(self, title):
        plot = zoom_plot(title=title)
        plot.setBackground(current().ground)
        plot.setMinimumSize(180, 145)
        for side in ('bottom', 'left'):
            plot.getAxis(side).enableAutoSIPrefix(False)
        return plot

    @staticmethod
    def _combo(row, label, choices):
        row.addWidget(QtWidgets.QLabel(label))
        combo = QtWidgets.QComboBox()
        combo.addItems(choices)
        row.addWidget(combo)
        return combo

    def _level_line(self):
        return self.data is not None and self.data['estimator'] == 'level_line'

    def _family_names(self):
        if self._level_line():
            return ['x-frontier', 'y-frontier']
        legacy = self.data is not None and self.data['estimator'] == 'legacy'
        return ['near diagonal', 'x-side', 'y-side'] if legacy else ['Γ union']

    def _families(self, fit):
        """The fit's families, each with its own abscissa: the level line's two frontiers (their
        points sit at m = r/α on their own axis), or the Γ families on a shared one."""
        if 'arms' in fit:
            out = []
            for name, key in (('x-frontier', 'x'), ('y-frontier', 'y')):
                arm = fit['arms'][key]
                x, env = np.asarray(arm['x'], dtype=np.float64), np.asarray(arm['envelope'], dtype=np.float64)
                slope = np.asarray(arm['slope'], dtype=np.float64)
                intercept = (np.nanmean(env, axis=1) - slope * x.mean()) if x.size else np.full(slope.size, np.nan)
                out.append({'name': name, 'x': x, 'env': env, 'members': arm['members'],
                            'slope': slope, 'intercept': intercept})
            return out
        names = self._family_names()
        return [{'name': names[f] if f < len(names) else f'family {f}', 'x': np.asarray(fit['x']),
                 'env': fit['envelopes'][:, f], 'members': [m[f] for m in fit['members']],
                 'slope': fit['slopes'][:, f], 'intercept': fit['intercepts'][:, f]}
                for f in range(fit['envelopes'].shape[1])]

    def _moments_tab(self):
        layout = self._page('Moments and scale fits')
        bar = QtWidgets.QHBoxLayout()
        d = self.data
        self.moment = self._combo(bar, 'Moment p', [f'{p:g}' for p in d['p']] if d else [])
        self.alpha = self._combo(bar, 'α₁ (α₂ = 2 − α₁)', [f'{a:g}' for a in d['alpha']] if d else [])
        self.family = self._combo(bar, 'Family', self._family_names())
        self.resolution = self._combo(bar, 'Fit sample', [])
        # The level line refits one anisotropy, or every one, over the band on the envelope plot.
        self.same_range = QtWidgets.QCheckBox('same range for every α')
        self.same_range.setVisible(self._level_line())
        bar.addWidget(self.same_range)
        bar.addStretch(1)
        layout.addLayout(bar)
        body = QtWidgets.QHBoxLayout()
        self.moment_plot = self._plot('Hyperbolic log₂ moment')
        self.envelope_plot = self._plot('Γ envelopes and retained fits')
        body.addWidget(self.moment_plot)
        body.addWidget(self.envelope_plot)
        layout.addLayout(body, 1)
        self.moment_note = QtWidgets.QLabel()
        self.moment_note.setWordWrap(True)
        layout.addWidget(self.moment_note)
        if d:
            self.alpha.setCurrentIndex(int(np.argmin(np.abs(d['alpha']-1))))
            self.moment.setCurrentIndex(int(np.argmin(np.abs(d['p']-2))))
        for combo in (self.moment, self.alpha, self.family):
            combo.currentIndexChanged.connect(self._draw_moments)
        self.resolution.currentIndexChanged.connect(self._draw_moments)

    def _draw_moments(self, *_):
        d = self.data
        if d is None:
            return
        pi, ai = self.moment.currentIndex(), self.alpha.currentIndex()
        fit = d['diagnostic_fits'][ai]
        families = self._families(fit)
        fi = min(self.family.currentIndex(), len(families) - 1)
        family = families[fi]
        self.resolution.blockSignals(True)
        previous = self.resolution.currentIndex()
        self.resolution.clear()
        self.resolution.addItems([f'{x:g}' for x in family['x']])
        self.resolution.setCurrentIndex(min(max(previous, 0), len(family['x'])-1))
        self.resolution.blockSignals(False)
        ri = self.resolution.currentIndex()
        self.moment_plot.clear()
        self.envelope_plot.clear()
        pairs = np.asarray(d['moment_pairs'])
        depth = int(d['levels'])
        table = np.full((depth+1, depth+1), np.nan)
        counts = np.zeros_like(table, dtype=int)
        for col, (jx, jy) in enumerate(pairs):
            table[jy, jx] = d['log2_moments'][pi, col]
            counts[jy, jx] = d['moment_counts'][col]
        self.moment_image = pg.ImageItem(np.where(np.isfinite(table), table, np.nan), axisOrder='row-major')
        finite = table[np.isfinite(table)]
        levels = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if levels[0] == levels[1]:
            levels = (levels[0]-.5, levels[1]+.5)
        self.moment_image.setLookupTable(pg.colormap.get('viridis').getLookupTable())
        self.moment_image.setLevels(levels)
        self.moment_image.setRect(QtCore.QRectF(-.5, -.5, depth+1, depth+1))
        self.moment_plot.addItem(self.moment_image)
        self.moment_plot.setLabel('bottom', 'x DWT index jₓ (0 = scaling φ)')
        self.moment_plot.setLabel('left', 'y DWT index jᵧ (0 = scaling φ)')
        self.moment_plot.setAspectLocked(True)
        members = family['members'][ri] if 0 <= ri < len(family['members']) else []
        if self._level_line():
            # The level line's members are resolution indices (φ = −1): the table's DWT index is J − r.
            members = [(0 if rx < 0 else depth - rx, 0 if ry < 0 else depth - ry) for rx, ry in members]
        if members:
            xy = np.asarray(members)
            self.moment_plot.plot(xy[:, 0], xy[:, 1], pen=None, symbol='s', symbolSize=13,
                                  symbolBrush=None, symbolPen=pg.mkPen(current().amber, width=2))
        self.envelope_plot.setTitle('Frontiers and their fits · drag the band to refit' if self._level_line()
                                    else 'Γ envelopes and retained fits')
        self.envelope_plot.setLabel('bottom', fit['xlabel'])
        self.envelope_plot.setLabel('left', 'log₂ S (frontier)' if self._level_line() else 'log₂ S (Γ envelope)')
        self.envelope_plot.addLegend(offset=(4, 4))
        shown = families if self._level_line() else [family]
        for fam in shown:
            colors = ({'x-frontier': Q_COLD, 'y-frontier': Q_WARM} if self._level_line() else {})
            for order, p in enumerate(d['p']):
                if self._level_line() and order != pi:
                    continue
                color = colors.get(fam['name'], _q_color(order, len(d['p'])))
                if not np.isfinite(fam['env'][order]).any():
                    continue
                name = f"{fam['name']}, p={p:g}" if self._level_line() else f'p={p:g}'
                self.envelope_plot.plot(fam['x'], fam['env'][order], pen=pg.mkPen(color),
                                        symbol='o', symbolSize=5, name=name)
                slope, intercept = fam['slope'][order], fam['intercept'][order]
                if np.isfinite(slope) and fam['x'].size:
                    x = np.array([min(fam['x']), max(fam['x'])])
                    self.envelope_plot.plot(x, slope*x+intercept, pen=pg.mkPen(color, style=QtCore.Qt.DashLine))
        if 0 <= ri < family['x'].size:
            self.envelope_plot.addItem(pg.InfiniteLine(float(family['x'][ri]), pen=pg.mkPen(current().amber)))
        self._draw_leaders()
        if self._level_line():
            xs = np.concatenate([f['x'] for f in families if f['x'].size]) if any(f['x'].size for f in families) else np.zeros(0)
            if xs.size:
                lo, hi = self._ranges.get(ai, (float(xs.min()), float(xs.max())))
                self.fit_region = pg.LinearRegionItem((lo, hi), brush=_translucent(current().amber, 34))
                self.fit_region.sigRegionChangeFinished.connect(self._on_fit_region)
                self.envelope_plot.addItem(self.fit_region)
        n = d['n_levels'][ai]
        values = [f'({x},{y}): n={counts[y,x]}' for x, y in members]
        admissible = np.isfinite(self._eta[:, ai]).any()
        self.moment_note.setText(f"log₂ S colour range {levels[0]:.3g} … {levels[1]:.3g} · "
                                 f"α₁={d['alpha'][ai]:g} · {len(family['x'])} retained samples · "
                                 f"{'fit admissible' if admissible else 'insufficient independent levels: no exponent fit'} · "
                                 f"ηα({d['p'][pi]:g})={self._eta[pi, ai]:.4g} · fitted levels {n}\n"
                                 'Highlighted members and valid-cell counts: ' + ('; '.join(values) if values else 'none'))
        self._draw_scaling()

    def _on_fit_region(self, region):
        lo, hi = (float(v) for v in region.getRegion())
        targets = range(len(self.data['alpha'])) if self.same_range.isChecked() else [self.alpha.currentIndex()]
        for ai in targets:
            self._refit(ai, lo, hi)
        self._recompute()
        self._draw_moments()

    def _refit(self, ai, lo, hi):
        """η_α over the frontier points inside [lo, hi] of −m: each frontier's slope over its points
        there (three at least), the smaller of the two, as the estimator reads it."""
        self._ranges[ai] = (lo, hi)
        arms = self.data['diagnostic_fits'][ai]['arms']
        n_p = len(self.data['p'])
        best, err = np.full(n_p, np.nan), np.full(n_p, np.nan)
        for arm in arms.values():
            x = np.asarray(arm['x'], dtype=np.float64)
            keep = (x >= lo - 1e-9) & (x <= hi + 1e-9)
            if keep.sum() < 3:
                continue
            env = np.asarray(arm['envelope'], dtype=np.float64)
            for i in range(n_p):
                slope, se = dr._slope_se(x[keep], env[i][keep])
                if np.isfinite(slope) and not slope >= best[i]:
                    best[i], err[i] = slope, se
        self._eta[:, ai], self._se[:, ai] = best, err

    def _recompute(self):
        d = self.data
        scaling = {'alpha': d['alpha'], 'eta_alpha': self._eta, 'eta_alpha_se': self._se,
                   'no_structure': d.get('no_structure', {})}
        self._view = {**d, 'eta_alpha': self._eta, **dr.directional_scaling(scaling, d['p']),
                      'L_e1': dr.directional_legendre(scaling, d['p'], direction=1),
                      'L_e2': dr.directional_legendre(scaling, d['p'], direction=2)}

    def _scaling_tab(self):
        layout = self._page('Scaling and directional bounds')
        grid = QtWidgets.QGridLayout()
        self.scaling_plots = [self._plot(t) for t in ('ηα(p) and x/y envelopes', 'Anisotropy profile',
                                                      'Upper bounds on D(H, e)', 'The sup over α')]
        grid.addWidget(self.scaling_plots[0], 0, 0)
        grid.addWidget(self.scaling_plots[3], 0, 1)
        grid.addWidget(self.scaling_plots[1], 1, 0)
        grid.addWidget(self.scaling_plots[2], 1, 1)
        layout.addLayout(grid)
        note = QtWidgets.QLabel('e₁ = native x axis · e₂ = native y axis · α is anisotropic dilation, not an angle. '
                                'The bounds are Theorem 6 (Ben Slimane et al. 2019), on the upper spectrum '
                                'D(H, e) = dim{h(·, e) ≤ H}, which never decreases: empty where nothing is drawn, '
                                'the dashed stretch set by p max; not a measured D(h).')
        note.setWordWrap(True)
        layout.addWidget(note)

    def _draw_scaling(self):
        if self.data is None or not hasattr(self, 'scaling_plots'):
            return
        d, v, ai, pi = self.data, self._view, self.alpha.currentIndex(), self.moment.currentIndex()
        for plot in self.scaling_plots:
            plot.clear()
            plot.addLegend(offset=(4, 4))
        curves, profile, bounds, sup = self.scaling_plots
        for col, a in enumerate(d['alpha']):
            curves.plot(d['p'], self._eta[:, col], pen=pg.mkPen(current().amber if col == ai else current().ink_muted, width=2 if col == ai else 1))
        # The isotropic curve first, so the directional ones are drawn over it.
        for key, name, color in [('eta_iso', 'η(p), α=(1,1)', current().ink), ('eta_e1', 'η(p,e₁)', Q_COLD), ('eta_e2', 'η(p,e₂)', Q_WARM)]:
            curves.plot(d['p'], v[key], pen=pg.mkPen(color, width=2), name=name)
        gamma = float(self.params.get('gamma', 0.0) or 0.0)
        integrated = (f' (coefficient integration γ = {gamma:g}, order γ/2 per axis)'
                      if gamma else '')
        curves.setLabel('bottom', 'Moment p')
        curves.setLabel('left', 'η' + integrated)
        # With one fitted α, η(p,e₁) = η(p,e₂) = η_iso by construction: that is no evidence of isotropy.
        fitted = int(np.isfinite(self._eta).any(axis=0).sum())
        curves.setTitle(f"ηα(p) and x/y envelopes · {fitted} of {len(d['alpha'])} anisotropies fitted")
        profile.setTitle('Anisotropy profile' if fitted >= 2 else f'Anisotropy profile needs two fitted anisotropies ({fitted} fitted)')
        for key, color, name in [('alpha_star_e1', Q_COLD, 'α₁* for e₁'), ('alpha_star_e2', Q_WARM, 'α₂* for e₂')]:
            if fitted >= 2 and np.isfinite(v[key]).any():
                profile.plot(d['p'], v[key], pen=pg.mkPen(color), symbol='o', symbolSize=4, name=name)
        profile.setLabel('bottom', 'Moment p')
        profile.setLabel('left', 'Maximizing axis weight α*' + integrated)
        # The sup over α at the chosen p: η_α/α₁ and η_α/α₂ against α₁, the α* intervals shaded.
        alphas = np.asarray(d['alpha'], dtype=np.float64)
        p_now = float(d['p'][pi])
        with np.errstate(invalid='ignore', divide='ignore'):
            ratios = ((self._eta[pi] / alphas, 'η_α/α₁ (e₁)', Q_COLD, 'alpha_star_e1_interval', lambda a: a),
                      (self._eta[pi] / (2.0 - alphas), 'η_α/α₂ (e₂)', Q_WARM, 'alpha_star_e2_interval', lambda a: 2.0 - a))
        for ratio, name, color, key, to_alpha1 in ratios:
            sup.plot(alphas, ratio, pen=pg.mkPen(color, width=2), symbol='o', symbolSize=4, name=name)
            interval = np.asarray(v.get(key, np.full((2, len(d['p'])), np.nan)))[:, pi]
            if np.isfinite(interval).all():
                lo, hi = sorted(to_alpha1(interval))
                shade = pg.LinearRegionItem((lo - 1e-3, hi + 1e-3), movable=False, brush=_translucent(color, 51))
                shade.setZValue(-10)
                sup.addItem(shade)
        sup.setTitle(f'The sup over α at p = {p_now:g} · shaded: α* within one standard error')
        sup.setLabel('bottom', 'α₁ (α₂ = 2 − α₁)')
        sup.setLabel('left', 'η_α(p)/α_e')
        knees = []
        for key, name, color in [('L_e1', 'D(H,e₁) bound', Q_COLD), ('L_e2', 'D(H,e₂) bound', Q_WARM)]:
            leg = v[key]
            h, L = np.asarray(leg['h']), np.asarray(leg['L'], dtype=np.float64)
            limited = np.asarray(leg.get('limited', np.zeros(h.size, bool)), dtype=bool)
            bounds.plot(h, np.where(limited, np.nan, L), pen=pg.mkPen(color, width=2), name=name)
            if limited.any():
                bounds.plot(h, np.where(limited, L, np.nan), pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine))
            if np.isfinite(leg.get('knee', np.nan)):
                knees.append(f"{name.split()[0]} knee H ≈ {leg['knee']:.2f}")
        finite_bound = any(np.isfinite(v[key]['L']).any() for key in ('L_e1', 'L_e2'))
        bounds.setTitle(('Upper bounds on D(H, e)' + (' · ' + ' · '.join(knees) if knees else ''))
                        if finite_bound else 'No finite directional bound in this computed range')
        bounds.setLabel('bottom', 'H' + integrated)
        bounds.setLabel('left', 'upper bound on D(H, e)')

    def _leaders_tab(self):
        layout = self._page('Leaders and local maps')
        self.leader_plot = self._plot('Level-set bound L_f(H, α) from the hyperbolic leaders')
        layout.addWidget(self.leader_plot, 1)
        note = QtWidgets.QLabel(
            "Abry et al.'s hyperbolic wavelet leaders (2015, eq. 3.3), read by the level line over p of both "
            "signs, bound the level-set spectrum of each anisotropy: L_f(H, α) = inf_p (Hp − ω(p, α) + 2) "
            "(eq. 3.7), empty below 0. The chosen α in amber, the isotropic α = (1, 1) in white. The layer's "
            "Outputs rows show the local maps H_x, H_y and the anisotropy α₁* = 2H_y/(H_x + H_y) from the "
            "strip leaders; their resolution is the coarsest fitted cell along the fine axis and three "
            "scale-max cells along the coarse one.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def _draw_leaders(self):
        lead = (self.data or {}).get('leaders')
        if lead is None or not hasattr(self, 'leader_plot'):
            return
        self.leader_plot.clear()
        self.leader_plot.addLegend(offset=(4, 4))
        alphas = np.asarray(self.data['alpha'], dtype=np.float64)
        ai, diag = self.alpha.currentIndex(), int(np.argmin(np.abs(alphas - 1.0)))
        for k, color, name in ((diag, current().ink, 'α = (1, 1)'), (ai, current().amber, f'α₁ = {alphas[ai]:g}')):
            self.leader_plot.plot(lead['h'], lead['L_alpha'][k], pen=pg.mkPen(color, width=2), name=name)
        self.leader_plot.setLabel('bottom', 'H')
        self.leader_plot.setLabel('left', 'L_f(H, α), upper bound on d(H, α)')

    def _basis_tab(self):
        layout = self._page('Hyperbolic basis preview')
        row = QtWidgets.QHBoxLayout()
        self.bank = self._combo(row, 'Preview bank', list(dwt.BANKS))
        self.bank.setCurrentText(self.preview_params.get('wavelet', 'sym4'))
        depth = int(np.log2(self.preview_params.get('scale_max', 64)))
        self.depth = QtWidgets.QSpinBox()
        self.depth.setRange(1, 10)
        self.depth.setValue(depth)
        row.addWidget(QtWidgets.QLabel('Depth J'))
        row.addWidget(self.depth)
        branches = QtWidgets.QHBoxLayout()
        self.jx = self._combo(branches, 'x branch', [])
        self.jy = self._combo(branches, 'y branch', [])
        self.use_bank = QtWidgets.QPushButton('Use bank in tool')
        self.use_bank.clicked.connect(lambda: self.bankRequested.emit(self.bank.currentText()))
        row.addWidget(self.use_bank)
        layout.addLayout(row)
        layout.addLayout(branches)
        grid = QtWidgets.QGridLayout()
        self.basis_plots = [self._plot(t) for t in ('x analysis filter', 'y analysis filter', 'Frequency responses', 'Tensor-product filter footprint')]
        for i, plot in enumerate(self.basis_plots):
            grid.addWidget(plot, i//2, i%2)
        layout.addLayout(grid, 1)
        self.basis_note = QtWidgets.QLabel()
        self.basis_note.setWordWrap(True)
        layout.addWidget(self.basis_note)
        self._set_branches()
        self.bank.currentIndexChanged.connect(self._draw_basis)
        self.depth.valueChanged.connect(self._set_branches)
        self.jx.currentIndexChanged.connect(self._draw_basis)
        self.jy.currentIndexChanged.connect(self._draw_basis)

    def _set_branches(self, *_):
        depth = self.depth.value()
        for combo in (self.jx, self.jy):
            old = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(['φ (coarsest scaling)']+[f'ψ, level {j}' for j in range(1, depth+1)])
            combo.setCurrentIndex(min(old if old >= 0 else min(2, depth), depth))
            combo.blockSignals(False)
        if hasattr(self, 'readout'):
            self._draw_basis()

    def _draw_basis(self, *_):
        if not hasattr(self, 'basis_plots'):
            return
        bank, depth = self.bank.currentText(), self.depth.value()
        x = axis_preview(bank, self.jx.currentIndex(), depth)
        y = axis_preview(bank, self.jy.currentIndex(), depth)
        self.preview = (x, y)
        for plot in self.basis_plots:
            plot.clear()
        for plot, data, name in zip(self.basis_plots[:2], (x, y), ('x', 'y')):
            step = max(1, int(np.ceil(data['taps'].size/4096)))
            plot.plot(data['x'][::step], data['profile'][::step], pen=pg.mkPen(current().ink))
            plot.setLabel('bottom', f'{name} offset (px)')
            plot.setLabel('left', f"{data['branch']} analysis atom (peak-normalized)")
        freq, tensor = self.basis_plots[2:]
        freq.addLegend(offset=(4, 4))
        for data, color, name in ((x, Q_COLD, 'x'), (y, Q_WARM, 'y')):
            freq.plot(data['frequency'], data['magnitude'], pen=pg.mkPen(color, width=2), name=f"{name}: {data['branch']}, level {data['level']}")
        freq.setLabel('bottom', 'Frequency (cycles/px)')
        freq.setLabel('left', 'Normalized gain')
        preview = tensor_preview(x, y)
        self.tensor_image = pg.ImageItem(preview['values'], axisOrder='row-major')
        self.tensor_image.setLookupTable(pg.colormap.get('CET-D1').getLookupTable())
        self.tensor_image.setLevels((-1, 1))
        sx, sy = preview['pitch']
        ox, oy = preview['origin']
        self.tensor_image.setRect(QtCore.QRectF(ox-.5*sx, oy-.5*sy, preview['values'].shape[1]*sx, preview['values'].shape[0]*sy))
        tensor.addItem(self.tensor_image)
        tensor.setAspectLocked(True)
        tensor.setLabel('bottom', 'x offset (px)')
        tensor.setLabel('left', 'y offset (px)')
        wave = lambda a: f"λ*≈{a['wavelength']:.3g}px" if np.isfinite(a['wavelength']) else 'low-pass, no bandpass λ*'
        physical = ""
        if self.field is not None and hasattr(self.field, "values"):
            from dynamix.core.bus import native_grid
            from dynamix.core.units import linear_unit_to_m

            _ny, _nx, _x0, dx, _y0, dy, _crs = native_grid(self.field)
            unit = str(getattr(getattr(self.field, "frame", None), "units", "px"))
            provenance = getattr(self.field, "provenance", {}) or {}
            factor = 1.
            if provenance.get("crs") and unit != "deg":
                factor, unit = linear_unit_to_m(provenance)
                factor = 1. if factor is None else factor
            physical = f" Native-axis cell spans {x['cell_px']*abs(dx)*factor:g} × {y['cell_px']*abs(dy)*factor:g} {unit}."
        self.basis_note.setText(f"Preview {bank}: {x['branch']}ₓ ⊗ {y['branch']}ᵧ · cells {x['cell_px']} × {y['cell_px']} px; "
                                f"analysis support {x['support_px']} × {y['support_px']} taps · x {wave(x)}; y {wave(y)}. "
                                'The preview does not change the computed plots. Use bank in tool, then Run to recompute; level pairs only select the displayed basis atom.' + physical)

    def export_data(self, path):
        d = self.data
        if d is None:
            return
        rows, members = [], []
        for ai, (alpha, fit) in enumerate(zip(d['alpha'], d['diagnostic_fits'])):
            for family, fam in enumerate(self._families(fit)):
                for i, xx in enumerate(fam['x']):
                    for pi, p in enumerate(d['p']):
                        rows.append([alpha, p, family, xx, fam['env'][pi][i], fam['slope'][pi], fam['intercept'][pi],
                                     np.isfinite(self._eta[:, ai]).any()])
                    members.extend([alpha, xx, family, jx, jy] for jx, jy in fam['members'][i])
        view = self._view
        context = {'source': self.source, 'analysis_key': (self.result.get('_landed') or {}).get('analysis_key'),
                   'shape': self.result.get('_shape'), 'params': self.result.get('params'), 'record': self.result.get('record'),
                   'normalization': d['moment_normalization'], 'envelope_columns': ['alpha1', 'p', 'family', 'x', 'log2_S', 'slope', 'intercept', 'admissible'],
                   'membership_columns': ['alpha1', 'x', 'family', 'jx', 'jy'],
                   'membership_index': ('resolution r = J − level, φ = −1' if self._level_line() else 'DWT level, φ = 0'),
                   'estimator': d['estimator'], 'x': d['diagnostic_fits'][0]['xlabel'] if d['diagnostic_fits'] else None,
                   'resolution_origin_J': d.get('resolution_origin'), 'families': self._family_names(),
                   'fit_ranges': {f'{d["alpha"][ai]:g}': list(r) for ai, r in sorted(self._ranges.items())},
                   'coarse_partner': d.get('coarse_partner'), 'no_structure': d.get('no_structure'),
                   'L_e1, L_e2': 'Theorem 6 upper bound on the upper spectrum D(H, e) = dim{h(·, e) <= H} over p >= p_alpha, '
                                 'capped at d = 2, empty below 0; not a measured D(h)',
                   'gamma': float(self.params.get('gamma', 0.0) or 0.0)}
        curves = {key: view[key] for key in ('eta_e1', 'eta_e2', 'eta_iso', 'alpha_star_e1', 'alpha_star_e2')}
        curves.update({f'{key}_{axis}': view[key][axis] for key in ('L_e1', 'L_e2') for axis in ('h', 'L')})
        np.savez_compressed(path, **curves, moment_pairs=d['moment_pairs'], log2_moments=d['log2_moments'], valid_cells=d['moment_counts'],
                            p=d['p'], alpha=d['alpha'], eta_alpha=self._eta,
                            envelopes=np.asarray(rows).reshape(-1, 8), membership=np.asarray(members).reshape(-1, 5),
                            context=json.dumps(context, ensure_ascii=False))

    def _save_data(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save directional diagnostic data', start_path('directional-moments.npz'), 'NumPy archive (*.npz)')
        if path:
            try:
                self.export_data(path)
            except (OSError, ValueError) as exc:
                self.readout.setText(str(exc))
            else:
                self.readout.setText('Moment tables, fit samples, Γ membership and source settings saved.')
