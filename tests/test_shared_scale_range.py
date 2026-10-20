# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The hidden shared scale range, a handshake between the fitting windows: no owner, any
window's range change publishes, every window that follows takes it, last change wins, and
each window detaches with its own switch."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.shell.shared_range import SharedScaleRange
from dynamix.shell.structure_functions_window import StructureFunctionsWindow
from test_fitting_tools_shell import _leaders_result, win  # noqa: F401
from test_structure_functions_window import _samples


def _structure(qtbot, hub, levels=(2, 5)):
    window = StructureFunctionsWindow(_samples(), levels=levels)
    qtbot.addWidget(window)
    window.attach_shared_range(hub)
    return window


def test_two_windows_shake_hands_with_no_owner_and_each_can_detach(qtbot):
    hub = SharedScaleRange()
    a, b = _structure(qtbot, hub), _structure(qtbot, hub)
    assert a.follow_check.isChecked() and b.follow_check.isChecked() and hub.value is None
    a.set_scale_window(2.0, 4.0)                       # a user's move on A reaches B
    assert tuple(b._region.getRegion()) == (2.0, 4.0) and hub.value == (2.0, 4.0)
    b.follow_check.setChecked(False)                   # B detaches: its own range stays
    a.set_scale_window(1.0, 3.0)
    assert tuple(b._region.getRegion()) == (2.0, 4.0) and hub.value == (1.0, 3.0)
    b.set_scale_window(2.5, 4.5)                       # a detached window publishes nothing
    assert hub.value == (1.0, 3.0) and tuple(a._region.getRegion()) == (1.0, 3.0)
    b.follow_check.setChecked(True)                    # re-attaching takes the shared range now
    assert tuple(b._region.getRegion()) == (1.0, 3.0)
    b.set_scale_window(3.0, 5.0)                       # no owner: B's move reaches A
    assert tuple(a._region.getRegion()) == (3.0, 5.0) and hub.value == (3.0, 5.0)


def test_a_published_range_is_applied_once_and_never_echoed(qtbot):
    hub = SharedScaleRange()
    a, b, c = (_structure(qtbot, hub) for _ in range(3))
    emitted = []
    hub.changed.connect(lambda lo, hi, source: emitted.append((lo, hi, source)))
    a.set_scale_window(2.0, 4.0)
    assert emitted == [(2.0, 4.0, a)]
    assert tuple(b._region.getRegion()) == tuple(c._region.getRegion()) == (2.0, 4.0)
    a.set_scale_window(2.0, 4.0)                       # the same range again: nothing to say
    assert len(emitted) == 1


def test_every_fitting_window_follows_and_publishes(qtbot):
    from dynamix.shell.generalized_spectrum_window import GeneralizedSpectrumWindow
    from dynamix.shell.large_deviations_window import LargeDeviationsWindow
    from dynamix.shell.log_cumulants_window import LogCumulantsWindow
    from dynamix.shell.multifractal_window import MultifractalWindow
    from test_log_cumulants_window import _samples as _cumulant_sets
    from test_multifractal_window import _pair
    from test_spectrum_window import _per_scale_set

    hub = SharedScaleRange()
    sf = _structure(qtbot, hub)
    sets, _h, _D = _per_scale_set()
    ld = LargeDeviationsWindow(sets, log2_L=7.0, levels=(1, 5))
    gen = GeneralizedSpectrumWindow(sets, log2_L=7.0, levels=(1, 5))
    lc = LogCumulantsWindow(_cumulant_sets(), levels=(1, 5))
    mf = MultifractalWindow(*_pair())
    for w in (ld, gen, lc, mf):
        qtbot.addWidget(w)
        w.attach_shared_range(hub)
    sf.set_scale_window(2.0, 4.0)
    assert (ld.lo_spin.value(), ld.hi_spin.value()) == (2.0, 4.0)
    assert (gen.spins["log₂a min"].value(), gen.spins["log₂a max"].value()) == (2.0, 4.0)
    assert lc.ranges() == [(2.0, 4.0)] * len(lc.ranges())
    assert tuple(mf._region.getRegion()) == (2.0, 4.0)
    ld.set_scale_window(3.0, 5.0)                      # any of them publishes
    assert tuple(sf._region.getRegion()) == (3.0, 5.0) and tuple(mf._region.getRegion()) == (3.0, 5.0)
    assert lc.ranges() == [(3.0, 5.0)] * len(lc.ranges())
    mf.set_scale_window(1.0, 4.0)
    assert tuple(sf._region.getRegion()) == (1.0, 4.0) and (ld.lo_spin.value(), ld.hi_spin.value()) == (1.0, 4.0)
    gen.set_scale_window(2.0, 5.0)
    assert tuple(sf._region.getRegion()) == (2.0, 5.0)
    lc.follow_check.setChecked(False)                  # the cumulants window detaches as one
    sf.set_scale_window(1.0, 3.0)
    assert lc.ranges() == [(2.0, 5.0)] * len(lc.ranges())


def test_the_main_window_attaches_every_fitting_window_to_one_hub(win, qtbot):
    win._active_result = _leaders_result()
    win._refresh_samples_buttons()
    win._on_structure_button_clicked()
    win._on_cumulants_button_clicked()
    win._on_ld_button_clicked()
    sf, lc, ldw = win._structure_window, win._cumulants_window, win._ld_window
    for w in (sf, lc, ldw):
        qtbot.addWidget(w)
    assert win._shared_range.value is None               # nothing published until a user moves one
    sf.set_scale_window(3.0, 5.0)
    assert win._shared_range.value == (3.0, 5.0)
    assert lc.ranges() == [(3.0, 5.0)] * len(lc.ranges())
    assert (ldw.lo_spin.value(), ldw.hi_spin.value()) == (3.0, 5.0)
    win._on_structure_button_clicked()                 # a window opened later takes the shared range
    fresh = win._structure_window
    qtbot.addWidget(fresh)
    assert fresh is not sf and tuple(fresh._region.getRegion()) == (3.0, 5.0)
