# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import pytest
from PySide6 import QtCore

from dynamix.model.param import Param, ParamKind
from dynamix.shell.knobs import control_spec
from dynamix.shell.knob_widgets import make_control


@pytest.fixture
def fparam():
    return Param("cutoff", ParamKind.FLOAT, default=0.5, min=0.0, max=10.0,
                 soft_min=0.0, soft_max=2.0, units="px")


def test_drag_value_shows_reading_and_emits_on_nudge(qtbot, fparam):
    w = make_control(control_spec(fparam), 0.5)
    qtbot.addWidget(w)
    assert w.text().endswith("px")
    with qtbot.waitSignal(w.valueChanged) as sig:
        qtbot.keyClick(w, QtCore.Qt.Key_Up)
    assert sig.args[0] > 0.5


def test_set_range_rebind_reaches_a_real_keyboard_gesture(qtbot, fparam):
    """CRITICAL fix. Before it, keyPressEvent passed ``self._spec.param`` into
    ``nudge``, which reconstructed a FRESH ControlSpec from the param's static declared soft
    range every call -- so a real Key_Up on a control whose bounds had just been rebound via
    ``set_range`` still applied the OLD, static step. fparam's static soft range is [0.0, 2.0]
    (step 0.02); rebinding to [0.0, 100.0] must make a REAL key press apply the rebound step
    (1.0), not the static one."""
    w = make_control(control_spec(fparam), 0.5)
    qtbot.addWidget(w)
    static_step = w._spec.step
    assert static_step == pytest.approx(0.02)

    w.set_range(0.0, 100.0)
    assert w._spec.step == pytest.approx(1.0)

    with qtbot.waitSignal(w.valueChanged) as sig:
        qtbot.keyClick(w, QtCore.Qt.Key_Up)
    delta = sig.args[0] - 0.5
    assert delta == pytest.approx(1.0)
    assert delta != pytest.approx(static_step)


def test_modifier_scaling(qtbot, fparam):
    w = make_control(control_spec(fparam), 0.5)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.valueChanged) as coarse:
        qtbot.keyClick(w, QtCore.Qt.Key_Up, QtCore.Qt.ShiftModifier)
    with qtbot.waitSignal(w.valueChanged) as fine:
        qtbot.keyClick(w, QtCore.Qt.Key_Up, QtCore.Qt.AltModifier)
    assert coarse.args[0] - 0.5 > fine.args[0] - 0.5


def test_inline_edit_accepts_hard_range_beyond_soft(qtbot, fparam):
    w = make_control(control_spec(fparam), 0.5)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.valueChanged) as sig:
        w.begin_edit(); w.commit_edit("9.5")        # beyond soft_max 2.0, inside max 10.0
    assert sig.args[0] == pytest.approx(9.5)


def test_inline_edit_seeds_the_exact_value_not_the_rounded_reading(qtbot, fparam):
    """A stored 0.001 displayed "0.00"; the editor used to seed from that rounded text, so
    double-click + Enter silently committed a real 0.0. The editor must carry the exact
    stored value (repr round-trips float64), and an untouched Enter must propose nothing."""
    w = make_control(control_spec(fparam), 0.001)
    qtbot.addWidget(w)
    w.begin_edit()
    assert w._editor.text() == repr(0.001)
    seen = []
    w.valueChanged.connect(seen.append)
    w.commit_edit(w._editor.text())      # Enter without touching the text
    assert seen == []                    # same value -> no proposal, nothing corrupted


def test_inline_edit_preserves_machine_precision(qtbot, fparam):
    w = make_control(control_spec(fparam), 0.5)
    qtbot.addWidget(w)
    typed = 0.1234567891012345
    with qtbot.waitSignal(w.valueChanged) as sig:
        w.begin_edit(); w.commit_edit(repr(typed))
    assert sig.args[0] == typed          # exact, not 2-sig-fig quantized


def test_cycle_steps_choices(qtbot):
    p = Param("wavelet", ParamKind.CHOICE, default="mexican", choices=("mexican", "morlet"))
    w = make_control(control_spec(p), "mexican")
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.valueChanged) as sig:
        qtbot.mouseClick(w, QtCore.Qt.LeftButton)
    assert sig.args[0] == "morlet"


def test_toggle(qtbot):
    p = Param("smooth", ParamKind.BOOL, default=True)
    w = make_control(control_spec(p), True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.valueChanged) as sig:
        qtbot.mouseClick(w, QtCore.Qt.LeftButton)
    assert sig.args[0] is False


def test_text_maps_to_non_interactive_label(qtbot):
    """ParamKind.TEXT (group_paint's spec_json/signature) is a reading, not an editor: no text
    cursor, no click-to-edit, but still a real value on screen and in the tooltip."""
    p = Param("spec_json", ParamKind.TEXT, default="", label="Groups")
    w = make_control(control_spec(p), "group:fault_a, group:fault_b")
    qtbot.addWidget(w)
    assert w.textInteractionFlags() == QtCore.Qt.NoTextInteraction
    assert w.toolTip() == "group:fault_a, group:fault_b"
    assert hasattr(w, "valueChanged")   # signature parity with every other control


def test_text_label_updates_via_set_value(qtbot):
    p = Param("group", ParamKind.TEXT, default="")
    w = make_control(control_spec(p), "a")
    qtbot.addWidget(w)
    w.set_value("b")
    assert w.toolTip() == "b"


def test_editable_text_maps_to_a_real_line_edit_and_emits_on_editing_finished(qtbot):
    """``Param.editable=True`` (group_filter.group had no way for a user to set
    it) earns a real ``QLineEdit`` -- typed text commits on ``editingFinished`` (Enter/focus-out),
    the same "propose on confirm, never on every keystroke" idiom every other control uses."""
    p = Param("group", ParamKind.TEXT, default="", label="Group", editable=True)
    w = make_control(control_spec(p), "")
    qtbot.addWidget(w)
    assert w.text() == ""

    w.setText("fault_a")
    with qtbot.waitSignal(w.valueChanged, timeout=1000) as sig:
        w.editingFinished.emit()
    assert sig.args[0] == "fault_a"


def test_editable_text_set_value_does_not_reemit(qtbot):
    p = Param("group", ParamKind.TEXT, default="", editable=True)
    w = make_control(control_spec(p), "a")
    qtbot.addWidget(w)
    seen = []
    w.valueChanged.connect(seen.append)

    w.set_value("b")

    assert w.text() == "b"
    assert seen == []                    # a programmatic update never re-proposes its own value


def test_editable_text_unchanged_commit_does_not_emit(qtbot):
    p = Param("group", ParamKind.TEXT, default="", editable=True)
    w = make_control(control_spec(p), "a")
    qtbot.addWidget(w)
    seen = []
    w.valueChanged.connect(seen.append)

    w.editingFinished.emit()             # Enter without touching the text

    assert seen == []
