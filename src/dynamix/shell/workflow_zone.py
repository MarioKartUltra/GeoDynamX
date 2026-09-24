# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""DeviceBox / SourceBox / WorkflowZone: The replacement for the chain-strip row.

Where ``chain_strip.py`` puts one flat ``DeviceStrip`` row per step, ``WorkflowZone`` puts one
taller ``DeviceBox`` per step, wide enough to hold a full control GRID (not one row) so a device
with many params -- ``wtmm2d`` now has ten -- can group them (``device.param_groups``) instead of
scrolling one param at a time. ``chain_strip.py`` and ``ChainStripZone``/``DeviceStrip`` are left
completely untouched (``tests/test_shell_chain_strip.py`` still exercises them directly); this
module is an ADDITION, not a replacement of that file.

**Mirrored, not imported, from ``chain_strip.py``:** ``DeviceBox._derived_text`` and its
``_on_control_changed``/``set_state``/``set_reading``/``set_selected`` bodies duplicate
``DeviceStrip``'s exactly. ``chain_strip.py`` must stay untouched (a project rule for this task --
its own suite pins the OLD classes), so there is no shared base class or free function to import
from without editing that file. The logic is a handful of lines with no state to drift out of sync
in either copy, the same trade-off ``wtmm_roi.py`` already made for its own ``derived_reading``
duplicate of ``wtmm.py``'s.

**The bypass/remove buttons are inert affordances for this task, on purpose.** Both are ``setEnabled(False)`` at construction: nothing downstream consumes
``bypassToggled``/``removeRequested`` yet -- ``WorkflowZone`` records the proposal in its own
descriptor list and re-emits ``chainEdited``, but ``main_window`` never connects to it, so a LIVE
remove button reindexes ``DeviceBox.step_index`` out from under ``main_window``'s own
``_names``/``_params`` bookkeeping (confirmed repro: removing step 1 of a 3-step chain shifts step
2 into index 1, so the next param edit on it silently writes into the WRONG step, inside a Qt slot
where the resulting ``ValueError`` is printed and swallowed), and a live bypass button latches
"checked" while the step keeps computing regardless -- a control asserting a state the engine
isn't in. The signals, and ``WorkflowZone``'s reaction to them, are left fully wired: the drag/reorder/bypass wiring only has to flip ``setEnabled(True)`` back on once it has
somewhere for the result to land.

**Bypass RENDERING vs. the bypass GESTURE are still deliberately different code paths** (independent
of the buttons being disabled above), mirroring the DragValue propose/confirm split
``chain_strip.py``'s own docstring documents: a ``DeviceBox`` constructed with ``bypassed=True``
(from a descriptor) disables only its CONTROL GRID (``self._body``), never ``self`` -- the title bar
(name, state dot, bypass/remove buttons) is a SEPARATE widget from ``self._body`` precisely so a
disabled box's own bypass button is never trapped inside the same disabled subtree it would need to
be clicked to escape (an earlier version of this class called
``self.setEnabled(False)``, which cascades to every descendant including the button that is
supposed to undo it -- structurally correct now even though the button is ALSO separately disabled for the reason above; re-enabling it will not inherit the trap). Clicking
the button on a box that is not yet bypassed only EMITS ``bypassToggled`` -- it never disables
``self._body`` reactively -- so applying that proposal (and thereby rendering the grid disabled) is
the owner's job, exactly as ``set_value`` is for a control.

**Drag-drop assembly, racks, illegal-drop refusal.** ``WorkflowZone`` now accepts real
drops of the two Task-5 mime types from ``dynamix.shell.browser`` (``DEVICE_MIME`` -- payload a
device name, inserted at ``defaults_for(get_device(name))``; ``PRESET_MIME`` -- payload a preset
name resolved through :meth:`WorkflowZone.set_presets`, injected by ``main_window`` with the same
dict the browser gets), plus a third, PRIVATE mime type (``_REORDER_MIME``) a box or rack encodes
itself as when its own title bar is dragged -- read back only by this zone, never by the browser.
pytest-qt cannot synthesize a real native drag reliably offscreen, so every drop test in
``tests/test_workflow_zone.py`` builds a ``QMimeData`` and calls ``dragEnterEvent``/``dropEvent``
directly -- the same "call the handler, not the input queue" convention this file's own widget
tests already use for every other gesture.

``self._descriptors`` IS the flattened chain order -- there is no separate "visual layout" kept in
sync with it. Rendering (:meth:`WorkflowZone.set_steps`) only has to GROUP consecutive entries
that share the same non-``None`` ``"rack"`` value into one ``RackBox``; a rack therefore always
occupies one contiguous span of the list, and a ``RackBox``'s member order is exactly that span's
own order. That is the whole of "rack members flatten IN PLACE at the rack's position, in
rack-internal order": the list's order already IS the flattened order, so flattening costs nothing
beyond reading it in sequence.

Before ANY zone mutation is committed -- a device/preset drop, a reorder, a bypass toggle, a
remove, all of them, uniformly -- :meth:`WorkflowZone._commit_or_revert` tries
``Chain(tuple(DeviceRef(d["device"], d["params"]) for d in self._descriptors)).validate()`` over
the FULL descriptor list, bypassed steps INCLUDED. That is deliberate, not an oversight: a bypassed step is present-but-INERT for what it computes, never for where
it sits in the order -- a bypassed FILTER still blocks a transform from landing after it. Checking
only the enabled subset let a transform land legally past a bypassed filter (nothing there to
object), and then RAISE the moment that filter was un-bypassed again, deep inside
``main_window._chain()``, with the window's ``_names``/``_params`` already reassigned out from
under it. Validating the full order means an arrangement is never allowed to become illegal in
the first place, active or not, so un-bypassing can never surface one later. ``Chain.validate``
stays the one authority on transforms-then-filters -- this is a REFUSAL at the zone, not a second
implementation of the rule.

Past that gate, the zone still has to REBUILD the boxes (``set_steps``) before it can honestly
call the mutation committed -- and that rebuild can itself fail for reasons ``Chain.validate``
never sees (a device's own control construction raising -- the live trigger was ``chain_classify``'s undeclared FLOAT param bounds crash ``knobs.control_spec``). Any
such exception is caught the same way a ``ValueError`` is: full revert to the pre-mutation
descriptor list, rebuilt, warned. A commit therefore either fully lands or fully doesn't -- the
zone is never left part-mutated (descriptors grown, boxes not) for the NEXT commit to inherit.
Either failure mode reverts ``self._descriptors`` to what it was before the mutation, rebuilds the
boxes from that reverted list, and shows the exception text in the zone's own ``reading_label``
for four seconds -- the same "state a fact, then let it fade" reading convention every strip's
honesty label already uses, just timed rather than replaced on the next resolve. Nothing is
emitted on refusal.

Bypass and remove are REAL now: both buttons are enabled at construction (see the ``DeviceBox``
docstring above, kept verbatim, for the confirmed desync repro that made disabling them necessary
before this task existed to consume their signals). Both now route through the SAME
``_commit_or_revert`` gate every drop does -- a remove or a bypass
toggle can never actually change whether the FULL order validates (removing a step, or excluding
one from computing, cannot turn a legal order illegal), so the gate is a no-op guard for them in
practice, but it is the same one gate rather than a second, ungated path that could drift. Every
param edit ALSO writes back into ``self._descriptors[i]["params"]`` now, not just into the
``paramChanged`` payload -- without that, a knob turned between two drag-drop gestures would be
silently discarded by the SECOND gesture's ``chainEdited`` payload, which is built from
``self._descriptors`` and had never been told about the first gesture's edit. Keeping the
descriptor list itself live is what lets ``main_window`` rebuild its own bookkeeping wholesale
from every ``chainEdited`` payload rather than patching it incrementally -- the only way the zone
and the window can never desync (the Critical was exactly that desync, from the opposite
direction: an unwired button instead of a stale descriptor).

A PRESET drop creates its OWN ``RackBox``, named after the preset, holding every one of its steps
("Racks" is a real category in the browser, and a preset dragged out
of it produced ungrouped boxes -- unreachable, since nothing else in this slice ever creates a
rack). The drop TARGET (a specific box, a specific rack) is not consulted for a preset: nesting
one preset's rack inside another named rack is not a supported gesture, so a preset always lands
as a new rack at the very end, or replaces an empty zone outright. A device drop or a reorder,
by contrast, DOES resolve a precise target now: landing on a
specific ``DeviceBox`` inserts directly before it; landing on a ``RackBox``'s own chrome (its
title bar, not a member) appends to that rack's end; landing on empty zone background appends at
the very end, top level. Routing is decided by WHICH WIDGET'S ``dropEvent`` Qt calls -- see the
"drag and drop" section below -- not by reading a pixel position back out of one, so this is exact
without any geometry involved.

Landing on a specific ``DeviceBox`` inherits that box's own rack membership -- nesting the arrival
into the SAME rack, at that exact position -- only when doing so is SAFE
(:meth:`WorkflowZone._safe_insertion_index`): the target box is
not racked at all, IS its rack's own first member (so its position already equals that rack's
start boundary), or the arriving item is explicitly destined for that SAME rack (a rack-background
drop, or a synthetic caller that passes matching ``into_rack``). Landing on any OTHER (interior)
member of a DIFFERENT rack is UNSAFE and snaps to that rack's own START boundary instead --
"dropping onto a rack's middle" becomes "insert before that rack" rather than inside it -- which is
what keeps every rack a single contiguous descriptor span no matter where within a DIFFERENT one
something is dropped. An earlier version of this fix only checked "dropped on itself"; it missed
that a moved RACK (which never inherits -- it always keeps its own identity) dropped on an
INTERIOR member of a different rack spliced itself into the middle of that rack's own span, which
``set_steps``' consecutive-grouping then silently rendered as TWO same-titled ``RackBox``es -- no
crash, no warning, just a violated invariant on screen.

Click-to-select is restored on ``DeviceBox`` (lost in the swap from ``DeviceStrip``, whose
``mousePressEvent`` covered the whole strip) and now shares the title bar with the NEW drag
gesture: a dedicated ``_TitleBar`` widget disambiguates the two by the same press-then-move
threshold every native title-bar drag uses (``QApplication.startDragDistance()``) -- released
before crossing it emits ``selected``; moved past it starts the box's (or rack's) own ``QDrag``
instead. A click anywhere else on the box -- its own background, not a generated control -- still
selects too, via ``DeviceBox.mousePressEvent`` directly, matching ``DeviceStrip``'s original
whole-strip behaviour for everywhere the title bar does not itself cover.

**Smart placement +
persistent refusal.** Bug triage traced the report
to the exact mechanism the paragraphs above describe as deliberate: every freshly opened raster's
default chain already ends in filters, so most drags of an unused TRANSFORM hit
``Chain.validate``'s transforms-then-filters refusal on the very first try -- and the only signal
was a small muted label that faded in four seconds. The fix is not to relax the rule (still the
one legality authority, still run in full by ``_commit_or_revert``) but to stop asking it to refuse
what it can instead be asked to place correctly: :meth:`WorkflowZone._legal_insert_index` computes,
BEFORE the candidate descriptor is built, the nearest position where the dropped device is already
legal -- a transform pushed past the end of the transform block clamps back to it, a filter pushed
before the start of the filter block clamps up to it -- so :meth:`_handle_device_drop` and a
single-box :meth:`_reorder` land the device where the user's gesture asked, in the only section it
could ever legally go, instead of silently doing nothing. What ``_commit_or_revert`` still refuses
after this: an unresolvable device, an illegal param, a device-construction failure, or a RACK
reorder (racks can legitimately mix transforms and filters, so there is no single "kind" to clamp a
whole rack by -- see the presets note above). Those refusals are now LOUDER, not
just visible: a brief red flash on the zone itself (:meth:`_flash_refusal`, styled via
``theme.py``'s ``[refused="true"]`` rule) plus a message that no longer fades on a fixed timer --
it PERSISTS until the next successful zone mutation (the design's three-tier fault-volume doctrine:
"refusal messages persist until the next user interaction rather than a fixed 4 s"). A successful
placement, clamped or not, still uses the old four-second fade (it is a status, not a fault) and
additionally scrolls the new or moved box into view (``ensureWidgetVisible`` on ``self._scroll`` --
the triage report's secondary friction point) and emits :attr:`WorkflowZone.dropPlaced` for a
future status-bar consumer.

**Data-aware readings and
hints.** :meth:`DeviceBox.sync_from_result` is the new seam: a device may optionally declare
``reading(result, params) -> str`` and/or ``data_hints(result, params) -> {param: (lo, hi, snap)}``
(``dynamix.devices.chain_filters``'s three chain filters do), and this box applies either or both
whenever a fresh result lands, in place of whatever generic reading the caller would otherwise
compute. See that method's own docstring for the full contract, including why a hint can only ever
narrow a control's SOFT bounds (never ``cache_key``-visible) and why the out-of-range snap goes
through the ordinary ``_on_control_changed`` edit path rather than a private one.

**Rack removal + undo stack.** ``RackBox`` gains a
second title-bar button, mirroring ``collapse_button``'s own construction -- ``×``, emitting
:attr:`RackBox.rackRemoveRequested` rather than mutating anything itself (the same propose/apply
split every other button here already keeps: a click only ever emits, the owning
:class:`WorkflowZone` decides what happens). :meth:`WorkflowZone._on_rack_remove_requested` removes
the rack's WHOLE span (:meth:`_rack_span`) through the same :meth:`_commit_or_revert` gate every
other mutation uses -- a pure descriptor-list deletion can never turn a validated transforms-then-
filters order illegal (removing entries only ever narrows a sequence that already had the property,
it cannot introduce a violation), so this always lands; the length check after the call is a cheap
defensive belt, not a real failure path, matching this module's existing "any exception still
reverts" contract rather than adding a new one.

On landing, the FULL pre-removal descriptor list -- not just the removed span -- is pushed onto
:attr:`WorkflowZone._removed_racks_stack` (LIFO, unbounded, mirrors ``TransectPanel._deleted_stack``
-- the precedent). :meth:`WorkflowZone.undo_removal` pops the most recent
snapshot and restores it WHOLESALE (again through ``_commit_or_revert``) rather than re-inserting
just the removed span at wherever the zone currently stands: a full-list snapshot has no "where do I
put it back relative to whatever else changed since" question to answer at all, and two sequential
removals produce two independent snapshots whose LIFO pops each restore exactly the state that
removal undid, regardless of what happened in between (each snapshot already carries forward every
earlier removal still in effect at the moment it was taken). ``MainWindow`` binds the actual
shortcut: **not** ``QKeySequence.Undo`` (⌘Z) -- that chord is already claimed for transect-delete undo at the same ``WindowShortcut`` context, and a second ``QShortcut`` on the identical key sequence
in that context does not stack, it makes BOTH ambiguous (Qt fires neither ``activated()`` handler).
Rack-removal undo binds Shift+⌘Z instead (documented at the ``QShortcut`` construction site in
``main_window.py``) -- a distinct, still-discoverable chord rather than inventing a focus-routing
rule for one shared key. The removal's own status message names it: "Removed rack <title> (N
devices) — ⇧⌘Z restores" ("reassurance is part of the message"), shown on this zone's own
``reading_label`` (:meth:`_show_warning`, normal non-persistent fade -- a removal is a success, not
a refusal) AND relayed outward via :attr:`WorkflowZone.rackRemoved` (``dropPlaced``-shaped:
``(title, message)``) for ``main_window._notify(msg, "status")`` to also post to the status bar --
the same dual zone-local-plus-window-level reporting :attr:`dropPlaced` already does for a clamped
placement. A separate signal from ``dropPlaced`` on purpose:
``tests/test_workflow_zone.py::test_droppplaced_signal_fires_only_on_a_clamped_placement`` already
pins ``dropPlaced`` to firing ONLY on a clamped drop/reorder placement, and reusing it here would
quietly broaden that contract.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import Device, defaults_for, get_device, is_transform
from dynamix.model.param import Param
from dynamix.shell.browser import DEVICE_MIME, PRESET_MIME
from dynamix.shell.knob_widgets import make_control
from dynamix.shell.knobs import control_spec

#: Column width for one param-pair slot -- ``DeviceBox`` is exactly ``n_columns`` of these wide
#: ("Fixed box width per param-column count").
_COLUMN_WIDTH = 108

#: Outer content margins, matching chain_strip.py's ``_MARGINS`` convention (generous enough that
#: a click near the corner lands on the box's own background, not a generated child control).
_MARGINS = (8, 4, 8, 4)

#: Mirrored from chain_strip.py's own (private) constant of the same name -- see the module
#: docstring's "mirrored, not imported" note.
_STATE_DOT_GLYPH = "●"

#: A ``DeviceBox`` has no fixed height (unlike ``chain_strip.STRIP_HEIGHT``'s single-row strip --
#: this is a variable-row grid), but ``main_window`` still needs ONE number to size the bottom
#: panel by. This is ``wtmm2d``/``wtmm2d_roi``'s own observed ``sizeHint().height()`` (161 px,
#: their PRE group's tallest row: ``a_min``'s derived sigma reading adds a third line to that
#: cell), rounded up for a margin of safety -- the same role ``STRIP_HEIGHT`` plays for
#: ``main_window``'s bottom-row height calculation, just measured rather than declared, since a
#: grid's height depends on content in a way a fixed-height single row's does not.
BOX_HEIGHT = 168

#: PRIVATE mime type a box or rack encodes itself as when its own title bar is dragged -- payload
#: is ``"box:<descriptor index>"`` or ``"rack:<title>"``. Never emitted by the browser and never
#: read by anything outside this module (contrast ``DEVICE_MIME``/``PRESET_MIME``, imported from
#: ``dynamix.shell.browser``, which the browser itself produces).
_REORDER_MIME = "application/x-dynamix-reorder"

#: How long a SUCCESS message (including a clamped-placement one) stays in the zone's
#: ``reading_label`` before it self-clears -- the "state a fact, then let it fade" convention the module docstring names. A REFUSAL no longer uses this timer at all: it persists until the next successful zone mutation -- see ``_show_warning``.
_WARNING_MS = 4000

#: How long the zone's red refusal flash stays on before reverting (the design's drop-refusal
#: feedback: "brief red flash at the drop site").
_FLASH_MS = 250


class _TitleBar(QtWidgets.QWidget):
    """A box's title-bar strip, wrapped in its own widget so a press there can be told apart from
    one meant for a generated control -- Qt delivers a mouse event to whichever child widget it
    geometrically lands on first, so nothing besides this widget's own children (name label, state
    dot, bypass/remove buttons) can ever see one of these presses.

    Click and drag share the same press: released before crossing
    ``QApplication.startDragDistance()`` is a click (``on_click``, if given); moved past it starts
    a :class:`QtGui.QDrag` instead (``on_drag``) -- the same disambiguation every native
    title-bar drag uses. Neither handler calls ``super()``: leaving the event in its default
    "accepted" state is what keeps it from also propagating up to the owning box's OWN
    ``mousePressEvent`` (the whole-box click-to-select fallback for everywhere the title bar does
    not itself cover).
    """

    def __init__(self, on_drag, on_click=None, parent=None):
        super().__init__(parent)
        self._on_drag = on_drag
        self._on_click = on_click
        self._press_pos: QtCore.QPoint | None = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._press_pos = event.pos()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (self._press_pos is not None and bool(event.buttons() & QtCore.Qt.LeftButton) and
                (event.pos() - self._press_pos).manhattanLength() >=
                QtWidgets.QApplication.startDragDistance()):
            self._press_pos = None
            self._on_drag()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (event.button() == QtCore.Qt.LeftButton and self._press_pos is not None and
                self._on_click is not None):
            self._on_click()
        self._press_pos = None


def _owning_zone(widget: QtWidgets.QWidget) -> "WorkflowZone | None":
    """Walk up from ``widget`` (a ``DeviceBox`` or a ``RackBox``) to the ``WorkflowZone`` that
    ultimately owns it -- shared by both classes' ``dragEnterEvent``/``dropEvent`` overrides,
    which route every drop they themselves receive to the zone's own :meth:`WorkflowZone.
    _handle_drop` (the zone is the one place ``self._descriptors`` lives, so nothing besides it
    can actually commit a mutation)."""
    while widget is not None and not isinstance(widget, WorkflowZone):
        widget = widget.parent()
    return widget


def _grouped_params(device: Device) -> list[tuple[str, tuple[Param, ...]]]:
    """``(group_label, params)`` pairs, in the order ``DeviceBox`` should render them.

    A device with no ``param_groups`` gets exactly one unlabeled group holding every VISIBLE
    param, in the device's own declared order -- the plain, ungrouped layout every device had
    before this task, unaffected by opting in or out. A device that DOES declare ``param_groups``
    renders each entry as its own labeled group, in dict order, using the PARAM ORDER GIVEN IN THE
    GROUP TUPLE -- not ``device.params`` order -- because that order is itself part of the
    declaration (see ``wtmm.py``'s ``param_groups`` docstring: the PRE pairing is deliberate). Any
    param not named by ANY group lands in one trailing, unlabeled group, in ``device.params``
    order.

    **Hidden-by-convention params.** A param whose name starts with
    ``_`` (``dynamix.devices.backproject.Backproject``'s seven shell-stamped scalars are the first
    example) never earns a control here, in ANY group, including the trailing catch-all --
    ``dynamix.model.param.Param``/``dynamix.shell.knobs.control_spec`` build one for it exactly
    like any other declared param (a TEXT one renders read-only, but still renders), so the
    filtering has to happen at the render boundary, here, rather than by relying on ``ParamKind``
    or ``editable`` to hide it. This is the SAME mechanism ``param_groups`` filtering already
    uses (a param simply absent from every group's tuple lands nowhere) -- underscore-prefixed
    names are just excluded from EVERY group's tuple, including the synthetic single-group case
    below, rather than added to one.
    """
    visible = tuple(p for p in device.params if not p.name.startswith("_"))
    groups_spec = getattr(device, "param_groups", None)
    if not groups_spec:
        return [("", visible)]
    schema = {p.name: p for p in device.params}
    listed: set[str] = set()
    result: list[tuple[str, tuple[Param, ...]]] = []
    for label, names in groups_spec.items():
        group_params = tuple(schema[n] for n in names if not n.startswith("_"))
        if group_params:
            result.append((label, group_params))
        listed.update(names)
    trailing = tuple(p for p in visible if p.name not in listed)
    if trailing:
        result.append(("", trailing))
    return result


class DeviceBox(QtWidgets.QFrame):
    """One chain step, as a control GRID rather than a single row.

    ``controls`` is keyed by param name, same as ``DeviceStrip`` -- ``box.controls["thresh"]`` --
    so ``main_window._sync_controls`` (and a test) can reach a specific generated widget without
    knowing the device's param order.

    Accepts drops itself (``setAcceptDrops(True)``) -- a device,
    preset, or reorder drop that lands anywhere on this box (its title bar, a mini-label, an
    otherwise-idle generated control -- Qt delivers the event to whichever child widget it lands
    on, and none of those accept drops themselves) inserts the arriving item directly BEFORE this
    box, routed to the owning ``WorkflowZone``'s ``_handle_drop(event, before=self)``. That is what
    makes a reorder land exactly where it was dropped rather than always at the end of whatever
    container caught it.
    """

    paramChanged = QtCore.Signal(str, object)
    removeRequested = QtCore.Signal()
    bypassToggled = QtCore.Signal(bool)
    selected = QtCore.Signal()

    def __init__(self, step_index: int, device: Device, params: dict, field=None,
                bypassed: bool = False, parent=None):
        super().__init__(parent)
        self._step_index = step_index
        self.device = device
        self._field = field
        self._params = params
        self.controls: dict[str, QtWidgets.QWidget] = {}
        self._derived_labels: dict[str, QtWidgets.QLabel] = {}

        self.setProperty("strip", "true")     # reuse the strip QSS chrome -- no new properties
        self.setProperty("selected", "false")
        self.setAcceptDrops(True)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(*_MARGINS)

        # Wrapped in its own widget (rather than a bare layout on ``outer``) so a press here can
        # be told apart from one meant for a generated control, and so its own drag gesture never
        # also falls through to this box's whole-background click-to-select below -- see
        # ``_TitleBar``'s docstring, and the module docstring's drag-drop section.
        self._title_bar = _TitleBar(on_drag=self._start_drag, on_click=self.selected.emit)
        title_row = QtWidgets.QHBoxLayout(self._title_bar)
        title_row.setContentsMargins(0, 0, 0, 0)
        self.name_label = QtWidgets.QLabel(device.name)
        title_row.addWidget(self.name_label)
        self.state_dot: QtWidgets.QLabel | None = None
        if is_transform(device):
            self.state_dot = QtWidgets.QLabel(_STATE_DOT_GLYPH)
            self.state_dot.setProperty("state", "idle")
            title_row.addWidget(self.state_dot)
        title_row.addStretch(1)
        self.bypass_button = QtWidgets.QToolButton()
        self.bypass_button.setCheckable(True)
        self.bypass_button.setChecked(bypassed)
        self.bypass_button.setText("B")
        self.bypass_button.setToolTip("Bypass")
        self.bypass_button.toggled.connect(self.bypassToggled.emit)
        # Both buttons were disabled here at first: WorkflowZone's own
        # reaction to bypassToggled/removeRequested only mutated its own internal descriptor list,
        # and nothing in main_window read that back into _names/_params/layer.chain, so a live
        # remove button desynced step_index from main_window's own chain-position bookkeeping
        # (confirmed repro: removing step 1 of a 3-step chain reindexed every later DeviceBox, so
        # the NEXT paramChanged(1, ...) from the box now sitting at index 1 wrote into the WRONG
        # step's params). The drag-drop wiring is that consumer now -- WorkflowZone.chainEdited round-trips into
        # main_window._on_chain_edited, which rebuilds _names/_params/bypass/rack wholesale from
        # the payload every time, so there is no stale bookkeeping left for a reindex to desync.
        # Both buttons are therefore enabled again, unconditionally, at construction.
        title_row.addWidget(self.bypass_button)
        self.remove_button = QtWidgets.QToolButton()
        self.remove_button.setText("×")
        self.remove_button.setToolTip("Remove")
        self.remove_button.clicked.connect(self.removeRequested.emit)
        title_row.addWidget(self.remove_button)
        outer.addWidget(self._title_bar)

        # The control grid lives in its OWN widget, disabled independently of ``self`` -- so a box
        # constructed ``bypassed=True`` grays out its PARAMS without cascading into the title bar
        # (name/state-dot/bypass/remove). If ``self`` itself were disabled instead (as an earlier
        # version of this class did), a box already bypassed could never be un-bypassed by clicking
        # its own (now-disabled-by-ancestor) button once re-enabled. The bypass/remove buttons are ALSO explicitly disabled above, independently of
        # this -- two different reasons producing the same "off" state on two different widgets.
        self._body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)

        col_cursor = 0
        for label, group_params in _grouped_params(device):
            n_cols = max(1, -(-len(group_params) // 2))     # ceil(len / 2), at least one column
            title_label = QtWidgets.QLabel(label)
            if label:
                title_label.setProperty("muted", "true")
            grid.addWidget(title_label, 0, col_cursor, 1, n_cols)
            for i, p in enumerate(group_params):
                col = col_cursor + i // 2
                row = 1 + (i % 2)                            # two params per column
                cell = QtWidgets.QVBoxLayout()
                cell.setContentsMargins(0, 0, 0, 0)
                cell.setSpacing(0)
                mini = QtWidgets.QLabel(p.label or p.name)
                mini.setProperty("muted", "true")
                cell.addWidget(mini)
                control = make_control(control_spec(p), params[p.name])
                # Same-thread GUI signal, default-arg trick to bind the param name at connect
                # time -- the pattern chain_strip.py's DeviceStrip already uses for this exact
                # shape of fan-out connection.
                control.valueChanged.connect(
                    lambda value, name=p.name: self._on_control_changed(name, value))
                cell.addWidget(control)
                self.controls[p.name] = control
                derived = self._derived_text(p.name, params[p.name])
                if derived is not None:
                    derived_label = QtWidgets.QLabel(derived)
                    derived_label.setProperty("reading", "true")
                    derived_label.setProperty("muted", "true")
                    self._derived_labels[p.name] = derived_label
                    cell.addWidget(derived_label)
                grid.addLayout(cell, row, col)
            col_cursor += n_cols
        body_layout.addLayout(grid)
        outer.addWidget(self._body)

        self.reading_label = QtWidgets.QLabel("")
        self.reading_label.setProperty("reading", "true")
        self.reading_label.setProperty("muted", "true")
        outer.addWidget(self.reading_label)

        total_cols = max(col_cursor, 1)
        self.setFixedWidth(_MARGINS[0] + _MARGINS[2] + total_cols * _COLUMN_WIDTH)

        # Rendering-only: see the module docstring's "bypass rendering vs. the bypass GESTURE"
        # note. Applied once, from the ctor argument -- never reactively from the button itself.
        # Scoped to the CONTROL GRID ONLY (``self._body``), not ``self`` -- see the comment above
        # ``self._body``'s construction.
        self._body.setEnabled(not bypassed)
        self._apply_active_when()

    def _apply_active_when(self) -> None:
        """Grey out every knob whose ``Param.active_when`` does not hold for the current params
        (e.g. tucker's Tape axis outside the 1-D tape embedding). The value is kept and still
        keyed -- this only says the knob does not apply right now."""
        for p in self.device.params:
            cond = getattr(p, "active_when", None)
            control = self.controls.get(p.name)
            if cond is None or control is None:
                continue
            conds = (cond,) if isinstance(cond[0], str) else cond     # one pair, or several
            control.setEnabled(all(self._params.get(other) in allowed
                                   for other, allowed in conds))

    @property
    def step_index(self) -> int:
        return self._step_index

    def _derived_text(self, name: str, value) -> str | None:
        """Mirrors ``DeviceStrip._derived_text`` (chain_strip.py) exactly -- see the module
        docstring for why this is a copy rather than a shared import."""
        derived_reading = getattr(self.device, "derived_reading", None)
        if derived_reading is None:
            return None
        return derived_reading(name, value, self._field, self._params)

    def _on_control_changed(self, name: str, value) -> None:
        """Mirrors ``DeviceStrip._on_control_changed``: updates ``self._params`` FIRST (a device's
        ``derived_reading`` may read a SIBLING param out of it -- ``WTMM2D``'s aₘᵢₙ line reads
        ``wavelet`` for λ), calls ``set_value`` on the originating control with the applied value,
        refreshes EVERY derived-reading label (not just the one beside this control, or a wavelet
        flip would leave the aₘᵢₙ line's λ stale until the whole box is torn down and rebuilt),
        then tells the outside world."""
        self._params[name] = value
        self.controls[name].set_value(value)
        self._apply_active_when()
        for label_name, label in self._derived_labels.items():
            text = self._derived_text(label_name, self._params[label_name])
            if text is not None:
                label.setText(text)
        self.paramChanged.emit(name, value)

    def set_selected(self, value: bool) -> None:
        self.setProperty("selected", "true" if value else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_state(self, state: str) -> None:
        """``"idle"|"computing"|"cached"|"error"``. No-op for a filter box, which has no dot."""
        if self.state_dot is None:
            return
        self.state_dot.setProperty("state", state)
        self.style().unpolish(self.state_dot)
        self.style().polish(self.state_dot)

    def set_reading(self, text: str) -> None:
        self.reading_label.setText(text)
        self.reading_label.setProperty("muted", "true" if text == "dropped 0" else "false")
        self.style().unpolish(self.reading_label)
        self.style().polish(self.reading_label)

    def sync_from_result(self, result: dict) -> None:
        """Applies a device's optional ``reading(result,
        params)``/``data_hints(result, params) -> {param: (lo, hi, snap)}`` protocol methods to
        this box -- duck-typed via ``getattr(self.device, ..., None)``, the SAME lookup
        ``_derived_text`` already uses for ``derived_reading``, so a device declaring neither is
        completely untouched. ``main_window._update_readings`` calls this once per landed result,
        in place of its own ``set_reading(_filter_reading(...))`` call, for any filter whose
        device provides either method (see that method's own comment for the dispatch).

        ``reading`` replaces this box's reading-label text outright with the device's own honesty
        statement (e.g. "kept 73/73 · h ∈ [-1.28, 0.69]").

        ``data_hints`` maps a SUBSET of this box's own param names to ``(lo, hi, snap)``. A
        control whose widget class supports ``set_range`` (``DragValue``; see
        ``knob_widgets.py``) has its soft bounds -- and therefore its drag/nudge step size --
        rebound to ``[lo, hi]``: pure view state (``Param``'s own docstring: "never enforced ...
        must not affect cache_key"), so this can never change what a Transform's ``cache_key``
        sees. A NaN ``lo`` (``chain_filters._NO_HINT``) is the "metric unavailable" sentinel --
        chains are present but the metric this control's cutoff depends on could not be computed
        for them (e.g. no ``log2_mod``) -- and disables the control rather than touching its
        bounds; any other value re-enables it.

        The snap fires ONLY in the pass-all direction: ``current < lo``, never ``current > hi``
        (CRITICAL/IMPORTANT fixes). Every filter here is "keep if metric >=
        cutoff", so a cutoff BELOW the data's own floor is behaviorally meaningless (already
        pass-all) and safe to snap up to ``lo`` for free. A cutoff ABOVE the data's own ceiling is
        NOT meaningless -- it is a deliberate, already-restrictive choice (e.g. the user set a high
        cutoff on purpose, and this box happens not to be the terminal filter, so the CURRENT
        result's own range -- see the "known limitation" paragraph on
        ``ChainHolderFilter.data_hints`` -- can look narrower than what the user actually set for
        reasons that have nothing to do with THIS control being wrong). Snapping on ``current >
        hi`` would silently yank a user's own high cutoff down to ``hi`` on every landing, which is
        never what a hint is for. That snap goes through ``_on_control_changed``, the exact path a
        user's own drag uses, so it updates ``self._params``, the control's display, and emits
        ``paramChanged`` like any other edit: one honest recompute, not a silent back-channel
        mutation. A param named in ``data_hints`` that this box has no control for (should not
        happen -- ``data_hints`` is the device's own doing) is silently skipped rather than
        raising, matching this method's overall "never disrupt a redraw" contract.
        """
        reading = getattr(self.device, "reading", None)
        if reading is not None:
            self.set_reading(reading(result, self._params))
        data_hints = getattr(self.device, "data_hints", None)
        if data_hints is None:
            return
        for name, hint in data_hints(result, self._params).items():
            control = self.controls.get(name)
            if control is None:
                continue
            lo, hi, snap = hint
            if lo != lo:            # NaN sentinel -- metric unavailable, see docstring above
                control.setEnabled(False)
                continue
            control.setEnabled(True)
            set_range = getattr(control, "set_range", None)
            if set_range is not None:
                set_range(lo, hi)
            current = self._params.get(name)
            # Pass-all direction ONLY -- current > hi is a deliberate, already-restrictive user
            # choice and must never be yanked down; see the docstring's own paragraph on why.
            if current is not None and current < lo:
                self._on_control_changed(name, snap)

    def set_body_visible(self, visible: bool) -> None:
        """Rack-collapse rendering: hides this box's control grid while its OWN title bar (name,
        state dot, bypass/remove) stays visible -- ``RackBox.set_collapsed`` calls this on every
        member, which is the whole of "collapsing hides bodies, keeps titles" (module
        docstring)."""
        self._body.setVisible(visible)

    def set_bypassed(self, value: bool) -> None:
        """Rendering-only, same propose/confirm split as ``set_value``/``set_selected``/
        ``set_state`` (module docstring's "bypass RENDERING vs. the bypass GESTURE"): applies the
        bypass PROPOSAL -- disabling only the control grid -- without touching the button's own
        checked state, which already reflects the click that produced this call.

        Not currently called: ``WorkflowZone._on_box_bypass_toggled`` now routes every bypass
        toggle through ``_commit_or_revert``, whose REBUILD
        reconstructs this box from scratch with ``bypassed`` baked into the constructor, making a
        separate "apply the render" step redundant for that one call site. Kept as the documented
        primitive for applying a bypass proposal WITHOUT a full rebuild, exactly the
        ``bypassed=True`` constructor argument's own reason to exist -- a future incremental (no
        full rebuild) bypass path has this ready rather than reinventing it."""
        self._body.setEnabled(not value)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Click-to-select for everywhere the title bar does not itself cover (its own background,
        a mini-label in the grid, the margins) -- mirrors ``DeviceStrip.mousePressEvent`` exactly.
        A press that lands on a generated control or the title bar never reaches here: Qt delivers
        it to that child widget first, and neither one propagates an accepted event upward."""
        if event.button() == QtCore.Qt.LeftButton:
            self.selected.emit()
        super().mousePressEvent(event)

    def _start_drag(self) -> None:
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData(_REORDER_MIME, f"box:{self._step_index}".encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        zone = _owning_zone(self)
        if zone is not None:
            zone.dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        zone = _owning_zone(self)
        if zone is not None:
            zone._handle_drop(event, before=self)


class SourceBox(QtWidgets.QFrame):
    """The zone's leftmost box: what layer this chain is running over, and where it came from.

    Populated after construction via :meth:`set_source`, the same after-the-fact pattern
    ``DeviceBrowser.set_presets`` already uses -- built once, empty, then filled once the caller
    knows what to show. Wiring real provenance text into this (the "source chip") is out of scope here; this class only has to exist and render whatever it is given.
    """

    chipHovered = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("strip", "true")
        self.setFixedWidth(_COLUMN_WIDTH * 2)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(*_MARGINS)

        self.name_label = QtWidgets.QLabel("")
        outer.addWidget(self.name_label)
        self.provenance_label = QtWidgets.QLabel("")
        self.provenance_label.setProperty("muted", "true")
        self.provenance_label.setWordWrap(True)
        outer.addWidget(self.provenance_label)
        self.chip_label = QtWidgets.QLabel("")
        self.chip_label.setProperty("muted", "true")
        self.chip_label.setVisible(False)
        outer.addWidget(self.chip_label)
        outer.addStretch(1)

        self._fed_by_layer_id: int | None = None

    def set_source(self, name: str, provenance, fed_by: str | None = None,
                  fed_by_layer_id: int | None = None) -> None:
        self.name_label.setText(name)
        self.provenance_label.setText(self._provenance_text(provenance))
        self._fed_by_layer_id = fed_by_layer_id
        if fed_by:
            self.chip_label.setText(f"◂ fed by {fed_by}")
            self.chip_label.setVisible(True)
        else:
            self.chip_label.setText("")
            self.chip_label.setVisible(False)

    @staticmethod
    def _provenance_text(provenance) -> str:
        if not provenance:
            return ""
        if isinstance(provenance, str):
            return provenance
        source = provenance.get("source") if isinstance(provenance, dict) else None
        return str(source) if source else ""

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        if self._fed_by_layer_id is not None and self.chip_label.isVisible():
            self.chipHovered.emit(self._fed_by_layer_id)
        super().enterEvent(event)


class RackBox(QtWidgets.QFrame):
    """A collapsible container for a contiguous run of ``DeviceBox``es -- UI grouping ONLY; the
    engine sees the flattened linear chain (the module docstring's drag-drop section has the flatten
    rule ``WorkflowZone.set_steps`` implements). ``title`` is both the display label and the
    identity every member descriptor's ``"rack"`` field carries.

    Collapsing (:meth:`set_collapsed`) hides every MEMBER's own control grid
    (``DeviceBox.set_body_visible``) while each member keeps its own title bar -- a collapsed rack
    reads as a row of slim titled strips, not one strip for the whole rack.

    Accepts drops itself (``setAcceptDrops(True)``) -- a drop landing on this rack's OWN chrome
    (the title bar, the padding around its members -- anywhere a member ``DeviceBox`` isn't) is a
    NEST at the END of this rack, routed straight to the owning ``WorkflowZone``'s
    ``_handle_drop(event, into_rack=self.title)``. A drop landing precisely on one of this rack's
    OWN members goes to that member's own ``dropEvent`` instead (every ``DeviceBox`` accepts
    drops too) -- ``_handle_drop(event, before=<that member>)``, which
    inserts directly before it, still inside this rack (the member's own ``"rack"`` value is what
    the zone inherits). Qt's own event delivery is what decides which of the two receives a given
    drop in real usage (offscreen size-hint propagation is not reliable enough in this environment
    to test PIXEL position for this, so drop routing is decided by WHICH WIDGET's ``dropEvent``
    fires, not by coordinates read back out of one) -- see the module docstring's drag-drop section
    and the tests, which call ``rack.dropEvent(...)`` or a member's ``dropEvent(...)`` directly
    for exactly this reason.

    A second title-bar button, ``×``, next to ``collapse_button`` -- emits
    :attr:`rackRemoveRequested` rather than removing anything itself (this box has no idea what its
    own span even is; only the owning :class:`WorkflowZone` can compute that via ``_rack_span``).
    See the module docstring's rack-removal section for the full remove/undo design.
    """

    #: The × button -- see the module docstring's rack-removal section. Named distinctly from
    #: ``DeviceBox.removeRequested`` (a plain no-arg signal too) only so a listener connected to
    #: both cannot mistake one for the other; both are handled the same "propose, owner decides"
    #: way.
    rackRemoveRequested = QtCore.Signal()

    def __init__(self, title: str, members: list[DeviceBox], parent=None):
        super().__init__(parent)
        self.title = title
        self.setProperty("strip", "true")
        self.setAcceptDrops(True)
        self._members = list(members)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(*_MARGINS)

        self._title_bar = _TitleBar(on_drag=self._start_drag)
        title_row = QtWidgets.QHBoxLayout(self._title_bar)
        title_row.setContentsMargins(0, 0, 0, 0)
        self.title_label = QtWidgets.QLabel(title)
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        self.collapse_button = QtWidgets.QToolButton()
        self.collapse_button.setCheckable(True)
        self.collapse_button.setText("▾")
        self.collapse_button.setToolTip("Collapse")
        self.collapse_button.toggled.connect(self.set_collapsed)
        title_row.addWidget(self.collapse_button)
        # Mirrors DeviceBox.remove_button's own construction (same glyph/tooltip style) --
        # a click only ever emits; WorkflowZone owns the actual span removal.
        self.remove_button = QtWidgets.QToolButton()
        self.remove_button.setText("×")
        self.remove_button.setToolTip("Remove rack")
        self.remove_button.clicked.connect(self.rackRemoveRequested.emit)
        title_row.addWidget(self.remove_button)
        outer.addWidget(self._title_bar)

        members_row = QtWidgets.QHBoxLayout()
        members_row.setContentsMargins(0, 0, 0, 0)
        for member in self._members:
            members_row.addWidget(member)
        outer.addLayout(members_row)

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapse_button.setChecked(collapsed)
        for member in self._members:
            member.set_body_visible(not collapsed)

    def _start_drag(self) -> None:
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData(_REORDER_MIME, f"rack:{self.title}".encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        zone = _owning_zone(self)
        if zone is not None:
            zone.dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        zone = _owning_zone(self)
        if zone is not None:
            zone._handle_drop(event, into_rack=self.title)


class WorkflowZone(QtWidgets.QFrame):
    """One ``DeviceBox``/``RackBox`` per chain step or rack, plus a leading ``SourceBox``, in a
    sideways-scrolling row.

    ``set_steps`` takes DESCRIPTORS (dicts shaped ``{"device": str, "params": dict, "bypassed":
    bool, "rack": str | None}``), the same shape :attr:`chainEdited` emits -- not
    ``dynamix.model.chain.DeviceRef`` instances, which have no ``bypassed``/``rack`` fields to
    carry. ``self._descriptors`` IS the flattened chain order; see the module docstring's drag-drop section for the full drag-drop/rack/refusal design.

    ``strip(i)`` is the deliberate alias ``main_window``'s existing call sites
    (``_set_transform_states``, ``_set_compute_reading``, ``_sync_controls``) already use against
    ``ChainStripZone`` -- keeping the name here is what lets this class drop into
    ``self.strips`` unmodified. It indexes ``self._boxes``, which holds every ``DeviceBox``
    (top-level AND nested in a rack) in descriptor-list order, so ``strip(i)`` is always the box
    for ``self._descriptors[i]`` regardless of rack membership.
    """

    paramChanged = QtCore.Signal(int, str, object)
    chainEdited = QtCore.Signal(object)
    #: The chip-hover feedback: relayed straight from ``self.source_box.chipHovered``, so
    #: ``main_window`` has one signal to connect to ``LayerPanel.flash_row`` per zone rebuild,
    #: rather than reaching past this zone into its ``source_box`` child.
    chipHovered = QtCore.Signal(int)
    #: Fired ``(device, message)`` whenever a drop or reorder had to
    #: be CLAMPED into a legal position (see :meth:`_legal_insert_index`) -- the main window relays this to its status bar; this zone shows the same message itself via
    #: ``reading_label`` (see :meth:`_commit_or_revert`), so nothing is lost by emitting it early.
    dropPlaced = QtCore.Signal(str, str)
    #: Fired ``(title, message)`` whenever a rack removal actually lands (see
    #: :meth:`_on_rack_remove_requested`) -- SAME shape as :attr:`dropPlaced`, deliberately a
    #: separate signal rather than reusing it (module docstring's rack-removal section explains why:
    #: ``dropPlaced`` is already pinned by a test to firing only on a clamped placement). Wired by
    #: ``main_window._build_strips`` to ``self._notify(msg, "status")``, same as ``dropPlaced``.
    rackRemoved = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("zone", "strip")
        self.setAcceptDrops(True)
        self._boxes: list[DeviceBox] = []
        self._descriptors: list[dict] = []
        self._presets: dict[str, tuple] = {}
        self._field = None
        # Set by ``_handle_device_drop``/``_reorder`` for the ``_commit_or_revert`` call
        # that immediately follows them (never outlives one drop cycle -- ``_commit_or_revert``
        # captures and resets both at its own top, win or lose, so a bypass/remove mutation -- which
        # never touches these -- always finds them ``None``).
        self._pending_placement: tuple[str, str] | None = None    # (device, message) if clamped
        self._pending_scroll_to: int | None = None                # descriptor index to scroll to
        # LIFO stack of FULL pre-removal descriptor-list snapshots, one push per rack ×
        # that actually lands -- see the module docstring's rack-removal section and :meth:`undo_removal`.
        self._removed_racks_stack: list[list[dict]] = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.source_box = SourceBox()
        self.source_box.chipHovered.connect(self.chipHovered.emit)
        self._host = QtWidgets.QWidget()
        self._row = QtWidgets.QHBoxLayout(self._host)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.addWidget(self.source_box)
        self._row.addStretch(1)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._host)
        outer.addWidget(self._scroll)

        # A refused drop's reason, timed -- see the module docstring's drag-drop section. Hidden
        # (which a QVBoxLayout excludes from its size calculation) whenever there is nothing to
        # say, so this costs no vertical room the rest of the time.
        self.reading_label = QtWidgets.QLabel("")
        self.reading_label.setProperty("reading", "true")
        self.reading_label.setProperty("muted", "true")
        self.reading_label.setVisible(False)
        outer.addWidget(self.reading_label)

        self._warning_timer = QtCore.QTimer(self)
        self._warning_timer.setSingleShot(True)
        self._warning_timer.setInterval(_WARNING_MS)
        self._warning_timer.timeout.connect(self._clear_warning)

    def set_presets(self, presets: dict) -> None:
        """Injected by ``main_window`` with the same dict the browser gets -- a preset drop
        resolves its device-step tuple through here, by name."""
        self._presets = dict(presets)

    def set_steps(self, descriptors, field=None) -> None:
        """(Re)build the zone from ``descriptors``, left to right after the ``SourceBox``: one
        ``DeviceBox`` for a ``rack=None`` entry, or one ``RackBox`` per contiguous run of entries
        sharing the same non-``None`` rack name. The descriptor list's OWN order already is the
        flattened chain order (module docstring), so grouping consecutive rack membership is all
        this has to do."""
        self._field = field
        self._clear_boxes()
        self._descriptors = [dict(d) for d in descriptors]
        n = len(self._descriptors)
        i = 0
        while i < n:
            rack = self._descriptors[i].get("rack")
            if rack is None:
                self._row.addWidget(self._make_box(i))
                i += 1
            else:
                j = i
                members = []
                while j < n and self._descriptors[j].get("rack") == rack:
                    members.append(self._make_box(j))
                    j += 1
                rack_box = RackBox(rack, members)
                rack_box.rackRemoveRequested.connect(self._on_rack_remove_requested)
                self._row.addWidget(rack_box)
                i = j
        self._row.addStretch(1)

    def _make_box(self, i: int) -> DeviceBox:
        d = self._descriptors[i]
        device = get_device(d["device"])
        params = {**defaults_for(device), **d.get("params", {})}
        box = DeviceBox(i, device, params, field=self._field,
                        bypassed=bool(d.get("bypassed", False)))
        box.paramChanged.connect(self._on_box_param_changed)
        box.bypassToggled.connect(self._on_box_bypass_toggled)
        box.removeRequested.connect(self._on_box_remove_requested)
        box.selected.connect(self._on_box_selected)
        self._boxes.append(box)
        return box

    def _clear_boxes(self) -> None:
        # index 0 is the SourceBox -- never removed by a rebuild -- so everything from 1 onward
        # (every DeviceBox/RackBox plus the trailing stretch) is what a set_steps call tears down.
        # Deleting a RackBox cascades to its own (Qt-parented) member boxes; nothing here needs to
        # walk into one separately.
        while self._row.count() > 1:
            item = self._row.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._boxes = []

    def strip(self, i: int) -> DeviceBox:
        return self._boxes[i]

    def select(self, i: int) -> None:
        """Enforce single selection: ``set_selected`` on every box, true only at index ``i``."""
        for box in self._boxes:
            box.set_selected(box.step_index == i)

    def set_source(self, name: str, provenance, fed_by: str | None = None,
                  fed_by_layer_id: int | None = None) -> None:
        self.source_box.set_source(name, provenance, fed_by=fed_by,
                                   fed_by_layer_id=fed_by_layer_id)

    def _on_box_param_changed(self, name: str, value) -> None:
        box: DeviceBox = self.sender()
        # Keeps the descriptor list live -- see the module docstring's drag-drop section on why a
        # knob edit has to land here too, not just in the paramChanged payload.
        self._descriptors[box.step_index]["params"][name] = value
        self.paramChanged.emit(box.step_index, name, value)

    def _on_box_selected(self) -> None:
        box: DeviceBox = self.sender()
        self.select(box.step_index)

    def _on_box_bypass_toggled(self, checked: bool) -> None:
        # Routed through the same gate every other mutation is:
        # toggling bypass can never actually change whether the FULL descriptor order validates
        # (see _commit_or_revert), so this is a no-op guard in practice -- but it is the SAME one
        # gate rather than a second, ungated path that could drift from it. The rebuild it
        # triggers also reconstructs this box fresh (bypassed=checked baked into the constructor),
        # so there is no separate "apply the render" step to take here any more.
        box: DeviceBox = self.sender()
        snapshot = [dict(d) for d in self._descriptors]
        self._descriptors[box.step_index]["bypassed"] = checked
        self._commit_or_revert(snapshot)

    def _on_box_remove_requested(self) -> None:
        box: DeviceBox = self.sender()
        snapshot = [dict(d) for d in self._descriptors]
        del self._descriptors[box.step_index]
        self._commit_or_revert(snapshot)

    def _on_rack_remove_requested(self) -> None:
        """Rack removal (the module docstring has the full design). ``self.sender()`` is the
        ``RackBox`` whose × was clicked; its ``title`` is looked up fresh through
        :meth:`_rack_span` rather than trusted from anywhere cached, so this is correct even if
        some other mutation already moved/renamed things since this box was built. ``start ==
        end`` (the "rack no longer present" sentinel :meth:`_rack_span` returns) is a defensive
        no-op -- a stale signal from a box mid-teardown, not a reachable case in normal use.

        The length check after ``_commit_or_revert`` is what decides whether the removal actually
        landed (vs. reverted) -- see the module docstring's rack-removal section for why a pure deletion
        can, in practice, never fail that gate, and why this check is a defensive belt rather than
        a real branch. Only a LANDED removal pushes onto the undo stack or reports outward: a
        reverted one already reported itself, loudly, zone-locally (``_revert``'s own persistent
        warning + flash) -- reporting again here would be the exact double-report the design rules out.
        """
        rack_box: RackBox = self.sender()
        title = rack_box.title
        start, end = self._rack_span(title)
        if start == end:
            return
        snapshot = [dict(d) for d in self._descriptors]
        n_devices = end - start
        del self._descriptors[start:end]
        self._commit_or_revert(snapshot)
        if len(self._descriptors) != len(snapshot) - n_devices:
            return                              # reverted -- _revert already reported it
        self._removed_racks_stack.append(snapshot)
        message = f"Removed rack {title} ({n_devices} devices) — ⇧⌘Z restores"
        self._show_warning(message)             # zone-local, normal fade -- a success, not a fault
        self.rackRemoved.emit(title, message)

    def undo_removal(self) -> None:
        """Pop the most recently removed rack's FULL pre-removal descriptor-list snapshot and
        restore it wholesale, through the same :meth:`_commit_or_revert` gate every other mutation
        uses -- see the module docstring's rack-removal section for why a whole-list restore (rather than
        re-inserting just the removed span at wherever the zone currently stands) is both simpler
        and correct across repeated removals. A no-op with an empty stack -- EQSelect's own
        silent-by-design refusal shape for "nothing to undo" -- there is nothing here for a
        caller to react to either way, matching ``TransectPanel.undo``'s identical guard.
        """
        if not self._removed_racks_stack:
            return
        before = [dict(d) for d in self._descriptors]
        self._descriptors = [dict(d) for d in self._removed_racks_stack.pop()]
        self._commit_or_revert(before)

    # -- drag and drop -----------------------------------------------------------------------
    #
    # Routing (which container a drop lands IN, and where within it) is decided by WHICH WIDGET'S
    # ``dropEvent`` Qt calls, not by reading a pixel position back out of one: this zone, every
    # ``RackBox``, and every ``DeviceBox`` all accept drops (``setAcceptDrops(True)``) now (a ``DeviceBox`` used to bubble every drop past itself). Every path converges on :meth:`_handle_drop`, parameterised by
    # ``into_rack`` (a rack's own chrome received the drop -- append to that rack's end) or
    # ``before`` (a specific ``DeviceBox`` received it -- insert directly before it, inheriting
    # that box's own rack membership) -- both ``None`` together means this zone's own top-level
    # background, which appends at the very end.

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        md = event.mimeData()
        if (md.hasFormat(DEVICE_MIME) or md.hasFormat(PRESET_MIME) or
                md.hasFormat(_REORDER_MIME)):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        self._handle_drop(event)

    def _handle_drop(self, event: QtGui.QDropEvent, into_rack: str | None = None,
                     before: "DeviceBox | None" = None) -> None:
        md = event.mimeData()
        snapshot = [dict(d) for d in self._descriptors]
        if md.hasFormat(DEVICE_MIME):
            self._handle_device_drop(bytes(md.data(DEVICE_MIME)).decode("utf-8"), into_rack, before)
        elif md.hasFormat(PRESET_MIME):
            self._handle_preset_drop(bytes(md.data(PRESET_MIME)).decode("utf-8"))
        elif md.hasFormat(_REORDER_MIME):
            self._reorder(bytes(md.data(_REORDER_MIME)).decode("utf-8"), into_rack, before)
        else:
            event.ignore()
            return
        self._commit_or_revert(snapshot)
        event.acceptProposedAction()

    def _rack_span(self, title: str) -> tuple[int, int]:
        """``[start, end)`` -- the contiguous run of ``self._descriptors`` tagged this rack, or
        ``(len, len)`` if no descriptor currently carries it (a moment mid-reorder, or a stale
        title)."""
        indices = [i for i, d in enumerate(self._descriptors) if d.get("rack") == title]
        if not indices:
            return len(self._descriptors), len(self._descriptors)
        return indices[0], indices[-1] + 1

    def _index_of_descriptor(self, d: dict) -> int:
        """Position of THIS descriptor OBJECT (identity, not equality -- two steps of the same
        device with the same params are not interchangeable for this lookup) in the current list.
        A landing target's own descriptor is captured by reference before any mutation, so this
        finds its CURRENT index even after ``span`` has been removed and shifted everything past
        it."""
        for i, cur in enumerate(self._descriptors):
            if cur is d:
                return i
        return len(self._descriptors)          # not found -- append rather than crash

    def _safe_insertion_index(self, into_rack: str | None,
                              before: "DeviceBox | None") -> tuple[int, bool]:
        """(index, landed_safely) for a drop targeting ``before`` (or ``into_rack``'s end, or the
        very end).

        Landing exactly at ``before``'s own position is SAFE whenever ``before`` is not racked, IS
        its rack's own FIRST member (so its position already equals that rack's start boundary --
        "already boundary-safe"), or the caller explicitly targeted that SAME rack via
        ``into_rack`` ("the dragged item is itself destined for that rack"). Landing on any OTHER
        (interior) member of a DIFFERENT rack is UNSAFE: returned as that rack's own START
        boundary instead, ``landed_safely=False`` -- "dropping onto a rack's middle" becomes
        "insert before that rack" (module docstring's drag-drop section) rather than splitting it
        into two same-titled ``RackBox``es, which is what an earlier version of this method could
        do (a moved rack dropped on an INTERIOR member of a different
        one spliced itself into the middle of that rack's own span, silently rendering it as two
        separate boxes -- ``set_steps``' consecutive-grouping has no way to know the two spans
        were ever meant to be one).
        """
        if before is not None:
            target_rack = self._descriptors[before.step_index].get("rack")
            if (target_rack is None or before.step_index == self._rack_span(target_rack)[0]
                    or into_rack == target_rack):
                return before.step_index, True
            return self._rack_span(target_rack)[0], False
        if into_rack is not None:
            return self._rack_span(into_rack)[1], True
        return len(self._descriptors), True

    def _insertion_target(self, into_rack: str | None,
                          before: "DeviceBox | None") -> tuple[int, str | None]:
        """(index, rack) for a NEW arrival. Position and safety come from
        :meth:`_safe_insertion_index`; a SAFE landing on ``before`` inherits its own rack
        membership (nesting into it, same as always); an UNSAFE one -- interior to a DIFFERENT
        rack -- detaches instead, falling back to ``into_rack`` (``None`` for an ordinary
        before-routed drop), landing before that rack rather than splitting it."""
        index, safe = self._safe_insertion_index(into_rack, before)
        if before is not None:
            target_rack = self._descriptors[before.step_index].get("rack")
            return index, (target_rack if safe else into_rack)
        return index, into_rack

    def _legal_insert_index(self, device_name: str, requested_index: int) -> int:
        """The smart-
        placement clamp. Pure descriptor-list math over ``self._descriptors`` as it stands RIGHT
        NOW (bypassed steps included, same as ``_commit_or_revert``'s own full-order validation --
        a bypassed FILTER still blocks a TRANSFORM from landing after it, module docstring's present-but-inert rule; the difference here is the transform is redirected rather than refused) -- no
        rack awareness, no Qt, no mutation. Called BEFORE the candidate descriptor is built/spliced
        in, by both a fresh device drop (:meth:`_handle_device_drop`) and a box reorder
        (:meth:`_reorder`), so both routes clamp identically.

        A TRANSFORM (:func:`dynamix.model.device.is_transform`) whose ``requested_index`` falls
        AFTER any filter clamps down to that filter's own index -- the end of the transform block.
        A FILTER whose ``requested_index`` falls BEFORE any transform clamps up to one past that
        transform's own index -- the start of the filter block. An already-legal index (an
        explicit, legal ``before=`` target, or anywhere already inside the correct block) is
        returned UNCHANGED: this only ever narrows an illegal candidate toward a position
        ``Chain.validate`` was always going to accept -- it is not a second implementation of the
        transforms-then-filters rule, ``Chain.validate`` (still called by ``_commit_or_revert``)
        stays the one authority on that.
        """
        if is_transform(get_device(device_name)):
            for i, d in enumerate(self._descriptors):
                if not is_transform(get_device(d["device"])):
                    return min(requested_index, i)
            return requested_index
        boundary = 0
        for i, d in enumerate(self._descriptors):
            if is_transform(get_device(d["device"])):
                boundary = i + 1
        return max(requested_index, boundary)

    def _placement_message(self, device_name: str) -> str:
        """The status-tier text for a clamped placement ([EQ §7] "refusals name the
        fix" -- this names the OUTCOME, since a clamp is a success, not a refusal)."""
        kind = "transform" if is_transform(get_device(device_name)) else "filter"
        return f"{device_name} placed in the {kind} section -- chains are transforms-then-filters"

    def _handle_device_drop(self, name: str, into_rack: str | None,
                            before: "DeviceBox | None") -> None:
        index, rack = self._insertion_target(into_rack, before)
        device = get_device(name)
        clamped = self._legal_insert_index(name, index)
        if clamped != index:
            self._pending_placement = (name, self._placement_message(name))
        index = clamped
        self._pending_scroll_to = index
        self._descriptors.insert(index, {"device": name, "params": defaults_for(device),
                                         "bypassed": False, "rack": rack})

    def _handle_preset_drop(self, name: str) -> None:
        """A preset drop creates its OWN named rack from the preset's steps (design: "Racks" is a
        real browser category, and dragging one of its rows out has to actually produce a rack). The drop TARGET is not consulted: nesting one preset's rack
        inside another named rack, or inserting it at a precise position, is not a supported
        gesture -- a preset always lands as a new rack at the very end, or replaces an empty zone
        outright."""
        steps = self._presets.get(name)
        if not steps:
            return
        new = [{"device": n, "params": dict(p), "bypassed": False, "rack": name} for n, p in steps]
        if not self._descriptors:
            self._descriptors = new
            return
        self._descriptors.extend(new)

    def _reorder(self, payload: str, into_rack: str | None, before: "DeviceBox | None") -> None:
        """A box or rack's own title-bar drag: pull its span out and reinsert it at the drop
        TARGET -- directly before ``before`` if landing there is SAFE (:meth:`_safe_insertion_index`
        -- inheriting its rack membership for a moved BOX, same rule as a new arrival; a moved RACK
        never inherits, it always keeps its own identity), snapped to that rack's own start
        boundary instead if not (module docstring's drag-drop section: this is what keeps a rack a
        single contiguous span no matter where within a DIFFERENT one something lands -- the fix for the
        silent-split defect). A moved BOX reordering within its OWN
        current rack is ALWAYS safe regardless of position -- landing on any sibling, first or not,
        is just repositioning within the contiguous run it already belongs to (a natural "reorder
        within a rack" gesture the general interior check would otherwise needlessly eject it
        from). Absent a specific box target, it lands at the end of ``into_rack``'s span, else at
        the very end, top level. Dropped on itself, or (for a box) on a fellow member of the SAME
        span being moved, or (for a rack) on one of its own members or its own chrome, is a
        deliberate no-op, checked before anything is mutated.
        """
        kind, _, ident = payload.partition(":")
        if kind == "box":
            if not ident.lstrip("-").isdigit():
                return
            idx = int(ident)
            if not (0 <= idx < len(self._descriptors)):
                return
            start, end = idx, idx + 1
        elif kind == "rack":
            start, end = self._rack_span(ident)
            if start == end or into_rack == ident:
                return
        else:
            return
        if before is not None and start <= before.step_index < end:
            return                              # dropped on itself / a member of the moved span

        # Resolved against the list as it stands NOW (before any mutation): the LANDING
        # descriptor -- ``before``'s own if safe, else the target rack's own first member -- is
        # captured by OBJECT REFERENCE, so its position after the span below is removed (which can
        # shift everything past it) is found by identity, not by an index computed too early.
        inherits_rack = kind == "box"
        # A moved BOX reordering within its OWN CURRENT rack is always safe, first member or not
        # -- it is simply repositioning within the contiguous run it is already (and will remain)
        # a part of, which is exactly the natural "reorder within a rack" gesture. Only a rack
        # that DIFFERS from the one being moved needs the general interior-landing check below.
        source_rack = self._descriptors[start].get("rack") if inherits_rack else None
        landing_descriptor = None
        landed_safely = True
        target_rack = None
        if before is not None:
            target_rack = self._descriptors[before.step_index].get("rack")
            if inherits_rack and target_rack is not None and source_rack == target_rack:
                landed_safely = True
            else:
                _, landed_safely = self._safe_insertion_index(into_rack, before)
            landing_index = (before.step_index if landed_safely
                             else self._rack_span(target_rack)[0])
            landing_descriptor = self._descriptors[landing_index]

        span = self._descriptors[start:end]
        del self._descriptors[start:end]

        if landing_descriptor is not None:
            index = self._index_of_descriptor(landing_descriptor)
            rack = (target_rack if landed_safely else into_rack) if inherits_rack else ident
        elif into_rack is not None:
            index, rack = self._rack_span(into_rack)[1], (into_rack if inherits_rack else ident)
        else:
            index, rack = len(self._descriptors), (None if inherits_rack else ident)

        if kind == "box":
            span = [dict(span[0])]
            span[0]["rack"] = rack
            # A single-box reorder clamps identically to a fresh device drop --
            # dragging a transform's title bar into filter territory (or vice versa) is redirected
            # to the nearest legal position, not left to ``Chain.validate`` to refuse. Computed
            # against ``self._descriptors`` as it stands HERE -- the moved span already removed
            # (``del`` above), not yet reinserted -- exactly the "before building the candidate
            # list" ordering the clamp needs. A RACK move (below) is NOT clamped: a rack can
            # legitimately mix transforms and filters (a preset's own step list, e.g.), so there is
            # no single "kind" to clamp it by -- ``Chain.validate`` remains the sole guard for
            # those, same as before this task.
            device_name = span[0]["device"]
            clamped = self._legal_insert_index(device_name, index)
            if clamped != index:
                self._pending_placement = (device_name, self._placement_message(device_name))
            index = clamped
        else:
            # A rack keeps its own identity when moved -- racks do not nest (the design's "UI
            # grouping ONLY", one level deep). Dropped at the top level or before a specific box
            # (the ordinary cases) it lands there unchanged; dropped onto a DIFFERENT rack
            # (``into_rack`` not ``None`` -- unusual, not a gesture the browser or a box's own
            # title bar produces) it lands adjacent to that rack's span rather than inside it,
            # since there is no "nested rack" for its members to actually join.
            span = [dict(d) for d in span]
            for d in span:
                d["rack"] = rack
        self._pending_scroll_to = index
        self._descriptors[index:index] = span

    def _commit_or_revert(self, before: list[dict]) -> None:
        """The refusal gate every zone mutation funnels through -- device/preset drop, reorder,
        bypass toggle, remove. Validates the FULL descriptor order (bypassed steps INCLUDED,
        present-but-inert for ordering purposes only -- module docstring's present-but-inert rule) so a
        bypassed filter still blocks a transform from landing after it: un-bypassing can therefore
        never surface an illegal chain later, because the arrangement was never allowed to become
        illegal in the first place, active or not. ``Chain.validate`` stays the one authority on
        transforms-then-filters -- this is a refusal at the zone, not a second implementation of
        the rule. Most transform/filter ordering mistakes never reach this gate at
        all -- :meth:`_handle_device_drop` and a box :meth:`_reorder` already clamped the candidate
        into a legal position before this runs. What still lands here as a genuine
        refusal: an unresolvable device, an illegal param, or a RACK reorder that the box-only
        clamp does not cover (a rack may legitimately mix transforms and filters).

        Past that gate, ``set_steps`` still has to REBUILD the boxes, and that can itself raise
        for reasons ``Chain.validate`` never sees -- a device's own control construction failing
        (see the module docstring). Any such exception is caught the same way: full revert,
        rebuilt, warned. A commit therefore either fully lands or fully doesn't; the zone is never
        left part-mutated for the NEXT commit to inherit.

        ``self._pending_placement``/``self._pending_scroll_to`` are captured and reset HERE, at the
        top, regardless of outcome -- so a bypass toggle or remove (which never set them) always
        finds them ``None``, and a clamp that ends up reverted (the rebuild itself failing, however
        unlikely) never leaks its message into some LATER, unrelated commit.
        """
        placement = self._pending_placement
        scroll_to = self._pending_scroll_to
        self._pending_placement = None
        self._pending_scroll_to = None
        try:
            Chain(tuple(DeviceRef(d["device"], d["params"])
                       for d in self._descriptors)).validate()
        except ValueError as exc:
            self._revert(before, str(exc))
            return
        try:
            self.set_steps(self._descriptors, self._field)
        except Exception as exc:                # noqa: BLE001 -- ANY box-construction failure
            self._revert(before, str(exc))
            return
        # Any PERSISTENT refusal message from an EARLIER mutation ends here -- the design: "refusal
        # messages persist until the next user interaction" -- whether or not THIS mutation has a
        # placement message of its own to show in its place.
        self._clear_warning()
        if placement is not None:
            device, message = placement
            self._show_warning(message)                 # success tier: normal 4 s fade
            self.dropPlaced.emit(device, message)
        if scroll_to is not None and 0 <= scroll_to < len(self._boxes):
            self._scroll.ensureWidgetVisible(self.strip(scroll_to))
        self.chainEdited.emit([dict(d) for d in self._descriptors])

    def _revert(self, before: list[dict], message: str) -> None:
        """Restore the last known-good arrangement and warn -- a genuine refusal: the
        message is PERSISTENT (no 4 s fade -- stays until the next successful mutation clears it in
        ``_commit_or_revert`` above), plus a brief red flash at the drop site
        (:meth:`_flash_refusal`). ``before`` is always renderable -- it was either the state
        ``set_steps`` last succeeded at, or the initial (also once successfully rendered) state --
        so this call is not itself guarded against raising."""
        self.set_steps(before, self._field)
        self._show_warning(message, persistent=True)
        self._flash_refusal()

    def _show_warning(self, message: str, persistent: bool = False) -> None:
        """``persistent=True`` (a refusal) leaves ``_warning_timer`` stopped -- nothing auto-clears
        the message; only the next successful mutation's ``_clear_warning()`` call does. The
        default (a success, including a clamped placement) restarts the normal 4 s fade timer."""
        self.reading_label.setText(message)
        self.reading_label.setVisible(True)
        self._warning_timer.stop()
        if not persistent:
            self._warning_timer.start()

    def _clear_warning(self) -> None:
        self.reading_label.setText("")
        self.reading_label.setVisible(False)

    def _flash_refusal(self) -> None:
        """Brief red flash at the drop site for a genuine refusal (the design's drop-refusal
        feedback) -- ``setProperty`` + an explicit style repolish is what makes Qt's QSS
        re-evaluate the ``[refused="true"]`` rule (``theme.py``) immediately, rather than waiting
        for some unrelated repaint to notice the property changed; :meth:`_end_flash`, on the same
        round trip ``_FLASH_MS`` later, reverts it."""
        self.setProperty("refused", True)
        self.style().unpolish(self)
        self.style().polish(self)
        QtCore.QTimer.singleShot(_FLASH_MS, self._end_flash)

    def _end_flash(self) -> None:
        self.setProperty("refused", False)
        self.style().unpolish(self)
        self.style().polish(self)
