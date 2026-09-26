# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The per-source floating inspector: an always-on-top window hosting the EXISTING canvas.

A layer-list toggle shows or hides a window of the raster that stays in front, so parameter changes can be watched live. The design refines it: the inspector is per
SOURCE dataset and it is a TREE -- the source raster plus every product derived from it, each row
with its own on/off and opacity, ordered by the processing provenance
(:mod:`dynamix.model.provenance` is the headless half of that ordering, so this window and the
layer list cannot disagree about it).

**This window MOVES a proven widget; it does not author a second one.** The picture inside it is
:class:`dynamix.shell.canvas.Canvas` -- the same class the center zone uses -- so every overlay,
gesture and, crucially, the geometry cache (``canvas.py``'s ``_cached_geometry``) come with it.
N open inspectors must not mean N rebuilds of the same geometry, and the way to get that
is to BE the canvas rather than to reimplement it.

**Window class, and the one place its precedent is deliberately diverged from.** The flags are
``Qt.Tool | Qt.WindowStaysOnTopHint`` on a TOP-LEVEL widget -- proven on this platform by
``shell/arrangement/view.py``'s ``_LassoRegionOverlay`` (and EQSelect's ``_LassoOverlay`` before
it), because a child ``QWidget`` will not reliably composite over the native render surface.
What this window must NOT carry is that overlay's three click-through settings --
``Qt.WindowTransparentForInput``, ``WA_TransparentForMouseEvents`` and ``WA_ShowWithoutActivating``
-- for the reason sec 5b gives in one line: an overlay wants clicks to pass THROUGH, an inspector
must RECEIVE them. An inspector that cannot be clicked cannot have its sub-layers toggled or its
scale scrubbed, which is the whole of what it is for. The divergence is asserted in
``tests/test_inspector.py`` rather than merely commented, so a future copy-paste of the overlay's
constructor cannot quietly reintroduce it.

Chrome is deliberately absent (sec 5c is PENDING on mockups): a thin title strip, the
canvas, the sub-layer tree, a transport and the follow-master button, nothing styled beyond the
theme's own properties. What is real here is the STATE and the signals that carry it; pixels are
the mockups' business.
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtWidgets

from dynamix.model.provenance import raster_key
from dynamix.shell.canvas import Canvas
from dynamix.shell.transport import Transport

#: What the status strip says before any result has ever reached this window. Honest, not blank
#: (the design doctrine #3): it names the state AND the fix. ``MainWindow`` owns the
#: other two readings -- "showing the live result" and "last result ... not active" -- and reads
#: this one from here so the two halves cannot come to word the same state differently.
NO_RESULT_TEXT = "no result yet — select a layer over this source and run its chain"

#: Appended to the status strip whenever this window's scale CONTROL names a scale its PICTURE is
#: not showing. That happens whenever the held result is post-``scale_select`` -- the shipped
#: ``DEMO_CHAIN``'s shape -- because that filter reduces the stack to the ONE layer the chain asked
#: for (``devices/filters.py``'s ``ScaleSelect.apply``), so there is no other scale in this window
#: to draw. Showing a moving reading over a still picture and saying nothing is precisely the
#: "silent refusal" sec 8 rule 6 forbids, so this names the state AND the fix -- and the fix it
#: names is a gesture that exists today: make that source's layer active and the master's scrub
#: re-runs its chain at the new index, which is what actually moves the picture.
SCALE_NOT_SHOWN_TEXT = ("scale {control} is not in this result — showing scale {picture}; "
                        "select this source's layer and scrub to re-run it there")

#: The same confession for a window whose picture IS the live result of the ACTIVE layer, where
#: the sentence above would name a gesture the user has already made ("select this source's
#: layer" — it is selected). Instructing someone to do what they have just done is the
#: fabricated-affordance failure this repo keeps paying for, so the live case names the
#: LIMITATION instead: there is no gesture today that holds this window at its own scale, because
#: the picture is the active chain's own output and the chain follows the master.
SCALE_NOT_SHOWN_LIVE_TEXT = (
    "scale {control} is not in this result — showing scale {picture}, the active chain's own; "
    "this window cannot hold a scale of its own yet")

#: The provenance row key (``provenance.raster_key`` / ``sublayer_key``) carried by a keyed tree
#: item. A GROUPING row (a layer) carries ``None`` here: those two functions are the only keys
#: this window and the persisted state (``model/inspector_state.py``) may use, and a layer is not
#: one of them.
_KEY_ROLE = QtCore.Qt.UserRole

#: The sub-layer tree's share of the window. The canvas is what the user came for, so the tree
#: takes a bounded strip and scrolls past it rather than growing without limit -- the one pixel
#: decision here, made because the render criterion needs the raster AND the rows both visible.
_TREE_MAX_H = 140


class InspectorWindow(QtWidgets.QWidget):
    """One source dataset's floating inspector. See the module docstring for the window class.

    ``tree_node`` is that source's :class:`dynamix.model.provenance.SourceNode`; ``scale_reading``
    is the same callable ``Transport`` already takes (``MainWindow._scale_reading``), since only
    the window knows the physical units.

    Every signal leads with the ``source_id``: the inspector is source-agnostic about everything
    else (sec 7 rule 3) -- it does not hold a ``Project``, does not know which layer is active and
    cannot look one up. Whoever wired it up is what owns that mapping.
    """

    #: The window was closed. Closing it and un-toggling the layer list's button are ONE state;
    #: this is the half the toggle listens to.
    closed = QtCore.Signal(str)

    #: ``(source_id, row_key, visible)`` -- a sub-layer row's checkbox changed.
    subLayerToggled = QtCore.Signal(str, str, bool)

    #: ``(source_id, row_key, opacity)`` -- a sub-layer row's opacity spin changed.
    subLayerOpacityChanged = QtCore.Signal(str, str, float)

    #: ``(source_id, scale_idx)`` -- THIS inspector's own transport moved. It drives this
    #: inspector's display only, never the active layer's chain.
    scaleChanged = QtCore.Signal(str, int)

    #: ``(source_id, following)`` -- the "M" button. Default ON.
    followMasterToggled = QtCore.Signal(str, bool)

    #: ``(source_id,)`` -- this window was moved or resized. Where a window IS is part
    #: of what it remembers (``Settings.view_options["inspectors"]``), and Qt announces that only
    #: through the two events below. The signal carries no geometry: whoever persists it reads
    #: the window's CURRENT geometry at write time, which is the same discipline the scale index
    #: is written under (a value read at write time cannot be stale by the time it lands).
    geometryChanged = QtCore.Signal(str)

    def __init__(self, source_id: str, source_label: str, tree_node,
                 scale_reading: Callable[[int], str], parent=None):
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.WindowStaysOnTopHint,
        )
        self.source_id = source_id
        self.setWindowTitle(source_label)

        self.title_label = QtWidgets.QLabel(source_label)
        self.status_label = QtWidgets.QLabel(NO_RESULT_TEXT)
        #: What ``MainWindow`` last said about this window (:meth:`set_status`), kept apart from
        #: what is RENDERED into the label -- :meth:`_render_status` composes the two, so this
        #: window's own scale confession can be appended without either half clobbering the
        #: other on its next write.
        self._status_base = NO_RESULT_TEXT
        #: The last result handed to :meth:`show_result`, kept so this window's own scale control
        #: can re-index it (:meth:`redraw_at`) with no resolve, no compute and no second worker --
        #: sec 5b's live-update contract is that an inspector NEVER opens a compute path.
        self._result: dict | None = None
        #: The scale index the CANVAS is actually drawn at; ``None`` until a result lands. The
        #: control's index (``transport.slider.value()``) is a different number whenever the held
        #: result cannot show what was asked for, and keeping the two apart is what lets the
        #: window say so (:data:`SCALE_NOT_SHOWN_TEXT`) instead of quietly disagreeing with itself.
        self.picture_scale_idx: int | None = None
        #: Whether the picture currently on the canvas came from the layer the user is driving
        #: right now (the ``show_result``). ``False`` before anything has ever arrived, and
        #: again the moment a KEPT result is redrawn -- sec 10 A2's honest non-active state.
        self.showing_live = False
        self.status_label.setProperty("muted", "true")
        # Stacked, not side by side: the status line is a sentence, and a sentence sharing a row
        # with the title gets elided to nothing at any honest window width.
        strip = QtWidgets.QVBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(0)
        strip.addWidget(self.title_label)
        strip.addWidget(self.status_label)

        self.canvas = Canvas()

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["layer / product", "opacity"])
        self.tree.setMaximumHeight(_TREE_MAX_H)
        # The name column takes what is left over; without this the spin column claims half the
        # width and every device name elides away to a bare checkbox (seen in an early render).
        header = self.tree.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        #: Keyed rows only, by provenance row key -- a grouping (layer) row is in neither dict.
        self.rows: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self.opacity_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._row_order: list[str] = []
        raster_item = self._add_row(None, source_label, raster_key(source_id))
        for layer_node in tree_node.layers:
            self._add_layer_rows(raster_item, layer_node)
        self.tree.expandAll()
        # Connected AFTER every row's check state is seeded, the same order ``_LayerRow`` uses
        # for its own buttons: seeding before the signal is wired can never fire a spurious
        # toggle, and a window that announced N sub-layer changes just by opening would write
        # exactly that much noise into the persisted state.
        self.tree.itemChanged.connect(self._on_item_changed)

        #: True while :meth:`set_n_scales` is re-ranging the transport. ``Transport.set_n_scales``
        #: emits ``scaleChanged`` as it clamps its own position into the new range, and that echo
        #: is not a gesture the user made in THIS window -- announcing it would report a scale
        #: change on every result. ``MainWindow._sync_transport`` carries the same flag for the
        #: same reason; this is that guard, one level down.
        self._syncing = False
        #: A REMEMBERED scale index that has nowhere to go yet. The one-shot restore
        #: runs at ``MainWindow.load_field``'s tail -- before the first result, while the
        #: transport still knows exactly one scale -- so a saved index of 2 would be clamped to
        #: 0 and lost. Held here instead and applied by :meth:`set_n_scales`, the moment the
        #: range it names actually exists. Cleared on the first application: after that the
        #: slider is the user's, and a restore that kept re-asserting itself on every recompute
        #: would be fighting them.
        self._pending_scale_idx: int | None = None

        self.transport = Transport(1, scale_reading)
        self.transport.scaleChanged.connect(self._on_scale_changed)

        self.follow_button = QtWidgets.QToolButton()
        self.follow_button.setCheckable(True)
        self.follow_button.setText("M")
        self.follow_button.setToolTip("Follow the master transport's scale index")
        self.follow_button.setChecked(True)
        self.follow_button.toggled.connect(self._on_follow_toggled)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self.transport, 1)
        bottom.addWidget(self.follow_button)

        column = QtWidgets.QVBoxLayout(self)
        column.addLayout(strip)
        column.addWidget(self.canvas, 1)
        column.addWidget(self.tree)
        column.addLayout(bottom)

    # -- the sub-layer tree ---------------------------------------------------------------------
    def row_keys(self) -> list[str]:
        """Every keyed row, in the order the tree shows them -- the processing provenance."""
        return list(self._row_order)

    def _add_layer_rows(self, parent_item, layer_node) -> None:
        """One grouping row for the layer, its products beneath it in chain-step order, then the
        layers cut FROM it -- ``provenance_tree``'s own nesting, followed rather than re-decided.
        """
        item = QtWidgets.QTreeWidgetItem(parent_item, [layer_node.name])
        item.setData(0, _KEY_ROLE, None)
        for product in layer_node.products:
            self._add_row(item, product.device, product.key)
        for child in layer_node.children:
            self._add_layer_rows(item, child)

    def _add_row(self, parent_item, label: str, key: str) -> QtWidgets.QTreeWidgetItem:
        """One KEYED row: a checkbox in column 0, an opacity spin in column 1.

        Defaults are visible and fully opaque -- ``SubLayerState``'s own defaults
        (``model/inspector_state.py``), so a row the user has never touched shows as the chain
        produced it and persists as nothing at all.
        """
        item = QtWidgets.QTreeWidgetItem(
            self.tree if parent_item is None else parent_item, [label])
        item.setData(0, _KEY_ROLE, key)
        item.setCheckState(0, QtCore.Qt.CheckState.Checked)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(2)
        spin.setValue(1.0)
        spin.valueChanged.connect(
            lambda value, k=key: self.subLayerOpacityChanged.emit(
                self.source_id, k, float(value)))
        self.tree.setItemWidget(item, 1, spin)

        self.rows[key] = item
        self.opacity_spins[key] = spin
        self._row_order.append(key)
        return item

    def sublayer_states(self) -> dict[str, tuple[bool, float]]:
        """Every keyed row's ``(visible, opacity)``, in row order -- what the project persists.

        Read off the widgets rather than mirrored into a second dict, for :attr:`follow_master`'s
        reason: the checkbox and the spin ARE the state, and a copy of them is one more thing
        that can come to disagree with what the user can see.
        """
        return {key: (self.rows[key].checkState(0) == QtCore.Qt.CheckState.Checked,
                      float(self.opacity_spins[key].value()))
                for key in self._row_order}

    def set_sublayer_state(self, key: str, *, visible: bool, opacity: float) -> None:
        """Put one remembered row back the way it was left (the restore).

        SILENT: a restore is not a gesture. Announcing it would emit ``subLayerToggled`` /
        ``subLayerOpacityChanged`` as though the user had just made the change, and every
        restored window would report N sub-layer edits merely by opening -- writing exactly that
        much noise back into the state it was just restored FROM. Both objects are blocked,
        because two different senders reach the two signals: the tree's ``itemChanged`` for the
        checkbox, and the spin's own lambda (which emits from ``self``) for the opacity.

        A key naming no row here is DROPPED, not raised: chains change, and a remembered row
        whose chain step has since been removed is a preference that no longer applies, not a
        fault.
        """
        item = self.rows.get(key)
        spin = self.opacity_spins.get(key)
        if item is None or spin is None:
            return
        self.blockSignals(True)
        self.tree.blockSignals(True)
        try:
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if visible
                               else QtCore.Qt.CheckState.Unchecked)
            spin.setValue(float(opacity))
        finally:
            self.tree.blockSignals(False)
            self.blockSignals(False)

    def _on_item_changed(self, item, column: int) -> None:
        key = item.data(0, _KEY_ROLE)
        if key is None or column != 0:
            return
        self.subLayerToggled.emit(
            self.source_id, key, item.checkState(0) == QtCore.Qt.CheckState.Checked)

    # -- results (the live-update contract) ------------------------------------------------
    def show_result(self, renderable, *, active: bool) -> None:
        """Draw the newest result for this source on the hosted canvas.

        Handed a ``Renderable`` that has ALREADY reached the screen once (``MainWindow._apply``,
        "the ONE place a result reaches the screen", via its ``resolved`` signal): this method
        never resolves, never computes and never polls -- sec 5b's live-update contract in one
        sentence. The draw is deliberately only ``_apply``'s overlay branch: no hide gate (that
        switch belongs to the ACTIVE layer's own overlay, not to a source's inspector) and no
        pick chains (routing the canvas's gestures across N open inspectors into the ONE
        ``GroupPalette`` is the next slice, sec 5b's inventory).

        ``active`` says whether the layer that produced this result is the one the user is
        driving. It is RECORDED here (:attr:`showing_live`), never worded here: the status
        sentence is ``MainWindow``'s (see :data:`NO_RESULT_TEXT`), since only the window can tell
        whether a later result for some OTHER source has since made this one a kept picture, and
        two halves wording one state is exactly how they come to disagree about it.
        """
        self.showing_live = bool(active)
        result = renderable.result
        # The scale range travels WITH the result (sec 8 rule 5): a transport built at
        # ``Transport(1, ...)`` and left there would offer one scale forever, and a follower that
        # cannot reach the master's index is not following it. ``scales`` is a numpy ARRAY from
        # the real transform, so no truthiness test may touch it -- ``x or ()`` on an ndarray
        # raises (the trap ``MainWindow._on_finished`` already records).
        scales = result.get("scales")
        if scales is not None and len(scales):
            self.set_n_scales(len(scales))
        # Held so this window's own control can re-index it later (:meth:`redraw_at`), and so the
        # scale it is DRAWN at is a number the window knows rather than one it has to infer.
        self._result = result
        drawable = self._drawable(self.transport.slider.value())
        if drawable is not None:
            self.canvas.set_result(result, drawable[0])
            self.picture_scale_idx = drawable[1]
        self._render_status()

    def set_status(self, text: str, *, live: bool) -> None:
        """Write the persistent status strip and the flag behind it, together.

        Persistent, not a timed toast: what it reports (sec 10 A2: "this is the LAST result, its
        layer is not active") stays true until something changes it, and a message that fades
        leaves the window silently claiming to be live again.

        Both halves are written here because a picture stops being live WITHOUT this window
        hearing anything -- the user selects a layer over some other source and nothing new
        arrives here at all. ``MainWindow`` re-derives both for every open inspector whenever a
        result lands (``_refresh_inspector_statuses``), which is why this takes ``live`` rather
        than trying to infer it from the sentence.
        """
        self.showing_live = bool(live)
        self._status_base = text
        self._render_status()

    def _render_status(self) -> None:
        """Write the strip: ``MainWindow``'s sentence, plus this window's own scale confession
        when it has one (:data:`SCALE_NOT_SHOWN_TEXT`).

        Composed rather than concatenated at each call site, because the two halves are written
        by different owners at different moments -- a result landing re-words one, a scrub the
        other -- and either writing the label directly would erase the other's half.
        """
        note = self._scale_note()
        self.status_label.setText(f"{self._status_base} · {note}" if note else self._status_base)

    def _scale_note(self) -> str:
        """The confession, or ``""`` when the control and the picture name the same scale.

        Which sentence depends on :attr:`showing_live`, because the FIX does: a kept picture
        moves again once its own layer is selected and re-run, while a live one is already
        following the master and has no such gesture left (see :data:`SCALE_NOT_SHOWN_LIVE_TEXT`).
        """
        if self.picture_scale_idx is None:
            return ""                       # nothing drawn yet: NO_RESULT_TEXT already says so
        control = int(self.transport.slider.value())
        if control == self.picture_scale_idx:
            return ""
        template = SCALE_NOT_SHOWN_LIVE_TEXT if self.showing_live else SCALE_NOT_SHOWN_TEXT
        return template.format(control=control, picture=self.picture_scale_idx)

    # -- scale + master -------------------------------------------------------------------------
    @property
    def follow_master(self) -> bool:
        """Whether this inspector follows the master transport (default ON).

        Read straight off the button rather than mirrored into a second attribute: the button IS
        the state, and a copy of it is one more thing that can come to disagree with what the
        user can see."""
        return self.follow_button.isChecked()

    def set_scale_index(self, idx: int) -> None:
        """Adopt a scale index this window did NOT originate -- the master driving a follower.

        Routes through ``Transport.sync_to`` (:meth:`dynamix.shell.transport.Transport.sync_to`),
        which is deliberately SILENT: the caller has already acted on this index, so emitting
        would echo it straight back and re-run a change that has already been applied. The
        reading label moves with the slider, which is what makes a follower visibly follow.

        Clamped to THIS window's own range first: ``sync_to`` is unclamped on the receiving side --
        ``SweepClock.scrub`` assigns ``_index`` outright and the label is formatted from whatever
        index it is handed, while ``QSlider.setValue`` clamps -- so a master index past this
        source's own scale count would otherwise leave the slider at the end while the reading named
        a scale this window cannot show, and the next playback tick would advance from out of range.

        The redraw is attempted with the same clamped index: where the held result still carries
        the whole stack it is a free re-index, and where it does not :meth:`redraw_at` says so
        rather than moving a reading over a still picture.
        """
        idx = min(int(idx), self.transport.slider.maximum())
        self.transport.sync_to(idx)
        self.redraw_at(idx)

    def restore_scale_index(self, idx: int) -> None:
        """Adopt a REMEMBERED index, whose range may not have arrived yet.

        Not :meth:`set_scale_index`, which clamps against the range the window knows RIGHT NOW:
        the one-shot restore runs at ``MainWindow.load_field``'s tail, before the first result,
        where that range is a single scale and every remembered index would clamp to 0. So it is
        held (:attr:`_pending_scale_idx`) and applied by :meth:`set_n_scales` once the result
        that defines the range lands -- clamped THEN, against a range that is real.

        Applied immediately when the range is already known, which is the case for a window
        restored over a source whose result had already arrived.
        """
        self._pending_scale_idx = int(idx)
        if self.transport.slider.maximum() > 0:
            self._apply_pending_scale_idx()

    def _apply_pending_scale_idx(self) -> None:
        """Spend a held restore index, once. Cleared BEFORE applying, so the ``set_n_scales``
        that :meth:`set_scale_index` can reach on its way through cannot re-enter this."""
        if self._pending_scale_idx is None:
            return
        idx, self._pending_scale_idx = self._pending_scale_idx, None
        self.set_scale_index(idx)

    def redraw_at(self, idx: int) -> None:
        """Show the HELD result at scale ``idx`` -- no resolve, no compute, no second worker.

        This is the whole of what this window's own scale control can honestly do (an
        inspector never opens a compute path). It succeeds exactly when the held result still
        carries the stack -- ``scale_select`` bypassed or absent -- because then re-indexing IS
        what selecting a scale means. When the chain already reduced the stack to one layer
        there is no other scale in this window to draw, and the status strip says which scale is
        on screen and how to move it (:data:`SCALE_NOT_SHOWN_TEXT`).
        """
        drawable = self._drawable(idx)
        if drawable is not None and drawable[1] != self.picture_scale_idx:
            self.canvas.set_result(self._result, drawable[0])
            self.picture_scale_idx = drawable[1]
        self._render_status()

    def _drawable(self, want: int) -> tuple[int, int] | None:
        """``(row, scale)`` for the held result given a WANTED scale index, or ``None`` when
        nothing drawable is held.

        The two numbers are deliberately separate, and conflating them is an ``IndexError``:
        ``Canvas.set_result``'s second argument indexes ``result["extrema"]`` (a ROW), while
        ``_scale_idx`` names a scale of the ORIGINAL stack. After ``scale_select`` the list holds
        exactly one row -- row 0 -- and that row IS scale ``_scale_idx``, which is commonly not 0.

        So: a one-row result can only ever be drawn at row 0, and the scale it shows is its own
        ``_scale_idx`` whatever was asked for. A multi-row result is the full stack, where row
        and scale coincide and the wanted index is honoured, clamped.
        """
        result = self._result
        if result is None:
            return None
        layers = result.get("extrema")
        if not layers or "_shape" not in result:
            return None
        if len(layers) == 1:
            return 0, int(result.get("_scale_idx", 0))
        row = max(0, min(int(want), len(layers) - 1))
        return row, row

    def set_n_scales(self, n: int) -> None:
        """Re-range the scale control from the DATA (sec 8 rule 5), mirroring
        ``Transport.set_n_scales``.

        Silent for this window's own signal: see :attr:`_syncing`. The transport's own clamp
        still runs, so a slider sitting past the end of a shorter new stack lands on a scale that
        actually exists rather than on one that no longer does.
        """
        self._syncing = True
        try:
            self.transport.set_n_scales(int(n))
        finally:
            self._syncing = False
        # The range a remembered index was waiting for may have just arrived.
        self._apply_pending_scale_idx()

    def _on_scale_changed(self, idx: int) -> None:
        """THIS window's own transport moved.

        Announced, never acted on here: ``MainWindow`` deliberately does not route this into
        ``_params``/``layer.chain`` (the containment guarantee -- an inspector must never
        move the ACTIVE layer's chain behind the user's back), so what an inspector's own slider
        drives is this window's own reading and its persisted state.

        ``MainWindow`` routes this signal back into :meth:`redraw_at` for THIS window only
        (``_on_inspector_scale_changed``). The status re-render happens here regardless -- before
        the ``_syncing`` return, since a ``set_n_scales`` clamp moves the slider too -- so a bare
        window (this file's own tests, and any future host) still cannot end up showing a reading
        that contradicts its picture.
        """
        self._render_status()
        if self._syncing:
            return
        self.scaleChanged.emit(self.source_id, int(idx))

    def _on_follow_toggled(self, checked: bool) -> None:
        self.followMasterToggled.emit(self.source_id, bool(checked))

    # -- lifetime -------------------------------------------------------------------------------
    def moveEvent(self, event) -> None:
        """Announce the move: where the window is, is part of what it remembers."""
        super().moveEvent(event)
        self.geometryChanged.emit(self.source_id)

    def resizeEvent(self, event) -> None:
        """Announce the resize, for :meth:`moveEvent`'s reason -- ``geometry`` is one tuple and
        Qt reports its two halves through two different events."""
        super().resizeEvent(event)
        self.geometryChanged.emit(self.source_id)

    def closeEvent(self, event) -> None:
        """Announce the close. The layer list's toggle un-checks itself off this signal, because
        a window that is gone while its button still reads "open" is two opinions about one state.
"""
        self.closed.emit(self.source_id)
        super().closeEvent(event)
