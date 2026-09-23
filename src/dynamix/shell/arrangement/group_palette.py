# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""GroupPalette: the arrangement view's group-designation panel (the design "Group designation": "ported ``selection.py`` screen-space picking -- click selects a chain,
shift-click adds; a group palette (create/name/color). Membership accumulates as arrangement
view-state while exploring.").

Pure Qt (no pyvista import), mirroring ``mask_row.py``'s own dependency-group-decoupling reasoning:
``dynamix[viz]`` (pyvista/pyvistaqt) and ``dynamix[gui]`` (PySide6/pyqtgraph) are separate
optional-dependency groups (``pyproject.toml``), and this widget only ever needs the latter.

**Gesture semantics -- this task's own reading of the spec's terse "click selects... shift-click
adds", pinned here since the design does not spell out the exact state machine:**

- A "pick" arrives from ``ArrangementView`` as ``(layer_id, chain_index) | None`` -- the result of
  ``Scene.pick`` -- plus whether Shift was held, via :meth:`add_pick`.
- **Shift-click ADDS** the picked chain to the CURRENT selection (a working multi-select buffer,
  distinct from any one group's own committed membership) -- the selection only ever grows on a
  shift-click.
- **Plain click REPLACES** the current selection with just the newly-picked chain -- or, on a
  miss, clears it to empty ("replacing" with nothing is still a replace, and gives the obvious way
  to deselect by clicking empty space). A shift-click MISS, by contrast, is a pure no-op on the
  selection: there is nothing new to add, and "add" never implied "also clear".
- Whichever chain was actually picked (on either gesture, but never on a miss -- nothing was
  picked) is ALSO added into the ACTIVE group's own membership, if one exists. This is a UNION,
  never a replace: a plain click's selection-reset only changes what is currently highlighted
  (:meth:`selection`, which ``ArrangementView`` forwards straight into ``Scene.set_selection`` for
  the white pick highlight) -- it never retracts a chain a group already picked up on an earlier
  click. The design's own words for this are "membership accumulates".
- With no active group (nothing created yet, via :meth:`new_group`), picking still updates the
  selection/highlight -- it just has nowhere to assign membership to yet.

**Colors.** A new group auto-assigns the next color from :data:`GROUP_COLORS` -- the Okabe & Ito
(2008) colorblind-safe categorical palette (the field-standard choice for "which bucket does this
belong to" coding), minus its 8th, black entry (not usable as a chain tint against an arbitrary
background) -- 7 colors, cycling with ``% len(GROUP_COLORS)`` past the 7th group.

**Commit boundary.** This widget only ever holds PRE-COMMIT, arrangement-view-state membership --
nothing here writes ``layer.tags`` or touches a cached result. That is the job entirely, fed
by :attr:`membershipChanged`'s payload.

**Batch ops.** :meth:`apply_picks` generalizes the
gesture above to a WHOLE LIST of picks at once (a box or lasso drag enclosing several chains in
one gesture, rather than one pick at a time) and to three explicit ops instead of the implicit
"replace-then-add" the click gesture hard-codes: ``"add"`` (union into the selection, EQSelect's
own box/lasso default), ``"subtract"`` (⌥-drag) and ``"replace"`` (a future gesture -- e.g. a
transect swath -- that means "this IS now the selection", not "on top of it"). :meth:`add_pick`
is now a thin wrapper: it keeps its own exact pre-existing miss/replace bookkeeping (see the
gesture semantics above -- a plain-click miss still clears the selection with no batch call at
all) and delegates the actual hit to ``apply_picks([picked], "add")``.

**Membership doctrine for ``subtract`` -- read this before assuming symmetry with ``add``.**
Selection (:meth:`selection`, the live highlight buffer) and an active group's MEMBERSHIP
(:meth:`groups`) are two different things (see "Gesture semantics" above: "membership
accumulates"). ``add``/``replace`` touch both -- a chain that was just picked, by any means,
still gets unioned into the active group's membership, exactly as a plain/shift click always
has. ``subtract`` touches ONLY the selection buffer. Un-picking a chain from the on-screen
highlight is not the same claim as "this chain was never a member of the group" -- membership is
a record of what the user has ever swept up, and an ⌥-drag over the highlight is a "stop
showing me that" gesture, not an "undo" of an earlier commitment. A group's membership only ever
shrinks through the group-management operations (not built here).
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

#: Okabe & Ito (2008), "Color Universal Design" -- the standard colorblind-safe categorical
#: palette, minus pure black (reserved, not a usable chain tint). Cycled with modulo past the 7th
#: group -- see the module docstring's "Colors" section.
GROUP_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 159, 0),      # orange
    (86, 180, 233),      # sky blue
    (0, 158, 115),       # bluish green
    (240, 228, 66),      # yellow
    (0, 114, 178),       # blue
    (213, 94, 0),        # vermillion
    (204, 121, 167),     # reddish purple
)

_NAME_ROLE = QtCore.Qt.UserRole


class GroupPalette(QtWidgets.QWidget):
    """Group list (name + color swatch + member count), a "New group" button, active-group
    selection (click a row), and the pick-gesture state machine (see the module docstring).

    Emits :attr:`membershipChanged` -- ``{name: {"chains": [(layer_id, chain_index), ...],
    "color": [r, g, b]}}`` -- on every :meth:`add_pick` call, whatever it changed (even a no-op
    miss), so a listener (``ArrangementView``, wiring this into ``Scene.set_group_preview``/
    ``Scene.set_selection``) always sees the palette's current, confirmed state.
    """

    membershipChanged = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._new_button = QtWidgets.QPushButton("New group")
        self._new_button.clicked.connect(lambda: self.new_group())
        layout.addWidget(self._new_button)

        self._list = QtWidgets.QListWidget()
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self._list)

        #: name -> {"chains": set[(layer_id, chain_index)], "color": [r, g, b]} -- the
        #: authoritative state; :meth:`groups` is a read-only snapshot of this.
        self._groups: dict[str, dict] = {}
        self._rows: dict[str, QtWidgets.QListWidgetItem] = {}
        self._active: str | None = None
        self._selection: set[tuple[int, int]] = set()

    # -- group management -------------------------------------------------------------------

    def new_group(self, name: str | None = None) -> str:
        """Create a group -- auto-named ``"Group N"`` (the first unused N) if ``name`` is
        omitted -- auto-colored from :data:`GROUP_COLORS`, and make it the active group. Raises
        ``ValueError`` for a name already in use: silently overwriting an existing group's
        membership would be a data-loss bug, not a convenience."""
        if name is None:
            name = self._next_auto_name()
        elif name in self._groups:
            raise ValueError(f"group {name!r} already exists")
        color = list(GROUP_COLORS[len(self._groups) % len(GROUP_COLORS)])
        self._groups[name] = {"chains": set(), "color": color}

        item = QtWidgets.QListWidgetItem(self._row_text(name))
        item.setData(_NAME_ROLE, name)
        item.setIcon(self._swatch_icon(color))
        self._list.addItem(item)
        self._rows[name] = item
        self._list.setCurrentItem(item)      # -> _on_current_item_changed -> self._active = name
        return name

    def set_active_group(self, name: str) -> None:
        """Programmatic equivalent of clicking ``name``'s row. Raises ``KeyError`` for an unknown
        name -- there is no silent "do nothing" reading of asking to activate a group that was
        never created."""
        self._list.setCurrentItem(self._rows[name])

    @property
    def active_group(self) -> str | None:
        return self._active

    def groups(self) -> dict:
        """A read snapshot -- ``{name: {"chains": [(layer_id, chain_index), ...], "color": [r, g,
        b]}}``, chains sorted for determinism -- the exact :attr:`membershipChanged` payload
        shape."""
        return {name: {"chains": sorted(spec["chains"]), "color": list(spec["color"])}
                for name, spec in self._groups.items()}

    def selection(self) -> set[tuple[int, int]]:
        """The current pick highlight -- what ``ArrangementView`` forwards to
        ``Scene.set_selection``. Returns a copy; the caller cannot mutate this palette's own state
        through it."""
        return set(self._selection)

    # -- the pick gesture --------------------------------------------------------------------

    def add_pick(self, picked, shift: bool) -> None:
        """One click's worth of state update. ``picked`` is ``Scene.pick``'s own return value,
        ``(layer_id, chain_index) | None``; ``shift`` is whether Shift was held. See the module
        docstring's "Gesture semantics" section for exactly what this does and why.

        The replace-on-a-plain-click and no-op-on-a-shift-miss bookkeeping stays HERE (it is
        about ``shift``, a concept :meth:`apply_picks`'s ``op`` does not have) -- a genuine hit
        delegates to :meth:`apply_picks` for the "add this pick, union it into the active
        group" half, so the two gestures (one pick vs. a batch) never drift onto two different
        implementations of that half."""
        if not shift:
            self._selection = set()
        if picked is None:
            self.membershipChanged.emit(self.groups())
            return
        self.apply_picks([picked], "add")

    def apply_picks(self, picks, op: str) -> None:
        """Batch selection-buffer update over a WHOLE LIST of ``(layer_id, chain_index)`` picks
        at once -- the box/lasso gestures' own entry point. ``op`` is one of ``"add"`` (union, the box/lasso default), ``"subtract"``
        (⌥-drag) or ``"replace"`` (a future whole-selection-becomes-this gesture); an unknown op
        raises ``ValueError`` -- the same "no silent no-op on a bad argument" contract
        :meth:`new_group`/:meth:`set_active_group` already keep.

        Active-group membership is unioned in for ``"add"``/``"replace"`` (a pick is a pick,
        regardless of which op moved it into the selection) and left UNTOUCHED for
        ``"subtract"`` -- see the module docstring's "Membership doctrine" section for why the
        two are not symmetric. Always emits :attr:`membershipChanged`, matching :meth:`add_pick`
        and :meth:`prune_layer`'s own "always emits" convention -- a listener's view stays in
        sync even when ``picks`` is empty (a degenerate box/lasso drag that enclosed nothing).
        """
        if op not in ("add", "subtract", "replace"):
            raise ValueError(f"unknown op {op!r}")
        keys = {(int(p[0]), int(p[1])) for p in picks}
        if op == "replace":
            self._selection = set(keys)
        elif op == "add":
            self._selection |= keys
        else:                                              # "subtract"
            self._selection -= keys
        if op != "subtract" and self._active is not None:
            spec = self._groups[self._active]
            newly = keys - spec["chains"]
            if newly:
                spec["chains"] |= newly
                self._rows[self._active].setText(self._row_text(self._active))
        self.membershipChanged.emit(self.groups())

    def prune_layer(self, layer_id: int) -> None:
        """Drop every ``(layer_id, *)`` entry from the current selection AND from every group's
        membership (the staleness guard). Driven by
        ``Scene.set_layers``'s own pruned-``layer_id`` report: a background recompute landing on
        ``layer_id`` produced a NEW chains list, and any chain index this palette is still holding
        for it may now name the wrong chain, or one that no longer exists at all.

        Always emits :attr:`membershipChanged` (even when nothing was actually dropped, matching
        :meth:`add_pick`'s own "always emits" convention) so a listener's view stays in sync
        without having to separately track whether this call had any effect."""
        layer_id = int(layer_id)
        self._selection = {key for key in self._selection if key[0] != layer_id}
        for name, spec in self._groups.items():
            before = len(spec["chains"])
            spec["chains"] = {key for key in spec["chains"] if key[0] != layer_id}
            if len(spec["chains"]) != before:
                self._rows[name].setText(self._row_text(name))
        self.membershipChanged.emit(self.groups())

    # -- internals -----------------------------------------------------------------------------

    def _next_auto_name(self) -> str:
        n = 1
        while f"Group {n}" in self._groups:
            n += 1
        return f"Group {n}"

    def _row_text(self, name: str) -> str:
        return f"{name} ({len(self._groups[name]['chains'])})"

    @staticmethod
    def _swatch_icon(color) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtGui.QColor(*color))
        return QtGui.QIcon(pixmap)

    def _on_current_item_changed(self, current, previous) -> None:
        self._active = current.data(_NAME_ROLE) if current is not None else None
