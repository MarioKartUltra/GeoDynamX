# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Devloop facade for ``dynamix.shell.arrangement.view`` (see ``devloop.py``'s module docstring).

``devloop``'s reload watcher is entirely flat: ``_MODULE_ORDER``/``_DEPENDS_ON`` list module
STEMS, and ``modules_to_reload`` builds reload targets as ``f"dynamix.shell.{stem}"`` -- a scheme
that cannot express a subpackage module like ``dynamix.shell.arrangement.view``. Rather than teach
that flat scheme a second shape for one package, this thin top-level module stands in for it:
``main_window.py`` imports ``ArrangementView`` from HERE (never straight from the subpackage), so
devloop has a real flat name -- ``arrangement_facade`` -- to reload and re-wire into
``main_window``'s dependents.

Known limitation, noted rather than solved here ("register a top-level facade
module instead and note it"): devloop's mtime scan (``_scan_mtimes``, ``rglob``) DOES see
``arrangement/view.py`` change (stem ``"view"``), but ``"view"`` is not in ``_MODULE_ORDER``, so
that change alone triggers no reload -- only touching THIS file (or something already in the
reload graph) does today. Teaching the scan to fold a subpackage's leaf stems into this facade's
identity is a separate, later change, not a Task-1 scaffolding concern.
"""
from __future__ import annotations

from dynamix.shell.arrangement.view import ArrangementView  # noqa: F401
