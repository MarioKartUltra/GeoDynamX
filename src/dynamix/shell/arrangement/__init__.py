# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dynamix.shell.arrangement -- the arrangement (session/arrangement Tab-flip) center-zone view.

Qt/pyvista only, per the project's shell boundary rule. Every pyvista touch is behind
:func:`dynamix.shell.arrangement.view._import_pyvista`, so importing this package never imports
pyvista itself -- only *using* the view (Tab) does, and only if it is installed
(``pip install dynamix[viz]``).
"""
