# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The transport's slider keeps its length when the scale reading beside it changes width. A
slider that resized under a held mouse would map the same cursor position to another value and
flip back and forth between two scales."""
from __future__ import annotations


def test_the_slider_keeps_its_length_whatever_the_reading_says(qtbot):
    from dynamix.shell.transport import Transport

    readings = {0: "a = 1 px", 1: "a = 4.0 px ≈ 9 m", 2: "a = 8.0 px ≈ 18.25 m EPSG:32613"}
    t = Transport(3, lambda i: readings[i])
    qtbot.addWidget(t)
    t.resize(600, 40)
    t.show()
    qtbot.waitExposed(t)
    widths = []
    for i in (0, 2, 1, 2, 0, 1):
        t.slider.setValue(i)
        qtbot.wait(20)                               # let the layout settle
        widths.append(t.slider.width())
    assert len(set(widths[1:])) == 1, widths
