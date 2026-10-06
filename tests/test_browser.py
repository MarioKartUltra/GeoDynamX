# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.browser -- the device browser.

Offscreen Qt (the runner sets ``QT_QPA_PLATFORM=offscreen``). Exercises the real registered
devices (``register_builtin_devices``), mirroring ``test_shell_chain_strip.py``'s
``registered_builtins`` pattern, so the family tree is proven against the real registry
rather than a hand-rolled stand-in.
"""
from __future__ import annotations

import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model import DEVICES, is_transform
from dynamix.model.device import get_device


@pytest.fixture
def registered_builtins(clean_registry):
    register_builtin_devices()
    return clean_registry


def _top_level(browser, title):
    for i in range(browser.topLevelItemCount()):
        item = browser.topLevelItem(i)
        if item.text(0) == title:
            return item
    raise AssertionError(f"no top-level item named {title!r}")


def _titles(browser):
    return [browser.topLevelItem(i).text(0) for i in range(browser.topLevelItemCount())]


def _families(browser):
    return [t for t in _titles(browser) if t not in ("Racks", "Dev")]


def _ids(header):
    from dynamix.shell.browser import _ID_ROLE

    return [header.child(i).data(0, _ID_ROLE) for i in range(header.childCount())]


def _is_dev(name):
    from dynamix.model.families import families_of
    from dynamix.shell.browser import _DEV_PREFIX, _SHELL_PLACED, _SUPERSEDED

    return (name.startswith(_DEV_PREFIX) or name in _SUPERSEDED or name in _SHELL_PLACED
            or not families_of(DEVICES[name]))


def test_top_level_items_are_the_families_then_racks_then_dev(qtbot, registered_builtins):
    from dynamix.model.families import FAMILY_ORDER
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    assert _titles(browser) == [f for f in FAMILY_ORDER if f != "Other"] + ["Racks", "Dev"]


def test_every_palette_device_is_listed_under_each_of_its_families(qtbot, registered_builtins):
    from dynamix.model.families import FAMILY_ORDER, families_of
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    for name, device in DEVICES.items():
        if _is_dev(name):
            continue
        declared = families_of(device)
        assert set(declared) <= set(FAMILY_ORDER) - {"Other"}, name
        listed = {t for t in _families(browser) if name in _ids(_top_level(browser, t))}
        assert listed == set(declared), name
    for title in _families(browser):
        ids = _ids(_top_level(browser, title))
        assert len(ids) == len(set(ids)), title          # once per family
    assert {"Edge detection", "Regularity", "Reconstruction"} <= {
        t for t in _families(browser) if "mz_edges" in _ids(_top_level(browser, t))}


def test_a_device_row_shows_its_label_with_the_id_muted_beside_it(qtbot, registered_builtins):
    from dynamix.model.families import display_name, label_of
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    row = browser._rows["pm_edges"][0]
    assert row.text(0) == label_of(get_device("pm_edges")) != "pm_edges"
    assert row.text(1) == "pm_edges"
    assert row.foreground(1).color() == browser._muted.color()
    assert row.toolTip(0) == display_name(get_device("pm_edges"))


def test_filter_row_mime_data_encodes_device_name(qtbot, registered_builtins):
    from dynamix.shell.browser import _ID_ROLE, DEVICE_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    row = _top_level(browser, "Selection: V-chains").child(0)
    name = row.data(0, _ID_ROLE)
    assert name in DEVICES and not is_transform(DEVICES[name])

    mime = browser.mimeData([row])

    assert mime.hasFormat(DEVICE_MIME)
    assert bytes(mime.data(DEVICE_MIME).data()) == name.encode("utf-8")


def test_transform_row_mime_data_encodes_device_name(qtbot, registered_builtins):
    """The payload is the device id, never the label the row shows."""
    from dynamix.shell.browser import DEVICE_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    row = browser._rows["wtmm2d"][0]

    mime = browser.mimeData([row])

    assert mime.hasFormat(DEVICE_MIME)
    assert bytes(mime.data(DEVICE_MIME).data()) == b"wtmm2d"
    assert row.text(0) != "wtmm2d"


def test_preset_row_mime_data_encodes_preset_name(qtbot, registered_builtins):
    from dynamix.shell.browser import PRESET_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    browser.set_presets({"WTMM standard": (("wtmm2d", {}), ("chain_topology", {}))})

    racks = _top_level(browser, "Racks")
    assert racks.childCount() == 1
    row = racks.child(0)
    assert row.text(0) == "WTMM standard"

    mime = browser.mimeData([row])

    assert mime.hasFormat(PRESET_MIME)
    assert bytes(mime.data(PRESET_MIME).data()) == b"WTMM standard"


def test_preset_row_does_not_carry_the_device_mime_type(qtbot, registered_builtins):
    from dynamix.shell.browser import DEVICE_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    browser.set_presets({"WTMM standard": (("wtmm2d", {}),)})
    racks = _top_level(browser, "Racks")
    row = racks.child(0)

    mime = browser.mimeData([row])

    assert not mime.hasFormat(DEVICE_MIME)


def test_category_headers_yield_no_drag_payload(qtbot, registered_builtins):
    from dynamix.shell.browser import DEVICE_MIME, PRESET_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    browser.set_presets({"WTMM standard": (("wtmm2d", {}),)})

    for title in _titles(browser):
        header = _top_level(browser, title)
        mime = browser.mimeData([header])
        assert not mime.hasFormat(DEVICE_MIME)
        assert not mime.hasFormat(PRESET_MIME)


def test_drag_is_enabled(qtbot, registered_builtins):
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    assert browser.dragEnabled() is True


def test_dev_group_contains_stub_and_superseded_devices(qtbot, registered_builtins):
    """Dev holds exactly the stubs, the superseded devices (registered for saved projects, out of
    the palette) and the devices only the shell places; every builtin device declaring an empty
    ``families`` is one of these."""
    from dynamix.model.families import families_of
    from dynamix.shell.browser import _SHELL_PLACED, _SUPERSEDED, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    seen = _ids(_top_level(browser, "Dev"))

    assert seen == [name for name in DEVICES if _is_dev(name)]
    assert set(seen) == ({"stub_wavelet", "stub_holder", "stub_wedge"} | set(_SUPERSEDED)
                         | set(_SHELL_PLACED))
    assert {name for name, d in DEVICES.items() if not families_of(d)} <= set(seen)
    assert "group_paint" in seen


def test_a_device_declaring_no_families_sits_under_dev(qtbot, registered_builtins):
    """An empty ``families`` sends a device to Dev without a table entry."""
    from dynamix.model.device import register_device
    from dynamix.shell.browser import DeviceBrowser

    class Retired:
        name = "retired_tool"
        params = ()
        families = ()

        def apply(self, result, params):
            return result

    register_device(Retired())
    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    assert "retired_tool" in _ids(_top_level(browser, "Dev"))
    for title in _families(browser):
        assert "retired_tool" not in _ids(_top_level(browser, title)), title


def test_dev_devices_are_listed_under_no_family(qtbot, registered_builtins):
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    dev = {name for name in DEVICES if _is_dev(name)}
    for title in _families(browser):
        assert dev.isdisjoint(_ids(_top_level(browser, title))), title


def test_dev_group_starts_collapsed(qtbot, registered_builtins):
    """Dev starts collapsed; every family and Racks start expanded."""
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    for title in _titles(browser):
        assert _top_level(browser, title).isExpanded() is (title != "Dev"), title


def test_presets_exist_and_have_valid_devices_and_params(qtbot, registered_builtins):
    """Test (c): presets.PRESETS contains expected presets with valid devices and params."""
    from dynamix.model.presets import PRESETS

    expected_names = {"M–Z edges", "WTMM + Hölder", "CDF edges", "PM edges", "Wavelet skeleton"}
    assert set(PRESETS.keys()) == expected_names

    for preset_name, steps in PRESETS.items():
        assert isinstance(steps, tuple), f"Preset {preset_name!r} must be a tuple"
        assert len(steps) > 0, f"Preset {preset_name!r} must have at least one step"

        for device_name, params in steps:
            assert device_name in DEVICES, (
                f"Preset {preset_name!r} references device {device_name!r} not in DEVICES"
            )
            device = get_device(device_name)
            assert device is not None, f"Device {device_name!r} not found"

            param_names = {p.name for p in device.params}
            for param_name in params.keys():
                assert param_name in param_names, (
                    f"Preset {preset_name!r} device {device_name!r}: "
                    f"param {param_name!r} not in device's schema {param_names}"
                )


def test_window_injects_all_presets_into_browser(qtbot, registered_builtins):
    """Test (d): the window injects all three presets into the browser."""
    from dynamix.model.presets import PRESETS
    from dynamix.shell.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    racks = _top_level(window.browser, "Racks")
    seen = [racks.child(i).text(0) for i in range(racks.childCount())]

    expected_names = {"WTMM standard"} | set(PRESETS.keys())
    assert set(seen) == expected_names


def test_superseded_devices_sit_under_dev_only(qtbot, registered_builtins):
    """holder_map stays REGISTERED (saved projects name it) but leaves the palette -- its
    per-method successors are what the Regularity family offers."""
    from dynamix.shell.browser import _SUPERSEDED, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    dev_names = set(_ids(_top_level(browser, "Dev")))
    for name in _SUPERSEDED:
        assert name in dev_names, name
        for title in _families(browser):
            assert name not in _ids(_top_level(browser, title)), (name, title)
    assert {"holder_measure", "holder_multiaffine"} <= set(
        _ids(_top_level(browser, "Regularity")))
