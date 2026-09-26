# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.browser -- the device browser.

Offscreen Qt (the runner sets ``QT_QPA_PLATFORM=offscreen``). Exercises the real registered
devices (``register_builtin_devices``), mirroring ``test_shell_chain_strip.py``'s
``registered_builtins`` pattern, so the category split is proven against the real registry
rather than a hand-rolled stand-in.
"""
from __future__ import annotations

import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model import DEVICES, is_transform


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


def test_top_level_items_are_the_four_categories(qtbot, registered_builtins):
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    titles = [browser.topLevelItem(i).text(0) for i in range(browser.topLevelItemCount())]
    assert titles == ["Transforms", "Filters", "Dev", "Racks"]


def test_every_registered_device_appears_exactly_once_under_the_right_category(
        qtbot, registered_builtins):
    from dynamix.shell.browser import DeviceBrowser, _DEV_PREFIX, _SHELL_PLACED, _SUPERSEDED

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    transforms = _top_level(browser, "Transforms")
    filters = _top_level(browser, "Filters")

    seen_transforms = [transforms.child(i).text(0) for i in range(transforms.childCount())]
    seen_filters = [filters.child(i).text(0) for i in range(filters.childCount())]

    # Exclude stub, superseded and shell-placed devices from the expected lists (all go to Dev)
    expected_transforms = sorted(
        name for name, d in DEVICES.items()
        if is_transform(d) and not name.startswith(_DEV_PREFIX) and name not in _SUPERSEDED
        and name not in _SHELL_PLACED
    )
    expected_filters = sorted(
        name for name, d in DEVICES.items()
        if not is_transform(d) and not name.startswith(_DEV_PREFIX)
        and name not in _SUPERSEDED
    )

    assert sorted(seen_transforms) == expected_transforms
    assert sorted(seen_filters) == expected_filters
    # exactly once: no duplicates, no cross-category leakage
    assert len(seen_transforms) == len(set(seen_transforms))
    assert len(seen_filters) == len(set(seen_filters))
    assert set(seen_transforms).isdisjoint(seen_filters)


def test_filter_row_mime_data_encodes_device_name(qtbot, registered_builtins):
    from dynamix.shell.browser import DEVICE_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    filters = _top_level(browser, "Filters")
    row = filters.child(0)

    mime = browser.mimeData([row])

    assert mime.hasFormat(DEVICE_MIME)
    assert bytes(mime.data(DEVICE_MIME).data()) == row.text(0).encode("utf-8")


def test_transform_row_mime_data_encodes_device_name(qtbot, registered_builtins):
    from dynamix.shell.browser import DEVICE_MIME, DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    transforms = _top_level(browser, "Transforms")
    row = transforms.child(0)

    mime = browser.mimeData([row])

    assert mime.hasFormat(DEVICE_MIME)
    assert bytes(mime.data(DEVICE_MIME).data()) == row.text(0).encode("utf-8")


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

    for title in ("Transforms", "Filters", "Dev", "Racks"):
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
    """Test (a): Dev holds exactly the stubs plus the superseded devices (registered for saved
    projects, out of the palette) plus the devices only the shell places (a derivative dataset's
    vector loader)."""
    from dynamix.shell.browser import DeviceBrowser, _DEV_PREFIX, _SHELL_PLACED, _SUPERSEDED

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    dev = _top_level(browser, "Dev")

    seen = [dev.child(i).text(0) for i in range(dev.childCount())]
    expected = sorted(name for name in DEVICES.keys()
                      if name.startswith(_DEV_PREFIX) or name in _SUPERSEDED
                      or name in _SHELL_PLACED)

    assert sorted(seen) == expected
    assert set(seen) == ({"stub_wavelet", "stub_holder", "stub_wedge"} | set(_SUPERSEDED)
                         | set(_SHELL_PLACED))


def test_dev_group_not_in_transforms_or_filters(qtbot, registered_builtins):
    """Test (a): stub devices are NOT under Transforms/Filters."""
    from dynamix.shell.browser import DeviceBrowser, _DEV_PREFIX

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    transforms = _top_level(browser, "Transforms")
    filters = _top_level(browser, "Filters")

    seen_transforms = [transforms.child(i).text(0) for i in range(transforms.childCount())]
    seen_filters = [filters.child(i).text(0) for i in range(filters.childCount())]

    stub_names = {name for name in DEVICES.keys() if name.startswith(_DEV_PREFIX)}
    assert set(seen_transforms).isdisjoint(stub_names)
    assert set(seen_filters).isdisjoint(stub_names)


def test_dev_group_starts_collapsed(qtbot, registered_builtins):
    """Test (b): Dev category starts collapsed while others start expanded."""
    from dynamix.shell.browser import DeviceBrowser

    browser = DeviceBrowser()
    qtbot.addWidget(browser)

    transforms = _top_level(browser, "Transforms")
    filters = _top_level(browser, "Filters")
    dev = _top_level(browser, "Dev")
    racks = _top_level(browser, "Racks")

    assert transforms.isExpanded()
    assert filters.isExpanded()
    assert not dev.isExpanded()
    assert racks.isExpanded()


def test_presets_exist_and_have_valid_devices_and_params(qtbot, registered_builtins):
    """Test (c): presets.PRESETS contains expected presets with valid devices and params."""
    from dynamix.model.presets import PRESETS
    from dynamix.model.device import get_device

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
    from dynamix.shell.main_window import MainWindow
    from dynamix.model.presets import PRESETS

    window = MainWindow()
    qtbot.addWidget(window)

    racks = _top_level(window.browser, "Racks")
    seen = [racks.child(i).text(0) for i in range(racks.childCount())]

    expected_names = {"WTMM standard"} | set(PRESETS.keys())
    assert set(seen) == expected_names


def test_superseded_devices_sit_under_dev_not_transforms(qtbot, registered_builtins):
    """holder_map stays REGISTERED (saved projects name it) but leaves the palette -- its
    per-method successors are what the Transforms category offers."""
    from dynamix.shell.browser import DeviceBrowser, _SUPERSEDED

    browser = DeviceBrowser()
    qtbot.addWidget(browser)
    transforms = _top_level(browser, "Transforms")
    dev = _top_level(browser, "Dev")
    transform_names = {transforms.child(i).text(0) for i in range(transforms.childCount())}
    dev_names = {dev.child(i).text(0) for i in range(dev.childCount())}
    for name in _SUPERSEDED:
        assert name in dev_names and name not in transform_names, name
    assert {"holder_measure", "holder_multiaffine"} <= transform_names
