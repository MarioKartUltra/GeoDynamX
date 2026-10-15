# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Plugin discovery: a user folder and an entry-point group; one bad file never stops the others."""
from __future__ import annotations

import hashlib
import textwrap

from dynamix.engine.cache import cache_key
from dynamix.model.device import DEVICES, get_device
from dynamix.model.plugins import discover, origin_of

GOOD = textwrap.dedent('''
    from dynamix.model.param import Param, ParamKind

    class Double:
        name = "plug_double"
        label = "Double"
        families = ("Labs",)
        params = (Param("gain", ParamKind.FLOAT, default=2.0, min=0.0, max=10.0),)
        consumes = (("Field",),)
        produces = frozenset({"Field"})

        def compute(self, field, params, progress=None):
            return {"values": field.values * params["gain"]}

        def cache_key(self, params):
            return ("plug_double", params["gain"])

    DEVICES = [Double()]
''')
BROKEN = "def oops(:\n"
BAD_CONTRACT = GOOD.replace('consumes = (("Field",),)', 'consumes = (("Fields",),)').replace("plug_double", "plug_bad")


def test_discovery_registers_the_good_file_and_reports_the_broken_one(tmp_path, clean_registry):
    (tmp_path / "double.py").write_text(GOOD)
    (tmp_path / "broken.py").write_text(BROKEN)
    results = discover(tmp_path, entry_group="geodynamix.tests.none")
    by_name = {r.path.rsplit("/", 1)[-1]: r for r in results}
    assert by_name["double.py"].error is None and "plug_double" in DEVICES
    assert "SyntaxError" in by_name["broken.py"].error and by_name["broken.py"].name == ""
    sha = hashlib.sha256((tmp_path / "double.py").read_bytes()).hexdigest()
    assert origin_of("plug_double") == f"user:{sha}" and origin_of("wtmm2d") == "builtin"


def test_a_plugin_naming_an_unknown_contract_is_refused(tmp_path, clean_registry):
    (tmp_path / "bad.py").write_text(BAD_CONTRACT)
    results = discover(tmp_path, entry_group="geodynamix.tests.none")
    assert len(results) == 1 and "Fields" in results[0].error and "plug_bad" not in DEVICES


def test_a_duplicate_id_is_an_error_and_the_first_stands(tmp_path, clean_registry):
    (tmp_path / "a.py").write_text(GOOD)
    (tmp_path / "b.py").write_text(GOOD)
    results = sorted(discover(tmp_path, entry_group="geodynamix.tests.none"), key=lambda r: r.path)
    assert results[0].error is None and "already registered" in results[1].error
    assert get_device("plug_double") is not None


def test_an_entry_point_registers_its_device(tmp_path, clean_registry, monkeypatch):
    import importlib.metadata as md

    class _EP:
        name, group, value = "double", "geodynamix.tests.ep", "plugmod:DEVICES"
        dist = type("D", (), {"name": "plugdist", "version": "1.2"})()

        def load(self):
            import types
            mod = types.ModuleType("plugmod")
            exec(GOOD, mod.__dict__)
            return mod.DEVICES

    monkeypatch.setattr(md, "entry_points",
                        lambda **kw: [_EP()] if kw.get("group") == "geodynamix.tests.ep" else [])
    results = discover(None, entry_group="geodynamix.tests.ep")
    assert [r.error for r in results] == [None]
    assert origin_of("plug_double") == "package:plugdist==1.2"


def test_the_cache_salt_leaves_builtin_keys_unchanged_and_separates_user_versions():
    base = cache_key("wtmm2d", "s", {"a": 1})
    assert cache_key("wtmm2d", "s", {"a": 1}, origin="builtin") == base
    assert cache_key("wtmm2d", "s", {"a": 1}, origin=None) == base
    assert cache_key("plug", "s", {}, origin="user:aa") != cache_key("plug", "s", {}, origin="user:bb")
