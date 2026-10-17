# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""``python -m dynamix.new_device``: a device file, its golden test and a README from one command."""
from __future__ import annotations

import subprocess
import sys

from dynamix.model.device import DEVICES, validate_params
from dynamix.model.families import families_of
from dynamix.model.plugins import discover
from dynamix.new_device import main


def test_a_generated_transform_registers_and_its_golden_test_records_then_compares(tmp_path, clean_registry):
    assert main(["plug_sum", "--kind", "transform", "--consumes", "Field", "--produces", "HMap",
                 "--dir", str(tmp_path)]) == 0
    assert (tmp_path / "plug_sum.py").exists() and (tmp_path / "test_plug_sum.py").exists()
    assert (tmp_path / "golden" / "plug_sum" / "README.md").exists()
    results = discover(tmp_path, entry_group="geodynamix.tests.none")
    assert [r.error for r in results] == [None] and "plug_sum" in DEVICES
    device = DEVICES["plug_sum"]
    assert families_of(device) == ("Labs",) and device.produces == frozenset({"HMap"})
    assert validate_params(device, {})["gain"] == 1.0
    for expect in ("records", "compares"):
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                              str(tmp_path / "test_plug_sum.py")], capture_output=True, text=True)
        assert run.returncode == 0, (expect, run.stdout[-800:], run.stderr[-800:])
    assert (tmp_path / "golden" / "plug_sum" / "output.npz").exists()


def test_a_generated_filter_has_an_apply_body_and_an_unknown_contract_is_refused(tmp_path, clean_registry, capsys):
    assert main(["plug_keep", "--kind", "filter", "--dir", str(tmp_path)]) == 0
    text = (tmp_path / "plug_keep.py").read_text(encoding="utf-8")
    assert "def apply(self, extrema, params)" in text and "def compute" not in text
    assert 'consumes = (("ExtremaStack",),)' in text and "produces = frozenset()" in text
    assert main(["plug_bad", "--consumes", "Fields", "--dir", str(tmp_path)]) == 2
    assert "Fields" in capsys.readouterr().err
    assert main(["plug_keep", "--kind", "filter", "--dir", str(tmp_path)]) == 2      # never overwrites
    assert main(["plug_pts", "--kind", "filter", "--consumes", "Points", "--dir", str(tmp_path)]) == 0
    assert 'consumes = (("Points",),)' in (tmp_path / "plug_pts.py").read_text(encoding="utf-8")   # as given


def test_a_leftover_test_or_golden_folder_refuses_the_id(tmp_path, clean_registry, capsys):
    (tmp_path / "test_plug_old.py").write_text("", encoding="utf-8")
    assert main(["plug_old", "--dir", str(tmp_path)]) == 2 and "test_plug_old.py" in capsys.readouterr().err
    (tmp_path / "golden" / "plug_gold").mkdir(parents=True)
    assert main(["plug_gold", "--dir", str(tmp_path)]) == 2 and "plug_gold" in capsys.readouterr().err
    assert not (tmp_path / "plug_old.py").exists() and not (tmp_path / "plug_gold.py").exists()
