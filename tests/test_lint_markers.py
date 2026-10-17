# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The marker lint's own tests: a file written to the rules yields no finding; each broken rule
yields its finding with the rule id and the line; the checker exits 0 in every case; a file under
tests/ yields nothing."""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "lint_markers.py"
spec = importlib.util.spec_from_file_location("lint_markers", SCRIPT)
lint = importlib.util.module_from_spec(spec)
sys.modules["lint_markers"] = lint          # a dataclass under __future__ annotations looks itself up here
spec.loader.exec_module(lint)

FACE, SAD, HEAD = lint.FACE, lint.SAD, lint.HEAD
GOOD = "\n".join([
    "# SPDX-License-Identifier: GPL-2.0-or-later",
    "# Copyright (C) 2026 Abraham Joseph Okayli Masaryk",
    f"# {HEAD} Extrema chaining across scales {HEAD}",
    '"""Docstring."""',
    "import numpy as np",
    "",
    f"# {FACE * 10} Extrema chaining",
    "",
    "",
    "def chain(x):",
    f"    # {FACE} refine chains",
    f"    # {SAD} TODO: validate bias at small scales",
    "    return np.asarray(x)",
    "",
    "",
    "# ♪♫♪♫♪♫♪♫♪♫♪♫♪♫Smoother registry♪♫♪♫♪♫♪♫♪♫♪♫♪♫",
    "",
    "",
    "def smoother():",
    f"    # {FACE} build the q-Gaussian filter",
    "    return None",
    "",
])


def _with(line_no: int, text: str) -> str:
    lines = GOOD.splitlines()
    lines[line_no - 1] = text
    return "\n".join(lines) + "\n"


def test_a_file_written_to_the_rules_yields_no_finding():
    assert lint.check_text(GOOD, "good.py") == []


@pytest.mark.parametrize("line_no, text, rule", [
    (7, f"# {FACE * 11} Extrema chaining", "S1"),
    (11, "    # ----------------------------------", "B1"),
    (11, "    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "B1"),
    (11, "    # done ✓", "B2"),
    (12, f"    # {SAD} validate bias at small scales", "T1"),
    (16, "# ♪♫♪♫♪♫♪♫♪♫♪♫Smoother registry♪♫♪♫♪♫♪♫♪♫♪♫", "W1"),
    (11, "    # ♪ refine chains", "W2"),
])
def test_each_broken_rule_yields_its_finding_at_its_line(line_no, text, rule):
    findings = lint.check_text(_with(line_no, text), "broken.py")
    assert [(f.rule, f.line) for f in findings] == [(rule, line_no)], findings


def test_a_missing_second_or_misplaced_header_is_reported():
    assert [f.rule for f in lint.check_text(_with(3, "# no header here"), "h.py")] == ["H1"]
    two = GOOD.replace('"""Docstring."""', f'"""Docstring."""\n# {HEAD} Again {HEAD}')
    assert [f.rule for f in lint.check_text(two, "h.py")] == ["H2"]
    late = GOOD.replace(f"# {HEAD} Extrema chaining across scales {HEAD}",
                        f"# a stray comment\n# {HEAD} Extrema chaining across scales {HEAD}")
    assert [f.rule for f in lint.check_text(late, "h.py")] == ["H3"]


def test_tests_and_verbatim_copies_are_never_checked(tmp_path):
    assert lint.check_file(ROOT / "tests" / "test_lint_markers.py") == []
    assert lint.check_file(ROOT / "src" / "dynamix" / "core" / "selection.py") == []
    assert lint.excluded(ROOT / "src" / "dynamix" / "_vendor" / "wtmm" / "cwt.py")
    assert not lint.excluded(ROOT / "src" / "dynamix" / "core" / "chaining.py")


def test_the_checker_exits_zero_with_findings_and_writes_the_summary(tmp_path, monkeypatch):
    bad = tmp_path / "bad.py"
    bad.write_text(_with(11, "    # ----------------------------------"), encoding="utf-8")
    summary = tmp_path / "summary.md"
    env = {"GITHUB_STEP_SUMMARY": str(summary), "PATH": "/usr/bin:/bin"}
    run = subprocess.run([sys.executable, str(SCRIPT), str(bad)], capture_output=True, text=True,
                         env=env)
    assert run.returncode == 0 and "B1" in run.stdout and "exit 0 by design" in run.stdout
    assert "Marker lint: 1 finding(s)" in summary.read_text(encoding="utf-8")
    assert subprocess.run([sys.executable, str(SCRIPT), str(bad)], capture_output=True).returncode == 0
