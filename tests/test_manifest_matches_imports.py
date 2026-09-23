# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every third-party module dynamix imports must be declared in pyproject.toml.

This is the test EQSelect lacked. Its requirements.txt declares numpy, pandas, pyarrow, scipy,
matplotlib, pyvista and PySide6 while its own wtmm_backend imports numba, pyfftw, mlx and the wtmm
packages — a manifest that silently drifted away from the code until a fresh checkout failed to
run. Deriving the truth from the AST makes that class of drift impossible to introduce quietly.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PYPROJECT = REPO / "pyproject.toml"

#: Deliberately undeclared, with the reason. These are proper packages with their own pyproject and
#: dependencies, shared across Creep / EQSelect / DynamiX, installed editable from a local checkout.
#: They are kept OUT of pyproject.toml on purpose so a private research repo's URL does not sit in
#: DynamiX's metadata. Both are lazily imported, so the core still installs with numpy alone.
UNDECLARABLE = {
    "wtmm": "documented local install: pip install -e ~/projects/Creep/wavelet/wtmm",
    "wtmm_ebsd": "documented local install: pip install -e ~/projects/Creep/wavelet/wtmm_ebsd",
}

#: Import name -> distribution name, where they differ.
IMPORT_TO_DIST = {"PIL": "pillow"}


def _third_party_imports() -> dict[str, set[str]]:
    """Top-level third-party module names imported anywhere under src/, mapped to the files."""
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name and name not in stdlib and name != "dynamix":
                    found.setdefault(name, set()).add(path.name)
    return found


def _declared_distributions() -> set[str]:
    """Every distribution named in pyproject, base plus all extras, normalised to lowercase."""
    cfg = tomllib.loads(PYPROJECT.read_text())["project"]
    specs = list(cfg.get("dependencies", []))
    for extra in cfg.get("optional-dependencies", {}).values():
        specs.extend(extra)
    out = set()
    for spec in specs:
        # strip version constraints / markers / extras: "numba>=0.60" -> "numba"
        name = spec.split(";")[0].split("[")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<"):
            name = name.split(sep)[0]
        out.add(name.strip().lower())
    return out


def test_every_import_is_declared_or_known_debt():
    imports = _third_party_imports()
    declared = _declared_distributions()
    missing = {}
    for mod, files in sorted(imports.items()):
        if mod in UNDECLARABLE:
            continue
        dist = IMPORT_TO_DIST.get(mod, mod).lower()
        if dist not in declared:
            missing[mod] = sorted(files)
    assert not missing, (
        "third-party imports absent from pyproject.toml: "
        + "; ".join(f"{m} (in {', '.join(f)})" for m, f in missing.items())
    )


def test_numpy_is_the_only_hard_dependency():
    """The analysis core must import with numpy alone. If this fails, a guarded import became a
    top-level one and the core silently grew a dependency."""
    cfg = tomllib.loads(PYPROJECT.read_text())["project"]
    base = {s.split(">")[0].split("=")[0].strip().lower() for s in cfg["dependencies"]}
    assert base == {"numpy"}, f"base dependencies drifted: {sorted(base)}"


def test_undeclarable_imports_are_still_present():
    """Guards against a stale allowlist. If nothing imports these any more the entry is dead and
    should be removed, not left recorded as a standing caveat that no longer applies."""
    imports = _third_party_imports()
    stale = [m for m in UNDECLARABLE if m not in imports]
    assert not stale, (
        f"UNDECLARABLE lists {stale}, but nothing imports them any more — drop the entry."
    )


@pytest.mark.parametrize("mod", sorted(UNDECLARABLE))
def test_undeclarable_imports_stay_lazy(mod):
    """The whole arrangement rests on these being lazily imported. If one ever becomes a top-level
    import, the core silently stops installing from a clean checkout — so fail loudly here first."""
    import ast

    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.iter_child_nodes(tree):          # module top level only
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or "").split(".")[0]]
            assert mod not in names, (
                f"{path.name}:{node.lineno} imports {mod} at module top level. It must stay lazy — "
                f"it is a documented local install, not a declared dependency."
            )
