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

#: Deliberately undeclared, with the reason. Empty since 2026-09-22: wtmm and wtmm_ebsd, the
#: former entries, are vendored under dynamix/_vendor, so their imports resolve inside the
#: package and no local editable install remains. The guard test below keeps this list honest
#: if a new local-install debt ever appears.
UNDECLARABLE: dict = {}

#: Import name -> distribution name, where they differ. mpl_toolkits (mplot3d, axes_grid1,
#: the scale-bar anchors) ships inside the matplotlib distribution.
IMPORT_TO_DIST = {"PIL": "pillow", "mpl_toolkits": "matplotlib"}


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


def test_base_dependencies_are_numpy_plus_the_fft_engines():
    """The analysis core must IMPORT with numpy alone (every other core import is lazy or
    guarded), but the base INSTALL deliberately carries the FFT engines since the platform
    switch (2026-09-22): pyfftw on every platform, mlx only behind its Apple Silicon marker.
    If this fails, either a dependency crept into the base list or the mlx marker was lost --
    losing it would make every Windows/Linux/Intel-Mac install try to fetch mlx."""
    cfg = tomllib.loads(PYPROJECT.read_text())["project"]
    base = {s.split(">")[0].split("=")[0].split(";")[0].strip().lower()
            for s in cfg["dependencies"]}
    assert base == {"numpy", "pyfftw", "mlx"}, f"base dependencies drifted: {sorted(base)}"
    mlx = next(s for s in cfg["dependencies"] if s.lower().startswith("mlx"))
    assert "sys_platform == 'darwin'" in mlx and "platform_machine == 'arm64'" in mlx, (
        f"mlx must stay behind its Apple Silicon environment marker; got: {mlx}"
    )


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
