# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The topology package's import discipline, AST-checked like test_manifest_matches_imports."""
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "dynamix" / "topology"
FORBIDDEN = {"PySide6", "pyvista", "pyqtgraph", "shapely", "geopandas", "wtmm", "wtmm_ebsd"}


def _imports(path):
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
            # GeoDynamix_Beta: the vendored wtmm/wtmm_ebsd live under dynamix._vendor
            out |= {"wtmm" for a in node.names if a.name.startswith("dynamix._vendor.")}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.add((node.module or "").split(".")[0])
            if (node.module or "").startswith("dynamix._vendor."):
                out.add("wtmm")
    return out


def test_topology_package_import_discipline():
    stdlib = set(sys.stdlib_module_names)
    for path in sorted(PKG.glob("*.py")):
        third = _imports(path) - stdlib - {"dynamix"}
        assert not (third & FORBIDDEN), f"{path.name} imports {third & FORBIDDEN}"
        if path.name in ("codes.py", "model.py"):
            assert third == set(), f"{path.name} must be stdlib-only; imports {third}"
        else:
            assert third <= {"numpy"}, f"{path.name} may import numpy only; got {third}"


def test_topology_devices_import_discipline():
    path = REPO / "src" / "dynamix" / "devices" / "topology.py"
    assert not (_imports(path) & FORBIDDEN)
