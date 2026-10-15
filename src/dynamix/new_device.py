# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""``python -m dynamix.new_device <id> [--kind transform|filter] [--consumes C] [--produces C] [--dir D]``:
writes ``<id>.py`` (a device with one Param, its contracts and its families, found at the next
start-up), ``test_<id>.py`` (a golden test: it records the output on its first run and compares on
every later one) and ``golden/<id>/README.md`` into the user's devices folder. Imports no Qt."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dynamix.model.contracts import ALL_CONTRACTS

USAGE = "python -m dynamix.new_device <id> [--kind transform|filter] [--consumes C] [--produces C] [--dir D]"

_DEVICE = '''# A GeoDynamix device. Drop this file in the devices folder; it is found at the next start-up.
# The one-page contract: docs/reference/devices_contract.md in the GeoDynamix tree.
from dynamix.model.param import Param, ParamKind


class {cls}:
    name = "{id}"
    label = "{label}"
    # Experimental: listed under Labs, folded, until you promote it to a family.
    families = ("Labs",)
    # Names from dynamix.model.capabilities.CAPABILITIES; with a declaration the engine probes
    # nothing, so a wants_cancel or an outputs you add needs its name here too.
    capabilities = frozenset()
    params = (Param("gain", ParamKind.FLOAT, default=1.0, min=0.0, max=10.0),)
    # Alternatives of contract names the step needs; what it adds ("<received>" passes the input on).
    consumes = (("{consumes}",),)
    produces = {produces_expr}
{body}

DEVICES = [{cls}()]
'''
_TRANSFORM_BODY = '''
    def compute(self, field, params, progress=None):
        """``field`` is a RasterField; return a dict the outputs and the next step read."""
        return {"values": field.values.astype("float32") * params["gain"]}

    def cache_key(self, params):
        return (self.name, params["gain"])
'''
_FILTER_BODY = '''
    def apply(self, extrema, params):
        """``extrema`` is the stack the transforms produced; return what you keep."""
        return extrema
'''
_TEST = '''"""Golden test for {id}: records golden/{id}/output.npz on the first run, compares after.
Run: python -m pytest {test_name}   (delete the .npz to record again)."""
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE / "golden" / "{id}" / "output.npz"


def _device():
    spec = importlib.util.spec_from_file_location("{id}_module", HERE / "{id}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DEVICES[0]


def test_{id}_matches_its_golden_output():
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.model.device import validate_params

    device = _device()
    params = validate_params(device, {{}})
    v = np.add.outer(np.arange(16.0), np.arange(16.0))
    field = RasterField(name="ramp", values=v, frame=LocalFrame(units="px"),
                        x_axis=np.arange(16.0), y_axis=np.arange(16.0))
    if hasattr(device, "compute"):
        out = device.compute(field, params)
        values = np.asarray(out["values"] if isinstance(out, dict) else out)
    else:
        values = np.asarray([len(device.apply([{{}}, {{}}, {{}}], params))])
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GOLDEN, values=values)
    np.testing.assert_allclose(values, np.load(GOLDEN)["values"], rtol=1e-6)
'''
_README = '''# Golden output of {id}

`output.npz` is what `{id}` computed on a 16×16 ramp the first time `test_{id}.py` ran; every later run
compares against it. A changed result is either a bug or a decision: delete the file to record again.
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dynamix.new_device", usage=USAGE)
    ap.add_argument("id")
    ap.add_argument("--kind", choices=("transform", "filter"), default="transform")
    ap.add_argument("--consumes", default=None, help="Field for a transform, ExtremaStack for a filter")
    ap.add_argument("--produces", default=None, help="Field for a transform, nothing for a filter")
    ap.add_argument("--dir", default=None)
    try:
        a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit:
        return 2
    if not re.fullmatch(r"[a-z][a-z0-9_]*", a.id):
        print(f"not a device id (lowercase, digits, underscores): {a.id!r}", file=sys.stderr)
        return 2
    consumes = a.consumes or ("Field" if a.kind == "transform" else "ExtremaStack")
    produces = a.produces if a.produces is not None else ("Field" if a.kind == "transform" else "")
    unknown = [c for c in (consumes, produces) if c and c != "<received>" and c not in ALL_CONTRACTS]
    if unknown:
        print(f"unknown contract name(s) {unknown}; known: {sorted(ALL_CONTRACTS)}", file=sys.stderr)
        return 2
    if a.dir is None:
        from dynamix.shell.settings import devices_dir
        folder = devices_dir()
    else:
        folder = Path(a.dir)
    device_path = folder / f"{a.id}.py"
    test_name = f"test_{a.id}.py"
    readme = folder / "golden" / a.id / "README.md"
    taken = [p for p in (device_path, folder / test_name, readme.parent) if p.exists()]
    if taken:
        print(f"{taken[0]} exists; not overwriting", file=sys.stderr)
        return 2
    folder.mkdir(parents=True, exist_ok=True)
    cls = "".join(part.capitalize() for part in a.id.split("_"))
    label = a.id.replace("_", " ").capitalize()
    body = _TRANSFORM_BODY if a.kind == "transform" else _FILTER_BODY
    produces_expr = f'frozenset({{"{produces}"}})' if produces else "frozenset()"
    device_path.write_text(_DEVICE.format(cls=cls, id=a.id, label=label, consumes=consumes,
                                          produces_expr=produces_expr, body=body))
    (folder / test_name).write_text(_TEST.format(id=a.id, test_name=test_name))
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(_README.format(id=a.id))
    for p in (device_path, folder / test_name, readme):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
