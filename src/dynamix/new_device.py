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
    families = ("Labs",)                 # experimental: listed folded until you promote it
    capabilities = frozenset()           # names from dynamix.model.capabilities.CAPABILITIES
    params = (Param("gain", ParamKind.FLOAT, default=1.0, min=0.0, max=10.0),)
    consumes = (("{consumes}",),)         # alternatives of contract names the step needs
    produces = frozenset({{"{produces}"}})  # what it adds; "<received>" passes the input on
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
        values = np.asarray([len(device.apply([], params))])
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
    ap.add_argument("--consumes", default="Field")
    ap.add_argument("--produces", default="Field")
    ap.add_argument("--dir", default=None)
    try:
        a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit:
        return 2
    if not re.fullmatch(r"[a-z][a-z0-9_]*", a.id):
        print(f"not a device id (lowercase, digits, underscores): {a.id!r}", file=sys.stderr)
        return 2
    unknown = [c for c in (a.consumes, a.produces) if c != "<received>" and c not in ALL_CONTRACTS]
    if unknown:
        print(f"unknown contract name(s) {unknown}; known: {sorted(ALL_CONTRACTS)}", file=sys.stderr)
        return 2
    if a.dir is None:
        from dynamix.shell.settings import devices_dir
        folder = devices_dir()
    else:
        folder = Path(a.dir)
    device_path = folder / f"{a.id}.py"
    if device_path.exists():
        print(f"{device_path} exists; not overwriting", file=sys.stderr)
        return 2
    folder.mkdir(parents=True, exist_ok=True)
    cls = "".join(part.capitalize() for part in a.id.split("_"))
    label = a.id.replace("_", " ").capitalize()
    body = _TRANSFORM_BODY if a.kind == "transform" else _FILTER_BODY
    consumes = a.consumes if a.kind == "transform" else "ExtremaStack"
    device_path.write_text(_DEVICE.format(cls=cls, id=a.id, label=label, consumes=consumes,
                                          produces=a.produces, body=body))
    test_name = f"test_{a.id}.py"
    (folder / test_name).write_text(_TEST.format(id=a.id, test_name=test_name))
    readme = folder / "golden" / a.id / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(_README.format(id=a.id))
    for p in (device_path, folder / test_name, readme):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
