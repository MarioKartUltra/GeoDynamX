# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tables for devices_reference.tex -- written from the device registry's own declarations.

    python docs/reference/make_tables.py        # writes docs/reference/tables/*.tex

Each table reads what the app reads: the families table each tool's ``label`` and ``families``,
the contract and declaration tables ``dynamix.model.contracts``, the applicability table
``step_states`` for each selection tool after each analysis, the output-rows table each tool's
declared ``outputs``. So a table follows the browser, the chain strip and the engine, and
tests/test_reference_tables.py fails when a file under ``tables/`` is older than a fresh render.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from dynamix.model.chain import DeviceRef  # noqa: E402
from dynamix.model.contracts import (ALPHA, CONTRACT_KEYS, CONTRACTS, FIELD, POINTS,  # noqa: E402
                                     RECEIVED, STACK, consumes_of, step_states)
from dynamix.model.device import (DEVICES, declared_outputs, defaults_for,  # noqa: E402
                                  is_transform)
from dynamix.model.families import FAMILY_ORDER, families_of, label_of  # noqa: E402
from dynamix.model.param import ParamKind  # noqa: E402

OUT = HERE / "tables"
#: The files this script writes; devices_reference.tex inputs each one.
TABLES = ("families", "contracts", "declarations", "applicability", "outputs")
#: What a layer's input can be.
_INPUTS = frozenset({FIELD, STACK, POINTS})
#: Where a contract lives in a result when it has no top-level key of its own.
_CARRIED = {FIELD: r"the \texttt{RasterField} itself", STACK: r"the \texttt{RasterField} itself",
            POINTS: r"the \texttt{PointSet} itself", ALPHA: r'\texttt{extrema[i]["alpha"]}'}
_HEADER = ("% Written by make_tables.py from the device registry: change a declaration, then\n"
           "% run python docs/reference/make_tables.py.\n")
_SPECIAL = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
            "$": r"\$", "{": r"\{", "}": r"\}", "^": r"\^{}", "~": r"\~{}",
            "<": r"\textless{}", ">": r"\textgreater{}"}
_SPELLED = {"–": "--", "—": "---", "ö": r"\"o", "é": r"\'e", "σ": r"$\sigma$",
            "α": r"$\alpha$", "…": r"\ldots{}"}


def tex(text: str) -> str:
    """``text`` as ASCII TeX for pdflatex: specials escaped, the non-ASCII characters the labels
    use spelled out. Any other non-ASCII character raises, naming it."""
    out = []
    for ch in str(text):
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        elif ch in _SPELLED:
            out.append(_SPELLED[ch])
        elif ch.isascii():
            out.append(ch)
        else:
            raise ValueError(f"no TeX spelling for {ch!r} (U+{ord(ch):04X}) in {text!r}")
    return "".join(out)


def _p(width: float) -> str:
    return rf">{{\raggedright\arraybackslash}}p{{{width}\textwidth}}"


def _longtable(colspec: str, caption: str, label: str, head: str, rows: list[str]) -> str:
    """A longtable with an unnumbered caption and its head repeated on every page. The label
    sits on an anchor of its own, so the text links to the table by name."""
    head = f"\\toprule\n{head} \\\\ \\midrule\n"
    return "".join([_HEADER, f"\\phantomsection\\label{{{label}}}%\n",
                    f"\\begin{{longtable}}{{{colspec}}}\n", f"\\caption*{{{caption}}}\\\\\n",
                    head, "\\endfirsthead\n", head, "\\endhead\n",
                    *(f"{row} \\\\\n" for row in rows), "\\bottomrule\n\\end{longtable}\n"])


def _name(device) -> str:
    """A tool as the reference names it: ``\\tool{label}{id}``, or ``\\dev{id}`` when the label
    is the id."""
    label = label_of(device)
    if label == device.name:
        return rf"\dev{{{tex(device.name)}}}"
    return rf"\tool{{{tex(label)}}}{{{tex(device.name)}}}"


def _listed() -> list:
    """The registered tools listed in a family, in registration order."""
    return [d for n, d in DEVICES.items() if not n.startswith("stub_") and families_of(d)]


def _dev() -> list:
    """The browser's Dev group: the stubs and every tool listed in no family."""
    return [d for n, d in DEVICES.items() if n.startswith("stub_") or not families_of(d)]


def _read(device, attr: str, params: dict):
    """The device's ``consumes`` (any-of alternatives) or ``produces`` (``None`` when it declares
    none) at ``params``."""
    if attr == "consumes":
        return consumes_of(device, params)
    declared = getattr(device, "produces", None)
    if declared is None:
        return None
    return tuple(declared(params) if callable(declared) else declared)


def _variants(device, attr: str) -> list[tuple[dict, object]]:
    """``[({}, declaration at the defaults)]``, then, when the declaration is a function of the
    params, one ``({param: value}, declaration)`` per CHOICE value that changes it."""
    defaults = defaults_for(device)
    base = _read(device, attr, defaults)
    out = [({}, base)]
    if not callable(getattr(device, attr, None)):
        return out
    for param in device.params:
        if param.kind is not ParamKind.CHOICE:
            continue
        for choice in param.choices:
            value = _read(device, attr, dict(defaults, **{param.name: choice}))
            if choice != param.default and value != base:
                out.append(({param.name: choice}, value))
    return out


def _setting(params: dict) -> str:
    return ", ".join(rf"\texttt{{{tex(k)}}} $=$ \texttt{{{tex(v)}}}" for k, v in params.items())


def families_table() -> str:
    rows = []
    for family in FAMILY_ORDER:
        members = [d for d in _listed() if family in families_of(d)]
        if members:
            rows.append(f"{tex(family)} & " + "; ".join(_name(d) for d in members))
    rows.append("Dev (collapsed) & " + "; ".join(_name(d) for d in _dev()))
    return _longtable(_p(0.2) + _p(0.74), r"\textbf{Families}: the tools the device browser "
                      "lists under each family.", "tab:families",
                      r"Family & Tools (display name \textperiodcentered{} device id)", rows)


def contracts_table() -> str:
    rows = []
    for contract, meaning in CONTRACTS.items():
        keys = CONTRACT_KEYS.get(contract) or ()
        where = (", ".join(rf"\texttt{{{tex(k)}}}" for k in keys) if keys
                 else _CARRIED[contract])
        rows.append(f"{tex(contract)} & {tex(meaning)} & {where}")
    return _longtable(_p(0.18) + _p(0.34) + _p(0.4), r"\textbf{Contracts}: the named kinds of "
                      "data and the result keys each one covers.", "tab:contracts",
                      "Contract & Meaning & Result keys", rows)


def _alternatives(alternatives) -> str:
    if not alternatives:
        return "anything"
    return " or ".join(" + ".join(tex(c) for c in alt) for alt in alternatives)


def _products(declared) -> str:
    if declared is None:
        return "(undeclared)"
    names = ["what it received" if c == RECEIVED else tex(c) for c in declared]
    return " + ".join(names) if names else "---"


def _declaration(device, attr: str) -> str:
    show = _alternatives if attr == "consumes" else _products
    variants = _variants(device, attr)
    cell = show(variants[0][1])
    if callable(getattr(device, attr, None)):
        cell += r"$^\dagger$"
    for params, value in variants[1:]:
        cell += f"; {_setting(params)}: {show(value)}"
    return cell


def declarations_table() -> str:
    rows = []
    for name, device in DEVICES.items():
        if name.startswith("stub_"):
            continue
        rows.append(" & ".join([_name(device), "T" if is_transform(device) else "F",
                                _declaration(device, "consumes"),
                                _declaration(device, "produces")]))
    return _longtable(_p(0.36) + "c" + _p(0.25) + _p(0.25), r"\textbf{Declarations}: what each "
                      r"tool consumes and produces (T: Transform, F: Filter; $\dagger$: depends "
                      "on the parameters).", "tab:declarations", "Tool & & Consumes & Produces",
                      rows)


def applicability() -> tuple[list, list]:
    """``(columns, rows)``. A column is ``(steps, layer input)``: each listed Transform that reads
    a layer input and produces something else (with each CHOICE value that changes what it
    produces), then each listed Transform that reads no layer input, after every column it
    applies after. A row is ``(step, [applies after each column])`` for each listed tool that
    reads no layer input (with each CHOICE value that changes what it reads)."""
    columns = []
    for device in _listed():
        if not is_transform(device):
            continue
        inputs = [alt[0] for alt in consumes_of(device, defaults_for(device))
                  if len(alt) == 1 and alt[0] in _INPUTS]
        if not inputs:
            continue
        for params, produced in _variants(device, "produces"):
            if produced and RECEIVED not in produced and _INPUTS.isdisjoint(produced):
                columns.append(((DeviceRef(device.name, dict(params)),), inputs[0]))
    readers = []
    for device in _listed():
        alternatives = consumes_of(device, defaults_for(device))
        if alternatives and all(_INPUTS.isdisjoint(alt) for alt in alternatives):
            readers.append(device)
    for stage in (d for d in readers if is_transform(d)):
        ref = DeviceRef(stage.name, {})
        columns += [(steps + (ref,), layer_input) for steps, layer_input in list(columns)
                    if step_states(steps + (ref,), layer_input)[-1].applies]
    rows = []
    for device in readers:
        for params, _declared in _variants(device, "consumes"):
            ref = DeviceRef(device.name, dict(params))
            rows.append((ref, [step_states(steps + (ref,), layer_input)[-1].applies
                               for steps, layer_input in columns]))
    return columns, rows


def _column_title(steps) -> str:
    parts = []
    for ref in steps:
        part = tex(label_of(DEVICES[ref.device]))
        if ref.params:
            part += ", " + ", ".join(tex(v) for v in ref.params.values())
        parts.append(part)
    return r"\rotatebox{90}{" + " + ".join(parts) + "}"


def applicability_table() -> str:
    columns, rows = applicability()
    head = r"\emph{applies after} $\rightarrow$ & " + " & ".join(
        _column_title(steps) for steps, _input in columns)
    body = []
    for ref, flags in rows:
        title = _name(DEVICES[ref.device])
        if ref.params:
            title += ", " + _setting(ref.params)
        body.append(title + " & " + " & ".join(r"$\bullet$" if f else "" for f in flags))
    return _longtable(_p(0.36) + f"*{{{len(columns)}}}{{c}}", r"\textbf{Applicability}: "
                      r"$\bullet$ where a selection tool applies after an analysis (the "
                      r"contracts rule, \S\ref{sec:contracts}).", "tab:applicability", head,
                      body)


def outputs_table() -> str:
    rows = []
    for device in _listed():
        first = True
        for out in declared_outputs(device):
            if not out.row:
                continue
            rows.append(" & ".join([_name(device) if first else "",
                                    rf"\texttt{{{tex(out.name)}}}", tex(out.label or out.name),
                                    out.kind, "when shown" if out.lazy else "with the analysis",
                                    "yes" if out.per_level else ""]))
            first = False
    return _longtable(_p(0.26) + _p(0.18) + _p(0.15) + "l" + _p(0.11) + "c",
                      r"\textbf{Output rows}: the rows each tool carries under its result.",
                      "tab:outputs", "Tool & Output & Row & Kind & Computed & Per level", rows)


PAGE = HERE / "devices_contract.md"


def contract_page() -> str:
    """The one-page device contract (Markdown, ships): what a device declares, the capabilities by
    name with the attributes the engine probes for them, the contract names, the families and the
    roles, read from the registry as the tables are."""
    from dynamix.model.capabilities import PROBES
    from dynamix.model.roles import chain_producers, primary_analyzers, shell_placed, superseded

    lines = ["# The device contract", "",
             "Written by `docs/reference/make_tables.py` from the device registry; edit a declaration,",
             "then run it. A device is one Python class; nothing here needs a base class.", "",
             "## What a device declares", "",
             "| attribute | meaning |", "|---|---|",
             "| `name` | the id: lowercase, digits, underscores; stable, it names the step in a saved chain, a cache key and a `.gprv` step |",
             "| `label`, `families` | display name and the families it is listed under (`Labs` for an experimental device, `()` to leave the palette) |",
             "| `params` | a tuple of `Param`; `validate_params` fills defaults and rejects unknown keys |",
             "| `compute(field, params, progress=None)` + `cache_key(params)` | a Transform; `apply(extrema, params)` makes a Filter |",
             "| `consumes`, `produces` | alternatives of contract names the step needs, and the set it adds (`<received>` passes the input on) |",
             "| `capabilities` | a frozenset of the names below; absent, the engine probes the attributes instead |",
             "| `role = \"primary\"`, `chain_producer`, `chain_consumer`, `superseded_by`, `shell_placed`, `band_variant_of` | the roles the shell derives its device sets from |",
             "| `outputs` | `Output` declarations the layer's rows show, computed by `compute_output` |",
             "", "A device file in the user's devices folder exposes `DEVICES = [...]`; a package registers through",
             "the `geodynamix.devices` entry-point group. `python -m dynamix.new_device <id>` writes a file with a",
             "golden test. A file that fails to import or validate is a greyed browser row with its error.", "",
             "## Capabilities", "", "| capability | probed attributes |", "|---|---|"]
    lines += [f"| `{name}` | " + ", ".join(f"`{a}`" for a in attrs) + " |" for name, attrs in PROBES.items()]
    lines += ["", "## Contracts", "", "| name | meaning |", "|---|---|"]
    lines += [f"| `{name}` | {meaning} |" for name, meaning in CONTRACTS.items()]
    lines += ["", "## Families", "", ", ".join(f"`{f}`" for f in FAMILY_ORDER), "",
              "## Roles today", "",
              "- primary analyzers: " + ", ".join(f"`{n}`" for n in primary_analyzers()),
              "- chain producers: " + ", ".join(f"`{n}`" for n in chain_producers()),
              "- superseded: " + ", ".join(f"`{n}`" + (f" by `{s}`" if s else " by the Transport") for n, s in superseded().items()),
              "- shell-placed: " + ", ".join(f"`{n}`" for n in shell_placed()), ""]
    return "\n".join(lines)


def render() -> dict[str, str]:
    """Every table's text, by its name in :data:`TABLES`, from the devices registered now."""
    return {"families": families_table(), "contracts": contracts_table(),
            "declarations": declarations_table(), "applicability": applicability_table(),
            "outputs": outputs_table()}


def main():
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    OUT.mkdir(exist_ok=True)
    for name, text in render().items():
        (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
        print("wrote", f"tables/{name}.tex")
    PAGE.write_text(contract_page(), encoding="utf-8")
    print("wrote", PAGE.name)


if __name__ == "__main__":
    main()
