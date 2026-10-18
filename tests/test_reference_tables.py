# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The device reference's generated tables (docs/reference/make_tables.py): written from the
registry's families, contracts and outputs, held in the tree as a fresh render, with the
applicability table saying what ``step_states`` says."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

REFERENCE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "reference"


@pytest.fixture(scope="module")
def make_tables():
    spec = importlib.util.spec_from_file_location("make_tables", REFERENCE / "make_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _key(steps) -> str:
    return "+".join(ref.device + "".join(f"[{k}={v}]" for k, v in sorted(ref.params.items()))
                    for ref in steps)


def _after(make_tables) -> dict[str, set[str]]:
    """Each row of the applicability table -> the columns it applies after, as step keys."""
    columns, rows = make_tables.applicability()
    return {_key((ref,)): {_key(steps) for (steps, _input), applies in zip(columns, flags)
                           if applies}
            for ref, flags in rows}


def test_tex_spells_the_label_characters_in_ascii(make_tables):
    assert make_tables.tex("H-line Hölder") == r"H-line H\"older"
    assert make_tables.tex("Mallat–Zhong edges") == "Mallat--Zhong edges"
    assert make_tables.tex("Re at σ") == r"Re at $\sigma$"
    assert make_tables.tex("a per-point α") == r"a per-point $\alpha$"
    assert make_tables.tex("tucker_HOOI_HOSVD & 5%") == r"tucker\_HOOI\_HOSVD \& 5\%"


def test_tex_refuses_a_character_it_cannot_spell(make_tables):
    with pytest.raises(ValueError, match=r"U\+2603"):
        make_tables.tex("snow ☃")


def test_the_tables_in_the_tree_are_a_fresh_render(make_tables, builtins):
    rendered = make_tables.render()
    assert tuple(rendered) == make_tables.TABLES
    held = sorted(p.stem for p in (REFERENCE / "tables").glob("*.tex"))
    assert held == sorted(make_tables.TABLES)
    for name, text in rendered.items():
        assert text.isascii(), name
        assert (REFERENCE / "tables" / f"{name}.tex").read_text(encoding="utf-8") == text, (
            f"tables/{name}.tex is stale: run python docs/reference/make_tables.py")


def test_every_tool_is_listed_under_each_of_its_families_or_in_dev(make_tables, builtins):
    from dynamix.model.device import DEVICES
    from dynamix.model.families import families_of

    rows = {line.split(" & ")[0]: line for line in make_tables.families_table().splitlines()
            if " & " in line}
    for name, device in DEVICES.items():
        listed = () if name.startswith("stub_") else families_of(device)
        for family in listed or ("Dev (collapsed)",):
            assert "{" + make_tables.tex(name) + "}" in rows[family], (name, family)


def test_the_applicability_table_reads_the_contracts(make_tables, builtins):
    after = _after(make_tables)
    topology = {"wtmm2d+chain_topology", "wtmm2d_roi+chain_topology"}
    chains = {"wtmm2d", "wtmm2d_roi"} | topology
    extrema = chains | {"mz_edges", "mz_edges[algorithm=printed]", "mz_edges[wavelet=frac_bspline]",
                        "wavelet_skeleton",
                        "cdf_edges", "pm_edges"}
    for name in ("orientation_wedge", "modulus_threshold", "hline_length", "hline_modulus"):
        assert after[name] == extrema, name
    for name in ("chain_topology", "chain_holder", "chain_modulus", "chain_length",
                 "chain_classify", "group_filter", "hline_modulus[mode=sup_all_scales]"):
        assert after[name] == chains, name
    assert after["min_vchains"] == topology
    assert after["hline_holder"] == {"mz_edges"}
    assert set(after) == {"chain_topology", "min_vchains", "orientation_wedge",
                          "modulus_threshold", "hline_length", "hline_modulus",
                          "hline_modulus[mode=sup_all_scales]", "hline_holder", "chain_holder",
                          "chain_modulus", "chain_length", "chain_classify", "group_filter"}


def test_the_declarations_spell_out_param_dependent_contracts(make_tables, builtins):
    rows = {line.split(" & ")[0]: line for line in make_tables.declarations_table().splitlines()
            if " & " in line}
    assert r"\texttt{algorithm} $=$ \texttt{printed}: ExtremaStack" in rows[
        r"\tool{Mallat--Zhong edges}{mz\_edges}"]
    assert r"\texttt{mode} $=$ \texttt{sup\_all\_scales}: ExtremaStack + VChains" in rows[
        r"\tool{H-chain modulus}{hline\_modulus}"]
    assert rows[r"\tool{Noise / dither}{noise}"].endswith(r"& what it received \\")


def test_the_outputs_table_lists_the_band_and_per_level_rows(make_tables, builtins):
    text = make_tables.outputs_table()
    for name in ("band", "band\\_mask", "modulus", "argument", "re\\_scale", "im\\_scale",
                 "filtered\\_scale", "approximation", "horizontal", "vertical", "diagonal",
                 "residual", "heldout"):
        assert rf"\texttt{{{name}}}" in text, name
    assert r"\texttt{snapshots}" not in text
    per_level = [line for line in text.splitlines() if line.endswith("& yes \\\\")]
    assert len(per_level) == 16                 # mz_edges 2, dwt2d 4, band_recon 3, holder 4, wavelet_ridge 3


def test_the_reference_inputs_every_table(make_tables):
    text = (REFERENCE / "devices_reference.tex").read_text(encoding="utf-8")
    for name in make_tables.TABLES:
        assert text.count(rf"\input{{tables/{name}}}") == 1, name


def test_every_public_param_is_named_in_the_reference(make_tables, builtins):
    from dynamix.model.device import DEVICES

    text = (REFERENCE / "devices_reference.tex").read_text(encoding="utf-8")
    missing = [f"{name}.{p.name}" for name, device in DEVICES.items()
               if not name.startswith("stub_")
               for p in device.params
               if not p.name.startswith("_") and make_tables.tex(p.name) not in text]
    assert not missing, missing


def test_the_contract_page_in_the_tree_is_a_fresh_render(make_tables, builtins):
    page = make_tables.contract_page()
    assert page.startswith("# The device contract") and "`FieldStage`" in page and "`wtmm2d`" in page
    assert make_tables.PAGE.read_text(encoding="utf-8") == page
