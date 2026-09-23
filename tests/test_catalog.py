# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Catalog tooling tests — schema validation, emission, indexes.

catalog/tools/ is not a package; import build.py by path.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

# The catalog tooling needs the optional `catalog` extra (pyyaml); an app-only install skips.
pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "catalog_build", ROOT / "catalog" / "tools" / "build.py"
)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

SCHEMA = json.loads((ROOT / "catalog" / "schema.json").read_text(encoding="utf-8"))

GOOD_RECORD = {
    "id": "gswa-500k-structures",
    "name": "GSWA 1:500k Interpreted Bedrock Geology — faults/structures",
    "operator": "[[GSWA]]",
    "portal": "https://dasc.dmirs.wa.gov.au/",
    "scale_tier": "regional",
    "resolution": "1:500,000",
    "data_types": ["structural-vectors", "geologic-map"],
    "regions": ["australia", "western-australia"],
    "coverage": {"name": "Western Australia", "bbox": [112.9, -35.2, 129.0, -13.6]},
    "fetch": [
        {"method": "http-download", "url": "https://dasc.dmirs.wa.gov.au/", "auth": "free-registration"}
    ],
    "license": "CC BY 4.0",
    "added": "2026-08-11",
    "verified": "2026-08-11",
}


def test_good_record_validates():
    assert build.validate_record(copy.deepcopy(GOOD_RECORD), SCHEMA) == []


def test_missing_required_field_reported():
    record = copy.deepcopy(GOOD_RECORD)
    del record["scale_tier"]
    errors = build.validate_record(record, SCHEMA)
    assert any("scale_tier" in e for e in errors)


def test_bad_enum_reported():
    record = copy.deepcopy(GOOD_RECORD)
    record["scale_tier"] = "galactic"
    errors = build.validate_record(record, SCHEMA)
    assert any("scale_tier" in e for e in errors)


def test_bad_data_type_reported():
    record = copy.deepcopy(GOOD_RECORD)
    record["data_types"] = ["vibes"]
    errors = build.validate_record(record, SCHEMA)
    assert any("data_types" in e for e in errors)


def test_empty_fetch_reported():
    record = copy.deepcopy(GOOD_RECORD)
    record["fetch"] = []
    errors = build.validate_record(record, SCHEMA)
    assert any("fetch" in e for e in errors)


def test_bad_bbox_reported():
    record = copy.deepcopy(GOOD_RECORD)
    record["coverage"]["bbox"] = [112.9, 40.0, 129.0, -13.6]  # S > N
    errors = build.validate_record(record, SCHEMA)
    assert any("bbox" in e for e in errors)


def test_antimeridian_bbox_allowed():
    record = copy.deepcopy(GOOD_RECORD)
    record["coverage"]["bbox"] = [170.0, -50.0, -170.0, -30.0]  # W > E: crosses dateline
    assert build.validate_record(record, SCHEMA) == []


def test_bad_date_reported():
    record = copy.deepcopy(GOOD_RECORD)
    record["verified"] = "August 11"
    errors = build.validate_record(record, SCHEMA)
    assert any("verified" in e for e in errors)


def test_parse_note_roundtrip(tmp_path):
    note = tmp_path / "x.md"
    note.write_text(
        "---\nid: x\nname: X\nadded: 2026-08-11\n---\nBody prose.\n", encoding="utf-8"
    )
    record, errors = build.parse_note(note)
    assert errors == []
    # PyYAML parses bare dates as datetime.date; parse_note must normalize to ISO strings
    assert record["added"] == "2026-08-11"


def test_parse_note_without_frontmatter_errors(tmp_path):
    note = tmp_path / "x.md"
    note.write_text("just prose\n", encoding="utf-8")
    record, errors = build.parse_note(note)
    assert record is None and errors


def _write_note(directory, record, stem=None):
    import yaml as _yaml

    stem = stem or record["id"]
    fm = _yaml.safe_dump(record, allow_unicode=True, sort_keys=False)
    (directory / f"{stem}.md").write_text(f"---\n{fm}---\n\nProse.\n", encoding="utf-8")


def test_load_sources_sorted_and_valid(tmp_path):
    a = copy.deepcopy(GOOD_RECORD)
    b = copy.deepcopy(GOOD_RECORD)
    b["id"] = "aaa-first"
    _write_note(tmp_path, a)
    _write_note(tmp_path, b)
    records, errors = build.load_sources(tmp_path)
    assert errors == []
    assert [r["id"] for r in records] == ["aaa-first", "gswa-500k-structures"]


def test_load_sources_reports_filename_mismatch(tmp_path):
    record = copy.deepcopy(GOOD_RECORD)
    _write_note(tmp_path, record, stem="wrong-name")
    records, errors = build.load_sources(tmp_path)
    assert records == []
    assert any("filename" in e for e in errors)


def test_real_sources_all_validate():
    records, errors = build.load_sources()
    assert errors == [], "\n".join(errors)
    assert records, "catalog/sources/ must contain at least the seed notes"


def test_load_sources_tolerates_non_string_id(tmp_path):
    """Non-string id (e.g., list) should not crash load_sources; should be reported as error."""
    note = tmp_path / "bad-id.md"
    # Write raw YAML with a list id to bypass _write_note validation
    note.write_text(
        "---\nid: [a, b]\nname: Bad ID\nadded: 2026-08-11\n---\nProse.\n",
        encoding="utf-8",
    )
    records, errors = build.load_sources(tmp_path)
    assert records == []
    # The error should be present (from schema validation or filename mismatch, not a crash)
    assert len(errors) > 0


def test_emit_catalog_deterministic(tmp_path):
    records = [copy.deepcopy(GOOD_RECORD)]
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    build.emit_catalog(records, out1)
    build.emit_catalog(records, out2)
    assert out1.read_bytes() == out2.read_bytes()
    payload = json.loads(out1.read_text(encoding="utf-8"))
    assert payload["sources"][0]["path"] == "sources/gswa-500k-structures.md"
    assert payload["sources"][0]["scale_tier"] == "regional"


def test_committed_catalog_json_up_to_date(tmp_path):
    records, errors = build.load_sources()
    assert errors == []
    fresh = tmp_path / "catalog.json"
    build.emit_catalog(records, fresh)
    committed = ROOT / "catalog" / "_generated" / "catalog.json"
    assert committed.exists(), "run: python catalog/tools/build.py"
    assert committed.read_bytes() == fresh.read_bytes(), (
        "catalog/_generated/catalog.json is stale — run: python catalog/tools/build.py"
    )


def test_write_indexes(tmp_path):
    a = copy.deepcopy(GOOD_RECORD)
    b = copy.deepcopy(GOOD_RECORD)
    b["id"] = "gebco-2025"
    b["name"] = "GEBCO global bathymetry grid"
    b["scale_tier"] = "global"
    b["data_types"] = ["bathymetry"]
    b["regions"] = ["global"]
    build.write_indexes([b, a], tmp_path)

    tier = (tmp_path / "By Tier.md").read_text(encoding="utf-8")
    assert build.GENERATED_HEADER in tier
    assert "## global (1)" in tier and "## regional (1)" in tier
    assert "[[gebco-2025|GEBCO global bathymetry grid]]" in tier

    region = (tmp_path / "By Region.md").read_text(encoding="utf-8")
    # a source appears under EVERY one of its region tags
    assert "## australia (1)" in region and "## western-australia (1)" in region

    dtype = (tmp_path / "By Data Type.md").read_text(encoding="utf-8")
    assert "## structural-vectors (1)" in dtype


def test_write_indexes_deterministic(tmp_path):
    records = [copy.deepcopy(GOOD_RECORD)]
    d1, d2 = tmp_path / "one", tmp_path / "two"
    d1.mkdir(), d2.mkdir()
    build.write_indexes(records, d1)
    build.write_indexes(records, d2)
    for name in ("By Tier.md", "By Region.md", "By Data Type.md"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes()


def test_committed_indexes_up_to_date(tmp_path):
    records, errors = build.load_sources()
    assert errors == []
    build.write_indexes(records, tmp_path)
    for name in ("By Tier.md", "By Region.md", "By Data Type.md"):
        committed = ROOT / "catalog" / "indexes" / name
        assert committed.exists(), "run: python catalog/tools/build.py"
        assert committed.read_bytes() == (tmp_path / name).read_bytes(), (
            f"catalog/indexes/{name} is stale — run: python catalog/tools/build.py"
        )
