# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A JSON save replaces its file whole or leaves the previous file as it was
(dynamix.model.atomic_write)."""
from __future__ import annotations

import json
import os

import pytest

from dynamix.model.atomic_write import write_json_atomically

PAYLOAD = {"format": "dynamix-project", "title": "Gravité", "scales": [1.5, 2, 4e-3],
           "nested": {"b": None, "a": True}}


def test_a_save_writes_the_bytes_a_plain_text_write_would(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(PAYLOAD, indent=2), encoding="utf-8")
    target = tmp_path / "target.json"
    write_json_atomically(target, PAYLOAD)
    assert target.read_bytes() == reference.read_bytes()


def test_a_save_over_an_existing_file_leaves_only_that_file(tmp_path):
    target = tmp_path / "doc.json"
    target.write_text("old", encoding="utf-8")
    files = sorted(tmp_path.iterdir())
    write_json_atomically(target, PAYLOAD)
    assert json.loads(target.read_text(encoding="utf-8")) == PAYLOAD
    assert sorted(tmp_path.iterdir()) == files


@pytest.mark.parametrize("step", ["fsync", "replace"])
def test_a_failure_before_the_file_is_replaced_keeps_the_previous_file(tmp_path, monkeypatch, step):
    target = tmp_path / "doc.json"
    write_json_atomically(target, {"version": 1})
    before, files = target.read_bytes(), sorted(tmp_path.iterdir())

    def fail(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m, pytest.raises(OSError, match="No space left"):
        m.setattr(os, step, fail)
        write_json_atomically(target, {"version": 2})
    assert target.read_bytes() == before
    assert sorted(tmp_path.iterdir()) == files


def test_an_unserialisable_payload_leaves_the_previous_file(tmp_path):
    target = tmp_path / "doc.json"
    write_json_atomically(target, {"version": 1})
    before, files = target.read_bytes(), sorted(tmp_path.iterdir())
    with pytest.raises(TypeError):
        write_json_atomically(target, {"version": object()})
    assert target.read_bytes() == before
    assert sorted(tmp_path.iterdir()) == files


def test_a_symlinked_file_is_written_through_and_the_link_kept(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("this platform does not let the test create a symlink")
    write_json_atomically(link, PAYLOAD)
    assert link.is_symlink()
    assert json.loads(real.read_text(encoding="utf-8")) == PAYLOAD
