# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.partition.Partition (reversible keep/prune over an id-universe)."""
from __future__ import annotations

from dynamix.core.partition import PRUNE_JSON, Partition


def test_prune_keep_complement_and_masks():
    p = Partition([1, 2, 3, 4, 5])
    assert p.prune([2, 4], "shallow") == 2
    assert p.pruned.tolist() == [2, 4] and p.kept.tolist() == [1, 3, 5]
    assert p.n_pruned == 2 and p.n_kept == 3
    assert p.keep_mask([1, 2, 3, 4, 5]).tolist() == [True, False, True, False, True]
    assert p.is_pruned([2, 3]).tolist() == [True, False]


def test_prune_clamps_to_universe_and_dedupes():
    p = Partition([1, 2, 3])
    assert p.prune([2, 2, 99, -1], "x") == 1        # only 2 is in-universe (deduped, clamped)
    assert p.pruned.tolist() == [2]


def test_prune_newly_count_and_reason_reassignment():
    p = Partition([1, 2, 3, 4])
    assert p.prune([1, 2], "a") == 2
    assert p.prune([2, 3], "b") == 1                # 3 is new; 2 is reassigned a->b (not newly pruned)
    assert p.reason_counts() == {"a": 1, "b": 2}    # a:{1}, b:{2,3} -- reasons stay disjoint
    assert p.pruned.tolist() == [1, 2, 3]


def test_unprune_and_clear_are_reversible():
    p = Partition([1, 2, 3, 4]); p.prune([1, 2, 3], "a")
    assert p.unprune([2]) == 1 and p.pruned.tolist() == [1, 3]
    assert p.clear() == 2 and p.n_pruned == 0


def test_default_reason_is_unspecified():
    p = Partition([1, 2]); p.prune([1])
    assert p.reason_counts() == {"unspecified": 1}


def test_json_roundtrip(tmp_path):
    p = Partition([10, 20, 30, 40], name="cat")
    p.prune([20], "shallow"); p.prune([40], "off-slab")
    p.save(tmp_path)
    r = Partition.load([10, 20, 30, 40], tmp_path)
    assert r.name == "cat" and r.pruned.tolist() == [20, 40]
    assert r.reason_counts() == {"off-slab": 1, "shallow": 1}


def test_save_is_noop_when_nothing_pruned(tmp_path):
    Partition([1, 2, 3]).save(tmp_path)
    assert not (tmp_path / PRUNE_JSON).exists()     # keep-everything -> no file written


def test_save_after_clear_removes_stale_file(tmp_path):
    """prune -> save -> restore-all -> save must REMOVE the file, not leave a stale one that would
    later resurrect the un-pruned points."""
    p = Partition([1, 2, 3])
    p.prune([1, 2], "x"); p.save(tmp_path)
    assert (tmp_path / PRUNE_JSON).exists()
    p.clear(); p.save(tmp_path)                     # full restore + re-save
    assert not (tmp_path / PRUNE_JSON).exists()     # stale file removed -> no resurrection on reload


def test_load_reclamps_when_universe_shrinks(tmp_path):
    p = Partition([1, 2, 3, 4]); p.prune([3, 4], "x"); p.save(tmp_path)
    r = Partition.load([1, 2, 3], tmp_path)         # 4 no longer in the universe
    assert r.pruned.tolist() == [3]


def test_empty_universe_is_safe():
    p = Partition([])
    assert p.prune([1], "x") == 0 and p.n_pruned == 0 and p.kept.size == 0
