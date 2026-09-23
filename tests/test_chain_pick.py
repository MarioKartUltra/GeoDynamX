# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np

from dynamix.core.chain_pick import chains_in_polygon, pick_chain

CHAINS = [
    {"x": np.array([10, 11, 12]), "y": np.array([5, 5, 5])},
    {"x": np.array([40, 41]), "y": np.array([20, 21])},
    {"x": np.array([7]), "y": np.array([7])},          # 1-point chain: still pickable
]


def test_pick_nearest_chain():
    assert pick_chain(CHAINS, (11.2, 5.4), max_dist=2.0) == 0
    assert pick_chain(CHAINS, (40.4, 20.1), max_dist=2.0) == 1


def test_pick_miss_returns_none():
    assert pick_chain(CHAINS, (100.0, 100.0), max_dist=2.0) is None


def test_pick_empty_chains():
    assert pick_chain([], (0.0, 0.0), max_dist=2.0) is None
    assert pick_chain([{"x": np.array([]), "y": np.array([])}], (0, 0), max_dist=2) is None


def test_lasso_selects_enclosed_chains():
    poly = [(5, 3), (15, 3), (15, 9), (5, 9)]        # encloses chains 0 and 2
    assert chains_in_polygon(CHAINS, poly) == [0, 2]


def test_lasso_degenerate_polygon():
    assert chains_in_polygon(CHAINS, [(0, 0), (1, 1)]) == []
