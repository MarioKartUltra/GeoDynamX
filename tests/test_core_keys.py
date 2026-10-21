# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""core.keys derives the cache key; engine.cache re-exports it (the layering gate). A key is the
same string from either import path, so every cached entry and recorded analysis key stays valid."""
from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("module", ["dynamix.core.keys", "dynamix.engine.cache"])
def test_cache_key_strings_are_pinned(module):
    cache_key = importlib.import_module(module).cache_key
    assert cache_key("wtmm2d", "a" * 64, {"n_oct": 4, "n_voice": 8, "precision": "f64"}) \
        == "1d4bdb71528e76018aaee3101592eb0b"
    params = {"shape": (64, 48), "scales": np.array([1.5, 3.0]), "wavelet": "gaussian",
              "count": np.int64(3)}
    assert cache_key("roi-cwt-levels", "", params, upstream="0f" * 16,
                     origin="/plugins/probe.py") == "a1f50a7f344c6e2bf7f086d4082e93d5"
