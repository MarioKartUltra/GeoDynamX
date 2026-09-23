# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The compute engine: resolve a layer's chain to something renderable.

Transforms are expensive and cached; filters are cheap and run on every redraw. That split is what
makes the scale slider instant -- moving it is a lookup into an already-computed stack, never a
recomputation.
"""
from dynamix.engine.cache import Cache, cache_key
from dynamix.engine.resolve import Renderable, resolve, source_identity

__all__ = ["Cache", "cache_key", "Renderable", "resolve", "source_identity"]
