"""Analysis-modality registry seam (pure Python; no GUI, no reflayers runtime imports).

The AnalysisModality Protocol defines the interface for optional analysis backends
(WTMM, nPCF, recurrence, etc.). Each modality is pure and headless: it computes
over RasterField inputs and produces a result dict; the scene overlay and detail-zone
widget are produced separately by the GUI layer and optional modality callbacks.
Phase A ships the seam + tests; modality implementations arrive in Phase D.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.reflayers import RefLayer


class AnalysisModality(Protocol):
    """Protocol for optional analysis modalities.

    name: str
        Unique identifier (e.g. "wtmm", "npcf", "recurrence").

    compute(field, params, *, out_dir=None, progress=None) -> dict
        Pure, headless computation over a raster field. Returns a result dict
        suitable for overlay() and detail_widget_factory() callbacks.

    overlay(result) -> RefLayer | None
        Convert a compute result to a scene overlay layer, or None.

    detail_widget_factory() -> Callable | None
        Return a zero-arg callable that constructs a QWidget, or None (headless).
    """
    name: str

    def compute(
        self,
        field: RasterField,
        params: dict,
        *,
        out_dir: str | None = None,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict:
        """Compute analysis result from a raster field."""
        ...

    def overlay(self, result: dict) -> RefLayer | None:
        """Convert result to a scene overlay layer, or None."""
        ...

    def detail_widget_factory(self) -> Callable[[], object] | None:
        """Return a zero-arg callable constructing a QWidget, or None."""
        ...


MODALITIES: dict[str, AnalysisModality] = {}
"""Registry of available analysis modalities. Phase A populates none."""


def register_modality(m: AnalysisModality) -> None:
    """Register a modality by name.

    Raises ValueError if a modality with the same name is already registered.
    """
    if m.name in MODALITIES:
        raise ValueError(f"modality '{m.name}' is already registered")
    MODALITIES[m.name] = m
