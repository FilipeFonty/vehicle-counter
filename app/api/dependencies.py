"""Instancias compartilhadas da aplicacao.

O detector carrega a sessao ONNX uma unica vez: recarregar por
requisicao adicionaria centenas de milissegundos.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.core.detector import Detector
from app.core.line_counter import LineCounter
from app.core.performance import PerformanceTracker


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector(get_settings())


@lru_cache(maxsize=1)
def get_line_counter() -> LineCounter:
    settings: Settings = get_settings()
    return LineCounter(settings.counting, camera_id=settings.application.camera_id)


@lru_cache(maxsize=1)
def get_performance_tracker() -> PerformanceTracker:
    return PerformanceTracker()