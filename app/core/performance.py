"""Coleta de latencias para o endpoint de performance."""

from __future__ import annotations

from collections import deque
from statistics import mean

from app.models.responses import PerformanceResponse


class PerformanceTracker:
    """Janela deslizante de latencias totais por inferencia."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._samples: deque[float] = deque(maxlen=max_samples)

    def record(self, total_ms: float) -> None:
        self._samples.append(total_ms)

    def summary(self) -> PerformanceResponse:
        if not self._samples:
            return PerformanceResponse(
                samples=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0,
                min_ms=0.0, max_ms=0.0, effective_fps=0.0,
            )

        ordenadas = sorted(self._samples)
        media = mean(ordenadas)

        return PerformanceResponse(
            samples=len(ordenadas),
            mean_ms=round(media, 2),
            p50_ms=round(self._percentil(ordenadas, 50), 2),
            p95_ms=round(self._percentil(ordenadas, 95), 2),
            min_ms=round(ordenadas[0], 2),
            max_ms=round(ordenadas[-1], 2),
            effective_fps=round(1000.0 / media, 2) if media > 0 else 0.0,
        )

    @staticmethod
    def _percentil(ordenadas: list[float], percentil: float) -> float:
        if not ordenadas:
            return 0.0
        indice = int(round((percentil / 100.0) * (len(ordenadas) - 1)))
        return ordenadas[max(0, min(indice, len(ordenadas) - 1))]

    def reset(self) -> None:
        self._samples.clear()