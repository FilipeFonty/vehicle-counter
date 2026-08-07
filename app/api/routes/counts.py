"""Endpoints de contagem, eventos e desempenho."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_line_counter, get_performance_tracker
from app.core.line_counter import LineCounter
from app.core.performance import PerformanceTracker
from app.models.event import CountsSummary
from app.models.responses import EventsResponse, PerformanceResponse

router = APIRouter(prefix="/api/v1", tags=["counting"])


@router.get("/counts", response_model=CountsSummary)
async def get_counts(counter: LineCounter = Depends(get_line_counter)) -> CountsSummary:
    """Contagens acumuladas por categoria e sentido."""
    return counter.summary()


@router.get("/events", response_model=EventsResponse)
async def get_events(
    limit: int = Query(default=100, ge=1, le=1000),
    counter: LineCounter = Depends(get_line_counter),
) -> EventsResponse:
    """Ultimos eventos de cruzamento registrados."""
    eventos = counter.events
    return EventsResponse(total=len(eventos), events=eventos[-limit:])


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance(
    perf: PerformanceTracker = Depends(get_performance_tracker),
) -> PerformanceResponse:
    """Latencias observadas: media, p50, p95 e FPS efetivo."""
    return perf.summary()


@router.post("/counts/reset", status_code=204)
async def reset_counts(counter: LineCounter = Depends(get_line_counter)) -> None:
    """Zera contagens e eventos (util em demonstracoes)."""
    counter.reset()