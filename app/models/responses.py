"""Contratos de resposta da API."""

from __future__ import annotations

from pydantic import BaseModel

from app.models.event import CountingEvent


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_format: str
    version: str = "0.1.0"


class EventsResponse(BaseModel):
    total: int
    events: list[CountingEvent]


class VideoAnalysisResponse(BaseModel):
    frames_processed: int
    processing_time_s: float
    effective_fps: float
    total_events: int
    events: list[CountingEvent]
    annotated_video_path: str | None = None


class PerformanceResponse(BaseModel):
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    effective_fps: float


class ErrorResponse(BaseModel):
    detail: str