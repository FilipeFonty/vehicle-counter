"""Aplicacao FastAPI do contador de veiculos."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.dependencies import get_detector
from app.api.routes import counts, inference, videos
from app.config import get_settings
from app.models.responses import HealthResponse

app = FastAPI(
    title="Vehicle Counter",
    description=(
        "Contagem e classificacao de veiculos em entradas de complexos "
        "industriais. Deteccao com YOLO26n via ONNX Runtime, rastreamento "
        "ByteTrack e contagem por cruzamento de linha virtual."
    ),
    version="0.1.0",
)

app.include_router(inference.router)
app.include_router(counts.router)
app.include_router(videos.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Health check usado pelo Docker."""
    try:
        detector = get_detector()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_format=detector.output_format,
        )
    except Exception:  # noqa: BLE001
        return HealthResponse(status="degraded", model_loaded=False, model_format="unknown")


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    settings = get_settings()
    return {
        "application": "vehicle-counter",
        "camera_id": settings.application.camera_id,
        "docs": "/docs",
    }