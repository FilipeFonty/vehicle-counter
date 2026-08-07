"""Endpoints de inferencia sobre imagem."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.dependencies import get_detector, get_performance_tracker
from app.core.annotation import draw_detections
from app.core.detector import Detector
from app.core.performance import PerformanceTracker
from app.models.detection import DetectionResult
from app.services.inference_service import decode_image, encode_png

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


def _ler_imagem(file: UploadFile | None):
    if file is None:
        raise HTTPException(status_code=422, detail="Nenhum arquivo enviado")
    return file


@router.post("/json", response_model=DetectionResult)
async def inference_json(
    file: UploadFile = File(...),
    detector: Detector = Depends(get_detector),
    perf: PerformanceTracker = Depends(get_performance_tracker),
) -> DetectionResult:
    """Detecta veiculos e devolve as deteccoes em JSON."""
    conteudo = await file.read()

    try:
        imagem = decode_image(conteudo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resultado = detector.detect(imagem)
    perf.record(resultado.performance.total_ms)
    return resultado


@router.post(
    "/annotated",
    responses={200: {"content": {"image/png": {}}, "description": "Imagem anotada"}},
)
async def inference_annotated(
    file: UploadFile = File(...),
    detector: Detector = Depends(get_detector),
    perf: PerformanceTracker = Depends(get_performance_tracker),
) -> Response:
    """Detecta veiculos e devolve a imagem anotada em PNG."""
    conteudo = await file.read()

    try:
        imagem = decode_image(conteudo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resultado = detector.detect(imagem)
    perf.record(resultado.performance.total_ms)

    anotada = draw_detections(imagem, resultado.detections)
    png = encode_png(anotada)

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Detections-Count": str(len(resultado.detections)),
            "X-Inference-Ms": str(resultado.performance.inference_ms),
        },
    )