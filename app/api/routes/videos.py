"""Endpoint de analise de video.

Processamento sincrono, conforme item 6 do enunciado. O video e
referenciado por caminho relativo ao diretorio data/, e nao enviado por
upload: videos sao grandes e o processamento e demorado, o que tornaria
uma requisicao multipart impraticavel.

Para limitar a duracao da requisicao, use max_frames e stride.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import (
    get_detector,
    get_line_counter,
    get_performance_tracker,
)
from app.config import BASE_DIR, Settings, get_settings
from app.core.detector import Detector
from app.core.line_counter import LineCounter
from app.core.performance import PerformanceTracker
from app.core.video_processor import VideoProcessor
from app.models.responses import VideoAnalysisResponse

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])

DATA_DIR = BASE_DIR / "data"


def _resolver_caminho(nome: str) -> Path:
    """Resolve o nome do arquivo dentro de data/, bloqueando path traversal."""
    candidato = (DATA_DIR / nome).resolve()

    # Impede que '../../etc/passwd' escape do diretorio permitido.
    if not candidato.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(
            status_code=400, detail="Caminho fora do diretorio permitido"
        )

    if not candidato.exists():
        raise HTTPException(
            status_code=404, detail=f"Video nao encontrado em data/: {nome}"
        )

    return candidato


@router.get("/available", response_model=list[str], summary="Lista videos disponiveis")
async def listar_videos() -> list[str]:
    """Nomes dos videos presentes no diretorio data/."""
    if not DATA_DIR.exists():
        return []
    extensoes = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(
        arquivo.name
        for arquivo in DATA_DIR.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes
    )


@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analisar_video(
    filename: str = Query(..., description="Nome do arquivo em data/, ex: test.mp4"),
    max_frames: int | None = Query(
        default=300, ge=1, description="Limite de frames; use null para o video inteiro"
    ),
    stride: int = Query(
        default=1, ge=1, le=30, description="Processa 1 a cada N frames"
    ),
    save_annotated: bool = Query(default=True, description="Grava o video anotado"),
    reset_counts: bool = Query(
        default=True, description="Zera contagens antes de processar"
    ),
    settings: Settings = Depends(get_settings),
    detector: Detector = Depends(get_detector),
    counter: LineCounter = Depends(get_line_counter),
    perf: PerformanceTracker = Depends(get_performance_tracker),
) -> VideoAnalysisResponse:
    """Executa o pipeline completo sobre um video: deteccao, rastreamento,
    contagem por cruzamento de linha e registro de eventos.

    ATENCAO: processamento sincrono. Um video longo mantem a conexao HTTP
    aberta durante todo o processamento. Use max_frames para limitar.
    """
    caminho = _resolver_caminho(filename)

    if reset_counts:
        counter.reset()

    processador = VideoProcessor(settings, detector, counter, perf)

    try:
        return processador.process(
            caminho,
            save_annotated=save_annotated,
            max_frames=max_frames,
            frame_stride=stride,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
