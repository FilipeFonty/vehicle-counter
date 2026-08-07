"""Processamento de video: detecta, rastreia, conta e anota.

Execucao sincrona, conforme item 6 do enunciado -- sem fila de
mensagens ou banco antes de o nucleo funcionar.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from app.config import Settings
from app.core.annotation import draw_counting_line, draw_counts_overlay, draw_detections
from app.core.detector import Detector
from app.core.line_counter import LineCounter
from app.core.performance import PerformanceTracker
from app.core.tracker import ByteTrackAdapter
from app.models.responses import VideoAnalysisResponse


class VideoProcessor:
    """Encadeia detector, tracker e contador ao longo de um video."""

    def __init__(
        self,
        settings: Settings,
        detector: Detector,
        counter: LineCounter,
        performance: PerformanceTracker | None = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.counter = counter
        self.performance = performance or PerformanceTracker()

    def process(
        self,
        video_path: Path | str,
        save_annotated: bool = True,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> VideoAnalysisResponse:
        """Processa o video e devolve os eventos de contagem.

        frame_stride > 1 processa 1 a cada N frames -- estrategia de
        reducao de custo computacional prevista para o Raspberry Pi 5.
        """
        caminho = Path(video_path)
        if not caminho.exists():
            raise FileNotFoundError(f"Video nao encontrado: {caminho}")

        captura = cv2.VideoCapture(str(caminho))
        if not captura.isOpened():
            raise ValueError(f"Nao foi possivel abrir o video: {caminho}")

        fps_origem = captura.get(cv2.CAP_PROP_FPS) or 30.0
        largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tracker novo a cada video: ids nao devem vazar entre execucoes.
        tracker = ByteTrackAdapter(self.settings.tracking, frame_rate=int(fps_origem))
        self.counter.reset()

        escritor = None
        caminho_saida: Path | None = None
        if save_annotated:
            diretorio = self.settings.output_absolute_path
            diretorio.mkdir(parents=True, exist_ok=True)
            caminho_saida = diretorio / f"{caminho.stem}_annotated.mp4"
            escritor = cv2.VideoWriter(
                str(caminho_saida),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_origem / frame_stride,
                (largura, altura),
            )

        linha_inicio = tuple(self.settings.counting.line_start)
        linha_fim = tuple(self.settings.counting.line_end)

        frames = 0
        lidos = 0
        inicio = time.perf_counter()

        try:
            while True:
                ok, frame = captura.read()
                if not ok:
                    break

                lidos += 1
                if (lidos - 1) % frame_stride != 0:
                    continue

                resultado = self.detector.detect(frame)
                self.performance.record(resultado.performance.total_ms)

                pares = tracker.update(resultado.detections)
                self.counter.update(pares, frame_number=frames)
                frames += 1

                if escritor is not None:
                    anotado = draw_counting_line(frame, linha_inicio, linha_fim)
                    anotado = draw_detections(
                        anotado,
                        [d for _, d in pares],
                        [tid for tid, _ in pares],
                    )
                    resumo = self.counter.summary()
                    anotado = draw_counts_overlay(
                        anotado, resumo.entry.total, resumo.exit.total
                    )
                    escritor.write(anotado)

                if max_frames is not None and frames >= max_frames:
                    break
        finally:
            captura.release()
            if escritor is not None:
                escritor.release()

        decorrido = time.perf_counter() - inicio
        eventos = self.counter.events

        return VideoAnalysisResponse(
            frames_processed=frames,
            processing_time_s=round(decorrido, 2),
            effective_fps=round(frames / decorrido, 2) if decorrido > 0 else 0.0,
            total_events=len(eventos),
            events=eventos,
            annotated_video_path=str(caminho_saida) if caminho_saida else None,
        )