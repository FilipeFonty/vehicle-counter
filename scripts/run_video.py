"""Executa o pipeline completo sobre um video local."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import load_settings
from app.core.detector import Detector
from app.core.line_counter import LineCounter
from app.core.performance import PerformanceTracker
from app.core.video_processor import VideoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(BASE_DIR / "data" / "test.mp4"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    detector = Detector(settings)
    contador = LineCounter(settings.counting, camera_id=settings.application.camera_id)
    perf = PerformanceTracker()

    processador = VideoProcessor(settings, detector, contador, perf)

    print(f"Video   : {args.video}")
    print(f"Linha   : {settings.counting.line_start} -> {settings.counting.line_end}")
    print(f"Stride  : {args.stride}")
    print("Processando...\n")

    resultado = processador.process(
        args.video,
        save_annotated=not args.no_save,
        max_frames=args.max_frames,
        frame_stride=args.stride,
    )

    print(f"Frames processados : {resultado.frames_processed}")
    print(f"Tempo total        : {resultado.processing_time_s}s")
    print(f"FPS efetivo        : {resultado.effective_fps}")
    print(f"Eventos            : {resultado.total_events}")

    resumo = contador.summary()
    print(f"\nENTRADAS: {resumo.entry.total}  {resumo.entry.model_dump()}")
    print(f"SAIDAS  : {resumo.exit.total}  {resumo.exit.model_dump()}")

    if resultado.events:
        print("\nPrimeiros eventos:")
        for evento in resultado.events[:10]:
            print(
                f"  #{evento.track_id:3d} {evento.vehicle_class:11s} "
                f"{evento.direction.value:5s} conf={evento.confidence:.2f} "
                f"frame={evento.frame_number}"
            )

    metricas = perf.summary()
    print(
        f"\nLatencia: media={metricas.mean_ms}ms  p50={metricas.p50_ms}ms  "
        f"p95={metricas.p95_ms}ms  FPS={metricas.effective_fps}"
    )

    if resultado.annotated_video_path:
        print(f"\nVideo anotado: {resultado.annotated_video_path}")


if __name__ == "__main__":
    main()