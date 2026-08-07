"""Adaptador do ByteTrack via biblioteca supervision.

DECISAO TECNICA
---------------
O ByteTrack embutido no Ultralytics foi avaliado primeiro, por ja estar
instalado. Optou-se por nao usa-lo diretamente: sua interface e privada
(BYTETracker._split_detections, _format_output, parse_bboxes) e mudou
de assinatura entre versoes menores da 8.4.x.

A biblioteca supervision expoe o MESMO algoritmo ByteTrack com API
publica e estavel. O algoritmo e os parametros sao equivalentes; muda
apenas o ponto de acoplamento.

Impacto: precisao inalterada, latencia inalterada, menos codigo proprio.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from app.config import TrackingConfig
from app.models.detection import Detection


class ByteTrackAdapter:
    """Encapsula o ByteTrack mantendo a interface do projeto.

    Recebe list[Detection] e devolve list[tuple[track_id, Detection]],
    de modo que o LineCounter permanece independente do tracker.
    """

    def __init__(self, config: TrackingConfig, frame_rate: int = 30) -> None:
        self.config = config
        self.frame_rate = frame_rate
        self._tracker = self._build()

    def _build(self) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=self.config.track_high_thresh,
            lost_track_buffer=self.config.track_buffer,
            minimum_matching_threshold=self.config.match_thresh,
            frame_rate=self.frame_rate,
        )

    def update(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        """Associa deteccoes a tracks e devolve pares (track_id, Detection)."""
        sv_detections = self._to_supervision(detections)
        rastreadas = self._tracker.update_with_detections(sv_detections)

        if len(rastreadas) == 0:
            return []

        pares: list[tuple[int, Detection]] = []
        for indice in range(len(rastreadas)):
            track_id = rastreadas.tracker_id[indice]
            if track_id is None:
                continue

            x1, y1, x2, y2 = rastreadas.xyxy[indice]
            class_id = int(rastreadas.class_id[indice])
            confianca = float(rastreadas.confidence[indice])

            # A caixa vem do tracker (suavizada pelo filtro de Kalman),
            # nao da deteccao bruta -- mais estavel para a contagem.
            pares.append(
                (
                    int(track_id),
                    Detection.from_raw(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=confianca,
                        class_id=class_id,
                    ),
                )
            )

        return pares

    @staticmethod
    def _to_supervision(detections: list[Detection]) -> sv.Detections:
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array(
            [
                [d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2]
                for d in detections
            ],
            dtype=np.float32,
        )
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def reset(self) -> None:
        """Reinicia o tracker (novo video, contagem do zero)."""
        self._tracker = self._build()