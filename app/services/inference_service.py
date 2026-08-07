"""Orquestracao entre detector, tracker, contador e anotacao."""

from __future__ import annotations

import cv2
import numpy as np

from app.models.detection import DetectionResult


def decode_image(raw: bytes) -> np.ndarray:
    """Converte bytes recebidos via HTTP em imagem BGR.

    Levanta ValueError se o conteudo nao for uma imagem decodificavel,
    cobrindo os casos de teste 13 e 14 do enunciado.
    """
    if not raw:
        raise ValueError("Arquivo vazio")

    buffer = np.frombuffer(raw, dtype=np.uint8)
    imagem = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if imagem is None:
        raise ValueError("Conteudo nao e uma imagem valida")

    return imagem


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Falha ao codificar PNG")
    return buffer.tobytes()


def result_to_track_ids(result: DetectionResult) -> list[int | None]:
    """Placeholder de ids para anotacao de imagem estatica."""
    return [None] * len(result.detections)