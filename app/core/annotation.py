"""Desenho de caixas, rotulos e linha de contagem."""

from __future__ import annotations

import cv2
import numpy as np

from app.models.detection import Detection

# BGR, uma cor por classe.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 200, 0),
    "motorcycle": (0, 200, 255),
    "bus": (255, 120, 0),
    "truck": (0, 0, 220),
}
DEFAULT_COLOR = (200, 200, 200)
LINE_COLOR = (0, 255, 255)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    track_ids: list[int] | None = None,
) -> np.ndarray:
    """Desenha caixas com classe, confianca e (opcionalmente) track id."""
    saida = image.copy()
    ids = track_ids or [None] * len(detections)

    for deteccao, track_id in zip(detections, ids):
        caixa = deteccao.bounding_box
        x1, y1 = int(caixa.x1), int(caixa.y1)
        x2, y2 = int(caixa.x2), int(caixa.y2)
        cor = CLASS_COLORS.get(deteccao.class_name, DEFAULT_COLOR)

        cv2.rectangle(saida, (x1, y1), (x2, y2), cor, 2)

        rotulo = f"{deteccao.class_name} {deteccao.confidence:.2f}"
        if track_id is not None:
            rotulo = f"#{track_id} {rotulo}"

        (largura, altura), _ = cv2.getTextSize(
            rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        topo = max(y1 - altura - 6, 0)
        cv2.rectangle(saida, (x1, topo), (x1 + largura + 4, topo + altura + 6), cor, -1)
        cv2.putText(
            saida,
            rotulo,
            (x1 + 2, topo + altura + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return saida


def draw_counting_line(
    image: np.ndarray,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
) -> np.ndarray:
    saida = image.copy()
    cv2.line(saida, tuple(map(int, line_start)), tuple(map(int, line_end)), LINE_COLOR, 3)
    return saida


def draw_counts_overlay(image: np.ndarray, entry_total: int, exit_total: int) -> np.ndarray:
    saida = image.copy()
    texto = f"IN: {entry_total}   OUT: {exit_total}"
    cv2.rectangle(saida, (10, 10), (260, 55), (0, 0, 0), -1)
    cv2.putText(
        saida, texto, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
    return saida