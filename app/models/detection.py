"""Contratos de dados das deteccoes.

Sao usados tanto internamente quanto nas respostas da API, garantindo
que o formato documentado no Swagger reflita o que o codigo produz.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Normalizacao exigida pelo item 3 do enunciado.
COCO_VEHICLE_NAMES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def normalize_class_name(class_id: int) -> str:
    """Converte o id COCO no nome normalizado da aplicacao."""
    return COCO_VEHICLE_NAMES.get(class_id, f"unknown_{class_id}")


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ponto de referencia para a contagem (item 5.1 do enunciado).

        O centro inferior aproxima o contato do veiculo com o solo,
        sendo mais estavel que o centroide quando a caixa muda de
        tamanho ao se aproximar da camera.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox

    @classmethod
    def from_raw(
        cls, x1: float, y1: float, x2: float, y2: float, confidence: float, class_id: int
    ) -> Detection:
        return cls(
            class_id=class_id,
            class_name=normalize_class_name(class_id),
            confidence=confidence,
            bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        )


class PerformanceMetrics(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class ImageInfo(BaseModel):
    width: int
    height: int


class DetectionResult(BaseModel):
    image: ImageInfo
    detections: list[Detection]
    performance: PerformanceMetrics