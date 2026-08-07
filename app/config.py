"""Carregamento e validacao da configuracao via Pydantic.

Centralizar a configuracao evita valores magicos espalhados pelo codigo
e atende ao item 5 do enunciado: a linha de contagem deve ser
configuravel, nao escrita no codigo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "default.yaml"


class ModelConfig(BaseModel):
    path: str
    format: str = "auto"
    input_size: int = 640
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    allowed_classes: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])
    intra_op_num_threads: int = Field(default=4, ge=1)

    @field_validator("format")
    @classmethod
    def validar_formato(cls, valor: str) -> str:
        permitidos = {"auto", "yolo26", "yolo11"}
        if valor not in permitidos:
            raise ValueError(f"format deve ser um de {permitidos}, recebido: {valor}")
        return valor


class FilteringConfig(BaseModel):
    apply_nms: bool = True
    nms_class_agnostic: bool = True
    min_box_area: float = Field(default=0.0, ge=0.0)
    roi_polygon: list[list[int]] | None = None

    @field_validator("roi_polygon")
    @classmethod
    def validar_poligono(cls, valor: list[list[int]] | None) -> list[list[int]] | None:
        if valor is None:
            return None
        if len(valor) < 3:
            raise ValueError("roi_polygon precisa de ao menos 3 vertices")
        for ponto in valor:
            if len(ponto) != 2:
                raise ValueError(f"Vertice invalido (esperado [x, y]): {ponto}")
        return valor


class TrackingConfig(BaseModel):
    tracker: str = "bytetrack"
    track_buffer: int = 60
    track_high_thresh: float = 0.40
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.55
    match_thresh: float = 0.85


class CountingConfig(BaseModel):
    line_start: list[int]
    line_end: list[int]
    hysteresis_pixels: int = Field(default=15, ge=0)
    track_expiry_frames: int = Field(default=90, ge=1)

    @field_validator("line_start", "line_end")
    @classmethod
    def validar_ponto(cls, valor: list[int]) -> list[int]:
        if len(valor) != 2:
            raise ValueError(f"Ponto da linha deve ser [x, y], recebido: {valor}")
        return valor


class ApplicationConfig(BaseModel):
    save_annotated_images: bool = True
    output_directory: str = "outputs"
    camera_id: str = "gate-01"


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}

    model: ModelConfig
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    counting: CountingConfig
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)

    @property
    def model_absolute_path(self) -> Path:
        caminho = Path(self.model.path)
        return caminho if caminho.is_absolute() else BASE_DIR / caminho

    @property
    def output_absolute_path(self) -> Path:
        caminho = Path(self.application.output_directory)
        return caminho if caminho.is_absolute() else BASE_DIR / caminho


def load_settings(config_path: Path | str | None = None) -> Settings:
    caminho = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not caminho.exists():
        raise FileNotFoundError(f"Configuracao nao encontrada: {caminho}")
    with caminho.open("r", encoding="utf-8") as arquivo:
        dados = yaml.safe_load(arquivo)
    return Settings(**dados)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cache para evitar reler o YAML a cada requisicao da API."""
    return load_settings()