"""Contratos de eventos de contagem."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


def _agora_utc() -> datetime:
    return datetime.now(timezone.utc)


class CountingEvent(BaseModel):
    """Registro unico de um veiculo cruzando a linha virtual."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    track_id: int
    vehicle_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    direction: Direction
    timestamp: datetime = Field(default_factory=_agora_utc)
    camera_id: str = "gate-01"
    frame_number: int | None = None


class ClassCounts(BaseModel):
    car: int = 0
    motorcycle: int = 0
    bus: int = 0
    truck: int = 0

    def increment(self, class_name: str) -> None:
        if hasattr(self, class_name):
            setattr(self, class_name, getattr(self, class_name) + 1)

    @property
    def total(self) -> int:
        return self.car + self.motorcycle + self.bus + self.truck


class CountsSummary(BaseModel):
    """Resposta do endpoint GET /api/v1/counts."""

    entry: ClassCounts = Field(default_factory=ClassCounts)
    exit: ClassCounts = Field(default_factory=ClassCounts)
    total_events: int = 0
    camera_id: str = "gate-01"