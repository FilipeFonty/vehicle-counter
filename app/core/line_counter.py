"""Contagem por cruzamento de linha virtual.

Implementa o item 5 do enunciado. A logica e puramente geometrica e
independe do detector, o que permite testa-la sem carregar o modelo.

Principio central: a distancia perpendicular COM SINAL entre o ponto
representativo do veiculo e a linha. O sinal indica o lado; o modulo
alimenta a histerese. Como usa produto vetorial, funciona com linhas
de qualquer inclinacao -- nao apenas horizontais.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.config import CountingConfig
from app.models.detection import Detection
from app.models.event import ClassCounts, CountingEvent, CountsSummary, Direction


def signed_distance_to_line(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    """Distancia perpendicular com sinal entre um ponto e uma reta.

    Usa o produto vetorial 2D: (B-A) x (P-A), normalizado pelo
    comprimento de AB. Positivo de um lado, negativo do outro.
    """
    ax, ay = line_start
    bx, by = line_end
    px, py = point

    dx, dy = bx - ax, by - ay
    comprimento = (dx * dx + dy * dy) ** 0.5
    if comprimento == 0:
        raise ValueError("line_start e line_end nao podem coincidir")

    cross = dx * (py - ay) - dy * (px - ax)
    return cross / comprimento


@dataclass
class TrackState:
    """Estado acumulado de um track entre frames."""

    track_id: int
    # Ultimo lado CONFIRMADO (fora da zona morta). None ate a primeira
    # observacao confiavel.
    confirmed_side: int | None = None
    counted: bool = False
    last_seen_frame: int = 0
    # Votos de classe ponderados pela confianca, para estabilizar
    # veiculos que oscilam entre categorias.
    class_votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    best_confidence: float = 0.0

    def register_class(self, class_name: str, confidence: float) -> None:
        self.class_votes[class_name] += confidence
        self.best_confidence = max(self.best_confidence, confidence)

    @property
    def stable_class(self) -> str:
        """Classe vencedora pela soma das confiancas."""
        if not self.class_votes:
            return "unknown"
        return max(self.class_votes.items(), key=lambda item: item[1])[0]


class LineCounter:
    """Mantem o estado dos tracks e emite eventos de cruzamento."""

    def __init__(self, config: CountingConfig, camera_id: str = "gate-01") -> None:
        self.config = config
        self.camera_id = camera_id

        self.line_start = (float(config.line_start[0]), float(config.line_start[1]))
        self.line_end = (float(config.line_end[0]), float(config.line_end[1]))
        self.hysteresis = float(config.hysteresis_pixels)
        self.expiry_frames = config.track_expiry_frames

        self._states: dict[int, TrackState] = {}
        self._events: list[CountingEvent] = []
        self._counts_entry = ClassCounts()
        self._counts_exit = ClassCounts()
        self._frame_number = 0

    # ------------------------------------------------------------------
    # Atualizacao
    # ------------------------------------------------------------------
    def update(
        self,
        tracked: list[tuple[int, Detection]],
        frame_number: int | None = None,
    ) -> list[CountingEvent]:
        """Processa um frame e devolve os eventos gerados nele.

        `tracked` e uma lista de pares (track_id, Detection).
        """
        if frame_number is not None:
            self._frame_number = frame_number
        else:
            self._frame_number += 1

        novos: list[CountingEvent] = []

        for track_id, deteccao in tracked:
            evento = self._processar_track(track_id, deteccao)
            if evento is not None:
                novos.append(evento)

        self._expirar_tracks()
        return novos

    def _processar_track(
        self, track_id: int, deteccao: Detection
    ) -> CountingEvent | None:
        estado = self._states.get(track_id)
        if estado is None:
            estado = TrackState(track_id=track_id)
            self._states[track_id] = estado

        estado.last_seen_frame = self._frame_number
        estado.register_class(deteccao.class_name, deteccao.confidence)

        ponto = deteccao.bounding_box.bottom_center
        distancia = signed_distance_to_line(ponto, self.line_start, self.line_end)

        # Zona morta: dentro da histerese nao confirmamos lado nenhum.
        # E o que impede um veiculo parado sobre a linha de oscilar.
        if abs(distancia) < self.hysteresis:
            return None

        lado_atual = 1 if distancia > 0 else -1

        # Primeira observacao confiavel: apenas inicializa. Um veiculo
        # que ja comeca do outro lado nao deve ser contado.
        if estado.confirmed_side is None:
            estado.confirmed_side = lado_atual
            return None

        if lado_atual == estado.confirmed_side:
            return None

        # Houve transicao confirmada de lado.
        anterior = estado.confirmed_side
        estado.confirmed_side = lado_atual

        if estado.counted:
            return None

        estado.counted = True
        direcao = Direction.ENTRY if anterior < 0 else Direction.EXIT

        evento = CountingEvent(
            track_id=track_id,
            vehicle_class=estado.stable_class,
            confidence=round(estado.best_confidence, 3),
            direction=direcao,
            camera_id=self.camera_id,
            frame_number=self._frame_number,
        )

        alvo = self._counts_entry if direcao is Direction.ENTRY else self._counts_exit
        alvo.increment(evento.vehicle_class)
        self._events.append(evento)

        return evento

    def _expirar_tracks(self) -> None:
        """Remove estados antigos para nao crescer indefinidamente."""
        limite = self._frame_number - self.expiry_frames
        expirados = [
            tid for tid, st in self._states.items() if st.last_seen_frame < limite
        ]
        for track_id in expirados:
            del self._states[track_id]

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    @property
    def events(self) -> list[CountingEvent]:
        return list(self._events)

    @property
    def active_tracks(self) -> int:
        return len(self._states)

    def summary(self) -> CountsSummary:
        return CountsSummary(
            entry=self._counts_entry.model_copy(),
            exit=self._counts_exit.model_copy(),
            total_events=len(self._events),
            camera_id=self.camera_id,
        )

    def reset(self) -> None:
        self._states.clear()
        self._events.clear()
        self._counts_entry = ClassCounts()
        self._counts_exit = ClassCounts()
        self._frame_number = 0