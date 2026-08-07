"""Testes da geometria de contagem.

Nenhum teste carrega o modelo: as deteccoes sao sinteticas, conforme
o item 10 do enunciado.
"""

from __future__ import annotations

import pytest

from app.config import CountingConfig
from app.core.line_counter import LineCounter, signed_distance_to_line
from app.models.detection import Detection
from app.models.event import Direction


def fazer_deteccao(cx: float, cy: float, class_id: int = 2, conf: float = 0.9) -> Detection:
    """Cria uma deteccao cujo centro inferior fica em (cx, cy)."""
    return Detection.from_raw(
        x1=cx - 25, y1=cy - 50, x2=cx + 25, y2=cy, confidence=conf, class_id=class_id
    )


@pytest.fixture
def config_horizontal() -> CountingConfig:
    return CountingConfig(
        line_start=[0, 400],
        line_end=[1000, 400],
        hysteresis_pixels=15,
        track_expiry_frames=90,
    )


# ----------------------------------------------------------------------
# Geometria basica
# ----------------------------------------------------------------------
def test_distancia_com_sinal_troca_de_lado():
    inicio, fim = (0.0, 100.0), (200.0, 100.0)
    acima = signed_distance_to_line((100.0, 50.0), inicio, fim)
    abaixo = signed_distance_to_line((100.0, 150.0), inicio, fim)
    assert acima * abaixo < 0, "Lados opostos devem ter sinais opostos"


def test_distancia_sobre_a_linha_e_zero():
    d = signed_distance_to_line((50.0, 100.0), (0.0, 100.0), (200.0, 100.0))
    assert abs(d) < 1e-9


def test_distancia_funciona_com_linha_inclinada():
    """A direcao nao pode depender apenas de Y (item 5 do enunciado)."""
    inicio, fim = (0.0, 0.0), (100.0, 100.0)
    um_lado = signed_distance_to_line((100.0, 0.0), inicio, fim)
    outro = signed_distance_to_line((0.0, 100.0), inicio, fim)
    assert um_lado * outro < 0


def test_linha_degenerada_levanta_erro():
    with pytest.raises(ValueError):
        signed_distance_to_line((0.0, 0.0), (10.0, 10.0), (10.0, 10.0))


# ----------------------------------------------------------------------
# Casos de teste do item 10
# ----------------------------------------------------------------------
def test_caso_1_carro_unico_atravessando(config_horizontal):
    """Um carro cruzando de cima para baixo gera exatamente um evento."""
    contador = LineCounter(config_horizontal)

    for y in (300, 350, 380, 420, 450, 500):
        contador.update([(1, fazer_deteccao(500, y))])

    assert contador.summary().total_events == 1
    assert contador.events[0].vehicle_class == "car"


def test_caso_4_veiculo_parado_sobre_a_linha(config_horizontal):
    """Oscilar dentro da histerese nao pode gerar contagem."""
    contador = LineCounter(config_horizontal)

    for y in (398, 402, 399, 401, 400, 403, 397):
        contador.update([(1, fazer_deteccao(500, y))])

    assert contador.summary().total_events == 0


def test_caso_5_veiculo_aproxima_e_recua(config_horizontal):
    """Chegar perto da linha e voltar nao conta."""
    contador = LineCounter(config_horizontal)

    for y in (300, 350, 380, 390, 380, 350, 300):
        contador.update([(1, fazer_deteccao(500, y))])

    assert contador.summary().total_events == 0


def test_veiculo_que_comeca_do_outro_lado(config_horizontal):
    """Comecar alem da linha apenas inicializa o estado."""
    contador = LineCounter(config_horizontal)

    for y in (500, 550, 600):
        contador.update([(1, fazer_deteccao(500, y))])

    assert contador.summary().total_events == 0


def test_direcoes_opostas_sao_distinguidas(config_horizontal):
    contador = LineCounter(config_horizontal)

    for y in (300, 500):
        contador.update([(1, fazer_deteccao(400, y))])
    for y in (500, 300):
        contador.update([(2, fazer_deteccao(600, y))])

    direcoes = {e.track_id: e.direction for e in contador.events}
    assert direcoes[1] != direcoes[2]


def test_contagem_duplicada_e_bloqueada(config_horizontal):
    """Ir e voltar pela linha conta apenas a primeira travessia."""
    contador = LineCounter(config_horizontal)

    for y in (300, 500, 300, 500):
        contador.update([(1, fazer_deteccao(500, y))])

    assert contador.summary().total_events == 1


def test_caso_2_dois_veiculos_proximos(config_horizontal):
    """IDs distintos geram eventos independentes."""
    contador = LineCounter(config_horizontal)

    for y in (300, 350, 450, 500):
        contador.update(
            [(1, fazer_deteccao(480, y)), (2, fazer_deteccao(560, y))]
        )

    assert contador.summary().total_events == 2


def test_caso_12_frame_sem_veiculos(config_horizontal):
    contador = LineCounter(config_horizontal)
    for _ in range(10):
        eventos = contador.update([])
        assert eventos == []
    assert contador.summary().total_events == 0


def test_caso_9_veiculo_desaparece_e_reaparece(config_horizontal):
    """Perder o veiculo por alguns frames nao deve duplicar a contagem."""
    contador = LineCounter(config_horizontal)

    contador.update([(1, fazer_deteccao(500, 300))])
    for _ in range(5):
        contador.update([])
    contador.update([(1, fazer_deteccao(500, 500))])

    assert contador.summary().total_events == 1


# ----------------------------------------------------------------------
# Estabilizacao de classe e contagens por categoria
# ----------------------------------------------------------------------
def test_classe_oscilante_e_estabilizada_por_votacao(config_horizontal):
    """Caminhao com alguns frames rotulados como carro continua caminhao."""
    contador = LineCounter(config_horizontal)

    contador.update([(1, fazer_deteccao(500, 300, class_id=7, conf=0.9))])
    contador.update([(1, fazer_deteccao(500, 350, class_id=7, conf=0.85))])
    contador.update([(1, fazer_deteccao(500, 380, class_id=2, conf=0.3))])
    contador.update([(1, fazer_deteccao(500, 450, class_id=7, conf=0.88))])

    assert contador.events[0].vehicle_class == "truck"


def test_contagens_separadas_por_categoria(config_horizontal):
    contador = LineCounter(config_horizontal)

    for track_id, class_id in ((1, 2), (2, 3), (3, 5), (4, 7)):
        for y in (300, 500):
            contador.update([(track_id, fazer_deteccao(400 + track_id * 40, y, class_id))])

    resumo = contador.summary()
    assert resumo.entry.car == 1
    assert resumo.entry.motorcycle == 1
    assert resumo.entry.bus == 1
    assert resumo.entry.truck == 1
    assert resumo.entry.total == 4


# ----------------------------------------------------------------------
# Gestao de estado
# ----------------------------------------------------------------------
def test_tracks_expiram(config_horizontal):
    config_horizontal.track_expiry_frames = 10
    contador = LineCounter(config_horizontal)

    contador.update([(1, fazer_deteccao(500, 300))], frame_number=1)
    assert contador.active_tracks == 1

    contador.update([], frame_number=100)
    assert contador.active_tracks == 0


def test_reset_limpa_tudo(config_horizontal):
    contador = LineCounter(config_horizontal)
    for y in (300, 500):
        contador.update([(1, fazer_deteccao(500, y))])
    assert contador.summary().total_events == 1

    contador.reset()
    assert contador.summary().total_events == 0
    assert contador.active_tracks == 0