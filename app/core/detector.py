"""Detector ONNX com pipeline explicito.

Implementa as etapas 1 a 13 do item 8 do enunciado sem depender de
model.predict(). Suporta duas familias de saida:

  YOLO26 -> (1, 300, 6)   [x1, y1, x2, y2, conf, cls] ja decodificado
  YOLO11 -> (1, 84, 8400) [cx, cy, w, h, 80 scores] tensor bruto

O caminho YOLO11 exige transposicao, decodificacao e NMS manual; o
caminho YOLO26 exige apenas NMS de limpeza, pois a saida contem caixas
sobrepostas e conflitos de classe apesar da arquitetura NMS-free.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.config import Settings
from app.models.detection import (
    Detection,
    DetectionResult,
    ImageInfo,
    PerformanceMetrics,
)

PAD_VALUE = 114  # cinza padrao do letterbox YOLO


def letterbox(
    image: np.ndarray, target_size: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona preservando o aspecto e preenche as bordas.

    Etapa 3 do pipeline. Distorcer o aspecto degradaria a precisao das
    caixas; por isso escalamos pelo menor fator e preenchemos o resto.

    Retorna a imagem, o fator de escala e o padding (esquerda, topo),
    necessarios depois para reverter as coordenadas.
    """
    altura, largura = image.shape[:2]
    escala = min(target_size / altura, target_size / largura)
    nova_l, nova_a = int(round(largura * escala)), int(round(altura * escala))

    redimensionada = (
        cv2.resize(image, (nova_l, nova_a), interpolation=cv2.INTER_LINEAR)
        if (nova_l, nova_a) != (largura, altura)
        else image
    )

    pad_l = (target_size - nova_l) / 2
    pad_a = (target_size - nova_a) / 2
    topo, base = int(round(pad_a - 0.1)), int(round(pad_a + 0.1))
    esq, dir_ = int(round(pad_l - 0.1)), int(round(pad_l + 0.1))

    saida = cv2.copyMakeBorder(
        redimensionada,
        topo,
        base,
        esq,
        dir_,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return saida, escala, (esq, topo)


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    """Non-Maximum Suppression em numpy puro.

    Etapa 11 do pipeline. Mantido explicito para fins didaticos, em vez
    de delegar a cv2.dnn.NMSBoxes.

    boxes no formato (N, 4) como [x1, y1, x2, y2].
    Retorna os indices mantidos.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    ordem = scores.argsort()[::-1]

    mantidos: list[int] = []
    while ordem.size > 0:
        atual = ordem[0]
        mantidos.append(int(atual))
        if ordem.size == 1:
            break

        restantes = ordem[1:]

        # Interseccao entre a caixa atual e todas as restantes
        xx1 = np.maximum(x1[atual], x1[restantes])
        yy1 = np.maximum(y1[atual], y1[restantes])
        xx2 = np.minimum(x2[atual], x2[restantes])
        yy2 = np.minimum(y2[atual], y2[restantes])

        larg = np.maximum(0.0, xx2 - xx1)
        alt = np.maximum(0.0, yy2 - yy1)
        intersecao = larg * alt

        uniao = areas[atual] + areas[restantes] - intersecao
        iou = np.where(uniao > 0, intersecao / uniao, 0.0)

        ordem = restantes[iou <= iou_threshold]

    return mantidos


class Detector:
    """Encapsula sessao ONNX, pre-processamento e pos-processamento."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_path: Path = settings.model_absolute_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX nao encontrado: {self.model_path}\n"
                "Rode antes: python scripts/export_onnx.py"
            )

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = settings.model.intra_op_num_threads
        opcoes.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opcoes,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        self.output_format = self._resolver_formato()

    def _resolver_formato(self) -> str:
        """Deduz o formato da saida quando config.format == 'auto'."""
        configurado = self.settings.model.format
        if configurado != "auto":
            return configurado

        # YOLO26: (1, 300, 6) -> ultima dimensao pequena
        # YOLO11: (1, 84, 8400) -> dimensao do meio == 4 + num_classes
        if len(self.output_shape) == 3 and self.output_shape[-1] == 6:
            return "yolo26"
        if len(self.output_shape) == 3 and self.output_shape[1] in (84, 85):
            return "yolo11"
        raise ValueError(
            f"Formato de saida nao reconhecido: {self.output_shape}. "
            "Defina model.format explicitamente na configuracao."
        )

    # ------------------------------------------------------------------
    # Pre-processamento (etapas 1 a 7)
    # ------------------------------------------------------------------
    def preprocess(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        tamanho = self.settings.model.input_size

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)          # etapa 2
        processada, escala, pad = letterbox(rgb, tamanho)          # etapa 3
        normalizada = processada.astype(np.float32) / 255.0        # etapa 4
        chw = np.transpose(normalizada, (2, 0, 1))                 # etapa 5
        tensor = np.expand_dims(chw, axis=0)                       # etapa 6
        return np.ascontiguousarray(tensor, dtype=np.float32), escala, pad  # etapa 7

    # ------------------------------------------------------------------
    # Decodificacao (etapa 9)
    # ------------------------------------------------------------------
    def _decode_yolo26(self, saida: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(1, 300, 6) -> boxes xyxy, scores, class_ids."""
        deteccoes = saida[0]
        boxes = deteccoes[:, :4].astype(np.float32)
        scores = deteccoes[:, 4].astype(np.float32)
        class_ids = deteccoes[:, 5].astype(np.int32)
        return boxes, scores, class_ids

    def _decode_yolo11(self, saida: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(1, 84, 8400) -> boxes xyxy, scores, class_ids.

        Nao ha coluna de objectness: a confianca e o maior score entre
        as 80 classes COCO.
        """
        predicoes = saida[0].T  # (8400, 84)

        caixas_cxcywh = predicoes[:, :4]
        scores_classes = predicoes[:, 4:]

        class_ids = np.argmax(scores_classes, axis=1).astype(np.int32)
        scores = np.max(scores_classes, axis=1).astype(np.float32)

        # cxcywh -> xyxy
        cx, cy, w, h = (
            caixas_cxcywh[:, 0],
            caixas_cxcywh[:, 1],
            caixas_cxcywh[:, 2],
            caixas_cxcywh[:, 3],
        )
        boxes = np.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1
        ).astype(np.float32)

        return boxes, scores, class_ids

    # ------------------------------------------------------------------
    # Pos-processamento (etapas 9 a 13)
    # ------------------------------------------------------------------
    def postprocess(
        self,
        saida: np.ndarray,
        escala: float,
        pad: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> list[Detection]:
        cfg_modelo = self.settings.model
        cfg_filtro = self.settings.filtering

        if self.output_format == "yolo26":
            boxes, scores, class_ids = self._decode_yolo26(saida)
        else:
            boxes, scores, class_ids = self._decode_yolo11(saida)

        # Etapa 10 -- filtragem por confianca
        mascara = scores >= cfg_modelo.confidence_threshold
        boxes, scores, class_ids = boxes[mascara], scores[mascara], class_ids[mascara]

        if len(boxes) == 0:
            return []

        # Etapa 11 -- NMS
        # No YOLO26 atua como limpeza: a saida traz caixas quase
        # identicas e ate a mesma caixa com classes diferentes.
        if cfg_filtro.apply_nms:
            if cfg_filtro.nms_class_agnostic:
                indices = nms(boxes, scores, cfg_modelo.iou_threshold)
            else:
                indices = []
                for classe in np.unique(class_ids):
                    idx_classe = np.where(class_ids == classe)[0]
                    mantidos = nms(
                        boxes[idx_classe], scores[idx_classe], cfg_modelo.iou_threshold
                    )
                    indices.extend(idx_classe[m] for m in mantidos)
            boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]

        # Etapa 12 -- reverter letterbox para as coordenadas originais
        pad_x, pad_y = pad
        altura_orig, largura_orig = original_shape
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / escala
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / escala
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, largura_orig)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, altura_orig)

        # Etapa 13 -- filtrar as classes de veiculo
        permitidas = set(cfg_modelo.allowed_classes)

        deteccoes: list[Detection] = []
        for caixa, score, class_id in zip(boxes, scores, class_ids):
            if int(class_id) not in permitidas:
                continue

            deteccao = Detection.from_raw(
                x1=float(caixa[0]),
                y1=float(caixa[1]),
                x2=float(caixa[2]),
                y2=float(caixa[3]),
                confidence=float(score),
                class_id=int(class_id),
            )

            # Filtro de area: veiculos minusculos nao rastreiam de forma
            # estavel e geram tracks efemeros.
            if deteccao.bounding_box.area < cfg_filtro.min_box_area:
                continue

            # Filtro de ROI pelo ponto de contato com o solo.
            if not self._dentro_da_roi(deteccao):
                continue

            deteccoes.append(deteccao)

        return deteccoes

    def _dentro_da_roi(self, deteccao: Detection) -> bool:
        poligono = self.settings.filtering.roi_polygon
        if not poligono:
            return True
        pontos = np.array(poligono, dtype=np.int32)
        x, y = deteccao.bounding_box.bottom_center
        return cv2.pointPolygonTest(pontos, (float(x), float(y)), False) >= 0

    # ------------------------------------------------------------------
    # Execucao completa
    # ------------------------------------------------------------------
    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        altura, largura = image_bgr.shape[:2]

        inicio = time.perf_counter()
        tensor, escala, pad = self.preprocess(image_bgr)
        fim_pre = time.perf_counter()

        saida = self.session.run(None, {self.input_name: tensor})[0]  # etapa 8
        fim_inf = time.perf_counter()

        deteccoes = self.postprocess(saida, escala, pad, (altura, largura))
        fim_pos = time.perf_counter()

        return DetectionResult(
            image=ImageInfo(width=largura, height=altura),
            detections=deteccoes,
            performance=PerformanceMetrics(
                preprocess_ms=round((fim_pre - inicio) * 1000, 2),
                inference_ms=round((fim_inf - fim_pre) * 1000, 2),
                postprocess_ms=round((fim_pos - fim_inf) * 1000, 2),
                total_ms=round((fim_pos - inicio) * 1000, 2),
            ),
        )