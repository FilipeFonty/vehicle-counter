"""Confirma o significado de cada coluna da saida ONNX usando imagem real.

Tensor sintetico nao permite validar semantica de colunas. Aqui usamos
um frame real e conferimos se as classes detectadas fazem sentido.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo26n_e2e.onnx"
IMAGE_PATH = BASE_DIR / "data" / "frame_teste.jpg"

COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IMGSZ = 640


def letterbox(imagem: np.ndarray, tamanho: int = IMGSZ):
    """Redimensiona preservando o aspecto e preenche com cinza."""
    altura, largura = imagem.shape[:2]
    escala = min(tamanho / altura, tamanho / largura)
    nova_l, nova_a = int(round(largura * escala)), int(round(altura * escala))

    redimensionada = cv2.resize(imagem, (nova_l, nova_a), interpolation=cv2.INTER_LINEAR)

    pad_l = (tamanho - nova_l) / 2
    pad_a = (tamanho - nova_a) / 2
    topo, base = int(round(pad_a - 0.1)), int(round(pad_a + 0.1))
    esq, dir_ = int(round(pad_l - 0.1)), int(round(pad_l + 0.1))

    saida = cv2.copyMakeBorder(
        redimensionada, topo, base, esq, dir_,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return saida, escala, (esq, topo)


def main() -> None:
    imagem = cv2.imread(str(IMAGE_PATH))
    if imagem is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {IMAGE_PATH}")

    altura_orig, largura_orig = imagem.shape[:2]
    print(f"Imagem original: {largura_orig}x{altura_orig}")

    # Etapas 2 a 7 do pipeline explicito
    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    processada, escala, (pad_x, pad_y) = letterbox(rgb)
    normalizada = processada.astype(np.float32) / 255.0
    chw = np.transpose(normalizada, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0).astype(np.float32)

    print(f"Tensor de entrada: {tensor.shape}, escala={escala:.4f}, pad=({pad_x},{pad_y})")

    sessao = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    nome_entrada = sessao.get_inputs()[0].name
    saida = sessao.run(None, {nome_entrada: tensor})[0]

    print(f"Saida bruta: {saida.shape}")

    deteccoes = saida[0]  # (300, 6)

    # Hipotese: colunas = [x1, y1, x2, y2, confianca, class_id]
    confiancas = deteccoes[:, 4]
    validas = deteccoes[confiancas > 0.25]

    print(f"\nDeteccoes com confianca > 0.25: {len(validas)}")
    print("\nPrimeiras 15 (coordenadas no espaco 640):")
    print(f"{'x1':>8} {'y1':>8} {'x2':>8} {'y2':>8} {'conf':>7} {'cls':>5}  nome")

    for linha in validas[:15]:
        x1, y1, x2, y2, conf, cls = linha
        cls_int = int(cls)
        nome = COCO_NAMES.get(cls_int, f"coco_{cls_int}")
        print(f"{x1:8.1f} {y1:8.1f} {x2:8.1f} {y2:8.1f} {conf:7.3f} {cls_int:5d}  {nome}")

    # Estatisticas de sanidade
    todas_cls = validas[:, 5].astype(int)
    print(f"\nClasses distintas encontradas: {sorted(set(todas_cls.tolist()))}")
    print(f"Faixa de confianca: {confiancas.min():.4f} a {confiancas.max():.4f}")
    print(f"Ordenadas por confianca decrescente? {np.all(np.diff(confiancas) <= 1e-6)}")


if __name__ == "__main__":
    main()