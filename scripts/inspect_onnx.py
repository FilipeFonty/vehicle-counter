"""Inspeciona entradas e saidas dos modelos ONNX exportados.

E este script que determina como o decodificador deve ser escrito.
Sem ele, a implementacao do pos-processamento seria adivinhacao.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def inspecionar(caminho: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"Modelo: {caminho.name}")
    print("=" * 60)

    sessao = ort.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])

    print("\nENTRADAS:")
    for entrada in sessao.get_inputs():
        print(f"  nome={entrada.name}  shape={entrada.shape}  tipo={entrada.type}")

    print("\nSAIDAS:")
    for saida in sessao.get_outputs():
        print(f"  nome={saida.name}  shape={saida.shape}  tipo={saida.type}")

    # Inferencia com tensor sintetico para ver o shape real
    entrada = sessao.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in entrada.shape]
    dummy = np.random.rand(*shape).astype(np.float32)

    resultados = sessao.run(None, {entrada.name: dummy})

    print("\nSAIDA REAL (tensor sintetico):")
    for saida, array in zip(sessao.get_outputs(), resultados):
        print(f"  {saida.name}: shape={array.shape}  dtype={array.dtype}")
        print(f"    min={array.min():.4f}  max={array.max():.4f}")
        achatado = array.reshape(-1)
        print(f"    primeiros 8 valores: {np.round(achatado[:8], 4)}")


def main() -> None:
    modelos = sorted(MODELS_DIR.glob("*.onnx"))
    if not modelos:
        print(f"Nenhum .onnx encontrado em {MODELS_DIR}")
        print("Rode antes: python scripts/export_onnx.py")
        return

    for modelo in modelos:
        inspecionar(modelo)


if __name__ == "__main__":
    main()