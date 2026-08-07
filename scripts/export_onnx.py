"""Exporta YOLO26n para ONNX nas duas variantes e compara.

end2end=True  -> NMS embutido no grafo (saida ja decodificada)
end2end=False -> saida bruta, exige decodificacao + NMS no nosso codigo

O item 8 do enunciado pede pipeline explicito, o que favorece end2end=False.
Mas a decisao final depende de qual variante realmente exporta e roda de
forma estavel neste ambiente.
"""

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_NAME = "yolo26n.pt"
IMGSZ = 640


def exportar(end2end: bool) -> Path | None:
    sufixo = "e2e" if end2end else "raw"
    print(f"\n{'=' * 60}")
    print(f"Exportando com end2end={end2end}")
    print("=" * 60)

    try:
        model = YOLO(MODEL_NAME)
        caminho = model.export(
            format="onnx",
            imgsz=IMGSZ,
            opset=17,
            simplify=True,
            dynamic=False,
            nms=end2end,
        )
        origem = Path(caminho)
        destino = MODELS_DIR / f"yolo26n_{sufixo}.onnx"
        origem.replace(destino)
        tamanho_mb = destino.stat().st_size / (1024 * 1024)
        print(f"[OK] Gerado: {destino.name} ({tamanho_mb:.1f} MB)")
        return destino
    except Exception as exc:  # noqa: BLE001
        print(f"[FALHOU] end2end={end2end}: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resultados = {
        "raw": exportar(end2end=False),
        "e2e": exportar(end2end=True),
    }

    print(f"\n{'=' * 60}")
    print("RESUMO")
    print("=" * 60)
    for nome, caminho in resultados.items():
        status = caminho.name if caminho else "FALHOU"
        print(f"  {nome:4s}: {status}")


if __name__ == "__main__":
    main()