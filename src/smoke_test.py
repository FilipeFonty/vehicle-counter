"""Prova de conceito: video -> deteccao -> classificacao -> tracking.

Objetivo desta etapa: validar que o modelo carrega, que as classes de
veiculos sao filtradas corretamente e que o ByteTrack atribui IDs
estaveis ao longo dos frames. Ainda NAO ha contagem de linha aqui.
"""

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "test.mp4"
OUTPUT_DIR = BASE_DIR / "outputs"

# Classes COCO de interesse: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Modelo principal e fallback, conforme item 3 do enunciado.
PRIMARY_MODEL = "yolo26n.pt"
FALLBACK_MODEL = "yolo11n.pt"


def load_model() -> YOLO:
    """Carrega o YOLO26n; se indisponivel, recorre ao YOLO11n."""
    try:
        model = YOLO(PRIMARY_MODEL)
        print(f"[OK] Modelo carregado: {PRIMARY_MODEL}")
        return model
    except Exception as exc:  # noqa: BLE001 - queremos ver qualquer falha aqui
        print(f"[AVISO] Falha ao carregar {PRIMARY_MODEL}: {exc}")
        print(f"[INFO] Tentando fallback: {FALLBACK_MODEL}")
        model = YOLO(FALLBACK_MODEL)
        print(f"[OK] Modelo carregado: {FALLBACK_MODEL}")
        return model


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video nao encontrado: {VIDEO_PATH}\n"
            "Coloque um arquivo chamado test.mp4 dentro da pasta data/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()

    results = model.track(
        source=str(VIDEO_PATH),
        tracker="config/bytetrack_tuned.yaml",
        classes=VEHICLE_CLASSES,
        conf=0.25,
        imgsz=640,
        persist=True,
        save=True,
        project=str(OUTPUT_DIR),
        name="smoke-test-tuned",
        exist_ok=True,
        stream=True,  # gerador: evita carregar o video inteiro na memoria
    )

    frames = 0
    seen_ids: set[int] = set()

    for result in results:
        frames += 1
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for track_id in boxes.id.int().tolist():
                seen_ids.add(track_id)

    print("\n--- Resumo do smoke test ---")
    print(f"Frames processados : {frames}")
    print(f"IDs unicos vistos  : {len(seen_ids)}")
    print(f"Video anotado em   : {OUTPUT_DIR / 'smoke-test'}")


if __name__ == "__main__":
    main()
