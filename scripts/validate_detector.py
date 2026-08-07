"""Compara o detector ONNX manual com o resultado do Ultralytics.

Sem esta validacao, nao ha como afirmar que o pipeline reimplementado
esta correto -- apenas que ele executa sem erro.
"""

from pathlib import Path

import cv2

from app.config import load_settings
from app.core.detector import Detector

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "data" / "frame_teste.jpg"


def main() -> None:
    imagem = cv2.imread(str(IMAGE_PATH))
    if imagem is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {IMAGE_PATH}")

    settings = load_settings()
    detector = Detector(settings)

    print(f"Modelo   : {detector.model_path.name}")
    print(f"Formato  : {detector.output_format}")
    print(f"Saida    : {detector.output_shape}")
    print(f"Threads  : {settings.model.intra_op_num_threads}")

    resultado = detector.detect(imagem)

    print(f"\n--- Detector ONNX manual ---")
    print(f"Deteccoes: {len(resultado.detections)}")
    for d in resultado.detections:
        b = d.bounding_box
        print(
            f"  {d.class_name:11s} conf={d.confidence:.3f} "
            f"box=({b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}) area={b.area:.0f}"
        )
    p = resultado.performance
    print(
        f"\nPre={p.preprocess_ms}ms  Inf={p.inference_ms}ms  "
        f"Pos={p.postprocess_ms}ms  Total={p.total_ms}ms"
    )

    # Referencia: Ultralytics sobre a mesma imagem
    try:
        from ultralytics import YOLO

        modelo = YOLO("yolo26n.pt")
        ref = modelo.predict(
            source=str(IMAGE_PATH),
            classes=settings.model.allowed_classes,
            conf=settings.model.confidence_threshold,
            imgsz=settings.model.input_size,
            verbose=False,
        )[0]

        print(f"\n--- Referencia Ultralytics ---")
        print(f"Deteccoes: {len(ref.boxes)}")
        for caixa in ref.boxes:
            cid = int(caixa.cls.item())
            conf = float(caixa.conf.item())
            x1, y1, x2, y2 = caixa.xyxy[0].tolist()
            print(
                f"  {ref.names[cid]:11s} conf={conf:.3f} "
                f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[AVISO] Comparacao indisponivel: {exc}")


if __name__ == "__main__":
    main()