"""Diagnostico de qualidade do tracking.

Uso:
    python src/track_stats.py                          # bytetrack padrao
    python src/track_stats.py config/bytetrack_tuned.yaml
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "test.mp4"

VEHICLE_CLASSES = [2, 3, 5, 7]
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main() -> None:
    tracker = sys.argv[1] if len(sys.argv) > 1 else "bytetrack.yaml"
    print(f"Tracker: {tracker}")

    model = YOLO("yolo26n.pt")

    results = model.track(
        source=str(VIDEO_PATH),
        tracker=tracker,
        classes=VEHICLE_CLASSES,
        conf=0.25,
        imgsz=640,
        persist=True,
        save=False,
        stream=True,
        verbose=False,
    )

    lifespan: Counter[int] = Counter()
    classes_por_track: dict[int, Counter[int]] = defaultdict(Counter)

    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.int().tolist()
        clss = boxes.cls.int().tolist()
        for track_id, class_id in zip(ids, clss):
            lifespan[track_id] += 1
            classes_por_track[track_id][class_id] += 1

    total = len(lifespan)
    efemeros = [t for t, n in lifespan.items() if n <= 5]
    curtos = [t for t, n in lifespan.items() if 5 < n <= 15]
    estaveis = [t for t, n in lifespan.items() if n > 15]
    instaveis = [t for t, c in classes_por_track.items() if len(c) > 1]

    print("\n--- Qualidade do tracking ---")
    print(f"Tracks totais            : {total}")
    print(f"  <= 5 frames (efemeros) : {len(efemeros)}")
    print(f"  6-15 frames (curtos)   : {len(curtos)}")
    print(f"  > 15 frames (estaveis) : {len(estaveis)}")
    print(f"Tracks com classe oscilante: {len(instaveis)}")

    print("\nDistribuicao de classes (por track, classe majoritaria):")
    majoritarias: Counter[str] = Counter()
    for c in classes_por_track.values():
        majoritarias[COCO_NAMES[c.most_common(1)[0][0]]] += 1
    for nome, qtd in majoritarias.most_common():
        print(f"  {nome:12s}: {qtd}")


if __name__ == "__main__":
    main()