"""Benchmark de latencia do detector.

Mede latencia por etapa e agrega estatisticas. Permite variar o numero
de threads do ONNX Runtime, o que e essencial para uma leitura honesta:
medir com 28 threads em um i7 nada diz sobre um dispositivo de 4 nucleos.

IMPORTANTE: os numeros produzidos aqui valem para o hardware em que o
script rodar. Nao devem ser apresentados como resultado de Raspberry Pi 5.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent


def percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    indice = int(round((p / 100.0) * (len(ordenados) - 1)))
    return ordenados[max(0, min(indice, len(ordenados) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "yolo26n_e2e.onnx"))
    parser.add_argument("--image", default=str(BASE_DIR / "data" / "frame_teste.jpg"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default=str(BASE_DIR / "benchmarks"))
    args = parser.parse_args()

    imagem = cv2.imread(args.image)
    if imagem is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {args.image}")

    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = args.threads
    opcoes.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sessao = ort.InferenceSession(
        args.model, sess_options=opcoes, providers=["CPUExecutionProvider"]
    )
    nome_entrada = sessao.get_inputs()[0].name

    # Pre-processa uma vez: aqui medimos a inferencia isoladamente.
    from app.core.detector import letterbox

    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    processada, _, _ = letterbox(rgb, 640)
    tensor = np.ascontiguousarray(
        np.expand_dims(np.transpose(processada.astype(np.float32) / 255.0, (2, 0, 1)), 0),
        dtype=np.float32,
    )

    print(f"Modelo  : {Path(args.model).name}")
    print(f"Threads : {args.threads}")
    print(f"Runs    : {args.runs} (warmup {args.warmup})")
    print("Executando...\n")

    for _ in range(args.warmup):
        sessao.run(None, {nome_entrada: tensor})

    latencias: list[float] = []
    for _ in range(args.runs):
        inicio = time.perf_counter()
        sessao.run(None, {nome_entrada: tensor})
        latencias.append((time.perf_counter() - inicio) * 1000)

    media = statistics.mean(latencias)
    resultado = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
        },
        "config": {
            "model": Path(args.model).name,
            "threads": args.threads,
            "runs": args.runs,
            "input_shape": list(tensor.shape),
        },
        "latency_ms": {
            "mean": round(media, 2),
            "median": round(statistics.median(latencias), 2),
            "p50": round(percentil(latencias, 50), 2),
            "p95": round(percentil(latencias, 95), 2),
            "p99": round(percentil(latencias, 99), 2),
            "min": round(min(latencias), 2),
            "max": round(max(latencias), 2),
            "stdev": round(statistics.stdev(latencias), 2) if len(latencias) > 1 else 0.0,
        },
        "throughput_fps": round(1000.0 / media, 2),
        "disclaimer": (
            "Resultado MEDIDO no hardware indicado em 'environment'. "
            "Nao representa desempenho em Raspberry Pi 5."
        ),
    }

    print(json.dumps(resultado["latency_ms"], indent=2))
    print(f"\nFPS (inferencia isolada): {resultado['throughput_fps']}")

    destino = Path(args.output)
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"benchmark_{platform.machine()}_{args.threads}threads.json"
    arquivo.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {arquivo}")


if __name__ == "__main__":
    main()