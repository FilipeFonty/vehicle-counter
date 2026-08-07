# Vehicle Counter

Contagem e classificação de veículos em entradas de complexos industriais.
Detecção com YOLO26n via ONNX Runtime, rastreamento ByteTrack, contagem por
cruzamento de linha virtual e API HTTP com FastAPI.

Projetado para execução local (sem inferência em nuvem) e para viabilidade
em hardware embarcado ARM64, com foco na Raspberry Pi 5.

---

## Sumário

- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Uso](#uso)
- [API](#api)
- [Docker](#docker)
- [Desempenho](#desempenho)
- [Adaptações para Raspberry Pi 5](#adaptações-para-raspberry-pi-5)
- [Decisões técnicas](#decisões-técnicas)
- [Limitações conhecidas](#limitações-conhecidas)
- [Testes](#testes)
- [Licenciamento](#licenciamento)

---

## Funcionalidades

- Detecção e classificação de quatro categorias: `car`, `motorcycle`, `bus`, `truck`
- Rastreamento multi-objeto com atribuição de ID temporário
- Contagem por cruzamento de linha virtual configurável
- Distinção entre sentido de entrada e saída
- Bloqueio de contagem duplicada
- Histerese para evitar contagem de veículo parado sobre a linha
- Estabilização de classe por votação ponderada pela confiança
- Endpoints HTTP com resposta JSON e imagem PNG anotada
- Métricas de latência (média, p50, p95) e FPS efetivo
- Imagem Docker multi-arquitetura (amd64 e arm64)

---

## Arquitetura

```
Vídeo ou imagem
    ↓
Captura do frame
    ↓
Pré-processamento (BGR→RGB, letterbox, normalização, HWC→CHW, batch, float32)
    ↓
Inferência (ONNX Runtime, CPU)
    ↓
Decodificação da saída
    ↓
Filtragem por confiança
    ↓
Non-Maximum Suppression (limpeza)
    ↓
Reversão do letterbox para coordenadas originais
    ↓
Filtragem das classes de veículo + área mínima + ROI
    ↓
Rastreamento (ByteTrack)
    ↓
Regra de cruzamento da linha + direção
    ↓
Registro do evento
    ↓
API, métricas e imagem anotada
```

### Estrutura de diretórios

```
vehicle-counter/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── inference.py      Endpoints JSON e PNG
│   │   │   ├── counts.py         Contagens, eventos, performance
│   │   │   └── videos.py         Análise de vídeo
│   │   └── dependencies.py       Instâncias compartilhadas
│   ├── core/
│   │   ├── detector.py           Pipeline de inferência explícito
│   │   ├── tracker.py            Adaptador ByteTrack
│   │   ├── line_counter.py       Geometria de cruzamento e contagem
│   │   ├── video_processor.py    Processamento de vídeo
│   │   ├── annotation.py         Desenho de caixas e linha
│   │   └── performance.py        Coleta de latências
│   ├── models/
│   │   ├── detection.py          Contratos de detecção
│   │   ├── event.py              Contratos de evento
│   │   └── responses.py          Contratos de resposta da API
│   ├── services/
│   ├── config.py                 Carregamento e validação da configuração
│   └── main.py                   Aplicação FastAPI
├── config/default.yaml           Configuração central
├── models/                       Modelo ONNX
├── scripts/                      Exportação, benchmark, validação
├── tests/                        Testes unitários
├── Dockerfile
├── docker-compose.yml
├── requirements.txt              Runtime (usado na imagem)
└── requirements-dev.txt          Desenvolvimento e exportação
```

---

## Instalação

### Requisitos

- Python 3.11
- Docker com plugin buildx (opcional, para containerização)

### Ambiente local

```bash
git clone <url-do-repositorio>
cd vehicle-counter

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements-dev.txt
pip install -e .
```

O `requirements.txt` contém apenas as dependências de runtime. O
`requirements-dev.txt` acrescenta PyTorch e Ultralytics, necessários somente
para exportar o modelo.

### Exportar o modelo ONNX

```bash
python scripts/export_onnx.py
```

O arquivo é gravado em `models/yolo26n_e2e.onnx` (9,5 MB). Os pesos
`yolo26n.pt` são baixados automaticamente pelo Ultralytics na primeira
execução.

---

## Uso

### Processar um vídeo

```bash
python scripts/run_video.py --video data/test.mp4
```

Opções:

| Flag | Efeito |
|---|---|
| `--max-frames N` | Limita a N frames |
| `--stride N` | Processa 1 a cada N frames |
| `--no-save` | Não grava o vídeo anotado |

O `--stride` é a principal estratégia de redução de custo computacional para
hardware embarcado: com `--stride 2`, metade dos frames é processada.

### Subir a API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Documentação interativa em `http://localhost:8000/docs`.

---

## API

| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/v1/inference/json` | Detecções em JSON |
| POST | `/api/v1/inference/annotated` | Imagem PNG anotada |
| GET | `/api/v1/counts` | Contagens por categoria e sentido |
| GET | `/api/v1/events` | Eventos de cruzamento |
| GET | `/api/v1/performance` | Latências e FPS |
| POST | `/api/v1/videos/analyze` | Processa vídeo: detecta, rastreia e conta |
| GET | `/api/v1/videos/available` | Lista vídeos disponíveis em `data/` |
| POST | `/api/v1/counts/reset` | Zera contagens |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger/OpenAPI |

### Exemplos com curl

Detecção em JSON:

```bash
curl -X POST http://localhost:8000/api/v1/inference/json \
  -F "file=@data/frame_teste.jpg"
```

Resposta:

```json
{
  "image": { "width": 1920, "height": 1080 },
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.9,
      "bounding_box": {
        "x1": 414.0, "y1": 635.3, "x2": 606.8, "y2": 873.6
      }
    }
  ],
  "performance": {
    "preprocess_ms": 3.33,
    "inference_ms": 24.4,
    "postprocess_ms": 0.44,
    "total_ms": 28.17
  }
}
```

Imagem anotada:

```bash
curl -X POST http://localhost:8000/api/v1/inference/annotated \
  -F "file=@data/frame_teste.jpg" \
  -o outputs/anotada.png
```

Análise de vídeo:

```bash
curl http://localhost:8000/api/v1/videos/available

curl -X POST "http://localhost:8000/api/v1/videos/analyze?filename=test.mp4&max_frames=505"
```

Parâmetros: `filename` (arquivo em `data/`), `max_frames`, `stride`,
`save_annotated`, `reset_counts`.

Contagens e eventos:

```bash
curl http://localhost:8000/api/v1/counts
curl http://localhost:8000/api/v1/events?limit=20
curl http://localhost:8000/api/v1/performance
curl http://localhost:8000/health
```

Tratamento de erros:

```bash
# Arquivo que não é imagem → HTTP 400
curl -X POST http://localhost:8000/api/v1/inference/json -F "file=@arquivo.txt"

# Requisição sem arquivo → HTTP 422
curl -X POST http://localhost:8000/api/v1/inference/json
```

---

## Docker

### Build e execução

```bash
docker build -t vehicle-counter:latest .
docker compose up -d
docker compose ps        # aguardar status "healthy"
```

### Build multi-arquitetura

```bash
docker buildx create --name vehicle-counter-builder --use
docker buildx inspect --bootstrap

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t usuario/vehicle-counter:latest \
  --push .
```

Sem registry, é possível construir e carregar uma arquitetura por vez:

```bash
docker buildx build --platform linux/arm64 -t vehicle-counter:arm64 --load .
```

### Execução de imagem ARM64 em host x86

Requer registro do QEMU/binfmt:

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker run --rm --platform linux/arm64 vehicle-counter:arm64 \
  python -c "import platform; print(platform.machine())"
```

### Características da imagem

- Base `python:3.11-slim-bookworm` (suporte oficial amd64 e arm64)
- Build multi-stage
- Usuário sem privilégios
- Health check integrado
- `restart: always` no Compose
- **383 MB** de conteúdo

A imagem **não** inclui PyTorch nem Ultralytics: o modelo já está exportado
para ONNX e a inferência usa apenas o ONNX Runtime. Isso reduz a imagem de
aproximadamente 2,5 GB para 383 MB — diferença decisiva para instalação em
cartão SD.

---

## Desempenho

Os resultados abaixo são separados em três níveis de evidência.

### 1. MEDIDO — ambiente de desenvolvimento (AMD64)

**Hardware:** Intel Core i7-14700HX · 14 GiB RAM · Ubuntu 26.04
**Modelo:** YOLO26n ONNX, entrada 640×640
**Vídeo de teste:** 1280×720 @ 25 FPS, 1195 frames (47,8 s), câmera fixa,
tráfego em dois sentidos

#### Escalabilidade por número de threads

Inferência isolada, 100 execuções com 10 de warmup:

| Threads | Média | p50 | p95 | Desvio | FPS | Speedup | Eficiência |
|---|---|---|---|---|---|---|---|
| 1 | 63,87 ms | 63,61 | 65,31 | 0,90 | 15,66 | 1,00× | 100% |
| 2 | 33,89 ms | 33,85 | 34,45 | 0,29 | 29,51 | 1,88× | 94% |
| 4 | 19,04 ms | 19,01 | 19,21 | 0,12 | 52,52 | 3,35× | 84% |
| 8 | 12,71 ms | 12,39 | 13,99 | 0,67 | 78,71 | 5,02× | 63% |

O modelo paraleliza bem até 4 núcleos, com 84% de eficiência. O desvio
padrão muito baixo indica latência previsível, característica desejável em
operação contínua.

#### Pipeline completo

Processamento dos 1192 frames com gravação do vídeo anotado:

- Latência do detector: média 22,67 ms · p50 22,57 ms · p95 23,17 ms
- Detector isolado: **44,1 FPS**
- Pipeline completo com gravação: **34,9 FPS**

Requisição isolada via API sobre um frame:

| Etapa | Tempo |
|---|---|
| Pré-processamento | 2,86 ms |
| Inferência | 22,98 ms |
| Pós-processamento | 0,32 ms |
| **Total** | **26,16 ms** |

#### Efeito da resolução de captura

O mesmo conteúdo foi processado em três resoluções ao longo do
desenvolvimento:

| Resolução | FPS do pipeline | Latência do detector |
|---|---|---|
| 3840×2160 | 14,5 | 22,7 ms |
| 1920×1080 | 28,0 | 21,97 ms |
| 1280×720 | **34,9** | 22,67 ms |

**A latência da inferência é praticamente constante**, porque o letterbox
reduz qualquer entrada para 640×640 antes do modelo. Todo o ganho vem de
decodificação, redimensionamento e encoding mais baratos.

Consequência prática: reduzir a resolução de captura de 4K para 720p mais
que dobra o throughput sem afetar a precisão do detector. É a otimização de
melhor custo-benefício para hardware embarcado.

#### Overhead de containerização

| Contexto | Pré | Inferência | Pós | Total |
|---|---|---|---|---|
| Nativo | 6,10 ms | 24,40 ms | 2,40 ms | 32,90 ms |
| Container | 3,33 ms | 24,40 ms | 0,44 ms | 28,17 ms |

A inferência é idêntica: o Docker isola namespaces, não virtualiza
processamento. O overhead de containerização em CPU é essencialmente nulo.

#### Comparação entre tamanhos de modelo

Ambos avaliados no mesmo vídeo e hardware, com 4 threads:

| | YOLO26n | YOLO26s |
|---|---|---|
| Eventos contados | 7 | 9 |
| Confiança típica | 0,85–0,92 | 0,90–0,96 |
| Latência média | 22,7 ms | 61,6 ms |
| FPS do detector | 44,0 | 16,2 |

O YOLO26s detecta mais e com maior confiança, mas quase triplica a latência.
Como referência, o YOLO26n com **uma única thread** leva 63,9 ms — valor
próximo ao que o YOLO26s consome com quatro. Em um dispositivo de quatro
núcleos e frequência substancialmente menor, o modelo `s` dificilmente
sustentaria taxa útil.

**Decisão: YOLO26n**, priorizando viabilidade embarcada, que é o objetivo do
projeto. O `s` permanece exportável para cenários com mais folga
computacional.

#### Memória

- Container em repouso: **153,7 MB**
- Pico RSS durante inferência: **210,6 MB**

O footprint cabe folgadamente até em uma Raspberry Pi 5 de 2 GB.

#### Contagem obtida

Vídeo de teste completo, 1192 frames processados, **36 eventos** registrados:

| Sentido | car | motorcycle | bus | truck | Total |
|---|---|---|---|---|---|
| Entrada | 2 | 0 | 0 | 0 | 2 |
| Saída | 30 | 0 | 3 | 1 | 34 |

A assimetria entre sentidos reflete o tráfego real do trecho, não uma falha
da regra de contagem. Uma análise independente das trajetórias dos tracks
confirmou 34 veículos deslocando-se em um sentido contra 6 no oposto,
proporção coerente com a contagem obtida.

**Convenção de sentido.** A direção é determinada pelo sinal do produto
vetorial em relação ao segmento `line_start → line_end`. Inverter a ordem
dos pontos na configuração inverte a convenção de entrada e saída,
permitindo adequar a nomenclatura ao posicionamento físico da câmera sem
alterar código.

### 2. VALIDADO FUNCIONALMENTE — ARM64 sob emulação QEMU

- Arquitetura confirmada em execução: `aarch64`
- `onnxruntime` 1.28.0 e `opencv-python-headless` 5.0.0 importam sem
  compilação da fonte (wheels ARM64 disponíveis)
- Inferência executada com saída `(1, 300, 6)`, idêntica ao AMD64
- API HTTP respondeu com **as mesmas 12 detecções** (`car: 11`, `bus: 1`)
  obtidas no AMD64 — paridade funcional entre arquiteturas

**Tempo observado sob QEMU: 2838,9 ms.**

⚠️ Este número **não representa a Raspberry Pi 5**. Sob QEMU, a tradução
dinâmica de instruções ARM→x86 introduz overhead de uma a duas ordens de
grandeza, especialmente em operações vetoriais (NEON). O valor serve apenas
como evidência de que a inferência executa em ARM64.

Um aviso `Unknown CPU vendor` aparece sob emulação porque o QEMU não expõe
identificação de fabricante de CPU. Não ocorre em hardware ARM real.

### 3. NÃO MEDIDO — Raspberry Pi 5

**Nenhum número de FPS, latência, temperatura ou throttling foi medido em
hardware físico.** Este projeto não teve acesso a uma Raspberry Pi 5.

O que os dados acima permitem afirmar com honestidade:

- O modelo escala bem até 4 núcleos (83% de eficiência), o mesmo número de
  núcleos da Pi 5 — o paralelismo será aproveitado
- A latência é muito estável (desvio de 0,12 ms com 4 threads), o que
  favorece previsibilidade em operação contínua
- O footprint de memória (~210 MB) cabe até no modelo de 2 GB
- A stack ARM64 funciona sem compilação da fonte

O que **não** pode ser afirmado: qualquer valor específico de FPS na Pi 5.
Estimar isso exigiria comparar IPC entre Raptor Lake e Cortex-A76 usando
benchmarks de terceiros, o que produziria um número especulativo apresentado
como medição. A validação precisa ocorrer em hardware físico.

---

## Adaptações para Raspberry Pi 5

### Diferenças entre AMD64 e ARM64

| Aspecto | Desenvolvimento (AMD64) | Raspberry Pi 5 (ARM64) |
|---|---|---|
| CPU | Intel i7-14700HX, 28 threads | Cortex-A76 quad-core |
| Extensões SIMD | AVX2 / AVX-512 | NEON |
| GPU para inferência | Não utilizada | VideoCore VII (não usada para ONNX) |
| RAM | 14 GiB | 2, 4, 8 ou 16 GB |
| Armazenamento | NVMe | microSD ou SSD via PCIe |
| Refrigeração | Ativa (notebook) | Passiva por padrão |

Não há GPU CUDA em nenhum dos dois casos: a inferência é integralmente CPU
via `CPUExecutionProvider`.

### Instalação na Raspberry Pi

```bash
# Raspberry Pi OS 64-bit ou Ubuntu Server ARM64
sudo apt update && sudo apt install -y python3-venv python3-pip

git clone <url-do-repositorio>
cd vehicle-counter

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Via Docker:

```bash
docker compose up -d
```

### Estratégias de redução de custo computacional

Em ordem de impacto e facilidade:

1. **Reduzir a resolução de entrada** — `model.input_size: 416` ou `320` no
   `config/default.yaml`. O custo cresce quadraticamente com a dimensão.
2. **Processar menos frames** — `--stride 2` ou `3`. Veículos a 5–30 km/h
   não exigem 30 FPS para contagem confiável.
3. **Limitar threads** — `model.intra_op_num_threads: 4`, alinhado aos
   núcleos disponíveis.
4. **Aumentar `min_box_area`** — descarta veículos distantes, reduzindo o
   trabalho do tracker.
5. **Restringir a ROI** — processa apenas a região útil do quadro.
6. **Quantização INT8** — redução adicional de latência e memória, com custo
   em precisão. Exige validação: comparar mAP e contagens antes e depois.

### Medição em hardware físico

Comandos para coletar as métricas ausentes:

```bash
# Temperatura
vcgencmd measure_temp
cat /sys/class/thermal/thermal_zone0/temp

# Throttling (0x0 = sem throttling)
vcgencmd get_throttled

# Frequência da CPU
vcgencmd measure_clock arm

# CPU e memória
htop
free -h

# Latência e FPS da aplicação
python scripts/benchmark.py --threads 4
curl http://localhost:8000/api/v1/performance
```

Recomenda-se **refrigeração ativa**. O Cortex-A76 sob carga contínua de
inferência aquece, e o throttling térmico reduz a frequência, degradando o
FPS de forma progressiva. Uma medição de FPS feita nos primeiros segundos
pode não refletir o desempenho sustentado.

### Captura de câmera

Para câmera USB ou módulo CSI, substituir o caminho do vídeo pelo índice do
dispositivo em `VideoProcessor.process()`:

```python
captura = cv2.VideoCapture(0)          # câmera USB
```

Para o módulo CSI em Raspberry Pi OS recente, a biblioteca `picamera2` é a
via recomendada, e exigiria um adaptador de captura adicional.

### Evolução prevista: exportação NCNN

A documentação da Ultralytics indica o NCNN como o formato de melhor
desempenho de inferência em Raspberry Pi, por ser otimizado especificamente
para arquitetura ARM. A migração exigiria substituir o backend de inferência
(ONNX Runtime → NCNN SDK) e revalidar todo o pipeline de pré e
pós-processamento.

Não foi implementada nesta entrega por dois motivos: está fora da stack
especificada para o projeto, e o ganho não poderia ser medido sem hardware
ARM físico — a emulação QEMU não produz números de desempenho utilizáveis.

### Evolução opcional: AI HAT+ com NPU Hailo

O AI HAT+ adiciona um acelerador Hailo que descarregaria a inferência da CPU.
Exigiria exportar o modelo para o formato Hailo e substituir o backend de
inferência. **Não é requisito** desta solução, que funciona apenas com CPU.

---

## Decisões técnicas

### YOLO26n como detector

Escolhido por ser a geração estável atual da Ultralytics, com arquitetura
NMS-free projetada para dispositivos de baixo consumo. O modelo `nano` foi
priorizado pelo alvo embarcado. YOLO11n permanece suportado no código como
alternativa, com decodificação de tensor bruto e NMS manual.

### NMS mesmo com arquitetura NMS-free

A arquitetura do YOLO26 elimina o NMS do grafo do modelo, mas a saída ainda
contém caixas sobrepostas. Foi observado, em imagem real, o mesmo objeto
recebendo classificações conflitantes na **mesma caixa** (`car` 0,371 e
`bus` 0,313 em coordenadas idênticas). O NMS class-agnostic foi mantido como
etapa de limpeza, removendo 3 detecções redundantes de 15.

Conclusão: NMS-free elimina o NMS da arquitetura, não a necessidade de
desambiguação.

### Pipeline de inferência explícito

O pré-processamento, a decodificação e o pós-processamento são implementados
manualmente com NumPy e OpenCV, em vez de delegados a `model.predict()`. A
implementação foi validada contra o Ultralytics na mesma imagem: as caixas
coincidem com diferença de 1 a 3 pixels, atribuível a arredondamento no
letterbox.

### Geometria da contagem

A posição do veículo em relação à linha usa a **distância perpendicular com
sinal**, calculada por produto vetorial 2D. Um único valor resolve três
problemas: o lado (sinal), o cruzamento (mudança de sinal) e a histerese
(módulo). Funciona com linhas de qualquer inclinação, não apenas horizontais.

O ponto de referência é o **centro inferior** da caixa, que aproxima o
contato com o solo e é mais estável que o centroide quando o veículo se
aproxima da câmera.

### ByteTrack via supervision

O ByteTrack embutido no Ultralytics foi avaliado primeiro. Optou-se por não
usá-lo diretamente: sua interface é privada (`_split_detections`,
`_format_output`, `parse_bboxes`) e mudou de assinatura entre versões menores
da série 8.4.x.

A biblioteca `supervision` expõe o mesmo algoritmo com API pública. A classe
`sv.ByteTrack` está marcada para remoção na v0.31.0; a versão está fixada em
`0.30.0` no `requirements.txt`, e a migração para um substituto é um próximo
passo de manutenção registrado.

### Ajuste do tracker

Os parâmetros padrão do ByteTrack produziram fragmentação significativa no
vídeo de teste: de 197 tracks, 73 sobreviviam 5 frames ou menos. Após ajuste
(`new_track_thresh` 0,55 · `track_buffer` 60 · `match_thresh` 0,85), os
tracks efêmeros caíram para 5.

### Sem fine-tuning inicial

O baseline utiliza os pesos pré-treinados no COCO. Fine-tuning permanece como
evolução prevista, justificável caso se confirmem problemas com o ângulo da
câmera, veículos parcialmente encobertos ou confusão sistemática entre ônibus
e caminhões — situação já observada pontualmente durante o desenvolvimento.

---

## Limitações conhecidas

Documentadas por transparência.

### 1. Motocicletas não são contadas

Nenhum dos vídeos de teste avaliados forneceu motocicletas em quantidade
suficiente para validar a classe. No material final, o detector registra
apenas 3 detecções de motocicleta ao longo de 1195 frames.

Uma investigação conduzida em um vídeo anterior é ilustrativa: varrendo os
frames com limiar de confiança reduzido a 0,15, o detector localizava
motocicletas 17 vezes, com confiança entre 0,17 e 0,50. A confiança máxima
observada ficava abaixo do `new_track_thresh`, impedindo a criação de um
track — e sem track não há ID nem contagem. Reduzir os limiares
(`new_track_thresh` 0,35 · `min_box_area` 12.000 · `track_high_thresh` 0,30)
foi testado e não resolveu, apenas degradou a estabilidade das demais
classes.

A causa provável é que o modelo pré-treinado no COCO tem dificuldade com
motocicletas em ângulo elevado, perspectiva sub-representada no conjunto de
treino. Este é exatamente o cenário que justifica o fine-tuning previsto
como evolução: um conjunto anotado do cenário-alvo permitiria ao modelo
aprender a aparência de motocicletas vistas de cima.

A classe está implementada, normalizada e coberta por testes unitários; o
que falta é evidência visual em vídeo.

### 2. Tráfego assimétrico no vídeo de teste

O material apresenta 34 veículos em um sentido contra 6 no oposto, o que
produz uma contagem de entrada pouco representativa (2 eventos). A lógica de
direção está validada por teste unitário
(`test_direcoes_opostas_sao_distinguidas`) e os 2 eventos de entrada
comprovam que o sentido oposto é detectado, mas um vídeo com fluxo
equilibrado produziria uma demonstração mais convincente.

### 3. Dashboard não implementado

Previsto no enunciado como evolução futura. As métricas estão disponíveis
via `/api/v1/counts`, `/api/v1/events` e `/api/v1/performance`, e podem ser
consultadas pela interface Swagger em `/docs`.

### 4. Sem validação em Raspberry Pi 5 física

Ver a seção de desempenho. Nenhuma métrica foi medida em hardware ARM real.

### 5. Métricas de detecção não avaliadas quantitativamente

Precision, recall, F1, mAP50 e mAP50-95 exigiriam um conjunto anotado do
cenário-alvo, inexistente neste escopo. As métricas de contagem, latência e
memória foram medidas.

### 6. Persistência apenas em memória

Eventos são perdidos ao reiniciar a aplicação. Persistência em SQLite ou
JSON é uma evolução prevista.

### 7. Processamento de vídeo síncrono

Adequado ao escopo atual. Vídeos longos mantêm a conexão HTTP aberta durante
todo o processamento; `max_frames` e `stride` limitam a duração.

### 8. Calibração dependente de resolução

As coordenadas da linha estão em pixels absolutos, exigindo recalibração ao
trocar a resolução da fonte. Converter para coordenadas normalizadas
(0,0 a 1,0) tornaria a configuração independente de resolução — evolução
prevista de baixo custo.

## Testes

```bash
pytest tests/ -v
```

17 testes unitários cobrindo geometria de cruzamento, determinação de
direção, histerese, bloqueio de contagem duplicada, expiração de tracks,
normalização de classes e estabilização de classe por votação.

Os testes usam detecções sintéticas e **não carregam o modelo**, executando
em cerca de 0,1 segundo. Isso mantém a lógica de contagem desacoplada do
detector.

Casos cobertos, referentes ao enunciado:

| Caso | Cobertura |
|---|---|
| 1 — Carro único atravessando | Teste unitário |
| 2 — Dois veículos próximos | Teste unitário |
| 4 — Veículo parado sobre a linha | Teste unitário |
| 5 — Veículo que aproxima e recua | Teste unitário |
| 6 e 7 — Entrada e saída | Teste unitário |
| 9 — Veículo que desaparece | Teste unitário |
| 12 — Frame sem veículos | Teste unitário |
| 13 — Imagem inválida | Validado via API (HTTP 400) |
| 15 — Requisição sem arquivo | Validado via API (HTTP 422) |

Casos não cobertos por falta de material de vídeo adequado: caminhão
parcialmente encoberto (3), moto próxima a carro (8), baixa iluminação (10) e
reflexos ou faróis (11).

---

## Licenciamento

Este projeto utiliza o modelo YOLO26n da Ultralytics, distribuído sob
**AGPL-3.0**. Consequentemente, este repositório também adota AGPL-3.0 e é
mantido aberto.

Para uma aplicação comercial proprietária, seria necessário avaliar uma
licença empresarial junto à Ultralytics ou adotar uma família de modelos com
licença compatível com o uso pretendido. Esta observação não constitui
aconselhamento jurídico.

O vídeo de teste utilizado durante o desenvolvimento não está versionado
neste repositório. Materiais de licença permissiva podem ser obtidos em
bancos como Pexels e Pixabay.
