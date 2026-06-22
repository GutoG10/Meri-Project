# Sistema Inteligente de Controle Semafórico com Visão Computacional

Projeto de TCC — Comparação de 10 detectores de objetos para detecção de pedestres parados, com protótipo de semáforo virtual que acende verde quando um pedestre fica parado aguardando a travessia.

---

## Sumário

1. [Pré-requisitos](#1-pré-requisitos)
2. [Instalação](#2-instalação)
3. [Pesos dos modelos](#3-pesos-dos-modelos)
4. [Estrutura do projeto](#4-estrutura-do-projeto)
5. [Anotação de Ground Truth](#5-anotação-de-ground-truth)
6. [Avaliação de um vídeo](#6-avaliação-de-um-vídeo)
7. [Avaliação em lote (todos os vídeos)](#7-avaliação-em-lote-todos-os-vídeos)
8. [Visualização das detecções](#8-visualização-das-detecções)
9. [Detecção em tempo real (webcam)](#9-detecção-em-tempo-real-webcam)
10. [Modelos disponíveis](#10-modelos-disponíveis)

---

## 1. Pré-requisitos

- Python **3.10** ou superior
- [Git](https://git-scm.com/)
- (Recomendado) NVIDIA GPU com CUDA — os modelos funcionam em CPU, porém muito mais lentos

---

## 2. Instalação

Link do repositório:
https://github.com/GutoG10/tcc-project

```bash
# Clone o repositório
git clone https://github.com/GutoG10/tcc-project.git
cd tcc-project

# Crie e ative um ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

> **Nota:** A biblioteca `effdet` (EfficientDet) pode exigir `pip install effdet timm` separadamente caso a instalação falhe via `requirements.txt`.

---

## 3. Pesos dos modelos

Os modelos baseados em **torchvision** (SSD, Faster R-CNN, RetinaNet, FCOS) e **Hugging Face** (DETR, Conditional DETR, Deformable DETR, EfficientDet) baixam os pesos automaticamente na primeira execução.

Os modelos **Ultralytics** precisam dos arquivos `.pt` na raiz do projeto:

| Modelo  | Arquivo         | Download                                                       |
|---------|-----------------|----------------------------------------------------------------|
| YOLOv8  | `yolov8s.pt`    | `python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"` |
| RT-DETR | `rtdetr-l.pt`   | `python -c "from ultralytics import RTDETR; RTDETR('rtdetr-l.pt')"` |

---

## 4. Estrutura do projeto

```
tcc-project/
├── src/
│   ├── annotate.py          # Ferramenta de anotação de ground truth
│   ├── evaluate.py          # Benchmark de um vídeo com modelos selecionáveis
│   ├── core/
│   │   ├── metrics.py       # Métricas, is_stopped(), compute_metrics()
│   │   └── tracker.py       # CentroidTracker para modelos sem rastreamento nativo
│   ├── models/              # Um módulo por modelo — todos expõem run()
│   └── detect/              # Versões de exibição em tempo real
├── run_all.py               # Avaliação em lote: todos os vídeos × todos os modelos
├── visualize.py             # Visualização quadro a quadro com detecções anotadas
├── videos/                  # Vídeos de teste + arquivos GT (*_gt.json)
├── results/                 # CSVs gerados pelas avaliações
├── weights/                 # Pesos dos modelos Ultralytics (opcional)
└── requirements.txt
```

**Convenção de nomes em `videos/`:**

```
videos/
├── meu_video.mp4
└── meu_video_gt.json   ← ground truth gerado pelo annotate.py
```

> **Importante:** A pasta `videos/` não vem com vídeos no repositório. Crie-a e adicione seus próprios vídeos de pedestres antes de usar `evaluate.py`, `run_all.py` ou `visualize.py`.
> ```bash
> mkdir videos
> # copie seus vídeos .mp4 para dentro de videos/
> ```

---

## 5. Anotação de Ground Truth

Use `src/annotate.py` para rotular manualmente os frames de um vídeo antes de avaliá-lo.

```bash
python src/annotate.py videos/meu_video.mp4 videos/meu_video_gt.json
```

### Controles durante a reprodução

| Tecla   | Ação                              |
|---------|-----------------------------------|
| `0`     | Sem pedestre                      |
| `1`     | Pedestre andando                  |
| `2`     | Pedestre **parado** (aguardando)  |
| `SPACE` | Pausar / Continuar                |
| `ESC`   | Salvar e encerrar                 |

O arquivo JSON gerado mapeia cada número de frame para um rótulo (0, 1 ou 2) e é o input obrigatório para `evaluate.py`.

---

## 6. Avaliação de um vídeo

Use `src/evaluate.py` para rodar um ou mais modelos em um único vídeo e obter métricas de desempenho.

### Todos os modelos (padrão)

```bash
python src/evaluate.py videos/meu_video.mp4 videos/meu_video_gt.json
```

### Modelos específicos

```bash
python src/evaluate.py videos/meu_video.mp4 videos/meu_video_gt.json \
    --models yolov8 faster_rcnn retinanet
```

### Todas as opções

```bash
python src/evaluate.py <video> <ground_truth.json> [opções]

Opções:
  --models  / -m   Modelos a avaliar (padrão: todos os 10)
  --output  / -o   Arquivo CSV de saída (padrão: results/resultados.csv)
  --conf    / -c   Limiar de confiança, 0–1 (padrão: 0.5)
  --skip-frames/-s Analisar 1 frame a cada N — ex.: 5 = 5× mais rápido (padrão: 1)
```

### Exemplo completo

```bash
python src/evaluate.py videos/mulher.mp4 videos/mulher_gt.json \
    --models yolov8 ssd faster_rcnn \
    --output results/mulher_3modelos.csv \
    --conf 0.4 \
    --skip-frames 5
```

A tabela de resultados é exibida no terminal e salva em CSV na pasta `results/`.

---

## 7. Avaliação em lote (todos os vídeos)

Use `run_all.py` para processar automaticamente todos os vídeos que possuem um `_gt.json` correspondente na pasta `videos/`. Gera um CSV por vídeo e um consolidado `todos_resultados.csv`.

### Uso básico

```bash
python run_all.py
```

### Todas as opções

```bash
python run_all.py [opções]

Opções:
  --videos-dir DIR      Pasta com vídeos e GT (padrão: videos/)
  --output-dir DIR      Pasta de saída dos CSVs (padrão: results/)
  --models / -m         Modelos a usar (padrão: todos)
  --conf / -c           Limiar de confiança (padrão: 0.5)
  --skip-frames / -s    Analisar 1 a cada N frames (padrão: 1)
  --skip-existing       Pular vídeos que já têm CSV gerado
  --no-merge            Não gerar o CSV consolidado todos_resultados.csv
```

### Exemplos

```bash
# Avaliar apenas 3 modelos, pulando vídeos já processados
python run_all.py --models yolov8 ssd faster_rcnn --skip-existing

# Processamento rápido (5× mais veloz) com saída customizada
python run_all.py --skip-frames 5 --output-dir resultados/experimento1

# Rodar tudo com limiar de confiança menor
python run_all.py --conf 0.3
```

Ao final, `results/todos_resultados.csv` consolida todos os resultados com uma coluna extra `Video` para identificar cada vídeo.

---

## 8. Visualização das detecções

Use `visualize.py` para inspecionar quadro a quadro como um modelo detecta e rastreia pedestres, com sobreposição do estado do semáforo virtual.

```bash
python visualize.py <video> <modelo> [opções]
```

### Exemplos

```bash
# Visualização básica — salva screenshots a cada 30 frames
python visualize.py videos/mulher.mp4 yolov8

# Com ground truth sobreposto no painel
python visualize.py videos/mulher.mp4 faster_rcnn --gt videos/mulher_gt.json

# Salvar 1 screenshot a cada 20 frames em pasta específica
python visualize.py videos/mulher.mp4 yolov8 --every-n 20 --output capturas/

# Exportar vídeo anotado completo
python visualize.py videos/mulher.mp4 yolov8 --video-out capturas/mulher_yolo.mp4

# Salvar apenas frames onde o estado do semáforo muda (--every-n 0)
python visualize.py videos/mulher.mp4 retinanet --every-n 0
```

### Todas as opções

```bash
python visualize.py <video> <modelo> [opções]

Argumentos obrigatórios:
  video          Caminho para o vídeo
  modelo         Um dos modelos listados abaixo

Opções:
  --gt FILE      Arquivo JSON de ground truth para comparação (opcional)
  --output / -o  Pasta raiz para screenshots (padrão: capturas/)
  --every-n / -n Salvar 1 frame a cada N; 0 = apenas mudanças de estado (padrão: 30)
  --conf / -c    Limiar de confiança (padrão: 0.5)
  --video-out    Caminho para salvar vídeo anotado (ex: capturas/saida.mp4)
```

**Legenda visual:**
- Caixa **verde** + "PARADO" → pedestre parado detectado → semáforo **VERDE**
- Caixa **laranja** + "pessoa" → pedestre em movimento → semáforo **VERMELHO**

---

## 9. Detecção em tempo real (webcam)

Os scripts em `src/detect/` rodam o semáforo virtual diretamente na câmera do computador, sem necessidade de vídeo ou ground truth. Cada arquivo corresponde a um modelo.

```bash
python src/detect/<modelo>.py
```

| Arquivo                          | Modelo           |
|----------------------------------|------------------|
| `src/detect/yolov8.py`           | YOLOv8           |
| `src/detect/rt_detr.py`          | RT-DETR          |
| `src/detect/ssd.py`              | SSD              |
| `src/detect/faster_rcnn.py`      | Faster R-CNN     |
| `src/detect/retinanet.py`        | RetinaNet        |
| `src/detect/fcos.py`             | FCOS             |
| `src/detect/efficientdet.py`     | EfficientDet     |
| `src/detect/detr.py`             | DETR             |
| `src/detect/conditional_detr.py` | Conditional DETR |
| `src/detect/deformable_detr.py`  | Deformable DETR  |

### Exemplo

```bash
# YOLOv8 com webcam padrão (câmera 0)
python src/detect/yolov8.py

# Faster R-CNN com webcam padrão
python src/detect/faster_rcnn.py
```

Duas janelas são abertas:
- **Detecção** — frame da câmera com caixas delimitadoras coloridas
  - Verde + "Aguardando travessia" → pedestre parado
  - Vermelho + "Andando" → pedestre em movimento
- **Semáforo Virtual** — círculo verde (VERDE) ou vermelho (VERMELHO)

Pressione `ESC` para encerrar.

> **Nota:** Para usar outra câmera, edite a linha `cv2.VideoCapture(0)` no script correspondente, trocando `0` pelo índice do dispositivo desejado.

---

## 10. Modelos disponíveis

| ID                | Nome completo        | Backend       |
|-------------------|----------------------|---------------|
| `yolov8`          | YOLOv8s              | Ultralytics   |
| `rt_detr`         | RT-DETR-L            | Ultralytics   |
| `ssd`             | SSDLite320-MobileNetV3 | torchvision |
| `faster_rcnn`     | Faster R-CNN ResNet50-FPN | torchvision |
| `retinanet`       | RetinaNet ResNet50-FPN | torchvision  |
| `fcos`            | FCOS ResNet50-FPN    | torchvision   |
| `efficientdet`    | EfficientDet-D0      | effdet        |
| `detr`            | DETR ResNet-50       | Hugging Face  |
| `conditional_detr`| Conditional DETR     | Hugging Face  |
| `deformable_detr` | Deformable DETR      | Hugging Face  |

---

## Parâmetros internos de detecção de parada

| Parâmetro       | Valor | Descrição                                                      |
|-----------------|-------|----------------------------------------------------------------|
| `LIMIAR_PARADO` | 10 px | Deslocamento médio máximo por frame para considerar parado     |
| `FRAMES_PARADO` | 45    | Janela de frames analisados (~1,5 s a 30 FPS sem skip)        |

A avaliação é **binária**: o modelo acerta quando o semáforo deveria ser verde (GT = 2, pedestre parado) e o modelo também prevê parado.


comandos para apresentação

python src/detect/yolov8.py
python src/evaluate.py videos/mulher.mp4 videos/mulher_gt.json --models yolov8 --output "resultados apresentação/mulher.csv"
python visualize.py videos/mulher.mp4 yolov8
