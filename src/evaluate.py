"""
Avaliacao comparativa de modelos de deteccao de pedestres parados.

Uso basico (todos os modelos):
    python src/evaluate.py video.mp4 ground_truth.json

Selecionar modelos especificos:
    python src/evaluate.py video.mp4 ground_truth.json --models yolov8 retinanet fcos

Salvar em arquivo especifico:
    python src/evaluate.py video.mp4 ground_truth.json --output results/meu_resultado.csv

Modelos disponiveis:
    yolov8, ssd, faster_rcnn, retinanet, fcos,
    efficientdet, detr, rt_detr, cascade_rcnn, deformable_detr

O ground_truth.json deve ser gerado pelo annotate.py com os rotulos:
    0 = sem pedestre | 1 = andando | 2 = parado (aguardando travessia)

A avaliacao e binaria: o modelo acertou quando o semaforo deveria ser verde (GT=2)?
"""

import argparse
import importlib
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from core.metrics import load_gt, print_table, save_csv
from models import AVAILABLE_MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliacao comparativa de detectores de pedestres"
    )
    parser.add_argument("video",      help="Caminho para o video de teste")
    parser.add_argument("ground_truth", help="Arquivo JSON com anotacoes ground truth")
    parser.add_argument(
        "--output", "-o",
        default="results/resultados.csv",
        help="Arquivo CSV de saida (padrao: results/resultados.csv)",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        default=list(AVAILABLE_MODELS.keys()),
        help="Modelos a avaliar (padrao: todos)",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Limiar de confianca (padrao: 0.5)",
    )
    parser.add_argument(
        "--skip-frames", "-s",
        type=int,
        default=1,
        dest="skip_frames",
        help="Analisar 1 frame a cada N (padrao: 1 = todos os frames)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir {args.video}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    gt_raw    = load_gt(args.ground_truth, total)
    gt_binary = [1 if v == 2 else 0 for v in gt_raw]

    if args.skip_frames > 1:
        gt_binary = gt_binary[::args.skip_frames]

    n_parado  = sum(1 for v in gt_raw if v == 2)
    n_andando = sum(1 for v in gt_raw if v == 1)
    n_ausente = sum(1 for v in gt_raw if v == 0)
    frames_analisados = len(gt_binary)
    print(f"\nVideo: {total} frames", end="")
    if args.skip_frames > 1:
        print(f" (analisando {frames_analisados} com skip={args.skip_frames})")
    else:
        print()
    print(f"  Parado:        {n_parado}")
    print(f"  Andando:       {n_andando}")
    print(f"  Sem pedestre:  {n_ausente}")
    print(f"\nModelos selecionados: {', '.join(args.models)}")

    # Garantir que a pasta de saida existe
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for model_id in args.models:
        label = AVAILABLE_MODELS[model_id]
        print(f"\n[{label}]")
        try:
            module = importlib.import_module(f"models.{model_id}")
            result = module.run(args.video, gt_binary, args.conf, args.skip_frames)
            results.append(result)
        except ImportError as e:
            print(f"  Dependencia nao instalada, pulando: {e}")
        except FileNotFoundError as e:
            print(f"  Arquivos de modelo nao encontrados, pulando:\n  {e}")
        except Exception as e:
            print(f"  Erro inesperado: {e}")

    if results:
        print_table(results)
        save_csv(results, args.output)
    else:
        print("\nNenhum modelo foi executado com sucesso.")


if __name__ == "__main__":
    main()
