import argparse
import importlib
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from core.metrics import load_gt, print_table, save_csv
from models import AVAILABLE_MODELS


def parse_args():
    # Define os argumentos aceitos pela linha de comando:
    # - video: caminho do vídeo de teste
    # - ground_truth: JSON com as anotações reais de cada frame (0=ausente, 1=andando, 2=parado)
    # - --output: onde salvar o CSV com os resultados (padrão: results/resultados.csv)
    # - --models: quais modelos rodar; se omitido, roda todos os disponíveis
    # - --conf: limiar de confiança mínimo para aceitar uma detecção
    # - --skip-frames: pula N frames entre cada análise, reduzindo o tempo de execução
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

    # Abre o vídeo apenas para contar o total de frames; fecha logo em seguida.
    # O total é necessário para alinhar o ground truth ao comprimento real do vídeo.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir {args.video}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Carrega as anotações do JSON e converte para binário:
    # gt_raw mantém os três estados (0/1/2) para as estatísticas descritivas,
    # gt_binary colapsa para 0 (pedestre ausente/andando) e 1 (pedestre parado),
    # que é o rótulo que os modelos precisam prever.
    gt_raw    = load_gt(args.ground_truth, total)
    gt_binary = [1 if v == 2 else 0 for v in gt_raw]

    # Aplica o mesmo skip ao ground truth para manter o alinhamento frame a frame
    # com as predições dos modelos, que também pulam frames.
    if args.skip_frames > 1:
        gt_binary = gt_binary[::args.skip_frames]

    # Exibe um resumo do vídeo e da distribuição de classes antes de rodar os modelos
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

    # Carrega e executa cada modelo dinamicamente via importlib.
    # Cada módulo em models/<id>.py deve expor uma função run(video, gt, conf, skip)
    # que processa o vídeo e devolve um dicionário com as métricas calculadas.
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

    # Imprime a tabela comparativa no terminal e salva o CSV com todas as métricas
    if results:
        print_table(results)
        save_csv(results, args.output)
    else:
        print("\nNenhum modelo foi executado com sucesso.")


if __name__ == "__main__":
    main()
