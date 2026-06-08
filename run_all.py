"""
Executa todos os modelos em todos os videos que possuem ground truth.

Cada video gera um CSV individual em results/{nome_video}.csv
Ao final, gera results/todos_resultados.csv consolidado com coluna "Video".

Convencao de nomes esperada em videos/:
    meu_video.mp4  +  meu_video_gt.json

Uso:
    python run_all.py
    python run_all.py --models yolov8 ssd faster_rcnn
    python run_all.py --conf 0.4 --output-dir meus_resultados/
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia todos os videos com todos os modelos automaticamente"
    )
    parser.add_argument(
        "--videos-dir", default="videos",
        help="Pasta com videos e GT (padrao: videos/)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Pasta de saida dos CSVs (padrao: results/)",
    )
    parser.add_argument(
        "--models", "-m", nargs="+",
        help="Modelos a usar (padrao: todos)",
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=0.5,
        help="Limiar de confianca (padrao: 0.5)",
    )
    parser.add_argument(
        "--skip-frames", "-s", type=int, default=1,
        dest="skip_frames",
        help="Analisar 1 frame a cada N (padrao: 1 = todos). Ex: 5 = 5x mais rapido",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Pular videos que ja possuem CSV em output-dir",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Nao gerar o CSV consolidado todos_resultados.csv",
    )
    return parser.parse_args()


def find_pairs(videos_dir: Path):
    pairs = []
    for gt_file in sorted(videos_dir.glob("*_gt.json")):
        video_stem = gt_file.stem[:-3]  # "meu_video_gt" -> "meu_video"
        video_file = videos_dir / f"{video_stem}.mp4"
        if video_file.exists():
            pairs.append((video_stem, video_file, gt_file))
        else:
            print(f"[AVISO] GT encontrado mas video nao existe: {video_file}")
    return pairs


def merge_csvs(output_dir: Path, pairs, merged_path: Path):
    all_rows = []
    headers = None
    for video_name, _, _ in pairs:
        csv_path = output_dir / f"{video_name}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if headers is None:
                headers = ["Video"] + list(reader.fieldnames)
            for row in reader:
                all_rows.append({"Video": video_name, **row})

    if not all_rows:
        print("[AVISO] Nenhum CSV gerado para consolidar.")
        return

    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nConsolidado salvo em: {merged_path}")


def main():
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(videos_dir)
    if not pairs:
        print(f"Nenhum par video+GT encontrado em '{videos_dir}/'")
        print("Esperado: video.mp4 e video_gt.json na mesma pasta.")
        sys.exit(1)

    print(f"Encontrados {len(pairs)} video(s) com ground truth:")
    for name, video, gt in pairs:
        print(f"  {name}: {video.name}  +  {gt.name}")

    evaluate_script = Path(__file__).parent / "src" / "evaluate.py"
    failed = []

    for i, (video_name, video_file, gt_file) in enumerate(pairs, 1):
        print(f"\n{'='*65}")
        print(f"[{i}/{len(pairs)}] Video: {video_name}")
        print(f"{'='*65}")

        if args.skip_existing and (output_dir / f"{video_name}.csv").exists():
            print(f"  CSV ja existe, pulando.")
            continue

        output_csv = output_dir / f"{video_name}.csv"
        cmd = [
            sys.executable, str(evaluate_script),
            str(video_file), str(gt_file),
            "--output", str(output_csv),
            "--conf", str(args.conf),
            "--skip-frames", str(args.skip_frames),
        ]
        if args.models:
            cmd += ["--models"] + args.models

        result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        if result.returncode != 0:
            print(f"[ERRO] Falha ao processar '{video_name}' (codigo {result.returncode})")
            failed.append(video_name)

    print(f"\n{'='*65}")
    print(f"Concluido: {len(pairs) - len(failed)}/{len(pairs)} videos processados com sucesso.")
    if failed:
        print(f"Falhas: {', '.join(failed)}")

    if not args.no_merge:
        merged_path = output_dir / "todos_resultados.csv"
        merge_csvs(output_dir, pairs, merged_path)


if __name__ == "__main__":
    main()
