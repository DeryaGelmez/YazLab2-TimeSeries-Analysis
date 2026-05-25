import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import config.config as cfg
from deep_learning.data_loader import build_batadal_cache, build_skab_cache
from deep_learning.figure_regeneration import regenerate_figures_for_run


def load_run_ids_from_manifest(path: Path) -> list[str]:
    run_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        run_ids.append(line.split("#", 1)[0].strip())
    return run_ids


def discover_run_ids(outputs_dir: Path) -> list[str]:
    run_ids: list[str] = []
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "history.json").exists() and (child / "model.pt").exists():
            run_ids.append(child.name)
    return run_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate figures from saved history.json and model.pt (no re-training)."
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="One or more run_id folder names under outputs/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Text file with one run_id per line (# comments allowed).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every run folder under outputs/ that has history.json and model.pt.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Defaults to config OUTPUTS_DIR.",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Only A1: loss.png and acc.png from history.json.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only A2: cm, roc, pr from model.pt (inference on test set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir or cfg.OUTPUTS_DIR

    if args.manifest:
        run_ids = load_run_ids_from_manifest(args.manifest.resolve())
    elif args.run_ids:
        run_ids = args.run_ids
    elif args.all:
        run_ids = discover_run_ids(outputs_dir)
    else:
        raise SystemExit(
            "Specify --run-ids, --manifest docs/dl_figures_curated_run_ids.txt, or --all."
        )

    if not run_ids:
        raise SystemExit("No run_ids to process.")

    needs_skab = any(r.startswith("skab_") for r in run_ids)
    needs_batadal = any(r.startswith("batadal_") for r in run_ids)

    skab_cache = build_skab_cache(cfg.DL_SKAB_N_FOLDS) if needs_skab else None
    batadal_cache = build_batadal_cache() if needs_batadal else None

    cfg.create_required_dirs()
    total = len(run_ids)
    print(
        f"Regenerating figures | runs={total}, outputs={outputs_dir}, "
        f"device={cfg.DL_DEVICE}, history_only={args.history_only}, test_only={args.test_only}"
    )

    for index, run_id in enumerate(run_ids, start=1):
        print(f"[{index}/{total}] {run_id}")
        regenerate_figures_for_run(
            run_id,
            outputs_dir=outputs_dir,
            skab_cache=skab_cache,
            batadal_cache=batadal_cache,
            history_only=args.history_only,
            test_only=args.test_only,
        )

    print(f"Done. Figures under {cfg.OUTPUTS_FIGURES_DIR}")


if __name__ == "__main__":
    main()
