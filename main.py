"""
CLI для пакетной обработки видео без UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from loguru import logger

from src.config import PipelineConfig
from src.pipeline import PricerPipeline
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lenta Pricer — CLI распознавания ценников"
    )
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--video", type=Path, nargs="+", help="Файл(ы) видео")
    src_grp.add_argument("--dir", type=Path, help="Директория с видеофайлами")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Путь к pipeline.yaml",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def collect_videos(args: argparse.Namespace) -> List[Path]:
    if args.video:
        return [Path(p) for p in args.video]
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(
        p for p in Path(args.dir).rglob("*")
        if p.suffix.lower() in extensions
    )


def progress_print(pct: float, msg: str) -> None:
    """Печатаем прогресс в одну строку с возвратом каретки."""
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r[{bar}] {pct * 100:5.1f}% — {msg[:60]:<60}")
    sys.stdout.flush()
    if pct >= 1.0:
        sys.stdout.write("\n")


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)

    cfg = PipelineConfig.from_yaml(args.config)
    pipeline = PricerPipeline(cfg)

    videos = collect_videos(args)
    if not videos:
        logger.error("Видеофайлы не найдены.")
        return 1

    logger.info(f"Будет обработано видео: {len(videos)}")
    for v in videos:
        try:
            out = pipeline.run(v, progress_cb=progress_print)
            logger.info(f"→ {out}")
        except Exception as e:
            logger.exception(f"Сбой на {v}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
