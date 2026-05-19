"""
Общие утилиты. Логирование через loguru, метрики качества кадра.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


# =============================================================================
#  Настройка логирования
# =============================================================================
def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Сконфигурировать loguru: stderr + ротация в файл."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<cyan>{name}:{line}</cyan> - {message}"
        ),
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "pricer_{time:YYYYMMDD}.log",
        rotation="50 MB",
        retention="14 days",
        level="DEBUG",
        encoding="utf-8",
    )


# =============================================================================
#  Метрики качества изображения
# =============================================================================
def laplacian_variance(image: np.ndarray) -> float:
    """Дисперсия лапласиана — индикатор резкости.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image: np.ndarray) -> float:
    """Средняя яркость в L-канале. Пенализует переэкспонированные / тёмные кадры."""
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return float(lab[..., 0].mean())
    return float(image.mean())


def is_near_edge(bbox: np.ndarray, frame_shape: tuple, margin: int) -> bool:
    """Bbox слишком близко к краю — ценник обрезан, в VLM лучше не отправлять."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return bool(x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin)


def safe_crop(image: np.ndarray, bbox: np.ndarray, pad: int = 8) -> np.ndarray:
    """Кроп с защитой от выхода за границы + небольшой padding.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2].copy()
