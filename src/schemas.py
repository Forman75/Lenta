"""
Доменные схемы данных. Один источник правды о том, как выглядит результат.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field


# =============================================================================
#  Финальная строка CSV
# =============================================================================
class PricerRecord(BaseModel):
    """Одна строка выходного CSV — один уникальный ценник.
    """

    # --- Идентификация и геометрия ------------------------------------------
    filename: str
    product_name: str = ""
    price_default: str = ""
    price_card: str = ""
    price_discount: str = ""
    barcode: str = ""
    discount_amount: str = ""
    id_sku: str = ""
    print_datetime: str = ""
    code: str = ""
    additional_info: str = ""
    color: str = ""           # red / yellow / white
    special_symbols: str = "" # Ш / Л / К  (Штука / Лоток / Короб)

    # --- Привязка к видео ---------------------------------------------------
    frame_timestamp: int = 0  # миллисекунды от начала видео
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0

    # --- Данные из QR-кода --------------------------------------------------
    qr_code_barcode: str = ""
    price1_qr: str = ""
    price2_qr: str = ""
    price3_qr: str = ""
    price4_qr: str = ""
    wholesale_level_1_count: str = ""
    wholesale_level_1_price: str = ""
    wholesale_level_2_count: str = ""
    wholesale_level_2_price: str = ""
    action_price_qr: str = ""
    action_code_qr: str = ""

    @classmethod
    def csv_columns(cls) -> List[str]:
        """Точный порядок колонок для CSV-вывода."""
        return list(cls.model_fields.keys())


# =============================================================================
#  VLM JSON — то, что должен вернуть Qwen2-VL (строго!)
# =============================================================================
class VLMResponse(BaseModel):
    """Схема ответа VLM. Используется и в промпте, и для парсинга."""

    product_name: Optional[str] = None
    price_default: Optional[str] = None
    price_card: Optional[str] = None
    price_discount: Optional[str] = None
    barcode: Optional[str] = None
    discount_amount: Optional[str] = None
    id_sku: Optional[str] = None
    print_datetime: Optional[str] = None
    code: Optional[str] = None
    additional_info: Optional[str] = None
    color: Optional[str] = None
    special_symbols: Optional[str] = None


# =============================================================================
#  Внутренние структуры пайплайна (НЕ сериализуются)
# =============================================================================
@dataclass
class Detection:
    """Одна детекция YOLO в одном кадре."""

    track_id: int
    bbox: np.ndarray            # shape (4,) — xyxy в координатах кадра
    confidence: float
    frame_idx: int
    frame_timestamp_ms: int
    crop: np.ndarray            # BGR numpy-кадр ценника
    quality_score: float = 0.0  # Рассчитывается в TrackManager

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))

    @property
    def aspect_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        return float(w / h) if h > 0 else 0.0


@dataclass
class TrackCandidate:

    track_id: int
    best_detection: Detection
    all_detections_count: int = 0  # Для статистики
    qr_payload: Optional[str] = None  # Сырая строка из QR-кода
