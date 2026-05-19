"""
Декодирование QR-кодов на ценниках без нейросетей.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import parse_qs

import cv2
import numpy as np
from loguru import logger
from pyzbar import pyzbar

from src.config import QRConfig


# =============================================================================
#  Маппинг короткие имена → канонические поля CSV
# =============================================================================
_QR_FIELD_MAP: Dict[str, str] = {
    "barcode": "qr_code_barcode", "b": "qr_code_barcode",
    "price1": "price1_qr", "p1": "price1_qr",
    "price2": "price2_qr", "p2": "price2_qr",
    "price3": "price3_qr", "p3": "price3_qr",
    "price4": "price4_qr", "p4": "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count", "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price", "wL1P": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count", "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price", "wL2P": "wholesale_level_2_price",
    "actionPrice": "action_price_qr", "aP": "action_price_qr",
    "actionCode": "action_code_qr", "aC": "action_code_qr",
}


@dataclass
class QRResult:
    raw_payload: Optional[str] = None
    fields: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.fields is None:
            self.fields = {}


class QRDecoder:
    """Каскадный декодер QR. CPU-only, без нейросетей."""

    def __init__(self, cfg: QRConfig) -> None:
        self.cfg = cfg
        self._cv2_detector = cv2.QRCodeDetector() if cfg.enable_cv2_fallback else None

    # -------------------------------------------------------------------
    def decode(self, image: np.ndarray) -> QRResult:
        """Применить каскад. Возвращает пустой QRResult, если ничего не найдено."""
        payload = self._try_pyzbar(image)
        if payload is None and self.cfg.enable_upscale_fallback:
            payload = self._try_pyzbar_upscaled(image)
        if payload is None and self._cv2_detector is not None:
            payload = self._try_cv2(image)

        if payload is None:
            return QRResult()

        fields = self._parse_payload(payload)
        return QRResult(raw_payload=payload, fields=fields)

    # -------------------------------------------------------------------
    def _try_pyzbar(self, image: np.ndarray) -> Optional[str]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            results = pyzbar.decode(gray)
            for r in results:
                if r.type == "QRCODE":
                    return r.data.decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"pyzbar упал: {e}")
        return None

    # -------------------------------------------------------------------
    def _try_pyzbar_upscaled(self, image: np.ndarray) -> Optional[str]:
        """Ресайз + CLAHE — помогает на маленьких/малоконтрастных QR."""
        f = self.cfg.upscale_factor
        new_size = (int(image.shape[1] * f), int(image.shape[0] * f))
        big = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if big.ndim == 3 else big
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return self._try_pyzbar(gray)

    # -------------------------------------------------------------------
    def _try_cv2(self, image: np.ndarray) -> Optional[str]:
        try:
            data, _, _ = self._cv2_detector.detectAndDecode(image)
            return data if data else None
        except Exception as e:
            logger.debug(f"cv2.QRCodeDetector упал: {e}")
            return None

    # -------------------------------------------------------------------
    def _parse_payload(self, raw: str) -> Dict[str, str]:
        """Парсим JSON или query-string. Игнорируем неизвестные поля."""
        parsed: Dict[str, str] = {}
        raw = raw.strip()

        # Вариант 1: JSON
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    canon = _QR_FIELD_MAP.get(k)
                    if canon is not None and v not in (None, ""):
                        parsed[canon] = str(v)
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Вариант 2: query-string (foo=bar&baz=qux)
        if "=" in raw and "&" in raw:
            for k, vs in parse_qs(raw).items():
                canon = _QR_FIELD_MAP.get(k)
                if canon is not None and vs:
                    parsed[canon] = vs[0]
            if parsed:
                return parsed

        # Вариант 3: простой EAN — только штрихкод
        if raw.isdigit() and 8 <= len(raw) <= 14:
            parsed["qr_code_barcode"] = raw

        return parsed
