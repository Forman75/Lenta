"""
Сборка `PricerRecord` из результатов VLM + QR + bbox.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from loguru import logger

from src.config import OutputConfig
from src.product_db import ProductDatabase
from src.qr_decoder import QRResult
from src.schemas import PricerRecord, TrackCandidate, VLMResponse


# Что Qwen может вернуть, имея в виду "поле отсутствует"
_ABSENT_TOKENS = {"нет", "no", "none", "—", "-", "n/a", "null"}

# Регулярка для нормализации цены: "2 345,99 руб" → "2345.99"
_PRICE_RE = re.compile(r"[\d.,]+")


class PostProcessor:
    """Превращает VLM+QR в `PricerRecord` и пишет финальный CSV."""

    def __init__(
        self,
        cfg: OutputConfig,
        product_db: Optional[ProductDatabase] = None,
    ) -> None:
        self.cfg = cfg
        self.product_db = product_db
        self.out_dir = Path(cfg.csv_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Счётчики качества для логирования сводки
        self._db_hits = 0
        self._db_misses = 0

    # -------------------------------------------------------------------
    def build_record(
        self,
        filename: str,
        candidate: TrackCandidate,
        vlm: VLMResponse,
        qr: QRResult,
    ) -> PricerRecord:
        """Собрать одну строку CSV."""
        det = candidate.best_detection
        rec = PricerRecord(
            filename=filename,
            frame_timestamp=det.frame_timestamp_ms,
            x_min=round(float(det.bbox[0]), 1),
            y_min=round(float(det.bbox[1]), 1),
            x_max=round(float(det.bbox[2]), 1),
            y_max=round(float(det.bbox[3]), 1),
        )

        # --- Поля от VLM ----------------------------------------------------
        rec.product_name = self._norm_text(vlm.product_name)
        rec.price_default = self._norm_price(vlm.price_default)
        rec.price_card = self._norm_price(vlm.price_card)
        rec.price_discount = self._norm_price(vlm.price_discount)
        rec.barcode = self._norm_barcode_visual(vlm.barcode)
        rec.discount_amount = self._norm_discount(vlm.discount_amount)
        rec.id_sku = self._norm_id_sku_visual(vlm.id_sku)
        rec.print_datetime = self._norm_datetime(vlm.print_datetime)
        rec.code = self._norm_text(vlm.code)
        rec.additional_info = self._norm_text(vlm.additional_info)
        rec.color = self._norm_color(vlm.color)
        rec.special_symbols = self._norm_special(vlm.special_symbols)

        # --- Поля от QR (имеют приоритет, т.к. это структурированный источник)
        for k, v in qr.fields.items():
            if hasattr(rec, k):
                setattr(rec, k, str(v))

        # --- DB-lookup: подмена product_name на эталон ---------------------
        # Приоритет источников штрихкода: QR (надёжно) → текст с ценника (OCR)
        if self.product_db is not None:
            for candidate_bc in (rec.qr_code_barcode, rec.barcode):
                if not candidate_bc or candidate_bc == self.cfg.not_present_marker:
                    continue
                ref_name = self.product_db.lookup(candidate_bc)
                if ref_name:
                    rec.product_name = ref_name
                    self._db_hits += 1
                    break
            else:
                # ни один штрихкод не сматчился — статистика для лога
                if rec.qr_code_barcode or rec.barcode:
                    self._db_misses += 1

        return rec

    # ===================================================================
    #  Нормализаторы (private)
    # ===================================================================
    def _present(self, raw: Optional[str]) -> Optional[str]:
        """Различает 'нет' / 'не распознано' / реальное значение."""
        if raw is None:
            return None  # → "" в финале (не распознано)
        s = str(raw).strip().lower()
        if s in _ABSENT_TOKENS or s == "":
            return self.cfg.not_present_marker
        return raw

    def _norm_text(self, raw: Optional[str]) -> str:
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v
        return str(v).strip()

    def _norm_price(self, raw: Optional[str]) -> str:
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v
        # "2 345,99 руб" → "2345.99"
        match = _PRICE_RE.search(str(v).replace(" ", ""))
        if not match:
            return self.cfg.not_recognized_marker
        num = match.group(0).replace(",", ".")
        try:
            return f"{float(num):.2f}".rstrip("0").rstrip(".") or "0"
        except ValueError:
            return self.cfg.not_recognized_marker

    def _norm_digits(self, raw: Optional[str], expected_len=None) -> str:
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v
        digits = re.sub(r"\D", "", str(v))
        if not digits:
            return self.cfg.not_recognized_marker
        if expected_len:
            lo, hi = expected_len
            if not (lo <= len(digits) <= hi):
                # длина не подходит — но всё равно сохраняем,
                # пусть валидируется на стороне сравнения
                logger.debug(f"Подозрительная длина {len(digits)}: {digits}")
        return digits

    # -------------------------------------------------------------------
    def _norm_barcode_visual(self, raw: Optional[str]) -> str:
        """Штрихкод в эталоне Ленты пишется с пробелами как НА ЦЕННИКЕ:
        "4 606272 000180". Если VLM вернула без пробелов — расставляем
        по стандартной разбивке EAN-13: 1 + 6 + 6.

        EAN-13 структура: первая цифра — система нумерации,
        следующие 6 — производитель, ещё 6 — товар + контрольная.
        """
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v

        digits = re.sub(r"\D", "", str(v))
        if not digits:
            return self.cfg.not_recognized_marker

        # EAN-13 стандарт: 13 цифр → "X XXXXXX XXXXXX"
        if len(digits) == 13:
            return f"{digits[0]} {digits[1:7]} {digits[7:13]}"
        # EAN-8 (короткий): 8 цифр → пишем как есть
        if len(digits) == 8:
            return digits
        # Прочие длины (внутренние коды, ITF-14 и т.п.) — без разбивки
        return digits

    # -------------------------------------------------------------------
    def _norm_id_sku_visual(self, raw: Optional[str]) -> str:
        """Артикул в эталоне с пробелом между двумя группами по 6 цифр:
        "110301 002876". Если VLM вернула слитно — разбиваем по 6.
        """
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v

        digits = re.sub(r"\D", "", str(v))
        if not digits:
            return self.cfg.not_recognized_marker

        if len(digits) == 12:
            return f"{digits[:6]} {digits[6:]}"
        return digits

    def _norm_discount(self, raw: Optional[str]) -> str:
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v
        s = str(v).replace(" ", "")
        # Достаём число
        m = re.search(r"-?\d+", s)
        if not m:
            return self.cfg.not_recognized_marker
        num = int(m.group(0))
        # Принудительно ставим знак минуса и %
        return f"-{abs(num)}%"

    def _norm_datetime(self, raw: Optional[str]) -> str:
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v
        # Принимаем самые разные форматы, приводим к "DD.MM.YYYY H:MM"
        # Из ТЗ-разметки видно, что часы могут быть однозначные (1:14 не 01:14)
        from dateutil import parser as date_parser
        try:
            dt = date_parser.parse(str(v), dayfirst=True, fuzzy=True)
            return f"{dt.day:02d}.{dt.month:02d}.{dt.year} {dt.hour}:{dt.minute:02d}"
        except (ValueError, OverflowError):
            return str(v).strip()

    def _norm_color(self, raw: Optional[str]) -> str:
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v
        s = str(v).strip().lower()
        mapping = {
            "red": "red", "красный": "red",
            "yellow": "yellow", "жёлтый": "yellow", "желтый": "yellow",
            "white": "white", "белый": "white",
        }
        return mapping.get(s, s)

    def _norm_special(self, raw: Optional[str]) -> str:
        """Из ТЗ известны три варианта: Ш (Штука), Л (Лоток), К (Короб)."""
        v = self._present(raw)
        if v is None:
            return self.cfg.not_recognized_marker
        if v == self.cfg.not_present_marker:
            return v
        s = str(v).strip()
        if not s:
            return self.cfg.not_recognized_marker
        # Берём первый русский символ из числа допустимых
        for ch in s:
            if ch.upper() in {"Ш", "Л", "К"}:
                return ch.upper()
        return s  # отдаём как есть, если не распознали — пусть проверяют

    # ===================================================================
    #  CSV writer
    # ===================================================================
    def write_csv(
        self,
        records: Iterable[PricerRecord],
        video_name: str,
    ) -> Path:
        """Запись финального CSV. Имя файла = по имени видео."""
        out_path = self.out_dir / f"{Path(video_name).stem}_result.csv"
        cols = PricerRecord.csv_columns()
        df = pd.DataFrame([r.model_dump() for r in records], columns=cols)

        df.to_csv(
            out_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
        )
        logger.info(f"CSV записан: {out_path} ({len(df)} строк)")

        # Сводка по DB-lookup за это видео
        if self.product_db is not None:
            total = self._db_hits + self._db_misses
            if total > 0:
                hit_rate = 100.0 * self._db_hits / total
                logger.info(
                    f"DB-lookup: {self._db_hits}/{total} ценников "
                    f"получили эталонное название ({hit_rate:.1f}%)"
                )
            # Сбрасываем счётчики для следующего видео
            self._db_hits = 0
            self._db_misses = 0

        return out_path
