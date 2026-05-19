"""
Lookup-сервис по справочнику товаров Ленты (db_hack.csv).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class ProductDatabase:
    """Загружает справочник в память, отдаёт fullname по штрихкоду."""

    def __init__(self, csv_path: str | Path, encoding: str = "cp1251") -> None:
        self.csv_path = Path(csv_path)
        self.encoding = encoding
        self._index: Dict[str, str] = {}
        self._load()

    # -------------------------------------------------------------------
    def _load(self) -> None:
        """Прочитать CSV и построить индекс `штрихкод → fullname`."""
        if not self.csv_path.exists():
            logger.warning(
                f"Справочник товаров не найден: {self.csv_path}. "
                f"Lookup по БД отключён, будут использоваться сырые ответы VLM."
            )
            return

        rows_count = 0
        with self.csv_path.open("r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                code = self._normalize_code(row.get("code", ""))
                fullname = (row.get("fullname") or "").strip()
                if code and fullname:
                    self._index[code] = fullname
                    rows_count += 1

        logger.info(
            f"Справочник товаров загружен: {rows_count} строк → "
            f"{len(self._index)} уникальных штрихкодов"
        )

    # -------------------------------------------------------------------
    @staticmethod
    def _normalize_code(raw: str) -> str:
        """Убираем пробелы и любые нецифровые символы.

        Это нужно, чтобы единообразно сравнивать штрихкоды, попавшие
        к нам разными путями:
          - из QR-кода → \"4606272000180\"
          - с ценника текстом → \"4 606272 000180\"
          - из БД → \"4606272000180\"
        """
        return re.sub(r"\D", "", str(raw))

    # -------------------------------------------------------------------
    def lookup(self, barcode: str) -> Optional[str]:
        """Найти эталонное название по штрихкоду. None — если не нашли."""
        norm = self._normalize_code(barcode)
        if not norm:
            return None
        return self._index.get(norm)

    # -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    # -------------------------------------------------------------------
    def __contains__(self, barcode: str) -> bool:
        return self._normalize_code(barcode) in self._index
