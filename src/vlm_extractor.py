"""
Извлечение полей ценника через Qwen2-VL-2B-Instruct (мультимодальная LLM).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image
from pydantic import ValidationError
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # пакет иногда называется иначе в разных версиях
    process_vision_info = None  # type: ignore[assignment]

from src.config import VLMConfig
from src.schemas import VLMResponse


# =============================================================================
#  Системный промпт: один источник правды о формате JSON
#
#  Содержит подробное описание визуальных особенностей ценников Ленты:
#  надстрочные копейки, скидка в круге, тип выкладки в маленьком кружке.
#  Без этих подсказок Qwen2-VL стабильно ошибается на 30-40% полей.
# =============================================================================
_SYSTEM_PROMPT = """Ты — точный OCR-ассистент для ценников супермаркета \"Лента\".
По изображению ценника верни СТРОГО ОДИН JSON-объект без пояснений.

ВИЗУАЛЬНЫЕ ОСОБЕННОСТИ ЦЕННИКОВ ЛЕНТЫ:
1. ЦЕНЫ записаны крупно с НАДСТРОЧНЫМИ КОПЕЙКАМИ: \"168⁹⁰\" означает 168.90,
   \"1 029⁹⁹\" означает 1029.99. В JSON записывай ОДНИМ числом с точкой.
2. СКИДКА показана в КРУГЕ с надписью типа \"-32% от цены без карты\".
   В JSON пиши со знаком минус и процентом: \"-32%\".
3. ТИП ВЫКЛАДКИ — буква в МАЛЕНЬКОМ КРУЖКЕ между датой и штрихкодом:
   Ш (Штука), Л (Лоток), К (Короб). Если кружка нет — поле null.
4. ЦЕНА ПО АКЦИИ — обычно с подписью \"По карте от N шт\" или \"Цена за ед. по акции\"
   (часто на жёлтых ценниках).
5. ШТРИХ-КОД EAN-13 печатается под штрих-полосой с пробелами вида
   \"4 606272 000180\". В JSON записывай С пробелами как на ценнике.
6. id_sku — артикул, мелкий шрифт над штрихкодом, формат \"110301 002876\"
   (две группы по 6 цифр через пробел). Записывай С пробелом как на ценнике.
7. ДАТА ПЕЧАТИ — мелкий шрифт левее штрихкода, формат \"24.12.2025 12:25\".
8. КОД ЗОНЫ ВЫКЛАДКИ (code) — мелкий шрифт под id_sku, бывают форматы
   \"06_062 003\", \"21_ФЕВ 002_1_2_2\" — может содержать буквы, подчёркивания.
9. ДОП. ИНФО (additional_info) — может быть либо текст в рамке на самом
   ценнике (\"Сухое\", \"БИО\"), либо текст промо (\"3 по цене 2 от цены без карты\"),
   либо отдельная карточка-шелфтокер (\"номер на весах: 214\").
10. ЦВЕТ ценника — по фоновой подложке: белый = white, жёлтый = yellow,
    красный = red.

СТРУКТУРА JSON (поля строго в этом порядке):
{
  \"product_name\": \"<полное название товара одной строкой как на ценнике>\",
  \"price_default\": \"<цена без карты числом, например 250.09>\",
  \"price_card\": \"<цена по карте числом, например 168.90>\",
  \"price_discount\": \"<акционная цена числом, если есть третья цена>\",
  \"barcode\": \"<штрихкод EAN-13 с пробелами, например 4 606272 000180>\",
  \"discount_amount\": \"<скидка в формате -32%>\",
  \"id_sku\": \"<артикул с пробелом, например 110301 002876>\",
  \"print_datetime\": \"<дата печати в формате DD.MM.YYYY H:MM>\",
  \"code\": \"<код зоны выкладки как написано на ценнике>\",
  \"additional_info\": \"<доп. информация: Сухое / БИО / промо-условия / номер на весах>\",
  \"color\": \"<white | yellow | red>\",
  \"special_symbols\": \"<Ш | Л | К>\"
}

ЖЁСТКИЕ ПРАВИЛА:
- Если поле НЕ ВИДНО на ценнике → поставь null. Не выдумывай.
- Цены — только число с точкой, БЕЗ \"руб\", БЕЗ пробелов внутри числа.
- discount_amount всегда со знаком минус и процентом: \"-27%\".
- Возвращай ТОЛЬКО JSON, без markdown, без комментариев до/после.
"""

_USER_PROMPT = "Извлеки все поля с этого ценника. Возвращай только JSON."


# =============================================================================
#  Класс-обёртка
# =============================================================================
class VLMExtractor:
    """Загружает Qwen2-VL-2B-Instruct в 4-bit и вытягивает поля из crop'а."""

    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self._load_model()

    # -------------------------------------------------------------------
    def _load_model(self) -> None:
        """Инициализация модели и процессора.

        BitsAndBytesConfig:
          - load_in_4bit  → NF4 квантизация, ~50% VRAM от FP16
          - bnb_4bit_compute_dtype=bfloat16 → стабильнее float16 на RTX 30xx
          - bnb_4bit_quant_type=\"nf4\" → лучший recall vs fp4 на VLM-задачах
        """
        logger.info(f"Загрузка VLM: {self.cfg.model_id}")
        cache_dir = Path(self.cfg.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.load_in_4bit and torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs = {"quantization_config": bnb_cfg}
        else:
            # CPU fallback или явный отказ от квантизации
            model_kwargs = {"torch_dtype": torch.float16}

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            cache_dir=str(cache_dir),
            device_map="auto",
            **model_kwargs,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            cache_dir=str(cache_dir),
            min_pixels=self.cfg.min_pixels,
            max_pixels=self.cfg.max_pixels,
        )

        # Логируем фактический VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1024 ** 3
            logger.info(f"VLM загружена, VRAM: {vram_gb:.2f} ГБ")

    # -------------------------------------------------------------------
    def extract(self, image_bgr: np.ndarray) -> VLMResponse:
        """Прогнать один кроп ценника через VLM и вернуть распарсенный объект."""
        # Конвертация BGR → RGB → PIL (того ожидает qwen-vl-utils)
        rgb = image_bgr[..., ::-1]
        pil = Image.fromarray(rgb)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = [pil], None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-5),
            )

        # Отрезаем prompt и декодируем только сгенерированную часть
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        raw_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._parse_json(raw_text)

    # -------------------------------------------------------------------
    def extract_batch(self, images_bgr: List[np.ndarray]) -> List[VLMResponse]:
        """Заглушка под батчинг. Сейчас — последовательно, чтобы не упереться в VRAM.

        На RTX 3060 + Qwen2-VL-2B-4bit честный batch=2 уже подъедает 6–7 ГБ
        вместе с YOLO. По умолчанию — последовательно. Включить честный
        батчинг можно, когда YOLO уже выгружен (см. pipeline._run).
        """
        return [self.extract(img) for img in images_bgr]

    # -------------------------------------------------------------------
    @staticmethod
    def _parse_json(raw_text: str) -> VLMResponse:
        """Стараемся достать JSON даже из грязного ответа."""
        # Иногда модель оборачивает ответ в ```json ... ``` — выкусываем
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            logger.warning(f"VLM не вернула JSON: {raw_text[:200]}")
            return VLMResponse()

        json_str = match.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON некорректен: {e}. Сырой текст: {raw_text[:200]}")
            return VLMResponse()

        try:
            return VLMResponse.model_validate(data)
        except ValidationError as e:
            logger.warning(f"VLMResponse schema mismatch: {e}")
            # Достаём всё, что валидно
            safe = {
                k: v for k, v in data.items()
                if k in VLMResponse.model_fields
            }
            return VLMResponse(**safe)

    # -------------------------------------------------------------------
    def unload(self) -> None:
        """Освободить VRAM (нужно, если хотим грузить другую модель)."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
