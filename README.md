# Lenta Pricer

Локальное решение для автоматического распознавания ценников супермаркета «Лента» из видеопотока робота. Хакатон **Lenta Tech Life Hack**.

---

## Быстрый старт

```bash
# 1) Системная зависимость для pyzbar (Ubuntu/Debian)
sudo apt-get install -y libzbar0

# 2) Установка зависимостей
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3) Положить артефакты от заказчика:
#    - models/yolov8s_pricetag.pt   ← дообученные веса (опционально, есть fallback)
#    - data/db_hack.csv             ← справочник товаров (cp1251, schema: fullname;code)

# 4) Web UI
streamlit run app.py

# или CLI
python main.py --video path/to/video.mp4
```

---

## Архитектура

```
Video
  ↓
[FrameExtractor]  ── 2.5 FPS · undistort (Brown-Conrady по коэф. заказчика) · Laplacian-blur filter
  ↓
[YOLOv8 + ByteTrack]  ── детекция ценников · стабильный track_id между кадрами
  ↓
[TrackManager]  ── скоринг качества каждой детекции, выбор ЛУЧШЕГО кадра на трек
  ↓                                                                      ↘
[QR Decoder]  ── pyzbar → cv2.QRCodeDetector fallback     [VLM Qwen2-VL-2B 4-bit]  → strict-JSON OCR
  ↓                                                                      ↙
[ProductDB lookup]  ── штрихкод → эталонное название из db_hack.csv (625K товаров)
  ↓
[PostProcessor]  ── нормализация · приоритет QR > VLM, DB > VLM для названия
  ↓
result.csv  (29 колонок по спецификации)
```

### Почему так

| Решение | Альтернатива | Почему наш выбор |
| --- | --- | --- |
| **2.5 FPS вместо 30** | Все кадры | На 30 FPS один ценник детектируется 30–60 раз. 2.5 FPS даёт 4–6 — трекеру достаточно, нагрузка на YOLO ×12 меньше. |
| **ByteTrack** | DeepSORT / SORT | Встроен в ultralytics, не требует Re-ID сетки, отлично работает на стабильно видимых статичных объектах. |
| **Best-frame-per-track** | Прогон VLM на каждом кадре | Qwen2-VL-2B 4-bit ≈ 1 с/кадр. 50 ценников × 30 кадров = 25 минут. С best-frame = 50 с. **30× быстрее.** |
| **Qwen2-VL-2B (4-bit)** | Donut / TrOCR | Универсален: и текст, и таблица, и цвет ценника одной моделью. 4-bit nf4 = 1.8 ГБ VRAM. Лицензия Apache 2.0. |
| **pyzbar + cv2 fallback** | Нейросетевой QR-декодер | QR — это уже формальный код. CPU-декодеры работают за миллисекунды. Тратить GPU незачем. |
| **Undistort обязателен** | Снять и не корректировать | Заказчик выдал коэф. Brown-Conrady; широкоугольный 2.8 mm даёт заметную "бочку" по краям. Без коррекции — потеря 5–10% детекций. |
| **ProductDB lookup по штрихкоду** | Доверять только OCR | Заказчик дал `db_hack.csv` на 625 172 товара. Lookup по штрихкоду даёт **точное** название там, где VLM могла бы ошибиться в одной букве. Покрытие = доля ценников с читаемым штрихкодом (≈80–95% на качественных кадрах). |

### Логика выбора лучшего кадра в треке

`TrackManager._score_detection` считает линейную комбинацию:

```
score = 0.45 * sharpness_norm      ← Laplacian variance кропа, главное для OCR
      + 0.25 * area_norm           ← больше пикселей = больше деталей
      + 0.20 * centerness_norm     ← по центру = меньше остаточной дисторсии
      + 0.10 * detector_confidence
```

Перед скорингом — три hard-фильтра: минимальная длина трека (`min_track_length=2`, отсев одиночных false-positive), минимальная площадь bbox, и проверка на касание края кадра (обрезанные ценники в VLM не идут).

---

## Структура проекта

```
lenta_pricer/
├── README.md
├── requirements.txt
├── configs/
│   └── pipeline.yaml          # Все гиперпараметры в одном месте
├── src/
│   ├── config.py              # Pydantic-настройки, загрузка YAML
│   ├── schemas.py             # PricerRecord, VLMResponse, TrackCandidate
│   ├── utils.py               # Логирование, laplacian_variance, safe_crop
│   ├── undistort.py           # DistortionCorrector (по коэф. заказчика)
│   ├── frame_extractor.py     # Сэмплинг + blur-filter
│   ├── detector.py            # YOLOv8 + ByteTrack обёртка
│   ├── track_manager.py       # Best-frame-per-track
│   ├── qr_decoder.py          # pyzbar + cv2 + парсер payload Ленты
│   ├── vlm_extractor.py       # Qwen2-VL-2B 4-bit + strict-JSON промпт
│   ├── product_db.py          # Справочник товаров (in-memory lookup)
│   ├── postprocessor.py       # CSV writer + нормализация + DB-lookup
│   └── pipeline.py            # Главный оркестратор
├── app.py                     # Streamlit UI
├── main.py                    # CLI
├── data/                      # ← положить сюда db_hack.csv (cp1251)
├── models/                    # YOLO веса + HF-кэш
└── output/                    # Готовые CSV
```

---

## Формат выходного CSV (29 колонок)

Полное соответствие `sample.csv` от заказчика:

```
filename,product_name,price_default,price_card,price_discount,barcode,
discount_amount,id_sku,print_datetime,code,additional_info,color,
special_symbols,frame_timestamp,x_min,y_min,x_max,y_max,
qr_code_barcode,price1_qr,price2_qr,price3_qr,price4_qr,
wholesale_level_1_count,wholesale_level_1_price,
wholesale_level_2_count,wholesale_level_2_price,
action_price_qr,action_code_qr
```

Соглашение из ТЗ (реализовано в `PostProcessor`):
- поля, которых **на ценнике нет** → `нет`;
- поля, которые **есть, но не распознались** → `""` (пусто).

---

## Конфигурация

Все настройки — в [`configs/pipeline.yaml`](configs/pipeline.yaml). Без правки кода можно менять:
- целевой FPS, порог размытия, undistortion вкл/выкл;
- порог уверенности детектора, imgsz, тип трекера;
- веса метрики качества для выбора лучшего кадра;
- модель VLM (можно подменить на Qwen2-VL-7B при наличии 16+ ГБ VRAM);
- маркеры «нет» / «не распознано».

---

## Метрики и тестирование

```bash
# Сравнение с эталоном из CSV-разметки заказчика
python scripts/evaluate.py \
    --pred output/26_12-20_result.csv \
    --gt   data/26_12-20.csv
```

Целевая метрика по ТЗ — доля ценников с точностью ≥ 80%. Реализация скрипта эвалуации — задача отдельного PR.

---

## Известные ограничения

1. **Холодный старт ~30 с** — Qwen2-VL квантуется при первой загрузке. Streamlit кэширует пайплайн, так что только один раз.
2. **VLM-галлюцинации на размазанных кропах** — снижаются жёстким `temperature=0` и валидацией JSON, но не исключаются.
3. **Глубокая занятость по VRAM при batch>1** — на RTX 3060 пока работаем последовательно. При переходе на RTX 4090 / A100 можно включить честный batching VLM (см. `VLMExtractor.extract_batch`).

---

## Дальнейшее развитие
- Дообучение YOLO на размеченных ценниках (заказчик предоставил CSV с координатами — можно конвертировать в YOLO-формат через `scripts/csv_to_yolo.py`).
- Эвалуация через fuzzy-matching по `barcode + product_name`.
- Экспорт YOLO → ONNX → RKNN (int8) для edge-деплоя на роботе (бонус по ТЗ).
- Параллельная обработка нескольких видео через `multiprocessing`.

---
