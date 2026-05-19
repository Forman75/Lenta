"""
Streamlit Web UI для Lenta Pricer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import PipelineConfig
from src.pipeline import PricerPipeline
from src.utils import setup_logging


# =============================================================================
#  Конфигурация страницы
# =============================================================================
st.set_page_config(
    page_title="Lenta Pricer — распознавание ценников",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
#  Кэшированная инициализация пайплайна (один раз на сессию сервера)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_pipeline(config_path: str) -> PricerPipeline:
    """Грузим модели один раз и держим в памяти."""
    setup_logging()
    cfg = PipelineConfig.from_yaml(config_path)
    return PricerPipeline(cfg)


# =============================================================================
#  Сайдбар: настройки
# =============================================================================
with st.sidebar:
    st.title("⚙️ Настройки")
    config_path = st.text_input(
        "Путь к pipeline.yaml",
        value="configs/pipeline.yaml",
        help="Все гиперпараметры в одном YAML-файле",
    )

    st.markdown("---")
    st.markdown("### О пайплайне")
    st.markdown(
        """
        - **Сэмплер**: 2.5 FPS + Laplacian blur filter
        - **Детектор**: YOLOv8 + ByteTrack
        - **VLM**: Qwen2-VL-2B (4-bit, ~2 ГБ VRAM)
        - **QR**: pyzbar + cv2 fallback
        - **Камера**: undistort по коэф. заказчика
        """
    )

    st.markdown("---")
    st.caption("Lenta Tech Life Hack · v1.0.0")


# =============================================================================
#  Главная панель
# =============================================================================
st.title("🏷️ Lenta Pricer")
st.markdown("##### Локальное распознавание ценников из видеопотока робота")

# --- Шаг 1. Загрузка пайплайна ----------------------------------------------
with st.spinner("Загружаю модели в VRAM... (первый запуск 30–60 с)"):
    try:
        pipeline = load_pipeline(config_path)
        st.success("✓ Пайплайн готов")
    except Exception as e:
        st.error(f"Не удалось загрузить пайплайн: {e}")
        st.stop()

# --- Шаг 2. Загрузка видео ---------------------------------------------------
uploaded = st.file_uploader(
    "Загрузите видео со стеллажа (.mp4, .avi, .mov)",
    type=["mp4", "avi", "mov", "mkv"],
    help="Видео обрабатывается локально, никуда не отправляется.",
)

if uploaded is None:
    st.info("👆 Загрузите видеофайл, чтобы начать.")
    st.stop()

# --- Шаг 3. Превью видео + кнопка запуска ------------------------------------
col_video, col_meta = st.columns([2, 1])
with col_video:
    st.video(uploaded)
with col_meta:
    st.metric("Размер файла", f"{uploaded.size / (1024 * 1024):.1f} МБ")
    st.metric("Имя файла", uploaded.name)
    run_button = st.button("▶ Запустить распознавание", type="primary", use_container_width=True)

if not run_button:
    st.stop()

# --- Шаг 4. Сохраняем видео во временный файл (cv2 требует path) -------------
with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=Path(uploaded.name).suffix,
) as tmp:
    tmp.write(uploaded.getvalue())
    tmp_path = Path(tmp.name)

# --- Шаг 5. Запуск пайплайна с прогресс-баром --------------------------------
progress_bar = st.progress(0.0)
status_box = st.empty()


def progress_callback(pct: float, msg: str) -> None:
    """Streamlit обновляет UI только из главного потока — мы и есть в нём."""
    progress_bar.progress(min(max(pct, 0.0), 1.0))
    status_box.info(msg)


try:
    csv_path = pipeline.run(tmp_path, progress_cb=progress_callback)
except Exception as e:
    st.error(f"Ошибка во время обработки: {e}")
    st.exception(e)
    st.stop()
finally:
    tmp_path.unlink(missing_ok=True)

# --- Шаг 6. Результаты -------------------------------------------------------
st.success(f"✓ Готово! Найдено ценников: см. ниже.")

df = pd.read_csv(csv_path, encoding="utf-8")

tab_table, tab_summary = st.tabs(["📋 Таблица", "📊 Сводка"])

with tab_table:
    st.dataframe(df, use_container_width=True, height=400)
    st.download_button(
        "⬇️ Скачать CSV",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

with tab_summary:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего ценников", len(df))
    col2.metric(
        "Распознано product_name",
        int(df["product_name"].astype(str).str.len().gt(0).sum()),
    )
    col3.metric(
        "Прочитано QR-кодов",
        int(df["qr_code_barcode"].astype(str).str.match(r"^\d{8,14}$").sum()),
    )
    # DB-lookup hit-rate как процент совпадений по штрихкоду
    n_with_bc = int(df["barcode"].astype(str).str.len().gt(2).sum())
    n_named = int(df["product_name"].astype(str).str.len().gt(5).sum())
    col4.metric(
        "С эталонным названием",
        f"{n_named}/{n_with_bc}" if n_with_bc else "0/0",
        help="Сколько ценников получили product_name из справочника db_hack.csv",
    )

    if "color" in df.columns:
        st.markdown("**Распределение по цвету ценника:**")
        st.bar_chart(df["color"].value_counts())
