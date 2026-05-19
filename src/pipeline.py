
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from src.config import PipelineConfig
from src.detector import PriceTagDetector
from src.frame_extractor import FrameExtractor
from src.postprocessor import PostProcessor
from src.product_db import ProductDatabase
from src.qr_decoder import QRDecoder
from src.schemas import PricerRecord, TrackCandidate
from src.track_manager import TrackManager
from src.undistort import DistortionCorrector
from src.vlm_extractor import VLMExtractor


# Тип колбэка прогресса для UI (Streamlit прокидывает свой)
ProgressCallback = Callable[[float, str], None]


class PricerPipeline:
    """Контейнер всех компонентов + метод `run` на одно видео."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._build_components()

    # -------------------------------------------------------------------
    def _build_components(self) -> None:
        """Инициализация всех компонентов."""
        logger.info("Инициализация пайплайна...")
        t0 = time.perf_counter()

        undistorter = DistortionCorrector(self.cfg.camera)
        self.frame_extractor = FrameExtractor(self.cfg.video, undistorter)
        self.detector = PriceTagDetector(self.cfg.detector)
        self.qr_decoder = QRDecoder(self.cfg.qr)
        self.vlm = VLMExtractor(self.cfg.vlm)

        # Справочник товаров — опциональный, не критичный
        product_db: Optional[ProductDatabase] = None
        if self.cfg.product_db.enabled:
            try:
                product_db = ProductDatabase(
                    csv_path=self.cfg.product_db.csv_path,
                    encoding=self.cfg.product_db.encoding,
                )
            except Exception as e:
                logger.warning(f"Не удалось загрузить справочник товаров: {e}")
                product_db = None

        self.postprocessor = PostProcessor(self.cfg.output, product_db=product_db)

        logger.info(f"Пайплайн готов за {time.perf_counter() - t0:.1f} с")

    # -------------------------------------------------------------------
    def run(
        self,
        video_path: str | Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        """Обработать одно видео целиком. Вернуть путь к выходному CSV."""
        video_path = Path(video_path)
        logger.info(f"▶ START: {video_path.name}")
        t_start = time.perf_counter()

        # Между видео сбрасываем состояние трекера, чтобы не текли ID
        self.detector.reset_tracker()
        track_mgr = TrackManager(self.cfg.track_selector)

        # ---- Этап 1: декодинг + детекция + трекинг -------------------
        self._notify(progress_cb, 0.05, "Сэмплирую видео и детектирую ценники...")
        frame_count = 0
        for detections, frame_shape in self._detect_stream(video_path):
            track_mgr.ingest(detections, frame_shape)
            frame_count += 1
            if frame_count % 10 == 0:
                self._notify(
                    progress_cb,
                    min(0.05 + 0.4 * (frame_count / 200), 0.45),
                    f"Обработано {frame_count} кадров...",
                )

        candidates: List[TrackCandidate] = track_mgr.finalize()
        logger.info(f"Уникальных ценников найдено: {len(candidates)}")

        # ---- Этап 2: VLM + QR на лучших кадрах -----------------------
        records: List[PricerRecord] = []
        for i, cand in enumerate(candidates):
            self._notify(
                progress_cb,
                0.5 + 0.45 * (i / max(len(candidates), 1)),
                f"Распознаю ценник {i + 1}/{len(candidates)}...",
            )
            record = self._process_candidate(video_path.name, cand)
            records.append(record)

        # ---- Этап 3: запись CSV --------------------------------------
        self._notify(progress_cb, 0.97, "Сохраняю результат...")
        out_csv = self.postprocessor.write_csv(records, video_path.name)

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"✓ DONE: {video_path.name} | "
            f"{len(records)} ценников | {elapsed:.1f} с"
        )
        self._notify(progress_cb, 1.0, f"Готово ({elapsed:.0f} с)")
        return out_csv

    # ===================================================================
    #  Вспомогательные методы
    # ===================================================================
    def _detect_stream(self, video_path: Path):
        """Однопроходный генератор: для каждого отобранного кадра
        отдаём пару (детекции, форма кадра).

        Логически — `FrameExtractor` → `Detector.track()`. Реализуем
        в одном цикле, потому что трекер `persist=True` должен видеть
        кадры последовательно и stateful.
        """
        for vf in self.frame_extractor.iter_frames(video_path):
            results = self.detector.model.track(
                vf.image,
                imgsz=self.cfg.detector.imgsz,
                conf=self.cfg.detector.conf_threshold,
                iou=self.cfg.detector.iou_threshold,
                device=self.cfg.detector.device,
                half=self.cfg.detector.half,
                tracker=self.cfg.detector.tracker_config,
                persist=True,
                verbose=False,
            )
            detections = self.detector._parse_results(results[0], vf)
            yield detections, vf.image.shape

    # -------------------------------------------------------------------
    def _process_candidate(
        self,
        filename: str,
        cand: TrackCandidate,
    ) -> PricerRecord:
        """Запустить VLM и QR на одном кандидате, собрать запись."""
        crop = cand.best_detection.crop

        # 1) QR-декодер: CPU, дёшево
        qr_result = self.qr_decoder.decode(crop)
        if qr_result.raw_payload:
            logger.debug(
                f"track {cand.track_id}: QR ок ({len(qr_result.fields)} полей)"
            )

        # 2) VLM: GPU, дорого, но только один раз на трек
        vlm_result = self.vlm.extract(crop)

        # 3) Сборка финальной записи
        return self.postprocessor.build_record(
            filename=filename,
            candidate=cand,
            vlm=vlm_result,
            qr=qr_result,
        )

    # -------------------------------------------------------------------
    @staticmethod
    def _notify(cb: Optional[ProgressCallback], pct: float, msg: str) -> None:
        if cb is not None:
            try:
                cb(pct, msg)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
