"""
Детектор + трекер ценников.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from src.config import DetectorConfig
from src.frame_extractor import VideoFrame
from src.schemas import Detection
from src.utils import safe_crop


class PriceTagDetector:
    """YOLO с ByteTrack — выдаёт `Detection` с trackID."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        weights = self._resolve_weights()
        logger.info(f"Загрузка YOLO: {weights}")
        self.model = YOLO(str(weights))

        # Тёплый прогон под фиксированный imgsz, чтобы первый кадр на пользователе не висел 10 с из-за компиляции CUDA-ядер.
        self._warmup()

    # -------------------------------------------------------------------
    def _resolve_weights(self) -> Path:
        primary = Path(self.cfg.weights_path)
        if primary.exists():
            return primary
        logger.warning(
            f"Веса не найдены: {primary}. "
            f"Используем fallback: {self.cfg.fallback_weights}"
        )
        return Path(self.cfg.fallback_weights)

    # -------------------------------------------------------------------
    def _warmup(self) -> None:
        """Один прогон по чёрному кадру — компилируем CUDA-графы заранее."""
        if not torch.cuda.is_available():
            logger.warning("CUDA недоступна — детекция пойдёт на CPU (медленно).")
            return
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_threshold,
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )

    # -------------------------------------------------------------------
    def detect_and_track(self, frames: Iterator[VideoFrame]) -> Iterator[List[Detection]]:
        """Для каждого кадра yield списка детекций с уже назначенными track_id.
        """
        for vf in frames:
            results = self.model.track(
                vf.image,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                device=self.cfg.device,
                half=self.cfg.half,
                tracker=self.cfg.tracker_config,
                persist=True,           # сохраняем состояние трекера между кадрами
                verbose=False,
            )

            detections = self._parse_results(results[0], vf)
            yield detections

    # -------------------------------------------------------------------
    def _parse_results(self, result, vf: VideoFrame) -> List[Detection]:
        """Достаём боксы, ID, конфиденсы из ultralytics-результата."""
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # IDs могут отсутствовать в первых кадрах (ByteTrack пока не уверен)
        if result.boxes.id is None:
            return []
        ids = result.boxes.id.cpu().numpy().astype(int)

        out: List[Detection] = []
        for bbox, conf, tid in zip(boxes, confs, ids):
            crop = safe_crop(vf.image, bbox, pad=12)
            if crop.size == 0:
                continue
            out.append(
                Detection(
                    track_id=int(tid),
                    bbox=bbox.astype(np.float32),
                    confidence=float(conf),
                    frame_idx=vf.index,
                    frame_timestamp_ms=vf.timestamp_ms,
                    crop=crop,
                )
            )
        return out

    # -------------------------------------------------------------------
    def reset_tracker(self) -> None:
        """Сбросить состояние трекера (вызывать между разными видео!)."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = None
