"""
Сэмплирование видеопотока и отсев размытых кадров.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from loguru import logger

from src.config import VideoConfig
from src.undistort import DistortionCorrector
from src.utils import laplacian_variance


@dataclass
class VideoFrame:
    """Один отобранный кадр + метаданные."""
    index: int                 # порядковый номер ОТОБРАННОГО кадра
    timestamp_ms: int          # позиция в видео, мс
    image: np.ndarray          # BGR-кадр (уже undistorted, если включено)
    sharpness: float           # Laplacian variance


class FrameExtractor:
    """Итерируется по видео, отдаёт только полезные кадры."""

    def __init__(
        self,
        cfg: VideoConfig,
        undistorter: Optional[DistortionCorrector] = None,
    ) -> None:
        self.cfg = cfg
        self.undistorter = undistorter if cfg.apply_undistort else None

    # -------------------------------------------------------------------
    def iter_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        """Главный генератор: yield по одному пригодному VideoFrame.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        try:
            yield from self._iterate(cap)
        finally:
            cap.release()

    # -------------------------------------------------------------------
    def _iterate(self, cap: cv2.VideoCapture) -> Iterator[VideoFrame]:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, int(round(src_fps / self.cfg.target_fps)))

        logger.info(
            f"Видео: {total} кадров @ {src_fps:.1f} FPS → "
            f"шаг {stride} (целевой {self.cfg.target_fps} FPS)"
        )

        sampled = 0
        kept = 0
        rejected_blur = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_idx % stride != 0:
                continue

            sampled += 1

            # --- препроцессинг -------------------------------------------------
            frame = self._preprocess(frame)
            sharp = laplacian_variance(frame)

            if sharp < self.cfg.blur_threshold:
                rejected_blur += 1
                continue

            kept += 1
            yield VideoFrame(
                index=kept - 1,
                timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                image=frame,
                sharpness=sharp,
            )

            if self.cfg.max_frames and kept >= self.cfg.max_frames:
                logger.info(f"Достигнут лимит max_frames={self.cfg.max_frames}")
                break

        logger.info(
            f"Сэмплинг завершён: отобрано {kept} из {sampled} "
            f"(отбраковано размытых: {rejected_blur})"
        )

    # -------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Undistort → опциональный resize. Порядок важен: сначала коррекция."""
        if self.undistorter is not None:
            frame = self.undistorter.undistort(frame)
        if self.cfg.resize_width and frame.shape[1] > self.cfg.resize_width:
            ratio = self.cfg.resize_width / frame.shape[1]
            new_size = (self.cfg.resize_width, int(frame.shape[0] * ratio))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return frame
