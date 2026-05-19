"""
Коррекция радиальной/тангенциальной дисторсии.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from src.config import CameraConfig


class DistortionCorrector:
    """Кэширует undistort-карты — пересчёт делается один раз на сессию.
    Для разрешения, отличного от штатного (3840×2160) — например, если
    видео пересжато в 1080p, — карты пересчитываются под текущий кадр
    при первом вызове.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.dist = np.array(cfg.distortion_coeffs, dtype=np.float32)
        self._maps_cache: dict[Tuple[int, int], tuple] = {}

    # -------------------------------------------------------------------
    def _camera_matrix(self, width: int, height: int) -> np.ndarray:
        """Восстанавливаем K из физических параметров матрицы и фокуса.
        """
        aspect = width / height
        height_mm = self.cfg.diagonal_mm / math.sqrt(aspect ** 2 + 1)
        width_mm = aspect * height_mm
        fx = (self.cfg.focal_length_mm * width) / width_mm
        fy = (self.cfg.focal_length_mm * height) / height_mm
        return np.array(
            [[fx, 0, width / 2],
             [0, fy, height / 2],
             [0, 0, 1]],
            dtype=np.float32,
        )

    # -------------------------------------------------------------------
    def _get_maps(self, width: int, height: int):
        """Ленивый расчёт карт remap для заданного разрешения (с кешем)."""
        key = (width, height)
        if key in self._maps_cache:
            return self._maps_cache[key]

        K = self._camera_matrix(width, height)
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, self.dist, (width, height), 0, (width, height)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K, self.dist, None, new_K, (width, height), cv2.CV_32FC1
        )
        self._maps_cache[key] = (map1, map2, roi)
        return map1, map2, roi

    # -------------------------------------------------------------------
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Скорректировать дисторсию в одном кадре. Возвращает обрезанный по ROI кадр."""
        h, w = frame.shape[:2]
        map1, map2, roi = self._get_maps(w, h)
        out = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            out = out[y:y + rh, x:x + rw]
        return out
