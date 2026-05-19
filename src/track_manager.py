"""
Менеджер треков: выбор ЛУЧШЕГО кадра ценника для подачи в тяжёлую VLM.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

import numpy as np
from loguru import logger

from src.config import TrackSelectorConfig
from src.schemas import Detection, TrackCandidate
from src.utils import is_near_edge, laplacian_variance


class TrackManager:
    """Накапливает все детекции по track_id и в конце выдаёт лучших кандидатов."""

    def __init__(self, cfg: TrackSelectorConfig) -> None:
        self.cfg = cfg
        # track_id -> [Detection, Detection, ...]
        self._tracks: Dict[int, List[Detection]] = defaultdict(list)

    # -------------------------------------------------------------------
    def ingest(self, detections: List[Detection], frame_shape: tuple) -> None:
        """Добавить детекции из одного кадра в накопитель.
        Параметр `frame_shape` нужен, чтобы понимать «у края или нет».
        """
        for det in detections:
            if not self._passes_prefilter(det, frame_shape):
                continue
            det.quality_score = self._score_detection(det, frame_shape)
            self._tracks[det.track_id].append(det)

    # -------------------------------------------------------------------
    def finalize(self) -> List[TrackCandidate]:
        """После прохода всех кадров — выдать лучших кандидатов на трек."""
        candidates: List[TrackCandidate] = []
        skipped_short = 0

        for tid, dets in self._tracks.items():
            if len(dets) < self.cfg.min_track_length:
                skipped_short += 1
                continue

            # Сортируем по убыванию качества, берём top_k (обычно 1)
            dets_sorted = sorted(dets, key=lambda d: d.quality_score, reverse=True)
            best = dets_sorted[0]

            candidates.append(
                TrackCandidate(
                    track_id=tid,
                    best_detection=best,
                    all_detections_count=len(dets),
                )
            )

        logger.info(
            f"TrackManager: всего треков {len(self._tracks)}, "
            f"в работу {len(candidates)}, отсеяно коротких {skipped_short}"
        )
        return candidates

    # -------------------------------------------------------------------
    def _passes_prefilter(self, det: Detection, frame_shape: tuple) -> bool:
        """Жёсткие правила: грубые фильтры до дорогого скоринга."""
        if det.area < self.cfg.min_bbox_area:
            return False
        ar = det.aspect_ratio
        if ar < self.cfg.min_aspect_ratio or ar > self.cfg.max_aspect_ratio:
            return False
        if is_near_edge(det.bbox, frame_shape, self.cfg.edge_margin):
            return False
        return True

    # -------------------------------------------------------------------
    def _score_detection(self, det: Detection, frame_shape: tuple) -> float:
        """Линейная комбинация четырёх нормализованных метрик качества."""
        # 1) Sharpness — основная метрика. Считаем по кропу, а не по кадру.
        sharp = laplacian_variance(det.crop)
        # Нормализация: 200+ — отлично, 60 — пограничное (пройдёт фильтр extractor'а)
        sharp_norm = min(sharp / 200.0, 1.0)

        # 2) Площадь. Чем больше пикселей — тем больше деталей для OCR.
        # Нормируем по площади ВСЕГО кадра.
        frame_area = frame_shape[0] * frame_shape[1]
        area_norm = min(det.area / (frame_area * 0.05), 1.0)  # 5% площади = 1.0

        # 3) Центрированность: расстояние от центра bbox до центра кадра.
        h, w = frame_shape[:2]
        cx_frame, cy_frame = w / 2, h / 2
        x1, y1, x2, y2 = det.bbox
        cx_box, cy_box = (x1 + x2) / 2, (y1 + y2) / 2
        max_dist = np.hypot(cx_frame, cy_frame)
        dist = np.hypot(cx_box - cx_frame, cy_box - cy_frame)
        center_norm = 1.0 - (dist / max_dist)

        # 4) Confidence — линейно, уже в [0,1].
        conf = det.confidence

        score = (
            0.45 * sharp_norm
            + 0.25 * area_norm
            + 0.20 * center_norm
            + 0.10 * conf
        )
        return float(score)
