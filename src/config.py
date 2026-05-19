"""
Конфигурация пайплайна с валидацией через Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator


# =============================================================================
#  Под-конфиги (модели валидации для каждой секции YAML)
# =============================================================================
class VideoConfig(BaseModel):
    target_fps: float = Field(gt=0, le=30)
    blur_threshold: float = Field(ge=0)
    resize_width: Optional[int] = Field(default=None, ge=320)
    apply_undistort: bool = True
    max_frames: Optional[int] = Field(default=None, ge=1)


class CameraConfig(BaseModel):
    image_size: Tuple[int, int]
    diagonal_mm: float
    focal_length_mm: float
    distortion_coeffs: List[float]

    @field_validator("distortion_coeffs")
    @classmethod
    def _five_coeffs(cls, v: List[float]) -> List[float]:
        if len(v) != 5:
            raise ValueError("distortion_coeffs должен содержать ровно 5 значений (k1,k2,p1,p2,k3)")
        return v


class DetectorConfig(BaseModel):
    weights_path: str
    fallback_weights: str
    imgsz: int = Field(ge=320, le=2048)
    conf_threshold: float = Field(ge=0, le=1)
    iou_threshold: float = Field(ge=0, le=1)
    device: str = "cuda:0"
    half: bool = True
    tracker_config: str = "bytetrack.yaml"


class TrackSelectorConfig(BaseModel):
    top_k: int = Field(ge=1)
    min_track_length: int = Field(ge=1)
    min_bbox_area: int = Field(ge=0)
    min_aspect_ratio: float = Field(gt=0)
    max_aspect_ratio: float = Field(gt=0)
    edge_margin: int = Field(ge=0)


class VLMConfig(BaseModel):
    model_id: str
    cache_dir: str
    load_in_4bit: bool
    max_new_tokens: int = Field(ge=64, le=4096)
    temperature: float = Field(ge=0, le=2)
    min_pixels: int
    max_pixels: int

    @field_validator("min_pixels", "max_pixels", mode="before")
    @classmethod
    def _eval_mul(cls, v):
        """YAML может содержать '256 * 28 * 28' — выполняем безопасный eval."""
        if isinstance(v, str):
            parts = [int(p.strip()) for p in v.split("*")]
            out = 1
            for p in parts:
                out *= p
            return out
        return v


class QRConfig(BaseModel):
    enable_upscale_fallback: bool
    upscale_factor: float = Field(gt=1)
    enable_cv2_fallback: bool


class OutputConfig(BaseModel):
    csv_dir: str
    save_crops: bool
    save_debug_frames: bool
    not_present_marker: str
    not_recognized_marker: str


class ProductDBConfig(BaseModel):
    """Опциональный справочник товаров (db_hack.csv от Lenta)."""

    enabled: bool = True
    csv_path: str
    encoding: str = "cp1251"


# =============================================================================
#  Корневой конфиг
# =============================================================================
class PipelineConfig(BaseModel):
    """Единый объект конфигурации — передаётся через DI во все компоненты."""

    video: VideoConfig
    camera: CameraConfig
    detector: DetectorConfig
    track_selector: TrackSelectorConfig
    vlm: VLMConfig
    qr: QRConfig
    product_db: ProductDBConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Загрузить и провалидировать конфиг из YAML-файла."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
