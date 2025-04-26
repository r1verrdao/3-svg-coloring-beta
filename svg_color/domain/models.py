from dataclasses import dataclass
from typing import Optional

@dataclass
class ColorConfig:
    max_colors: int = 80
    color_threshold: float = 30


@dataclass
class ProcessingInput:
    album_id: str
    svg_path: str
    png_path: str
    config: ColorConfig

@dataclass
class ProcessingResult:
    success: bool
    output_svg_path: Optional[str] = None
    message: str = ""
    duration: Optional[float] = None