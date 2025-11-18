"""
Data models used in the system
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from datetime import datetime


@dataclass
class FrameData:
    """Represents a single frame with metadata"""
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    source_info: Optional[str] = None
    
    def copy(self):
        """Creates a copy of FrameData"""
        return FrameData(
            frame=self.frame.copy() if self.frame is not None else None,
            timestamp=self.timestamp,
            frame_number=self.frame_number,
            source_info=self.source_info
        )


@dataclass
class DataSourceInfo:
    """Information about the data source"""
    name: str
    source_type: str  # "camera", "image_sequence", "dicom"
    width: int
    height: int
    total_frames: Optional[int] = None
    fps: float = 30.0
    supports_seek: bool = False
    color_mode: str = "RGB"  # "RGB", "GRAY"
    
    def get_resolution_str(self) -> str:
        """Returns resolution as string"""
        return f"{self.width}x{self.height}"
