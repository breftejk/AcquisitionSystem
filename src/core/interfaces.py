"""
Base interfaces for data sources and processing algorithms
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np


class IDataSource(ABC):
    """Generic interface for data source"""

    @abstractmethod
    def open(self) -> bool:
        """Opens the data source"""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """Starts acquisition"""
        pass
    
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a single frame"""
        pass
    
    @abstractmethod
    def seek(self, position: int) -> bool:
        """Seeks to specified position (if supported)"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns information about the data source"""
        pass
    
    @abstractmethod
    def close(self):
        """Closes the data source"""
        pass
    
    @abstractmethod
    def supports_seek(self) -> bool:
        """Whether the source supports seeking"""
        pass
    
    @abstractmethod
    def get_total_frames(self) -> Optional[int]:
        """Returns the total number of frames (if known)"""
        pass
    
    @abstractmethod
    def get_current_position(self) -> int:
        """Returns the current position"""
        pass


class IDetectionAlgorithm(ABC):
    """Generic interface for detection/processing algorithm"""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configures the algorithm"""
        pass
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Processes the frame"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Returns the algorithm name"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Returns algorithm parameters for UI display"""
        pass
