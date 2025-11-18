"""
Algorytm bez przetwarzania (pass-through)
"""
from typing import Dict, Any
import numpy as np
from ..core.interfaces import IDetectionAlgorithm


class NoProcessingAlgorithm(IDetectionAlgorithm):
    """Algorytm który nie przetwarza obrazu (zwraca oryginał)"""
    
    def __init__(self):
        self.name = "No Processing"
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Brak konfiguracji"""
        return True
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Zwraca kopię oryginalnej klatki"""
        return frame.copy()
    
    def get_name(self) -> str:
        """Zwraca nazwę algorytmu"""
        return self.name
    
    def get_parameters(self) -> Dict[str, Any]:
        """Brak parametrów"""
        return {}

