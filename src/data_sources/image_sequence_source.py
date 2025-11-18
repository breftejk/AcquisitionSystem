"""
Implementacja źródła danych z sekwencji obrazów PNG
"""
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from pathlib import Path
from ..core.interfaces import IDataSource


class ImageSequenceSource(IDataSource):
    """Źródło danych z sekwencji obrazów PNG"""
    
    def __init__(self, folder_path: str):
        """
        Args:
            folder_path: Ścieżka do folderu z obrazami
        """
        self.folder_path = Path(folder_path)
        self.image_files: List[Path] = []
        self.current_position = 0
        self._is_opened = False
    
    def open(self) -> bool:
        """Wczytuje listę plików PNG z folderu"""
        if not self.folder_path.exists() or not self.folder_path.is_dir():
            return False
        
        # Wczytaj wszystkie pliki PNG, posortowane alfabetycznie
        self.image_files = sorted(self.folder_path.glob("*.png"))
        
        if len(self.image_files) == 0:
            # Jeśli nie ma PNG, spróbuj JPG
            self.image_files = sorted(self.folder_path.glob("*.jpg"))
            self.image_files.extend(sorted(self.folder_path.glob("*.jpeg")))
        
        self._is_opened = len(self.image_files) > 0
        return self._is_opened
    
    def start(self) -> bool:
        """Rozpoczyna akwizycję"""
        return self._is_opened
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Reads current image at current_position"""
        if not self._is_opened or self.current_position >= len(self.image_files):
            return None
        
        image_path = self.image_files[self.current_position]
        frame = cv2.imread(str(image_path))
        
        if frame is not None:
            # OpenCV returns BGR, convert to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return None
    
    def seek(self, position: int) -> bool:
        """Przewija do określonego obrazu"""
        if not self._is_opened:
            return False
        
        if 0 <= position < len(self.image_files):
            self.current_position = position
            return True
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Zwraca informacje o sekwencji obrazów"""
        if not self._is_opened or len(self.image_files) == 0:
            return {
                "name": self.folder_path.name,
                "source_type": "image_sequence",
                "width": 0,
                "height": 0
            }
        
        # Odczytaj pierwszy obraz aby poznać wymiary
        first_image = cv2.imread(str(self.image_files[0]))
        height, width = first_image.shape[:2] if first_image is not None else (0, 0)
        
        return {
            "name": self.folder_path.name,
            "source_type": "image_sequence",
            "width": width,
            "height": height,
            "fps": 30.0,  # Domyślne FPS
            "supports_seek": True,
            "total_frames": len(self.image_files),
            "color_mode": "RGB"
        }
    
    def close(self):
        """Zamyka źródło danych"""
        self._is_opened = False
        self.current_position = 0
    
    def supports_seek(self) -> bool:
        """Sekwencje obrazów wspierają przewijanie"""
        return True
    
    def get_total_frames(self) -> Optional[int]:
        """Zwraca liczbę obrazów w sekwencji"""
        return len(self.image_files) if self._is_opened else None
    
    def get_current_position(self) -> int:
        """Zwraca aktualną pozycję"""
        return self.current_position

