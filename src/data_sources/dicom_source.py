"""
Implementacja źródła danych z plików DICOM
"""
from typing import Optional, Dict, Any, List
import numpy as np
import pydicom
from pathlib import Path
from ..core.interfaces import IDataSource


class DicomSource(IDataSource):
    """Źródło danych z plików DICOM (single/multi-frame)"""
    
    def __init__(self, path: str):
        """
        Args:
            path: Ścieżka do pliku DICOM lub folderu z plikami DICOM
        """
        self.path = Path(path)
        self.dicom_files: List[Path] = []
        self.datasets: List[pydicom.Dataset] = []
        self.frames: List[np.ndarray] = []
        self.current_position = 0
        self._is_opened = False
    
    def open(self) -> bool:
        """Wczytuje pliki DICOM"""
        try:
            if self.path.is_file():
                # Pojedynczy plik DICOM
                self.dicom_files = [self.path]
            elif self.path.is_dir():
                # Folder z plikami DICOM
                self.dicom_files = sorted(self.path.glob("*.dcm"))
                if len(self.dicom_files) == 0:
                    # Spróbuj bez rozszerzenia
                    self.dicom_files = sorted([f for f in self.path.iterdir() 
                                               if f.is_file() and not f.name.startswith('.')])
            
            if len(self.dicom_files) == 0:
                return False
            
            # Wczytaj wszystkie pliki DICOM
            for dcm_file in self.dicom_files:
                try:
                    ds = pydicom.dcmread(str(dcm_file))
                    self.datasets.append(ds)
                    
                    # Sprawdź czy to multi-frame
                    if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                        # Multi-frame DICOM
                        pixel_array = ds.pixel_array
                        for i in range(ds.NumberOfFrames):
                            frame = pixel_array[i]
                            self.frames.append(self._process_dicom_frame(frame, ds))
                    else:
                        # Single-frame DICOM
                        frame = ds.pixel_array
                        self.frames.append(self._process_dicom_frame(frame, ds))
                        
                except Exception as e:
                    print(f"Error loading DICOM file {dcm_file}: {e}")
                    continue
            
            self._is_opened = len(self.frames) > 0
            return self._is_opened
            
        except Exception as e:
            print(f"Error opening DICOM source: {e}")
            return False
    
    def _process_dicom_frame(self, pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
        """
        Przetwarza klatkę DICOM do formatu RGB/Grayscale
        
        Args:
            pixel_array: Surowa tablica pikseli z DICOM
            ds: Dataset DICOM z metadanymi
            
        Returns:
            Przetworzona klatka
        """
        # Normalizuj do 0-255
        frame = pixel_array.astype(np.float32)
        
        # Zastosuj window/level jeśli dostępne
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
            width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
            
            min_val = center - width / 2
            max_val = center + width / 2
            
            frame = np.clip(frame, min_val, max_val)
            frame = ((frame - min_val) / (max_val - min_val) * 255.0)
        else:
            # Normalizacja bez window/level
            min_val = frame.min()
            max_val = frame.max()
            if max_val > min_val:
                frame = ((frame - min_val) / (max_val - min_val) * 255.0)
        
        frame = frame.astype(np.uint8)
        
        # Konwertuj do RGB jeśli grayscale
        if len(frame.shape) == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        
        return frame
    
    def start(self) -> bool:
        """Rozpoczyna akwizycję"""
        return self._is_opened
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Reads current frame at current_position"""
        if not self._is_opened or self.current_position >= len(self.frames):
            return None
        
        frame = self.frames[self.current_position]
        return frame
    
    def seek(self, position: int) -> bool:
        """Przewija do określonej klatki"""
        if not self._is_opened:
            return False
        
        if 0 <= position < len(self.frames):
            self.current_position = position
            return True
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Zwraca informacje o źródle DICOM"""
        if not self._is_opened or len(self.frames) == 0:
            return {
                "name": self.path.name,
                "source_type": "dicom",
                "width": 0,
                "height": 0
            }
        
        first_frame = self.frames[0]
        height, width = first_frame.shape[:2]
        
        return {
            "name": self.path.name,
            "source_type": "dicom",
            "width": width,
            "height": height,
            "fps": 30.0,  # Domyślne FPS
            "supports_seek": True,
            "total_frames": len(self.frames),
            "color_mode": "RGB"
        }
    
    def close(self):
        """Zamyka źródło danych"""
        self._is_opened = False
        self.current_position = 0
        self.frames.clear()
        self.datasets.clear()
    
    def supports_seek(self) -> bool:
        """DICOM wspiera przewijanie"""
        return True
    
    def get_total_frames(self) -> Optional[int]:
        """Zwraca liczbę klatek"""
        return len(self.frames) if self._is_opened else None
    
    def get_current_position(self) -> int:
        """Zwraca aktualną pozycję"""
        return self.current_position

