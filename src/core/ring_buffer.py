"""
Ring buffer do przechowywania klatek
"""
from typing import Optional, List
from threading import Lock
from collections import deque
from .models import FrameData


class FrameRingBuffer:
    """Ring buffer z nadpisywaniem najstarszych klatek"""
    
    def __init__(self, capacity: int = 100):
        """
        Args:
            capacity: Maksymalna liczba klatek w buforze
        """
        self.capacity = capacity
        self._buffer: deque[FrameData] = deque(maxlen=capacity)
        self._lock = Lock()
    
    def add(self, frame_data: FrameData):
        """Dodaje klatkę do bufora"""
        with self._lock:
            self._buffer.append(frame_data)
    
    def get(self, index: int) -> Optional[FrameData]:
        """
        Pobiera klatkę z bufora po indeksie
        
        Args:
            index: Indeks względem najstarszej klatki (0 = najstarsza)
        """
        with self._lock:
            if 0 <= index < len(self._buffer):
                return self._buffer[index]
            return None
    
    def get_latest(self) -> Optional[FrameData]:
        """Pobiera najnowszą klatkę"""
        with self._lock:
            if len(self._buffer) > 0:
                return self._buffer[-1]
            return None
    
    def get_by_frame_number(self, frame_number: int) -> Optional[FrameData]:
        """Pobiera klatkę po numerze klatki"""
        with self._lock:
            for frame_data in reversed(self._buffer):
                if frame_data.frame_number == frame_number:
                    return frame_data
            return None
    
    def get_size(self) -> int:
        """Zwraca aktualny rozmiar bufora"""
        with self._lock:
            return len(self._buffer)
    
    def get_capacity(self) -> int:
        """Zwraca pojemność bufora"""
        return self.capacity
    
    def get_fill_percentage(self) -> float:
        """Zwraca procent zapełnienia bufora"""
        return (self.get_size() / self.capacity) * 100.0
    
    def clear(self):
        """Czyści bufor"""
        with self._lock:
            self._buffer.clear()
    
    def get_all_frame_numbers(self) -> List[int]:
        """Zwraca listę wszystkich numerów klatek w buforze"""
        with self._lock:
            return [fd.frame_number for fd in self._buffer]
    
    def get_frame_range(self) -> tuple[Optional[int], Optional[int]]:
        """Zwraca zakres numerów klatek (min, max)"""
        with self._lock:
            if len(self._buffer) == 0:
                return None, None
            return self._buffer[0].frame_number, self._buffer[-1].frame_number

