"""
Camera data source implementation (OpenCV)
"""
from typing import Optional, Dict, Any
import numpy as np
import cv2
from ..core.interfaces import IDataSource


class CameraSource(IDataSource):
    """Camera data source"""

    def __init__(self, camera_id: int = 0):
        """
        Args:
            camera_id: Camera ID (0 = default camera)
        """
        self.camera_id = camera_id
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self._is_opened = False
        self._width = 0
        self._height = 0
        self._fps = 30.0

    def open(self) -> bool:
        """Opens camera connection"""
        try:
            # Use default backend
            self.capture = cv2.VideoCapture(self.camera_id)

            if not self.capture.isOpened():
                print(f"Failed to open camera {self.camera_id}")
                return False

            # Configure camera for low latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Get camera properties
            self._width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = self.capture.get(cv2.CAP_PROP_FPS)
            if self._fps <= 0 or self._fps > 120:
                self._fps = 30.0

            self._is_opened = True
            print(f"Camera {self.camera_id} opened: {self._width}x{self._height} @ {self._fps}fps")
            return True

        except Exception as e:
            print(f"Error opening camera {self.camera_id}: {e}")
            self._is_opened = False
            if self.capture is not None:
                try:
                    self.capture.release()
                except:
                    pass
            return False

    def start(self) -> bool:
        """Starts acquisition"""
        return self._is_opened
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Reads frame from camera"""
        # Standard OpenCV pattern: while cap.isOpened() -> cap.read()
        if not self._is_opened or self.capture is None:
            return None
        
        if not self.capture.isOpened():
            return None

        try:
            success, img = self.capture.read()
            if success and img is not None and img.size > 0:
                self.frame_count += 1
                # OpenCV returns BGR, convert to RGB
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            if self.frame_count == 0:
                print(f"Error reading first frame from camera {self.camera_id}: {e}")
            return None

    def seek(self, position: int) -> bool:
        """Cameras don't support seeking"""
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Returns camera information"""
        return {
            "name": f"Camera {self.camera_id}",
            "source_type": "camera",
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "supports_seek": False,
            "total_frames": None,
            "color_mode": "RGB"
        }

    def close(self):
        """Closes camera connection"""
        if self.capture is not None:
            try:
                self.capture.release()
            except:
                pass
            self._is_opened = False
    
    def supports_seek(self) -> bool:
        """Cameras don't support seeking"""
        return False
    
    def get_total_frames(self) -> Optional[int]:
        """Cameras don't have a fixed number of frames"""
        return None
    
    def get_current_position(self) -> int:
        """Returns number of frames read"""
        return self.frame_count
    
    @staticmethod
    def list_available_cameras(max_cameras: int = 2) -> list[int]:
        """
        Finds available cameras

        Args:
            max_cameras: Maximum number of cameras to check (default: 2)

        Returns:
            List of available camera IDs
        """
        import os

        # Suppress OpenCV errors during detection
        old_log_level = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'

        available = []

        try:
            for i in range(max_cameras):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to read to verify camera works
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                    cap.release()
                else:
                    cap.release()
                    # Stop after first failure
                    break
        finally:
            # Restore log level
            if old_log_level:
                os.environ['OPENCV_LOG_LEVEL'] = old_log_level
            else:
                os.environ.pop('OPENCV_LOG_LEVEL', None)

        return available

