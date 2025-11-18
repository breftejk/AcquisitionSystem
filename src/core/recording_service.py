"""
Recording service for camera stream
"""
from pathlib import Path
from typing import Optional
from datetime import datetime
import cv2
import numpy as np


class RecordingService:
    """Service for recording frames to PNG sequence"""

    def __init__(self, output_folder: str = "recordings"):
        """
        Args:
            output_folder: Base folder for recordings
        """
        self.output_folder = Path(output_folder)
        self.current_recording_folder: Optional[Path] = None
        self.frame_counter = 0
        self.is_recording = False
    
    def start_recording(self, name: Optional[str] = None) -> str:
        """
        Starts recording

        Args:
            name: Optional recording name (default: timestamp)

        Returns:
            Path to recording folder
        """
        if self.is_recording:
            return str(self.current_recording_folder)
        
        # Create folder for recording
        if name is None:
            name = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
        
        self.current_recording_folder = self.output_folder / name
        self.current_recording_folder.mkdir(parents=True, exist_ok=True)
        
        self.frame_counter = 0
        self.is_recording = True
        
        return str(self.current_recording_folder)
    
    def record_frame(self, frame: np.ndarray) -> bool:
        """
        Saves frame

        Args:
            frame: Frame to save (RGB)

        Returns:
            True if successful
        """
        if not self.is_recording or self.current_recording_folder is None:
            return False
        
        # Create filename with proper padding (0000.png, 0001.png, etc.)
        filename = f"frame_{self.frame_counter:06d}.png"
        filepath = self.current_recording_folder / filename
        
        # OpenCV requires BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save image
        success = cv2.imwrite(str(filepath), frame_bgr)
        
        if success:
            self.frame_counter += 1
        
        return success
    
    def stop_recording(self) -> tuple[str, int]:
        """
        Stops recording

        Returns:
            Tuple (folder path, number of frames recorded)
        """
        if not self.is_recording:
            return "", 0
        
        folder_path = str(self.current_recording_folder)
        frames_recorded = self.frame_counter
        
        self.is_recording = False
        self.current_recording_folder = None
        self.frame_counter = 0
        
        return folder_path, frames_recorded
    
    def is_recording_active(self) -> bool:
        """Whether recording is active"""
        return self.is_recording
    
    def get_frame_count(self) -> int:
        """Returns number of recorded frames"""
        return self.frame_counter if self.is_recording else 0

    def get_recording_folder(self) -> Optional[str]:
        """Returns current recording folder"""
        return str(self.current_recording_folder) if self.current_recording_folder else None

