"""
Playback controller
"""
from enum import Enum
from typing import Optional, Callable
from threading import Thread, Event, Lock
import time


class PlaybackState(Enum):
    """Playback states"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class PlaybackController:
    """Playback controller with play/pause/step/seek support"""

    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps: Frames per second
        """
        self._fps = fps
        self._state = PlaybackState.STOPPED
        self._current_frame = 0
        self._loop_enabled = False
        self._total_frames: Optional[int] = None
        
        self._lock = Lock()
        self._play_thread: Optional[Thread] = None
        self._stop_event = Event()
        
        # Callback called on each frame
        self._frame_callback: Optional[Callable[[int], None]] = None
    
    def set_frame_callback(self, callback: Callable[[int], None]):
        """Sets callback called on each new frame"""
        self._frame_callback = callback
    
    def play(self):
        """Starts playback"""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                return
            
            self._state = PlaybackState.PLAYING
            self._stop_event.clear()
            
            if self._play_thread is None or not self._play_thread.is_alive():
                self._play_thread = Thread(target=self._play_loop, daemon=True)
                self._play_thread.start()
    
    def pause(self):
        """Pauses playback"""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.PAUSED
                self._stop_event.set()
    
    def stop(self):
        """Stops playback"""
        with self._lock:
            self._state = PlaybackState.STOPPED
            self._stop_event.set()
            self._current_frame = 0
    
    def toggle_play_pause(self):
        """Toggles between play and pause"""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self.pause()
            else:
                self.play()
    
    def step_forward(self):
        """Step forward"""
        with self._lock:
            if self._total_frames is None or self._current_frame < self._total_frames - 1:
                self._current_frame += 1
                if self._frame_callback:
                    self._frame_callback(self._current_frame)
    
    def step_backward(self):
        """Step backward"""
        with self._lock:
            if self._current_frame > 0:
                self._current_frame -= 1
                if self._frame_callback:
                    self._frame_callback(self._current_frame)
    
    def seek(self, frame_number: int):
        """Seeks to specified frame"""
        with self._lock:
            if self._total_frames is not None:
                frame_number = max(0, min(frame_number, self._total_frames - 1))
            else:
                frame_number = max(0, frame_number)
            
            self._current_frame = frame_number
            if self._frame_callback:
                self._frame_callback(self._current_frame)
    
    def set_fps(self, fps: float):
        """Sets FPS"""
        with self._lock:
            self._fps = max(1.0, min(fps, 120.0))
    
    def get_fps(self) -> float:
        """Returns current FPS"""
        return self._fps
    
    def set_loop(self, enabled: bool):
        """Enables/disables looping"""
        with self._lock:
            self._loop_enabled = enabled
    
    def is_loop_enabled(self) -> bool:
        """Whether looping is enabled"""
        return self._loop_enabled
    
    def set_total_frames(self, total: Optional[int]):
        """Sets total number of frames"""
        with self._lock:
            self._total_frames = total
    
    def get_total_frames(self) -> Optional[int]:
        """Returns total number of frames"""
        return self._total_frames
    
    def get_current_frame(self) -> int:
        """Returns current frame number"""
        return self._current_frame
    
    def get_state(self) -> PlaybackState:
        """Returns current state"""
        return self._state
    
    def _play_loop(self):
        """Playback loop (executed in separate thread)"""
        while not self._stop_event.is_set():
            with self._lock:
                if self._state != PlaybackState.PLAYING:
                    break
                
                # Read FPS dynamically so changes take effect immediately
                frame_time = 1.0 / self._fps

                # Check if end reached
                if self._total_frames is not None and self._current_frame >= self._total_frames - 1:
                    if self._loop_enabled:
                        self._current_frame = 0
                    else:
                        self._state = PlaybackState.PAUSED
                        break
                
                # Go to next frame
                self._current_frame += 1
                current_frame = self._current_frame
            
            # Call callback
            if self._frame_callback:
                self._frame_callback(current_frame)
            
            # Wait appropriate time (based on current FPS)
            time.sleep(frame_time)
