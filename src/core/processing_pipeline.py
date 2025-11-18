"""
Image processing pipeline with multi-threading
"""
from typing import Optional, Callable
from threading import Thread, Lock
from queue import Queue, Empty
import time
from datetime import datetime

from .interfaces import IDataSource, IDetectionAlgorithm
from .models import FrameData
from .ring_buffer import FrameRingBuffer
from .recording_service import RecordingService


class ProcessingPipeline:
    """Acquisition and processing pipeline with multi-threading"""

    def __init__(self, 
                 data_source: IDataSource,
                 algorithm: IDetectionAlgorithm,
                 buffer_size: int = 100):
        """
        Args:
            data_source: Data source
            algorithm: Processing algorithm
            buffer_size: Frame buffer size
        """
        self.data_source = data_source
        self.algorithm = algorithm
        
        # Buffers
        self.source_buffer = FrameRingBuffer(buffer_size)
        self.processed_buffer = FrameRingBuffer(buffer_size)
        
        # Queues between threads
        self.acquisition_queue = Queue(maxsize=10)
        self.processing_queue = Queue(maxsize=10)
        
        # Threads
        self.acquisition_thread: Optional[Thread] = None
        self.processing_thread: Optional[Thread] = None
        
        # Thread control
        self.is_running = False
        self.lock = Lock()
        
        # Callbacks
        self.on_new_frame: Optional[Callable[[FrameData, FrameData], None]] = None
        
        # Recording
        self.recording_service = RecordingService()
        
        # Statistics
        self.frames_acquired = 0
        self.frames_processed = 0
        self.last_acquisition_time = 0.0
        self.last_processing_time = 0.0
    
    def start(self) -> bool:
        """Starts the pipeline"""
        with self.lock:
            if self.is_running:
                return True
            
            # Open data source
            if not self.data_source.open():
                return False
            
            if not self.data_source.start():
                self.data_source.close()
                return False
            
            self.is_running = True
            
            # Start threads
            self.acquisition_thread = Thread(target=self._acquisition_loop, daemon=True)
            self.processing_thread = Thread(target=self._processing_loop, daemon=True)
            
            self.acquisition_thread.start()
            self.processing_thread.start()
            
            return True
    
    def stop(self):
        """Stops the pipeline"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
        
        # Wait for threads to finish (with timeout)
        if self.acquisition_thread:
            self.acquisition_thread.join(timeout=2.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Close data source
        self.data_source.close()
        
        # Stop recording if active
        if self.recording_service.is_recording_active():
            self.recording_service.stop_recording()
    
    def _acquisition_loop(self):
        """Acquisition loop (separate thread)"""
        while self.is_running:
            start_time = time.time()
            
            # Read frame
            frame = self.data_source.read_frame()
            
            if frame is not None:
                # Create FrameData
                frame_data = FrameData(
                    frame=frame,
                    timestamp=datetime.now(),
                    frame_number=self.frames_acquired,
                    source_info=self.data_source.get_info().get("name", "Unknown")
                )
                
                # Add to source buffer
                self.source_buffer.add(frame_data)
                
                # Pass to processing
                try:
                    self.processing_queue.put(frame_data, block=False)
                except:
                    pass  # Queue full, skip

                self.frames_acquired += 1
                
                # Statistics
                self.last_acquisition_time = time.time() - start_time
            else:
                # No frame - wait a bit
                time.sleep(0.01)


    def _processing_loop(self):
        """Processing loop (separate thread)"""
        while self.is_running:
            try:
                # Get frame to process
                frame_data = self.processing_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process frame
                processed_frame = self.algorithm.process(frame_data.frame)
                
                # Create FrameData for processed frame
                processed_data = FrameData(
                    frame=processed_frame,
                    timestamp=datetime.now(),
                    frame_number=frame_data.frame_number,
                    source_info=self.algorithm.get_name()
                )
                
                # Add to processed buffer
                self.processed_buffer.add(processed_data)
                
                # Record processed frame if active
                if self.recording_service.is_recording_active():
                    self.recording_service.record_frame(processed_frame)

                self.frames_processed += 1
                
                # Statistics
                self.last_processing_time = time.time() - start_time
                
                # Call callback
                if self.on_new_frame:
                    try:
                        self.on_new_frame(frame_data, processed_data)
                    except Exception as e:
                        print(f"Error in callback: {e}")
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    def get_latest_frames(self) -> tuple[Optional[FrameData], Optional[FrameData]]:
        """
        Returns latest frames (source and processed)

        Returns:
            Tuple (source_frame, processed_frame)
        """
        source = self.source_buffer.get_latest()
        processed = self.processed_buffer.get_latest()
        return source, processed
    
    def get_frame_by_number(self, frame_number: int) -> tuple[Optional[FrameData], Optional[FrameData]]:
        """
        Returns frames with specified number

        Returns:
            Tuple (source_frame, processed_frame)
        """
        source = self.source_buffer.get_by_frame_number(frame_number)
        processed = self.processed_buffer.get_by_frame_number(frame_number)
        return source, processed
    
    def get_buffer_info(self) -> dict:
        """Returns buffer information"""
        return {
            "source_size": self.source_buffer.get_size(),
            "source_capacity": self.source_buffer.get_capacity(),
            "source_fill": self.source_buffer.get_fill_percentage(),
            "processed_size": self.processed_buffer.get_size(),
            "processed_capacity": self.processed_buffer.get_capacity(),
            "processed_fill": self.processed_buffer.get_fill_percentage(),
            "frames_acquired": self.frames_acquired,
            "frames_processed": self.frames_processed,
            "acquisition_time_ms": self.last_acquisition_time * 1000.0,
            "processing_time_ms": self.last_processing_time * 1000.0
        }
    
    def set_algorithm(self, algorithm: IDetectionAlgorithm):
        """Changes processing algorithm"""
        with self.lock:
            self.algorithm = algorithm
