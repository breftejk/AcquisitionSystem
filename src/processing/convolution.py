"""
Algorytm konwolucji z wyborem maski
"""
from typing import Dict, Any
import numpy as np
import cv2
from ..core.interfaces import IDetectionAlgorithm


class ConvolutionAlgorithm(IDetectionAlgorithm):
    """Convolution algorithm with predefined kernels"""

    # Predefined kernels
    KERNELS = {
        "Average 3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
        "Average 5x5": np.ones((5, 5), dtype=np.float32) / 25.0,
        "Gaussian 3x3": np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0,
        "Gaussian 5x5": cv2.getGaussianKernel(5, -1) @ cv2.getGaussianKernel(5, -1).T,
        "Sobel X": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32),
        "Sobel Y": np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32),
        "Laplacian": np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32),
        "Laplacian 5x5": np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=np.float32),
        "Sharpen": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32),
        "Edge Detection": np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float32),
        "Canny Edge Detection": "canny",  # Special marker for Canny algorithm
    }
    
    def __init__(self):
        self.current_kernel_name = "Average 3x3"
        self.current_kernel = self.KERNELS[self.current_kernel_name]
        self.normalize_output = True
        # Canny edge detection parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure convolution algorithm

        Args:
            config: Configuration dictionary:
                - kernel_name: kernel name
                - normalize: whether to normalize output
                - canny_threshold1: Canny lower threshold
                - canny_threshold2: Canny upper threshold
        """
        if "kernel_name" in config:
            kernel_name = config["kernel_name"]
            if kernel_name in self.KERNELS:
                self.current_kernel_name = kernel_name
                self.current_kernel = self.KERNELS[kernel_name]
            else:
                return False
        
        if "normalize" in config:
            self.normalize_output = config["normalize"]
        
        if "canny_threshold1" in config:
            self.canny_threshold1 = int(config["canny_threshold1"])

        if "canny_threshold2" in config:
            self.canny_threshold2 = int(config["canny_threshold2"])

        return True
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame using convolution or Canny edge detection

        Args:
            frame: Frame to process (RGB or Grayscale)

        Returns:
            Processed frame
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Check if Canny edge detection is selected (compare by name, not array)
        if self.current_kernel_name == "Canny Edge Detection":
            return self._apply_canny(frame)

        # Standard convolution processing
        # Check if image is color
        is_color = len(frame.shape) == 3 and frame.shape[2] == 3
        
        if is_color:
            # For color image, convert to grayscale, process, and return as RGB
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            result = cv2.filter2D(gray, -1, self.current_kernel)
            
            # Normalize if needed
            if self.normalize_output:
                result = self._normalize_image(result)
            
            # Convert back to RGB
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            # Image already in grayscale
            result = cv2.filter2D(frame, -1, self.current_kernel)
            
            # Normalize if needed
            if self.normalize_output:
                result = self._normalize_image(result)
        
        return result
    
    def _apply_canny(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection

        Args:
            frame: Input frame (RGB or Grayscale)

        Returns:
            Edge detected frame
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)

        # Convert back to RGB if input was RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-255 range

        Args:
            img: Image to normalize

        Returns:
            Normalized image
        """
        # For operations that can produce negative values (Sobel, Laplacian)
        if img.min() < 0 or img.max() > 255:
            img_min = img.min()
            img_max = img.max()
            
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255.0)
            else:
                img = np.zeros_like(img)
            
            img = img.astype(np.uint8)
        
        return img
    
    def get_name(self) -> str:
        """Returns algorithm name"""
        return f"Convolution"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Returns algorithm parameters for UI display"""
        params = {
            "kernel_name": {
                "type": "choice",
                "value": self.current_kernel_name,
                "choices": list(self.KERNELS.keys()),
                "label": "Kernel"
            },
            "normalize": {
                "type": "bool",
                "value": self.normalize_output,
                "label": "Normalize Output"
            }
        }

        # Add Canny-specific parameters if Canny is selected
        if self.current_kernel_name == "Canny Edge Detection":
            params["canny_threshold1"] = {
                "type": "slider",
                "value": self.canny_threshold1,
                "min": 0,
                "max": 255,
                "label": "Canny Threshold 1"
            }
            params["canny_threshold2"] = {
                "type": "slider",
                "value": self.canny_threshold2,
                "min": 0,
                "max": 255,
                "label": "Canny Threshold 2"
            }

        return params

    @staticmethod
    def get_available_kernels() -> list[str]:
        """Returns list of available kernels"""
        return list(ConvolutionAlgorithm.KERNELS.keys())

