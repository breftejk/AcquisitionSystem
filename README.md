# Acquisition System

Image acquisition system with multiple data sources and image processing algorithms.

![Camera Demo Screenshot](./images/Screenshot%202025-11-18%20at%2007.55.33.png)

![DICOM Demo Screenshot](./images/Screenshot%202025-11-18%20at%2007.59.01.png)

## Features

- **Data sources:**
  - Camera (OpenCV)
  - PNG image sequence
  - DICOM files (single/multi-frame)

- **Control:**
  - Play/Pause (Space)
  - Step ±1 (←/→)
  - Seek (slider)
  - Loop
  - Configurable FPS

- **Cache:**
  - Ring buffer with configurable size (default 100 frames)
  - Fill indicator
  - Rewind to N frames back

- **Recording:**
  - Saving camera stream to PNG sequence
  - Playing recorded sequences

- **Processing algorithms:**
  - No processing
  - Convolution with masks:
    - Averaging (3x3, 5x5)
    - Gaussian
    - Sobel X/Y
    - Laplacian

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Project structure

```
src/
├── core/              # Base interfaces and models
├── data_sources/      # Data source implementations
├── processing/        # Processing algorithms
└── ui/               # User interface (PyQt6)
```

## Architecture

System based on generic interfaces:
- `IDataSource` - data sources
- `IDetectionAlgorithm` - processing algorithms
- `FrameRingBuffer` - frame buffer
- `PlaybackController` - playback controller

Multi-threaded architecture:
- Acquisition thread
- Processing thread
- UI thread (responsive <100ms)
