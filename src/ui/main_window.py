"""
Main application window - Modern compact UI
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QComboBox, QSpinBox,
                             QFileDialog, QGroupBox, QGridLayout, QCheckBox,
                             QProgressBar, QSplitter, QTabWidget, QScrollArea,
                             QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent
import numpy as np
from typing import Optional, List

from ..core.interfaces import IDataSource, IDetectionAlgorithm
from ..core.processing_pipeline import ProcessingPipeline
from ..core.playback_controller import PlaybackController, PlaybackState
from ..data_sources.camera_source import CameraSource
from ..data_sources.image_sequence_source import ImageSequenceSource
from ..data_sources.dicom_source import DicomSource
from ..processing.no_processing import NoProcessingAlgorithm
from ..processing.convolution import ConvolutionAlgorithm


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Application state
        self.current_data_source: Optional[IDataSource] = None
        self.current_algorithm: IDetectionAlgorithm = NoProcessingAlgorithm()
        self.pipeline: Optional[ProcessingPipeline] = None
        self.playback_controller = PlaybackController(fps=30.0)

        # Available algorithms
        self.algorithms: List[IDetectionAlgorithm] = [
            NoProcessingAlgorithm(),
            ConvolutionAlgorithm()
        ]

        # UI update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.setInterval(33)  # ~30 FPS

        self.init_ui()
        self.setup_playback_controller()

        # Start timer
        self.update_timer.start()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Acquisition System")
        # Compact size that fits screens
        self.resize(1200, 700)
        self.setMinimumSize(1000, 600)

        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left panel - Controls (compact)
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel)

        # Right panel - Image display
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def create_control_panel(self) -> QWidget:
        """Create compact control panel on the left"""
        panel = QWidget()
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Tabs for different controls
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Tab 1: Source
        tabs.addTab(self.create_source_tab(), "📷 Source")

        # Tab 2: Processing
        tabs.addTab(self.create_processing_tab(), "⚙️ Process")

        # Tab 3: Recording
        tabs.addTab(self.create_recording_tab(), "🔴 Record")

        layout.addWidget(tabs)

        # Status at bottom
        layout.addWidget(self.create_status_panel())

        return panel

    def create_source_tab(self) -> QWidget:
        """Create source selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Source type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["Camera", "Image Sequence", "DICOM"])
        self.source_type_combo.currentTextChanged.connect(self.on_source_type_changed)
        type_layout.addWidget(self.source_type_combo, 1)
        layout.addLayout(type_layout)

        # Source selection
        source_layout = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        source_layout.addWidget(self.source_combo, 1)

        self.refresh_sources_btn = QPushButton("🔄")
        self.refresh_sources_btn.setMaximumWidth(35)
        self.refresh_sources_btn.clicked.connect(self.refresh_sources)
        source_layout.addWidget(self.refresh_sources_btn)
        layout.addLayout(source_layout)

        # Browse and Open buttons
        btn_layout = QHBoxLayout()
        self.browse_folder_btn = QPushButton("📁 Browse...")
        self.browse_folder_btn.clicked.connect(self.browse_folder)
        btn_layout.addWidget(self.browse_folder_btn)

        self.open_source_btn = QPushButton("▶️ Open")
        self.open_source_btn.clicked.connect(self.open_source)
        self.open_source_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        btn_layout.addWidget(self.open_source_btn)
        layout.addLayout(btn_layout)

        # Separator
        separator = QLabel()
        separator.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        layout.addWidget(separator)

        # Playback controls (only shown for seekable sources)
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout()

        # Buttons
        btn_row = QHBoxLayout()
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setMaximumWidth(50)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        btn_row.addWidget(self.play_pause_btn)

        self.step_backward_btn = QPushButton("◄")
        self.step_backward_btn.setMaximumWidth(40)
        self.step_backward_btn.clicked.connect(self.step_backward)
        btn_row.addWidget(self.step_backward_btn)

        self.step_forward_btn = QPushButton("►")
        self.step_forward_btn.setMaximumWidth(40)
        self.step_forward_btn.clicked.connect(self.step_forward)
        btn_row.addWidget(self.step_forward_btn)

        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.stateChanged.connect(self.on_loop_changed)
        btn_row.addWidget(self.loop_checkbox)
        btn_row.addStretch()
        playback_layout.addLayout(btn_row)

        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.valueChanged.connect(self.on_fps_changed)
        fps_layout.addWidget(self.fps_spinbox)
        fps_layout.addStretch()
        playback_layout.addLayout(fps_layout)

        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)
        self.position_slider.valueChanged.connect(self.on_position_changed)
        playback_layout.addWidget(self.position_slider)

        self.position_label = QLabel("0 / 0")
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        playback_layout.addWidget(self.position_label)

        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)

        layout.addStretch()

        # Initialize
        self.refresh_sources()

        return tab

    def create_processing_tab(self) -> QWidget:
        """Create processing algorithm tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Algorithm selection
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        for algo in self.algorithms:
            self.algorithm_combo.addItem(algo.get_name())
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        algo_layout.addWidget(self.algorithm_combo, 1)
        layout.addLayout(algo_layout)

        # Separator
        separator = QLabel()
        separator.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        layout.addWidget(separator)

        # Parameters panel (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.algorithm_params_widget = QWidget()
        self.algorithm_params_layout = QVBoxLayout(self.algorithm_params_widget)
        self.algorithm_params_layout.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(self.algorithm_params_widget)

        layout.addWidget(scroll)

        # Initialize
        self.update_algorithm_parameters()

        return tab

    def create_recording_tab(self) -> QWidget:
        """Create recording tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Recording button
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("QPushButton { font-size: 14px; }")
        layout.addWidget(self.record_btn)

        # Status
        self.recording_label = QLabel("Not recording")
        self.recording_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_label.setStyleSheet("QLabel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; }")
        layout.addWidget(self.recording_label)

        layout.addStretch()

        return tab

    def create_display_panel(self) -> QWidget:
        """Create image display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Splitter for two images
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Source image
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout()
        self.source_image_label = QLabel()
        self.source_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_image_label.setMinimumSize(400, 300)
        self.source_image_label.setStyleSheet("QLabel { background-color: #1a1a1a; }")
        self.source_image_label.setScaledContents(False)
        source_layout.addWidget(self.source_image_label)

        self.source_info_label = QLabel("No image")
        self.source_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        source_layout.addWidget(self.source_info_label)
        source_group.setLayout(source_layout)
        splitter.addWidget(source_group)

        # Processed image
        processed_group = QGroupBox("Processed")
        processed_layout = QVBoxLayout()
        self.processed_image_label = QLabel()
        self.processed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image_label.setMinimumSize(400, 300)
        self.processed_image_label.setStyleSheet("QLabel { background-color: #1a1a1a; }")
        self.processed_image_label.setScaledContents(False)
        processed_layout.addWidget(self.processed_image_label)

        self.processed_info_label = QLabel("No image")
        self.processed_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        processed_layout.addWidget(self.processed_info_label)
        processed_group.setLayout(processed_layout)
        splitter.addWidget(processed_group)

        layout.addWidget(splitter)

        return panel

    def create_status_panel(self) -> QGroupBox:
        """Create compact status panel"""
        group = QGroupBox("Status")
        layout = QGridLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Resolution
        layout.addWidget(QLabel("Res:"), 0, 0)
        self.resolution_label = QLabel("--")
        self.resolution_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.resolution_label, 0, 1)

        # Cache
        layout.addWidget(QLabel("Cache:"), 1, 0)
        self.cache_progress = QProgressBar()
        self.cache_progress.setMaximumHeight(15)
        self.cache_progress.setTextVisible(True)
        self.cache_progress.setFormat("%p%")
        layout.addWidget(self.cache_progress, 1, 1)

        # Performance
        layout.addWidget(QLabel("Acq:"), 2, 0)
        self.acq_time_label = QLabel("-- ms")
        layout.addWidget(self.acq_time_label, 2, 1)

        layout.addWidget(QLabel("Proc:"), 3, 0)
        self.proc_time_label = QLabel("-- ms")
        layout.addWidget(self.proc_time_label, 3, 1)

        group.setLayout(layout)
        return group

    # Event handlers (keep existing implementations)
    def on_source_type_changed(self, source_type: str):
        """Handle source type change"""
        self.refresh_sources()

    def refresh_sources(self):
        """Refresh available sources"""
        source_type = self.source_type_combo.currentText()
        self.source_combo.clear()

        if source_type == "Camera":
            cameras = CameraSource.list_available_cameras()
            for cam_id in cameras:
                self.source_combo.addItem(f"Camera {cam_id}", cam_id)

            if len(cameras) == 0:
                self.source_combo.addItem("No cameras found", -1)
        elif source_type == "Image Sequence":
            self.source_combo.addItem("Select folder...", None)
        elif source_type == "DICOM":
            self.source_combo.addItem("Select file/folder...", None)

    def browse_folder(self):
        """Open folder selection dialog"""
        source_type = self.source_type_combo.currentText()

        if source_type == "Image Sequence":
            folder = QFileDialog.getExistingDirectory(self, "Select Image Sequence Folder")
            if folder:
                self.source_combo.clear()
                self.source_combo.addItem(folder, folder)
        elif source_type == "DICOM":
            path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
            if not path:
                # Try file selection
                path, _ = QFileDialog.getOpenFileName(self, "Select DICOM File", "", "DICOM Files (*.dcm);;All Files (*)")
            if path:
                self.source_combo.clear()
                self.source_combo.addItem(path, path)

    def open_source(self):
        """Open selected data source"""
        source_type = self.source_type_combo.currentText()

        # Close previous source
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

        # Create new source
        if source_type == "Camera":
            camera_id = self.source_combo.currentData()
            if camera_id is None or camera_id < 0:
                return
            self.current_data_source = CameraSource(camera_id)
        elif source_type == "Image Sequence":
            folder = self.source_combo.currentData()
            if folder is None:
                return
            self.current_data_source = ImageSequenceSource(folder)
        elif source_type == "DICOM":
            path = self.source_combo.currentData()
            if path is None:
                return
            self.current_data_source = DicomSource(path)

        # Create pipeline
        self.pipeline = ProcessingPipeline(
            self.current_data_source,
            self.current_algorithm,
            buffer_size=100
        )

        # Start pipeline
        if self.pipeline.start():
            # Update source info
            info = self.current_data_source.get_info()
            self.resolution_label.setText(f"{info['width']}x{info['height']}")

            # Configure playback controller only for seekable sources
            if self.current_data_source.supports_seek():
                total_frames = self.current_data_source.get_total_frames()
                self.playback_controller.set_total_frames(total_frames)
                self.position_slider.setMaximum(total_frames - 1 if total_frames else 100)

                # Enable playback controls
                self.play_pause_btn.setEnabled(True)
                self.step_backward_btn.setEnabled(True)
                self.step_forward_btn.setEnabled(True)
                self.position_slider.setEnabled(True)
                self.loop_checkbox.setEnabled(True)
                self.fps_spinbox.setEnabled(True)

                # Start from beginning
                self.current_data_source.seek(0)
                self.playback_controller.seek(0)

                # Start playback automatically
                self.playback_controller.play()
                self.play_pause_btn.setText("⏸")
            else:
                # Disable playback controls for camera (continuous streaming)
                self.play_pause_btn.setEnabled(False)
                self.step_backward_btn.setEnabled(False)
                self.step_forward_btn.setEnabled(False)
                self.position_slider.setEnabled(False)
                self.loop_checkbox.setEnabled(False)

                # Lock FPS to 30 for camera (camera has fixed frame rate)
                self.fps_spinbox.setValue(30)
                self.fps_spinbox.setEnabled(False)
        else:
            self.current_data_source = None
            self.pipeline = None

    def toggle_play_pause(self):
        """Toggle play/pause"""
        # Only for seekable sources
        if not self.current_data_source or not self.current_data_source.supports_seek():
            return

        state = self.playback_controller.get_state()

        if state == PlaybackState.PLAYING:
            # Pause
            self.playback_controller.pause()
            self.play_pause_btn.setText("▶")
        else:
            # Play - start from current position
            current_pos = self.current_data_source.get_current_position()
            self.playback_controller.seek(current_pos)
            self.playback_controller.play()
            self.play_pause_btn.setText("⏸")

    def step_forward(self):
        """Step forward one frame"""
        if self.pipeline and self.current_data_source and self.current_data_source.supports_seek():
            current = self.playback_controller.get_current_frame()
            self.playback_controller.seek(current + 1)

    def step_backward(self):
        """Step backward one frame"""
        if self.pipeline and self.current_data_source and self.current_data_source.supports_seek():
            current = self.playback_controller.get_current_frame()
            self.playback_controller.seek(current - 1)

    def on_loop_changed(self, state):
        """Handle loop checkbox change"""
        self.playback_controller.set_loop(state == Qt.CheckState.Checked.value)

    def on_fps_changed(self, value):
        """Handle FPS change"""
        # Update playback controller FPS immediately (works during playback)
        self.playback_controller.set_fps(float(value))

    def on_position_changed(self, value):
        """Handle position slider change"""
        if self.pipeline and self.current_data_source and self.current_data_source.supports_seek():
            # Only seek if user is dragging (not programmatic change)
            if self.position_slider.isSliderDown():
                # Pause playback when user starts dragging
                if self.playback_controller.get_state() == PlaybackState.PLAYING:
                    self.playback_controller.pause()
                    self.play_pause_btn.setText("▶")

                # Seek to position
                self.current_data_source.seek(value)
                self.playback_controller.seek(value)

    def on_algorithm_changed(self, index):
        """Handle algorithm change"""
        self.current_algorithm = self.algorithms[index]
        self.update_algorithm_parameters()

        if self.pipeline:
            self.pipeline.set_algorithm(self.current_algorithm)

    def update_algorithm_parameters(self):
        """Update algorithm parameter controls"""
        # Clear existing
        while self.algorithm_params_layout.count():
            item = self.algorithm_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get parameters
        params = self.current_algorithm.get_parameters()

        for param_name, param_info in params.items():
            param_layout = QVBoxLayout()
            label = QLabel(param_info.get("label", param_name) + ":")
            param_layout.addWidget(label)

            param_type = param_info.get("type", "text")
            value = param_info.get("value")

            if param_type == "choice":
                combo = QComboBox()
                combo.addItems(param_info.get("choices", []))
                combo.setCurrentText(str(value))
                combo.currentTextChanged.connect(
                    lambda v, pn=param_name: self.on_parameter_changed(pn, v)
                )
                param_layout.addWidget(combo)
            elif param_type == "bool":
                checkbox = QCheckBox()
                checkbox.setChecked(bool(value))
                checkbox.stateChanged.connect(
                    lambda s, pn=param_name: self.on_parameter_changed(pn, s == Qt.CheckState.Checked.value)
                )
                param_layout.addWidget(checkbox)
            elif param_type == "slider":
                # Create slider with value label
                slider_row = QHBoxLayout()

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(param_info.get("min", 0))
                slider.setMaximum(param_info.get("max", 255))
                slider.setValue(int(value))

                value_label = QLabel(str(int(value)))
                value_label.setMinimumWidth(35)

                slider.valueChanged.connect(
                    lambda v, pn=param_name, lbl=value_label: (
                        lbl.setText(str(v)),
                        self.on_parameter_changed(pn, v)
                    )
                )

                slider_row.addWidget(slider)
                slider_row.addWidget(value_label)
                param_layout.addLayout(slider_row)

            self.algorithm_params_layout.addLayout(param_layout)

        self.algorithm_params_layout.addStretch()

    def on_parameter_changed(self, param_name: str, value):
        """Handle parameter change"""
        # Get current parameters
        params = self.current_algorithm.get_parameters()
        # Update the changed parameter
        params[param_name]["value"] = value

        # Create config dict with just the values (not the full param info)
        config = {}
        for name, info in params.items():
            config[name] = info["value"]

        # Configure with the values
        self.current_algorithm.configure(config)

    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.pipeline:
            return

        if self.pipeline.recording_service.is_recording_active():
            # Stop recording
            folder, count = self.pipeline.recording_service.stop_recording()
            self.record_btn.setText("🔴 Start Recording")
            self.record_btn.setStyleSheet("QPushButton { font-size: 14px; }")
            self.recording_label.setText(f"Saved {count} frames to:\n{folder}")
        else:
            # Start recording
            folder = self.pipeline.recording_service.start_recording()
            self.record_btn.setText("⏹ Stop Recording")
            self.record_btn.setStyleSheet("QPushButton { font-size: 14px; background-color: #f44336; color: white; }")
            self.recording_label.setText(f"Recording to:\n{folder}")

    def setup_playback_controller(self):
        """Setup playback controller callback"""
        self.playback_controller.set_frame_callback(self.on_playback_frame_changed)

    def on_playback_frame_changed(self, frame_number: int):
        """Handle playback frame change"""
        if self.pipeline and self.current_data_source and self.current_data_source.supports_seek():
            # Seek to the requested frame
            self.current_data_source.seek(frame_number)

            # Update position slider without triggering seek
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(frame_number)
            self.position_slider.blockSignals(False)

            # Update position label
            total = self.playback_controller.get_total_frames()
            self.position_label.setText(f"{frame_number} / {total if total else '?'}")

    def update_display(self):
        """Update displayed images"""
        if not self.pipeline:
            return

        # Get latest frames
        source_frame, processed_frame = self.pipeline.get_latest_frames()

        # Update source image
        if source_frame:
            self.display_frame(source_frame.frame, self.source_image_label, self.source_info_label, source_frame.frame_number)

        # Update processed image
        if processed_frame:
            self.display_frame(processed_frame.frame, self.processed_image_label, self.processed_info_label, processed_frame.frame_number)

        # Update status
        self.update_status()

    def display_frame(self, frame: np.ndarray, label: QLabel, info_label: QLabel, frame_number: int):
        """Display frame on label"""
        if frame is None or frame.size == 0:
            return

        # Make a copy to ensure data isn't garbage collected
        frame = frame.copy()

        # Convert to QImage
        height, width = frame.shape[:2]

        if len(frame.shape) == 3:
            # RGB - ensure contiguous array
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        else:
            # Grayscale
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            bytes_per_line = width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8).copy()

        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(scaled_pixmap)
        info_label.setText(f"Frame {frame_number} - {width}x{height}")

    def update_status(self):
        """Update status panel"""
        if not self.pipeline:
            return

        buffer_info = self.pipeline.get_buffer_info()

        # Cache progress
        fill_pct = buffer_info.get('source_fill', 0)
        self.cache_progress.setValue(int(fill_pct))

        # Performance
        acq_time = buffer_info.get('acquisition_time_ms', 0)
        proc_time = buffer_info.get('processing_time_ms', 0)

        self.acq_time_label.setText(f"{acq_time:.1f} ms")
        self.proc_time_label.setText(f"{proc_time:.1f} ms")

        # Color code performance
        if acq_time > 100:
            self.acq_time_label.setStyleSheet("color: red;")
        elif acq_time > 50:
            self.acq_time_label.setStyleSheet("color: orange;")
        else:
            self.acq_time_label.setStyleSheet("color: green;")

        if proc_time > 100:
            self.proc_time_label.setStyleSheet("color: red;")
        elif proc_time > 50:
            self.proc_time_label.setStyleSheet("color: orange;")
        else:
            self.proc_time_label.setStyleSheet("color: green;")

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key.Key_Left:
            self.step_backward()
        elif event.key() == Qt.Key.Key_Right:
            self.step_forward()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close"""
        if self.pipeline:
            self.pipeline.stop()
        event.accept()

