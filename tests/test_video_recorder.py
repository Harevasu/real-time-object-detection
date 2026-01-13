"""
Unit tests for Video Recorder Module.
"""

import pytest
import numpy as np
import cv2
import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_recorder import VideoRecorder


class TestVideoRecorderInit:
    """Tests for VideoRecorder initialization."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        recorder = VideoRecorder()
        assert recorder.fps == 30  # Default from settings
        assert recorder.frame_size == (1280, 720)
        assert recorder.is_recording is False
        assert recorder.writer is None
    
    def test_init_custom_fps(self):
        """Test initialization with custom FPS."""
        recorder = VideoRecorder(fps=60)
        assert recorder.fps == 60
    
    def test_init_custom_frame_size(self):
        """Test initialization with custom frame size."""
        recorder = VideoRecorder(frame_size=(640, 480))
        assert recorder.frame_size == (640, 480)
    
    def test_init_custom_output_path(self):
        """Test initialization with custom output path."""
        recorder = VideoRecorder(output_path="test_output.mp4")
        assert recorder.output_path == "test_output.mp4"
    
    def test_init_all_custom(self):
        """Test initialization with all custom values."""
        recorder = VideoRecorder(
            output_path="custom.mp4",
            fps=24,
            frame_size=(1920, 1080),
            codec='XVID'
        )
        assert recorder.output_path == "custom.mp4"
        assert recorder.fps == 24
        assert recorder.frame_size == (1920, 1080)
        assert recorder.codec == 'XVID'


class TestVideoRecorderRecording:
    """Tests for recording functionality."""
    
    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary output path."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    def test_start_recording(self, temp_output_path):
        """Test starting recording."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        
        success = recorder.start()
        
        assert success is True
        assert recorder.is_recording is True
        assert recorder.writer is not None
        assert recorder.frame_count == 0
        
        recorder.stop()
    
    def test_start_creates_directory(self):
        """Test that start creates output directory if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "output.mp4")
            recorder = VideoRecorder(
                output_path=output_path,
                fps=30,
                frame_size=(640, 480)
            )
            
            success = recorder.start()
            
            assert success is True
            assert os.path.exists(os.path.dirname(output_path))
            
            recorder.stop()
    
    def test_write_frame(self, temp_output_path, sample_frame):
        """Test writing frames."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        success = recorder.write_frame(sample_frame)
        
        assert success is True
        assert recorder.frame_count == 1
        
        recorder.stop()
    
    def test_write_multiple_frames(self, temp_output_path, sample_frame):
        """Test writing multiple frames."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        for _ in range(30):
            recorder.write_frame(sample_frame)
        
        assert recorder.frame_count == 30
        
        recorder.stop()
    
    def test_write_frame_before_start(self, sample_frame):
        """Test writing frame before starting."""
        recorder = VideoRecorder()
        
        success = recorder.write_frame(sample_frame)
        
        assert success is False
    
    def test_stop_recording(self, temp_output_path, sample_frame):
        """Test stopping recording."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        for _ in range(30):
            recorder.write_frame(sample_frame)
        
        stats = recorder.stop()
        
        assert recorder.is_recording is False
        assert stats is not None
        assert stats['frame_count'] == 30
        assert stats['output_path'] == temp_output_path
        assert os.path.exists(temp_output_path)
    
    def test_stop_without_start(self):
        """Test stopping without starting."""
        recorder = VideoRecorder()
        
        stats = recorder.stop()
        
        assert stats is None
    
    def test_output_file_created(self, temp_output_path, sample_frame):
        """Test that output file is created."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        for _ in range(10):
            recorder.write_frame(sample_frame)
        
        recorder.stop()
        
        assert os.path.exists(temp_output_path)
        assert os.path.getsize(temp_output_path) > 0


class TestVideoRecorderStatus:
    """Tests for recording status functionality."""
    
    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary output path."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_get_status_not_recording(self):
        """Test get_status when not recording."""
        recorder = VideoRecorder()
        
        status = recorder.get_status()
        
        assert status['is_recording'] is False
    
    def test_get_status_recording(self, temp_output_path):
        """Test get_status when recording."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        # Write some frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            recorder.write_frame(frame)
        
        status = recorder.get_status()
        
        assert status['is_recording'] is True
        assert status['frame_count'] == 10
        assert status['output_path'] == temp_output_path
        assert 'elapsed_seconds' in status
        
        recorder.stop()


class TestVideoRecorderToggle:
    """Tests for toggle functionality."""
    
    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary output path."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_toggle_start(self):
        """Test toggle to start recording."""
        recorder = VideoRecorder(fps=30, frame_size=(640, 480))
        
        result = recorder.toggle()
        
        assert result is True
        assert recorder.is_recording is True
        
        recorder.stop()
    
    def test_toggle_stop(self, temp_output_path):
        """Test toggle to stop recording."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        result = recorder.toggle()
        
        assert result is False
        assert recorder.is_recording is False


class TestVideoRecorderFormats:
    """Tests for different output formats."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    def test_mp4_format(self, sample_frame):
        """Test MP4 format output."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            recorder = VideoRecorder(
                output_path=output_path,
                fps=30,
                frame_size=(640, 480)
            )
            recorder.start()
            recorder.write_frame(sample_frame)
            recorder.stop()
            
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_avi_format(self, sample_frame):
        """Test AVI format output."""
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as f:
            output_path = f.name
        
        try:
            recorder = VideoRecorder(
                output_path=output_path,
                fps=30,
                frame_size=(640, 480)
            )
            recorder.start()
            recorder.write_frame(sample_frame)
            recorder.stop()
            
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestVideoRecorderEdgeCases:
    """Edge case tests for VideoRecorder."""
    
    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary output path."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_double_start(self, temp_output_path):
        """Test starting twice (should warn)."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        # Second start should return False
        result = recorder.start()
        
        assert result is False
        
        recorder.stop()
    
    def test_frame_resize(self, temp_output_path):
        """Test that frames are resized to match expected size."""
        recorder = VideoRecorder(
            output_path=temp_output_path,
            fps=30,
            frame_size=(640, 480)
        )
        recorder.start()
        
        # Write a frame with different size
        large_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        success = recorder.write_frame(large_frame)
        
        assert success is True
        
        recorder.stop()
    
    def test_auto_generate_path(self):
        """Test auto-generation of output path."""
        recorder = VideoRecorder(fps=30, frame_size=(640, 480))
        recorder.start()
        
        assert recorder.output_path is not None
        assert recorder.output_path.endswith('.mp4')
        
        recorder.stop()
        
        # Cleanup
        if os.path.exists(recorder.output_path):
            os.remove(recorder.output_path)
    
    def test_context_manager(self):
        """Test using VideoRecorder as context manager."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            with VideoRecorder(
                output_path=output_path,
                fps=30,
                frame_size=(640, 480)
            ) as recorder:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                recorder.write_frame(frame)
            
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
