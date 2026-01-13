"""
Unit tests for Video Capture Module.
"""

import pytest
import numpy as np
import cv2
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_capture import VideoCapture


class TestVideoCapture:
    """Tests for VideoCapture class."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        vc = VideoCapture()
        assert vc.source == 0
        assert vc.frame_width == 1280
        assert vc.frame_height == 720
        assert vc.cap is None
        assert vc.is_file is False
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        vc = VideoCapture(camera_index=1, width=640, height=480)
        assert vc.source == 1
        assert vc.frame_width == 640
        assert vc.frame_height == 480
        assert vc.is_file is False
    
    def test_init_partial_custom_values(self):
        """Test initialization with some custom values."""
        vc = VideoCapture(camera_index=2)
        assert vc.source == 2
        assert vc.frame_width == 1280  # Default
        assert vc.frame_height == 720  # Default
    
    def test_init_video_file(self):
        """Test initialization with video file path."""
        vc = VideoCapture(video_path="test_video.mp4")
        assert vc.source == "test_video.mp4"
        assert vc.is_file is True
        assert vc.video_path == "test_video.mp4"
    
    def test_video_path_overrides_camera(self):
        """Test that video_path takes precedence over camera_index."""
        vc = VideoCapture(camera_index=1, video_path="video.mp4")
        assert vc.source == "video.mp4"
        assert vc.is_file is True
    
    def test_read_frame_before_start(self):
        """Test reading frame before starting camera."""
        vc = VideoCapture()
        success, frame = vc.read_frame()
        assert success is False
        assert frame is None
    
    def test_release_before_start(self):
        """Test releasing before starting (should not raise error)."""
        vc = VideoCapture()
        # Should not raise an exception
        vc.release()
    
    def test_get_progress_camera(self):
        """Test get_progress for camera (should return zeros)."""
        vc = VideoCapture()
        current, total, pct = vc.get_progress()
        assert current == 0
        assert total == 0
        assert pct == 0.0
    
    def test_get_fps(self):
        """Test get_fps method."""
        vc = VideoCapture()
        fps = vc.get_fps()
        assert fps > 0
    
    def test_get_frame_size(self):
        """Test get_frame_size method."""
        vc = VideoCapture(width=640, height=480)
        width, height = vc.get_frame_size()
        assert width == 640
        assert height == 480
    
    @pytest.mark.skipif(
        not cv2.VideoCapture(0).isOpened(),
        reason="No camera available"
    )
    def test_start_with_camera(self):
        """Test starting video capture with real camera."""
        vc = VideoCapture()
        success = vc.start()
        if success:
            assert vc.cap is not None
            assert vc.cap.isOpened()
            vc.release()
    
    @pytest.mark.skipif(
        not cv2.VideoCapture(0).isOpened(),
        reason="No camera available"
    )
    def test_read_frame_with_camera(self):
        """Test reading frame from real camera."""
        vc = VideoCapture()
        if vc.start():
            success, frame = vc.read_frame()
            assert success is True
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert len(frame.shape) == 3  # Height, Width, Channels
            vc.release()
    
    @pytest.mark.skipif(
        not cv2.VideoCapture(0).isOpened(),
        reason="No camera available"
    )
    def test_release_with_camera(self):
        """Test releasing camera resources."""
        vc = VideoCapture()
        if vc.start():
            vc.release()
            assert vc.cap is None or not vc.cap.isOpened()
    
    def test_invalid_camera_index(self):
        """Test with invalid camera index."""
        vc = VideoCapture(camera_index=999)
        success = vc.start()
        assert success is False
    
    def test_nonexistent_video_file(self):
        """Test with non-existent video file."""
        vc = VideoCapture(video_path="nonexistent_video.mp4")
        success = vc.start()
        assert success is False


class TestVideoFileCapture:
    """Tests for video file capture functionality."""
    
    @pytest.fixture
    def test_video_path(self):
        """Create a test video file."""
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        # Create a short test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
        
        for i in range(30):  # 1 second at 30 FPS
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        
        writer.release()
        
        yield video_path
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
    
    def test_start_video_file(self, test_video_path):
        """Test starting video file capture."""
        vc = VideoCapture(video_path=test_video_path)
        success = vc.start()
        
        assert success is True
        assert vc.is_file is True
        assert vc.total_frames == 30
        assert vc.fps == 30.0
        
        vc.release()
    
    def test_read_video_frame(self, test_video_path):
        """Test reading frame from video file."""
        vc = VideoCapture(video_path=test_video_path)
        vc.start()
        
        success, frame = vc.read_frame()
        
        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert vc.current_frame == 1
        
        vc.release()
    
    def test_video_progress(self, test_video_path):
        """Test video progress tracking."""
        vc = VideoCapture(video_path=test_video_path)
        vc.start()
        
        # Read some frames
        for _ in range(15):
            vc.read_frame()
        
        current, total, pct = vc.get_progress()
        
        assert current == 15
        assert total == 30
        assert pct == 50.0
        
        vc.release()
    
    def test_video_seek(self, test_video_path):
        """Test seeking in video file."""
        vc = VideoCapture(video_path=test_video_path)
        vc.start()
        
        # Seek to frame 20
        success = vc.seek(20)
        
        assert success is True
        assert vc.current_frame == 20
        
        vc.release()
    
    def test_video_restart(self, test_video_path):
        """Test restarting video file."""
        vc = VideoCapture(video_path=test_video_path)
        vc.start()
        
        # Read some frames
        for _ in range(10):
            vc.read_frame()
        
        # Restart
        success = vc.restart()
        
        assert success is True
        assert vc.current_frame == 0
        
        vc.release()
    
    def test_video_end_detection(self, test_video_path):
        """Test video end detection."""
        vc = VideoCapture(video_path=test_video_path)
        vc.start()
        
        # Read all frames
        frames_read = 0
        while True:
            success, frame = vc.read_frame()
            if not success:
                break
            frames_read += 1
        
        assert frames_read == 30
        
        vc.release()


class TestVideoCaptureEdgeCases:
    """Edge case tests for VideoCapture."""
    
    def test_zero_resolution(self):
        """Test with zero resolution (should use defaults)."""
        vc = VideoCapture(width=0, height=0)
        assert vc.frame_width == 0
        assert vc.frame_height == 0
    
    def test_negative_camera_index(self):
        """Test with negative camera index."""
        vc = VideoCapture(camera_index=-1)
        success = vc.start()
        assert success is False
    
    def test_multiple_start_calls(self):
        """Test calling start multiple times."""
        vc = VideoCapture()
        # First call
        vc.start()
        # Second call should work without issues
        vc.start()
        vc.release()
    
    def test_multiple_release_calls(self):
        """Test calling release multiple times."""
        vc = VideoCapture()
        vc.release()
        vc.release()  # Should not raise error
        vc.release()  # Should not raise error
    
    def test_context_manager(self):
        """Test using VideoCapture as context manager."""
        with VideoCapture() as vc:
            # Just test that it enters and exits without error
            pass
    
    def test_seek_without_start(self):
        """Test seeking before starting."""
        vc = VideoCapture(video_path="test.mp4")
        success = vc.seek(10)
        assert success is False
    
    def test_seek_on_camera(self):
        """Test seeking on camera (should fail)."""
        vc = VideoCapture(camera_index=0)
        success = vc.seek(10)
        assert success is False

