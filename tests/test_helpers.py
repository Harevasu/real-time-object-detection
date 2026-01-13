"""
Unit tests for Utils Helper Module.
"""

import pytest
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    get_timestamp,
    format_fps,
    format_detection_count,
    calculate_area,
    calculate_iou,
    ensure_directory,
    get_model_path,
    get_output_path,
    format_time_elapsed,
    validate_frame,
    clamp_bbox,
    scale_bbox,
    get_center,
    get_dimensions,
    format_confidence,
    create_detection_summary,
)


class TestTimestamp:
    """Tests for timestamp functions."""
    
    def test_get_timestamp_default(self):
        """Test default timestamp format."""
        ts = get_timestamp()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert ts[8] == '_'
    
    def test_get_timestamp_custom_format(self):
        """Test custom timestamp format."""
        ts = get_timestamp("%Y-%m-%d")
        assert len(ts) == 10
        assert ts[4] == '-'


class TestFormatting:
    """Tests for formatting functions."""
    
    def test_format_fps_default(self):
        """Test FPS formatting with default decimals."""
        result = format_fps(29.97)
        assert result == "FPS: 30.0"
    
    def test_format_fps_custom_decimals(self):
        """Test FPS formatting with custom decimals."""
        result = format_fps(29.97, decimals=2)
        assert result == "FPS: 29.97"
    
    def test_format_fps_zero(self):
        """Test FPS formatting with zero."""
        result = format_fps(0)
        assert result == "FPS: 0.0"
    
    def test_format_detection_count(self):
        """Test detection count formatting."""
        counts = {'person': 2, 'car': 1}
        result = format_detection_count(counts)
        assert 'person: 2' in result
        assert 'car: 1' in result
    
    def test_format_detection_count_empty(self):
        """Test detection count with empty dict."""
        result = format_detection_count({})
        assert result == "No detections"
    
    def test_format_confidence_percentage(self):
        """Test confidence formatting as percentage."""
        result = format_confidence(0.876)
        assert result == "87.6%"
    
    def test_format_confidence_decimal(self):
        """Test confidence formatting as decimal."""
        result = format_confidence(0.876, as_percentage=False)
        assert result == "0.88"
    
    def test_format_time_elapsed_minutes(self):
        """Test time formatting for minutes."""
        result = format_time_elapsed(125)
        assert result == "2:05"
    
    def test_format_time_elapsed_hours(self):
        """Test time formatting for hours."""
        result = format_time_elapsed(3661)
        assert result == "1:01:01"


class TestBBoxCalculations:
    """Tests for bounding box calculations."""
    
    def test_calculate_area(self):
        """Test area calculation."""
        area = calculate_area((0, 0, 100, 100))
        assert area == 10000.0
    
    def test_calculate_area_rectangle(self):
        """Test area calculation for rectangle."""
        area = calculate_area((10, 20, 60, 120))
        assert area == 5000.0  # 50 * 100
    
    def test_calculate_area_zero(self):
        """Test area calculation for zero area."""
        area = calculate_area((100, 100, 100, 100))
        assert area == 0.0
    
    def test_calculate_iou_overlap(self):
        """Test IoU with overlapping boxes."""
        iou = calculate_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.14 < iou < 0.15
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap."""
        iou = calculate_iou((0, 0, 50, 50), (100, 100, 150, 150))
        assert iou == 0.0
    
    def test_calculate_iou_complete_overlap(self):
        """Test IoU with complete overlap."""
        iou = calculate_iou((0, 0, 100, 100), (0, 0, 100, 100))
        assert iou == 1.0
    
    def test_get_center(self):
        """Test center calculation."""
        center = get_center((0, 0, 100, 100))
        assert center == (50.0, 50.0)
    
    def test_get_center_offset(self):
        """Test center calculation with offset."""
        center = get_center((10, 20, 110, 220))
        assert center == (60.0, 120.0)
    
    def test_get_dimensions(self):
        """Test dimension calculation."""
        dims = get_dimensions((10, 20, 110, 220))
        assert dims == (100.0, 200.0)
    
    def test_clamp_bbox(self):
        """Test bounding box clamping."""
        clamped = clamp_bbox((-10, -10, 700, 500), 640, 480)
        assert clamped == (0, 0, 640, 480)
    
    def test_clamp_bbox_no_change(self):
        """Test bounding box clamping when already valid."""
        clamped = clamp_bbox((10, 10, 100, 100), 640, 480)
        assert clamped == (10, 10, 100, 100)
    
    def test_scale_bbox(self):
        """Test bounding box scaling."""
        scaled = scale_bbox((100, 100, 200, 200), 0.5, 0.5)
        assert scaled == (50.0, 50.0, 100.0, 100.0)
    
    def test_scale_bbox_upscale(self):
        """Test bounding box upscaling."""
        scaled = scale_bbox((10, 10, 20, 20), 2.0, 2.0)
        assert scaled == (20.0, 20.0, 40.0, 40.0)


class TestValidation:
    """Tests for validation functions."""
    
    def test_validate_frame_valid(self):
        """Test valid frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert validate_frame(frame) is True
    
    def test_validate_frame_none(self):
        """Test None frame."""
        assert validate_frame(None) is False
    
    def test_validate_frame_empty(self):
        """Test empty frame."""
        frame = np.array([])
        assert validate_frame(frame) is False
    
    def test_validate_frame_1d(self):
        """Test 1D array."""
        frame = np.array([1, 2, 3])
        assert validate_frame(frame) is False
    
    def test_validate_frame_not_array(self):
        """Test non-array input."""
        assert validate_frame("not an array") is False
    
    def test_validate_frame_grayscale(self):
        """Test grayscale frame (2D)."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        assert validate_frame(frame) is True


class TestFileOperations:
    """Tests for file operation functions."""
    
    def test_ensure_directory_creates(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "test_subdir")
            result = ensure_directory(new_dir)
            assert os.path.exists(new_dir)
            assert result == new_dir
    
    def test_ensure_directory_exists(self):
        """Test with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ensure_directory(temp_dir)
            assert result == temp_dir
    
    def test_get_model_path_existing(self):
        """Test model path with existing file."""
        # Use a file that exists in the project
        path = get_model_path("yolov8n.pt")
        assert path == "yolov8n.pt" or path.endswith("yolov8n.pt")
    
    def test_get_model_path_nonexistent(self):
        """Test model path with nonexistent file."""
        path = get_model_path("nonexistent.pt")
        assert path == "nonexistent.pt"
    
    def test_get_output_path(self):
        """Test output path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = get_output_path("test", ".mp4", temp_dir)
            assert path.startswith(temp_dir)
            assert "test_" in path
            assert path.endswith(".mp4")


class TestDetectionSummary:
    """Tests for detection summary functions."""
    
    def test_create_detection_summary(self):
        """Test detection summary creation."""
        detections = [
            {'class_name': 'person', 'confidence': 0.9},
            {'class_name': 'person', 'confidence': 0.8},
            {'class_name': 'car', 'confidence': 0.7}
        ]
        summary = create_detection_summary(detections)
        
        assert summary['total'] == 3
        assert summary['by_class']['person'] == 2
        assert summary['by_class']['car'] == 1
        assert 0.79 < summary['avg_confidence'] < 0.81
        assert summary['max_confidence'] == 0.9
        assert summary['min_confidence'] == 0.7
    
    def test_create_detection_summary_empty(self):
        """Test detection summary with empty list."""
        summary = create_detection_summary([])
        
        assert summary['total'] == 0
        assert summary['by_class'] == {}
        assert summary['avg_confidence'] == 0.0
    
    def test_create_detection_summary_single(self):
        """Test detection summary with single detection."""
        detections = [{'class_name': 'dog', 'confidence': 0.95}]
        summary = create_detection_summary(detections)
        
        assert summary['total'] == 1
        assert summary['by_class']['dog'] == 1
        assert summary['avg_confidence'] == 0.95
