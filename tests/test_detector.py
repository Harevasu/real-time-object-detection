"""
Unit tests for Object Detection Module.
"""

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import ObjectDetector


class TestObjectDetectorInit:
    """Tests for ObjectDetector initialization."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        detector = ObjectDetector()
        assert detector.model_name == "yolov8m.pt"  # From settings
        assert detector.device == "cpu"  # From settings
        assert detector.confidence == 0.5  # From settings
        assert detector.model is None
    
    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        assert detector.model_name == "yolov8n.pt"
    
    def test_init_custom_device(self):
        """Test initialization with custom device."""
        detector = ObjectDetector(device="cuda:0")
        assert detector.device == "cuda:0"
    
    def test_init_custom_confidence(self):
        """Test initialization with custom confidence."""
        detector = ObjectDetector(confidence=0.7)
        assert detector.confidence == 0.7
    
    def test_init_all_custom(self):
        """Test initialization with all custom values."""
        detector = ObjectDetector(
            model_name="yolov8s.pt",
            device="cpu",
            confidence=0.6
        )
        assert detector.model_name == "yolov8s.pt"
        assert detector.device == "cpu"
        assert detector.confidence == 0.6


class TestObjectDetectorModel:
    """Tests for model loading and detection."""
    
    @pytest.fixture
    def loaded_detector(self):
        """Create a detector with loaded model."""
        detector = ObjectDetector(model_name="yolov8n.pt", device="cpu")
        detector.load_model()
        return detector
    
    def test_load_model(self):
        """Test model loading."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        success = detector.load_model()
        
        assert success is True
        assert detector.model is not None
        assert detector.class_names is not None
        assert len(detector.class_names) == 80  # COCO classes
    
    def test_load_model_class_names(self):
        """Test that class names are loaded correctly."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        detector.load_model()
        
        # Check some known COCO classes
        assert 'person' in detector.class_names.values()
        assert 'car' in detector.class_names.values()
        assert 'dog' in detector.class_names.values()
    
    def test_detect_returns_results(self, loaded_detector, sample_frame):
        """Test that detect returns results."""
        results = loaded_detector.detect(sample_frame)
        
        assert results is not None
    
    def test_detect_with_custom_confidence(self, loaded_detector, sample_frame):
        """Test detection with custom confidence threshold."""
        results = loaded_detector.detect(sample_frame, conf=0.8)
        
        assert results is not None
    
    def test_detect_on_black_frame(self, loaded_detector, black_frame):
        """Test detection on a black frame."""
        results = loaded_detector.detect(black_frame)
        
        # Should return results (possibly empty)
        assert results is not None
    
    def test_detect_on_white_frame(self, loaded_detector, white_frame):
        """Test detection on a white frame."""
        results = loaded_detector.detect(white_frame)
        
        assert results is not None
    
    def test_get_class_names(self, loaded_detector):
        """Test getting class names dict."""
        class_names = loaded_detector.get_class_names()
        assert class_names[0] == 'person'
        assert class_names[2] == 'car'
    
    def test_get_class_names_contains_all(self, loaded_detector):
        """Test that class names contains 80 COCO classes."""
        class_names = loaded_detector.get_class_names()
        assert len(class_names) == 80


class TestObjectDetectorParsing:
    """Tests for parsing detection results."""
    
    @pytest.fixture
    def detector_with_model(self):
        """Create a detector with loaded model."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        detector.load_model()
        return detector
    
    def test_get_detections_format(self, detector_with_model, sample_frame):
        """Test that parsed results have correct format."""
        results = detector_with_model.detect(sample_frame)
        detections = detector_with_model.get_detections(results)
        
        assert isinstance(detections, list)
        
        if len(detections) > 0:
            detection = detections[0]
            assert 'class_id' in detection
            assert 'class_name' in detection
            assert 'confidence' in detection
            assert 'bbox' in detection
    
    def test_get_detections_bbox_format(self, detector_with_model, sample_frame):
        """Test that bounding boxes have correct format."""
        results = detector_with_model.detect(sample_frame)
        detections = detector_with_model.get_detections(results)
        
        if len(detections) > 0:
            bbox = detections[0]['bbox']
            assert len(bbox) == 4  # x1, y1, x2, y2
            assert all(isinstance(v, (int, float)) for v in bbox)
    
    def test_get_detections_confidence_range(self, detector_with_model, sample_frame):
        """Test that confidence values are in valid range."""
        results = detector_with_model.detect(sample_frame)
        detections = detector_with_model.get_detections(results)
        
        for detection in detections:
            assert 0.0 <= detection['confidence'] <= 1.0


class TestObjectDetectorEdgeCases:
    """Edge case tests for ObjectDetector."""
    
    def test_detect_before_load(self, sample_frame):
        """Test detection before model is loaded."""
        detector = ObjectDetector()
        
        with pytest.raises((AttributeError, Exception)):
            detector.detect(sample_frame)
    
    def test_load_invalid_model(self):
        """Test loading an invalid model name."""
        detector = ObjectDetector(model_name="invalid_model.pt")
        success = detector.load_model()
        
        # Should either return False or raise an exception
        # depending on implementation
        assert success is False or detector.model is None
    
    def test_detect_wrong_input_type(self):
        """Test detection with wrong input type."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        detector.load_model()
        
        with pytest.raises((TypeError, Exception)):
            detector.detect("not an image")
    
    def test_detect_single_channel(self):
        """Test detection with grayscale image."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        detector.load_model()
        
        grayscale = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        
        # YOLO might handle this or raise an error
        try:
            results = detector.detect(grayscale)
            assert results is not None
        except Exception:
            pass  # Expected if model requires 3 channels


class TestObjectDetectorPerformance:
    """Performance-related tests for ObjectDetector."""
    
    def test_detect_multiple_frames(self, sample_frame):
        """Test detection on multiple frames."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        detector.load_model()
        
        for _ in range(5):
            results = detector.detect(sample_frame)
            assert results is not None
    
    def test_detect_different_resolutions(self):
        """Test detection on different resolutions."""
        detector = ObjectDetector(model_name="yolov8n.pt")
        detector.load_model()
        
        resolutions = [(320, 240), (640, 480), (1280, 720)]
        
        for width, height in resolutions:
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            results = detector.detect(frame)
            assert results is not None
