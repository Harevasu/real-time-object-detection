"""
Unit tests for Image Preprocessing Module.
"""

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import Preprocessor


class TestPreprocessor:
    """Tests for Preprocessor class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        pp = Preprocessor()
        assert pp.target_size == (640, 640)
    
    def test_init_custom_size(self):
        """Test initialization with custom target size."""
        pp = Preprocessor(target_size=(320, 320))
        assert pp.target_size == (320, 320)
    
    def test_resize_to_default(self, sample_frame):
        """Test resizing to default target size."""
        pp = Preprocessor()
        resized = pp.resize(sample_frame)
        assert resized.shape == (640, 640, 3)
    
    def test_resize_to_custom(self, sample_frame):
        """Test resizing to custom size."""
        pp = Preprocessor()
        resized = pp.resize(sample_frame, size=(320, 240))
        assert resized.shape == (240, 320, 3)  # Height, Width, Channels
    
    def test_resize_maintains_dtype(self, sample_frame):
        """Test that resize maintains data type."""
        pp = Preprocessor()
        resized = pp.resize(sample_frame)
        assert resized.dtype == sample_frame.dtype
    
    def test_bgr_to_rgb(self, sample_frame):
        """Test BGR to RGB conversion."""
        pp = Preprocessor()
        rgb = pp.bgr_to_rgb(sample_frame)
        
        # Check shape is same
        assert rgb.shape == sample_frame.shape
        
        # Check channels are swapped (B and R swapped)
        assert np.array_equal(rgb[:, :, 0], sample_frame[:, :, 2])  # R = B
        assert np.array_equal(rgb[:, :, 2], sample_frame[:, :, 0])  # B = R
        assert np.array_equal(rgb[:, :, 1], sample_frame[:, :, 1])  # G = G
    
    def test_rgb_to_bgr(self, sample_frame):
        """Test RGB to BGR conversion."""
        pp = Preprocessor()
        # Convert to RGB first
        rgb = pp.bgr_to_rgb(sample_frame)
        # Convert back to BGR
        bgr = pp.rgb_to_bgr(rgb)
        
        # Should be same as original
        assert np.array_equal(bgr, sample_frame)
    
    def test_bgr_rgb_roundtrip(self, sample_frame):
        """Test BGR -> RGB -> BGR roundtrip."""
        pp = Preprocessor()
        rgb = pp.bgr_to_rgb(sample_frame)
        bgr = pp.rgb_to_bgr(rgb)
        assert np.array_equal(bgr, sample_frame)


class TestPreprocessorNormalization:
    """Tests for normalization functions."""
    
    def test_normalize_default(self, sample_frame):
        """Test normalization with default range."""
        pp = Preprocessor()
        normalized = pp.normalize(sample_frame)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_normalize_output_type(self, sample_frame):
        """Test normalization output type."""
        pp = Preprocessor()
        normalized = pp.normalize(sample_frame)
        
        assert normalized.dtype == np.float32
    
    def test_denormalize(self, sample_frame):
        """Test denormalization."""
        pp = Preprocessor()
        normalized = pp.normalize(sample_frame)
        denormalized = pp.denormalize(normalized)
        
        assert denormalized.dtype == np.uint8
        # Allow small differences due to float conversion
        assert np.allclose(denormalized, sample_frame, atol=1)


class TestPreprocessorFilters:
    """Tests for image filtering functions."""
    
    def test_apply_gaussian_blur(self, sample_frame):
        """Test Gaussian blur application."""
        pp = Preprocessor()
        blurred = pp.apply_gaussian_blur(sample_frame, kernel_size=(5, 5))
        
        assert blurred.shape == sample_frame.shape
        assert blurred.dtype == sample_frame.dtype
    
    def test_apply_gaussian_blur_different_kernels(self, sample_frame):
        """Test Gaussian blur with different kernel sizes."""
        pp = Preprocessor()
        
        blur3 = pp.apply_gaussian_blur(sample_frame, kernel_size=(3, 3))
        blur7 = pp.apply_gaussian_blur(sample_frame, kernel_size=(7, 7))
        
        # Different kernel sizes should produce different results
        assert not np.array_equal(blur3, blur7)
    
    def test_enhance_contrast(self, sample_frame):
        """Test contrast enhancement function."""
        pp = Preprocessor()
        enhanced = pp.enhance_contrast(sample_frame)
        
        assert enhanced.shape == sample_frame.shape
        assert enhanced.dtype == sample_frame.dtype


class TestPreprocessorPipeline:
    """Tests for preprocessing pipeline."""
    
    def test_preprocess_default(self, sample_frame):
        """Test default preprocessing."""
        pp = Preprocessor()
        processed = pp.preprocess(sample_frame)
        
        assert processed.shape == sample_frame.shape
    
    def test_preprocess_with_blur(self, sample_frame):
        """Test preprocessing with blur."""
        pp = Preprocessor()
        processed = pp.preprocess(sample_frame, apply_blur=True)
        
        assert processed.shape == sample_frame.shape
    
    def test_preprocess_with_enhance(self, sample_frame):
        """Test preprocessing with contrast enhancement."""
        pp = Preprocessor()
        processed = pp.preprocess(sample_frame, enhance=True)
        
        assert processed.shape == sample_frame.shape


class TestPreprocessorEdgeCases:
    """Edge case tests for Preprocessor."""
    
    def test_resize_single_pixel(self):
        """Test resizing a single pixel image."""
        pp = Preprocessor()
        single_pixel = np.array([[[255, 128, 64]]], dtype=np.uint8)
        resized = pp.resize(single_pixel, size=(10, 10))
        
        assert resized.shape == (10, 10, 3)
    
    def test_resize_large_image(self):
        """Test resizing a large image."""
        pp = Preprocessor()
        large_frame = np.random.randint(0, 256, (4000, 6000, 3), dtype=np.uint8)
        resized = pp.resize(large_frame, size=(640, 480))
        
        assert resized.shape == (480, 640, 3)
    
    def test_grayscale_input(self):
        """Test handling of grayscale input."""
        pp = Preprocessor()
        grayscale = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        
        # Resize should still work
        resized = pp.resize(grayscale, size=(320, 240))
        assert resized.shape == (240, 320)
    
    def test_empty_frame(self):
        """Test handling of empty frame."""
        pp = Preprocessor()
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((cv2.error, Exception)):
            pp.resize(empty)
