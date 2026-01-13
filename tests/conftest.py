"""
Pytest fixtures and configuration for tests.
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame for testing."""
    # Create a 640x480 BGR frame with random colors
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_frame_720p():
    """Create a 720p BGR frame for testing."""
    frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def black_frame():
    """Create a black frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """Create a white frame for testing."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def sample_detections():
    """Create sample detection results for testing."""
    return [
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 300]
        },
        {
            'class_id': 2,
            'class_name': 'car',
            'confidence': 0.85,
            'bbox': [300, 150, 500, 350]
        },
        {
            'class_id': 16,
            'class_name': 'dog',
            'confidence': 0.45,
            'bbox': [50, 200, 150, 350]
        },
        {
            'class_id': 67,
            'class_name': 'cell phone',
            'confidence': 0.72,
            'bbox': [400, 50, 450, 120]
        }
    ]


@pytest.fixture
def high_confidence_detections():
    """Create high-confidence detection results."""
    return [
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.98,
            'bbox': [100, 100, 200, 300]
        },
        {
            'class_id': 1,
            'class_name': 'bicycle',
            'confidence': 0.92,
            'bbox': [250, 200, 400, 400]
        }
    ]


@pytest.fixture
def overlapping_detections():
    """Create overlapping detections for NMS testing."""
    return [
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 300]
        },
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': [110, 105, 205, 295]  # Overlapping with first
        },
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.75,
            'bbox': [105, 102, 198, 298]  # Overlapping with first
        }
    ]


@pytest.fixture
def empty_detections():
    """Create empty detection list."""
    return []
