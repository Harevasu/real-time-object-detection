"""
Utility Helper Functions for Real-Time Object Detection System.

This module provides common utility functions used across the application
for file handling, formatting, calculations, and validation.
"""

import os
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: strftime format string
        
    Returns:
        Formatted timestamp string
        
    Example:
        >>> get_timestamp()
        '20260113_143052'
        >>> get_timestamp("%Y-%m-%d")
        '2026-01-13'
    """
    return datetime.now().strftime(format_str)


def format_fps(fps: float, decimals: int = 1) -> str:
    """
    Format FPS value for display.
    
    Args:
        fps: Frames per second value
        decimals: Number of decimal places
        
    Returns:
        Formatted FPS string
        
    Example:
        >>> format_fps(29.97)
        'FPS: 30.0'
    """
    return f"FPS: {fps:.{decimals}f}"


def format_detection_count(counts: Dict[str, int]) -> str:
    """
    Format detection counts for display.
    
    Args:
        counts: Dictionary of class names to counts
        
    Returns:
        Formatted string of detection counts
        
    Example:
        >>> format_detection_count({'person': 2, 'car': 1})
        'person: 2, car: 1'
    """
    if not counts:
        return "No detections"
    return ", ".join(f"{cls}: {count}" for cls, count in counts.items())


def calculate_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Area of the bounding box in pixels
        
    Example:
        >>> calculate_area((0, 0, 100, 100))
        10000.0
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box as (x1, y1, x2, y2)
        box2: Second bounding box as (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
        
    Example:
        >>> calculate_iou((0, 0, 100, 100), (50, 50, 150, 150))
        0.14285714285714285
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    if intersection == 0:
        return 0.0
    
    # Calculate union
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def ensure_directory(path: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        The path that was ensured to exist
        
    Example:
        >>> ensure_directory("outputs/screenshots")
        'outputs/screenshots'
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_model_path(model_name: str, models_dir: str = "models") -> str:
    """
    Get full path to model file.
    
    Args:
        model_name: Name of the model file
        models_dir: Directory containing models
        
    Returns:
        Full path to the model
        
    Example:
        >>> get_model_path("yolov8n.pt")
        'models/yolov8n.pt'
    """
    # If already a full path, return as-is
    if os.path.exists(model_name):
        return model_name
    
    # Check in models directory
    model_path = os.path.join(models_dir, model_name)
    if os.path.exists(model_path):
        return model_path
    
    # Return original name (YOLO will auto-download)
    return model_name


def get_output_path(prefix: str = "output", 
                    extension: str = ".mp4",
                    output_dir: str = "outputs") -> str:
    """
    Generate output file path with timestamp.
    
    Args:
        prefix: Prefix for the filename
        extension: File extension (including dot)
        output_dir: Output directory
        
    Returns:
        Full path to output file
        
    Example:
        >>> get_output_path("recording", ".mp4")
        'outputs/recording_20260113_143052.mp4'
    """
    ensure_directory(output_dir)
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}{extension}"
    return os.path.join(output_dir, filename)


def format_time_elapsed(seconds: float) -> str:
    """
    Format elapsed time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (HH:MM:SS or MM:SS)
        
    Example:
        >>> format_time_elapsed(3661)
        '1:01:01'
        >>> format_time_elapsed(125)
        '2:05'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def validate_frame(frame: Any) -> bool:
    """
    Validate that frame is a valid image array.
    
    Args:
        frame: Image frame to validate
        
    Returns:
        True if frame is valid, False otherwise
        
    Example:
        >>> import numpy as np
        >>> validate_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        True
        >>> validate_frame(None)
        False
    """
    if frame is None:
        return False
    
    if not isinstance(frame, np.ndarray):
        return False
    
    # Check for empty frame
    if frame.size == 0:
        return False
    
    # Check for valid dimensions (at least 2D)
    if frame.ndim < 2:
        return False
    
    return True


def clamp_bbox(bbox: Tuple[float, float, float, float], 
               frame_width: int, 
               frame_height: int) -> Tuple[int, int, int, int]:
    """
    Clamp bounding box coordinates to frame dimensions.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Clamped bounding box as (x1, y1, x2, y2) integers
        
    Example:
        >>> clamp_bbox((-10, -10, 700, 500), 640, 480)
        (0, 0, 640, 480)
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), frame_width))
    y1 = max(0, min(int(y1), frame_height))
    x2 = max(0, min(int(x2), frame_width))
    y2 = max(0, min(int(y2), frame_height))
    return (x1, y1, x2, y2)


def scale_bbox(bbox: Tuple[float, float, float, float], 
               scale_x: float, 
               scale_y: float) -> Tuple[float, float, float, float]:
    """
    Scale bounding box coordinates.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        scale_x: Scale factor for x coordinates
        scale_y: Scale factor for y coordinates
        
    Returns:
        Scaled bounding box
        
    Example:
        >>> scale_bbox((100, 100, 200, 200), 0.5, 0.5)
        (50.0, 50.0, 100.0, 100.0)
    """
    x1, y1, x2, y2 = bbox
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def get_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get center point of bounding box.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Center point as (cx, cy)
        
    Example:
        >>> get_center((0, 0, 100, 100))
        (50.0, 50.0)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_dimensions(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get width and height of bounding box.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Dimensions as (width, height)
        
    Example:
        >>> get_dimensions((10, 20, 110, 220))
        (100.0, 200.0)
    """
    x1, y1, x2, y2 = bbox
    return (abs(x2 - x1), abs(y2 - y1))


def format_confidence(confidence: float, as_percentage: bool = True) -> str:
    """
    Format confidence score for display.
    
    Args:
        confidence: Confidence value between 0 and 1
        as_percentage: If True, display as percentage
        
    Returns:
        Formatted confidence string
        
    Example:
        >>> format_confidence(0.876)
        '87.6%'
        >>> format_confidence(0.876, as_percentage=False)
        '0.88'
    """
    if as_percentage:
        return f"{confidence * 100:.1f}%"
    else:
        return f"{confidence:.2f}"


def create_detection_summary(detections: List[Dict]) -> Dict[str, Any]:
    """
    Create summary of detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Summary dictionary with counts and statistics
        
    Example:
        >>> detections = [{'class_name': 'person', 'confidence': 0.9}]
        >>> create_detection_summary(detections)
        {'total': 1, 'by_class': {'person': 1}, 'avg_confidence': 0.9}
    """
    if not detections:
        return {
            'total': 0,
            'by_class': {},
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0
        }
    
    by_class = {}
    confidences = []
    
    for det in detections:
        class_name = det.get('class_name', 'unknown')
        confidence = det.get('confidence', 0.0)
        
        by_class[class_name] = by_class.get(class_name, 0) + 1
        confidences.append(confidence)
    
    return {
        'total': len(detections),
        'by_class': by_class,
        'avg_confidence': sum(confidences) / len(confidences),
        'max_confidence': max(confidences),
        'min_confidence': min(confidences)
    }
