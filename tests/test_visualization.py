"""
Unit tests for Visualization Module.
"""

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import Visualizer


class TestVisualizerInit:
    """Tests for Visualizer initialization."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        viz = Visualizer()
        
        assert viz.box_color is not None
        assert viz.text_color is not None
        assert viz.font is not None
        assert viz.class_colors is not None
    
    def test_init_class_colors(self):
        """Test that class colors are generated."""
        viz = Visualizer()
        
        assert len(viz.class_colors) == 80  # COCO classes
        for color in viz.class_colors:
            assert len(color) == 3  # BGR
            assert all(0 <= c <= 255 for c in color)
    
    def test_init_fps_tracking(self):
        """Test FPS tracking initialization."""
        viz = Visualizer()
        
        assert viz.fps_start_time > 0
        assert viz.fps_frame_count == 0
        assert viz.current_fps == 0


class TestVisualizerColors:
    """Tests for color handling."""
    
    def test_get_class_color_valid(self):
        """Test getting color for valid class ID."""
        viz = Visualizer()
        
        color = viz.get_class_color(0)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    def test_get_class_color_wrap_around(self):
        """Test color wraps around for high class IDs."""
        viz = Visualizer()
        
        # Class ID larger than number of colors
        color = viz.get_class_color(150)
        assert len(color) == 3
    
    def test_different_classes_different_colors(self):
        """Test that different classes have different colors."""
        viz = Visualizer()
        
        color0 = viz.get_class_color(0)
        color1 = viz.get_class_color(1)
        color2 = viz.get_class_color(2)
        
        # At least some colors should be different
        assert color0 != color1 or color1 != color2


class TestVisualizerDrawBox:
    """Tests for bounding box drawing."""
    
    def test_draw_box(self, sample_frame):
        """Test drawing a bounding box."""
        viz = Visualizer()
        bbox = [100, 100, 200, 200]
        
        result = viz.draw_box(sample_frame.copy(), bbox)
        
        assert result is not None
        assert result.shape == sample_frame.shape
    
    def test_draw_box_custom_color(self, sample_frame):
        """Test drawing box with custom color."""
        viz = Visualizer()
        bbox = [100, 100, 200, 200]
        color = (255, 0, 0)  # Blue
        
        result = viz.draw_box(sample_frame.copy(), bbox, color=color)
        
        assert result is not None
    
    def test_draw_box_custom_thickness(self, sample_frame):
        """Test drawing box with custom thickness."""
        viz = Visualizer()
        bbox = [100, 100, 200, 200]
        
        result = viz.draw_box(sample_frame.copy(), bbox, thickness=5)
        
        assert result is not None
    
    def test_draw_box_edge_of_frame(self, sample_frame):
        """Test drawing box at edge of frame."""
        viz = Visualizer()
        h, w = sample_frame.shape[:2]
        bbox = [w - 50, h - 50, w, h]  # Bottom-right corner
        
        result = viz.draw_box(sample_frame.copy(), bbox)
        
        assert result is not None
    
    def test_draw_box_float_coordinates(self, sample_frame):
        """Test drawing box with float coordinates."""
        viz = Visualizer()
        bbox = [100.5, 100.7, 200.3, 200.9]
        
        result = viz.draw_box(sample_frame.copy(), bbox)
        
        assert result is not None


class TestVisualizerDrawLabel:
    """Tests for label drawing."""
    
    def test_draw_label(self, sample_frame):
        """Test drawing a label."""
        viz = Visualizer()
        
        result = viz.draw_label(
            sample_frame.copy(),
            "person 0.95",
            (100, 100)
        )
        
        assert result is not None
        assert result.shape == sample_frame.shape
    
    def test_draw_label_custom_colors(self, sample_frame):
        """Test drawing label with custom colors."""
        viz = Visualizer()
        
        result = viz.draw_label(
            sample_frame.copy(),
            "car 0.85",
            (100, 100),
            color=(255, 255, 255),
            bg_color=(0, 0, 255)
        )
        
        assert result is not None
    
    def test_draw_label_long_text(self, sample_frame):
        """Test drawing label with long text."""
        viz = Visualizer()
        
        result = viz.draw_label(
            sample_frame.copy(),
            "This is a very long class name with confidence 0.99",
            (100, 100)
        )
        
        assert result is not None
    
    def test_draw_label_edge_position(self, sample_frame):
        """Test drawing label near edge of frame."""
        viz = Visualizer()
        
        result = viz.draw_label(
            sample_frame.copy(),
            "person 0.95",
            (10, 10)  # Near top-left
        )
        
        assert result is not None


class TestVisualizerDrawDetections:
    """Tests for drawing complete detections."""
    
    def test_draw_detections(self, sample_frame, sample_detections):
        """Test drawing multiple detections."""
        viz = Visualizer()
        
        result = viz.draw_detections(sample_frame.copy(), sample_detections)
        
        assert result is not None
        assert result.shape == sample_frame.shape
    
    def test_draw_detections_empty(self, sample_frame, empty_detections):
        """Test drawing with no detections."""
        viz = Visualizer()
        
        result = viz.draw_detections(sample_frame.copy(), empty_detections)
        
        assert result is not None
        # Frame should be unchanged
        assert np.array_equal(result, sample_frame)
    
    def test_draw_detections_single(self, sample_frame):
        """Test drawing single detection."""
        viz = Visualizer()
        single = [{
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.95,
            'bbox': [100, 100, 200, 300]
        }]
        
        result = viz.draw_detections(sample_frame.copy(), single)
        
        assert result is not None
    
    def test_draw_detections_with_colors(self, sample_frame, sample_detections):
        """Test that different classes have different colors."""
        viz = Visualizer()
        
        result = viz.draw_detections(
            sample_frame.copy(),
            sample_detections,
            use_class_color=True
        )
        
        assert result is not None


class TestVisualizerFPS:
    """Tests for FPS display."""
    
    def test_draw_fps(self, sample_frame):
        """Test drawing FPS counter."""
        viz = Visualizer()
        
        result = viz.draw_fps(sample_frame.copy(), fps=30.0)
        
        assert result is not None
    
    def test_draw_fps_zero(self, sample_frame):
        """Test drawing zero FPS."""
        viz = Visualizer()
        
        result = viz.draw_fps(sample_frame.copy(), fps=0.0)
        
        assert result is not None
    
    def test_draw_fps_high_value(self, sample_frame):
        """Test drawing high FPS value."""
        viz = Visualizer()
        
        result = viz.draw_fps(sample_frame.copy(), fps=120.5)
        
        assert result is not None
    
    def test_update_fps(self):
        """Test FPS update calculation."""
        viz = Visualizer()
        
        # Simulate frame processing
        for _ in range(30):
            viz.update_fps()
        
        # FPS should be calculated
        assert viz.fps_frame_count > 0


class TestVisualizerObjectCounts:
    """Tests for object count display."""
    
    def test_draw_object_count(self, sample_frame, sample_detections):
        """Test drawing object counts."""
        viz = Visualizer()
        counts = {'person': 2, 'car': 1, 'dog': 1}
        
        result = viz.draw_object_count(sample_frame.copy(), counts)
        
        assert result is not None
    
    def test_draw_object_count_empty(self, sample_frame):
        """Test drawing with no objects."""
        viz = Visualizer()
        
        result = viz.draw_object_count(sample_frame.copy(), {})
        
        assert result is not None
    
    def test_draw_object_count_many(self, sample_frame):
        """Test drawing many object counts."""
        viz = Visualizer()
        counts = {
            'person': 10,
            'car': 5,
            'dog': 3,
            'cat': 2,
            'bird': 1
        }
        
        result = viz.draw_object_count(sample_frame.copy(), counts)
        
        assert result is not None


class TestVisualizerInfo:
    """Tests for info display."""
    
    def test_draw_info(self, sample_frame):
        """Test drawing info text."""
        viz = Visualizer()
        
        result = viz.draw_info(
            sample_frame.copy(),
            "Press 'q' to quit",
            position=(10, 30)
        )
        
        assert result is not None
    
    def test_draw_info_different_positions(self, sample_frame):
        """Test drawing info at different positions."""
        viz = Visualizer()
        positions = [(10, 30), (100, 100), (200, 200)]
        
        for pos in positions:
            result = viz.draw_info(sample_frame.copy(), "Test", position=pos)
            assert result is not None


class TestVisualizerComplete:
    """Tests for complete frame visualization."""
    
    def test_draw_all_elements(self, sample_frame, sample_detections):
        """Test drawing all elements together."""
        viz = Visualizer()
        
        frame = sample_frame.copy()
        frame = viz.draw_detections(frame, sample_detections)
        frame = viz.draw_fps(frame, fps=30.0)
        
        assert frame is not None
        assert frame.shape == sample_frame.shape
    
    def test_draw_detections_only(self, sample_frame, sample_detections):
        """Test drawing detections only."""
        viz = Visualizer()
        
        result = viz.draw_detections(
            sample_frame.copy(),
            sample_detections
        )
        
        assert result is not None
    
    def test_draw_fps_only(self, sample_frame):
        """Test drawing FPS only."""
        viz = Visualizer()
        
        result = viz.draw_fps(
            sample_frame.copy(),
            fps=30.0
        )
        
        assert result is not None


class TestVisualizerEdgeCases:
    """Edge case tests for Visualizer."""
    
    def test_draw_box_negative_coords(self, sample_frame):
        """Test drawing box with negative coordinates."""
        viz = Visualizer()
        bbox = [-10, -10, 100, 100]  # Partially outside frame
        
        # Should handle gracefully
        result = viz.draw_box(sample_frame.copy(), bbox)
        assert result is not None
    
    def test_draw_box_outside_frame(self, sample_frame):
        """Test drawing box completely outside frame."""
        viz = Visualizer()
        h, w = sample_frame.shape[:2]
        bbox = [w + 100, h + 100, w + 200, h + 200]
        
        result = viz.draw_box(sample_frame.copy(), bbox)
        assert result is not None
    
    def test_draw_zero_size_box(self, sample_frame):
        """Test drawing zero-size box."""
        viz = Visualizer()
        bbox = [100, 100, 100, 100]  # Zero width and height
        
        result = viz.draw_box(sample_frame.copy(), bbox)
        assert result is not None
    
    def test_draw_on_grayscale(self):
        """Test drawing on grayscale image."""
        viz = Visualizer()
        grayscale = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        
        # Convert to 3-channel for drawing
        frame = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        bbox = [100, 100, 200, 200]
        
        result = viz.draw_box(frame, bbox)
        assert result is not None
    
    def test_draw_label_empty_string(self, sample_frame):
        """Test drawing empty label."""
        viz = Visualizer()
        
        result = viz.draw_label(sample_frame.copy(), "", (100, 100))
        assert result is not None
    
    def test_many_detections_performance(self, sample_frame):
        """Test performance with many detections."""
        viz = Visualizer()
        
        # Create 100 detections
        detections = []
        for i in range(100):
            detections.append({
                'class_id': i % 80,
                'class_name': f'class_{i}',
                'confidence': 0.9,
                'bbox': [i * 5, i * 3, i * 5 + 50, i * 3 + 80]
            })
        
        result = viz.draw_detections(sample_frame.copy(), detections)
        assert result is not None
