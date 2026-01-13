"""
Visualization Module
Handles drawing bounding boxes, labels, and FPS on frames.
"""

import cv2
import sys
import time
import numpy as np

sys.path.append('..')
from config import settings


class Visualizer:
    """
    Handles visualization of detection results on frames.
    """
    
    def __init__(self):
        """Initialize the visualizer with settings."""
        self.box_color = settings.BOX_COLOR
        self.text_color = settings.TEXT_COLOR
        self.text_bg_color = settings.TEXT_BG_COLOR
        self.box_thickness = settings.BOX_THICKNESS
        self.font_scale = settings.FONT_SCALE
        self.font_thickness = settings.FONT_THICKNESS
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_fps = settings.SHOW_FPS
        self.show_confidence = settings.SHOW_CONFIDENCE
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # Color palette for different classes
        self.class_colors = self._generate_colors(80)
    
    def _generate_colors(self, num_classes):
        """
        Generate distinct colors for each class.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            List of BGR color tuples
        """
        colors = []
        for i in range(num_classes):
            # Generate colors using HSV for better distribution
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(
                np.uint8([[[ hue, 255, 255 ]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors
    
    def get_class_color(self, class_id):
        """
        Get color for a specific class.
        
        Args:
            class_id: Class ID
            
        Returns:
            BGR color tuple
        """
        return self.class_colors[class_id % len(self.class_colors)]
    
    def draw_box(self, frame, bbox, color=None, thickness=None):
        """
        Draw a bounding box on the frame.
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2]
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with box drawn
        """
        color = color if color is not None else self.box_color
        thickness = thickness if thickness is not None else self.box_thickness
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        return frame
    
    def draw_label(self, frame, text, position, color=None, bg_color=None):
        """
        Draw a label with background at the specified position.
        
        Args:
            frame: Input frame
            text: Label text
            position: (x, y) top-left corner
            color: Text color (BGR)
            bg_color: Background color (BGR)
            
        Returns:
            Frame with label drawn
        """
        color = color if color is not None else self.text_color
        bg_color = bg_color if bg_color is not None else self.text_bg_color
        
        x, y = [int(v) for v in position]
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 5),
            (x + text_width + 5, y),
            bg_color,
            -1  # Filled
        )
        
        # Draw text
        cv2.putText(
            frame, text,
            (x + 2, y - baseline - 2),
            self.font, self.font_scale, color, self.font_thickness
        )
        
        return frame
    
    def draw_detection(self, frame, detection, use_class_color=True):
        """
        Draw a complete detection (box + label) on the frame.
        
        Args:
            frame: Input frame
            detection: Detection dictionary with bbox, class_name, confidence
            use_class_color: Use unique color per class
            
        Returns:
            Frame with detection drawn
        """
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        class_id = detection.get('class_id', 0)
        
        # Get color
        if use_class_color:
            color = self.get_class_color(class_id)
        else:
            color = self.box_color
        
        # Draw bounding box
        self.draw_box(frame, bbox, color=color)
        
        # Prepare label text
        if self.show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        # Draw label
        self.draw_label(frame, label, (bbox[0], bbox[1]), bg_color=color)
        
        return frame
    
    def draw_detections(self, frame, detections, use_class_color=True):
        """
        Draw all detections on the frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            use_class_color: Use unique color per class
            
        Returns:
            Frame with all detections drawn
        """
        for detection in detections:
            self.draw_detection(frame, detection, use_class_color)
        
        return frame
    
    def update_fps(self):
        """
        Update FPS calculation.
        
        Returns:
            Current FPS value
        """
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:  # Update every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        return self.current_fps
    
    def draw_fps(self, frame, fps=None):
        """
        Draw FPS counter on the frame.
        
        Args:
            frame: Input frame
            fps: FPS value (uses internal calculation if None)
            
        Returns:
            Frame with FPS drawn
        """
        if fps is None:
            fps = self.current_fps
        
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw in top-right corner
        text_size = cv2.getTextSize(fps_text, self.font, self.font_scale, self.font_thickness)[0]
        x = frame.shape[1] - text_size[0] - 10
        y = 30
        
        # Draw background
        cv2.rectangle(
            frame,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, fps_text,
            (x, y),
            self.font, self.font_scale, (0, 255, 0), self.font_thickness
        )
        
        return frame
    
    def draw_info(self, frame, info_text, position=(10, 30)):
        """
        Draw info text on the frame.
        
        Args:
            frame: Input frame
            info_text: Text to display
            position: (x, y) position
            
        Returns:
            Frame with info drawn
        """
        cv2.putText(
            frame, info_text,
            position,
            self.font, self.font_scale, (255, 255, 255), self.font_thickness
        )
        return frame
    
    def draw_object_count(self, frame, counts):
        """
        Draw object counts on the frame.
        
        Args:
            frame: Input frame
            counts: Dictionary of class_name to count
            
        Returns:
            Frame with counts drawn
        """
        y_offset = 60
        for class_name, count in counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(
                frame, text,
                (10, y_offset),
                self.font, 0.5, (255, 255, 255), 1
            )
            y_offset += 20
        
        return frame
    
    def visualize(self, frame, detections, show_fps=True, show_counts=False):
        """
        Apply full visualization pipeline.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            show_fps: Whether to show FPS
            show_counts: Whether to show object counts
            
        Returns:
            Visualized frame
        """
        result = frame.copy()
        
        # Draw detections
        self.draw_detections(result, detections)
        
        # Update and draw FPS
        if show_fps and self.show_fps:
            self.update_fps()
            self.draw_fps(result)
        
        # Draw object counts
        if show_counts:
            counts = {}
            for d in detections:
                name = d['class_name']
                counts[name] = counts.get(name, 0) + 1
            self.draw_object_count(result, counts)
        
        return result


# Test the module
if __name__ == "__main__":
    import numpy as np
    
    print("Testing Visualization Module...")
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 50, 50)  # Gray background
    
    # Create test detections
    test_detections = [
        {'bbox': [100, 100, 250, 300], 'confidence': 0.95, 'class_id': 0, 'class_name': 'person'},
        {'bbox': [300, 150, 500, 350], 'confidence': 0.87, 'class_id': 2, 'class_name': 'car'},
        {'bbox': [450, 50, 550, 150], 'confidence': 0.72, 'class_id': 16, 'class_name': 'dog'},
    ]
    
    visualizer = Visualizer()
    
    # Test visualization
    result = visualizer.visualize(test_frame, test_detections, show_counts=True)
    
    # Display result
    cv2.imshow("Visualization Test", result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Visualization module test complete!")
