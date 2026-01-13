"""
Object Detection Module
Handles YOLO model loading and inference.
"""

import os
import sys
from ultralytics import YOLO

sys.path.append('..')
from config import settings


class ObjectDetector:
    """
    YOLO-based object detector.
    
    Attributes:
        model: YOLO model instance
        model_name: Name of the model file
        device: Device to run inference on (cpu/cuda)
        confidence: Confidence threshold for detections
    """
    
    def __init__(self, model_name=None, device=None, confidence=None):
        """
        Initialize the object detector.
        
        Args:
            model_name: YOLO model name (default from settings)
            device: Device for inference (default from settings)
            confidence: Confidence threshold (default from settings)
        """
        self.model_name = model_name if model_name else settings.MODEL_NAME
        self.device = device if device else settings.DEVICE
        self.confidence = confidence if confidence else settings.CONFIDENCE_THRESHOLD
        self.model = None
        self.class_names = None
        
    def load_model(self):
        """
        Load the YOLO model.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            print(f"Loading model: {self.model_name}...")
            
            # Check if model exists locally in models directory
            model_path = os.path.join(settings.MODELS_DIR, self.model_name)
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Model loaded from: {model_path}")
            else:
                # Download model (will be saved automatically)
                self.model = YOLO(self.model_name)
                print(f"Model downloaded: {self.model_name}")
            
            # Get class names
            self.class_names = self.model.names
            print(f"Model loaded successfully with {len(self.class_names)} classes")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, frame, conf=None):
        """
        Run object detection on a frame.
        
        Args:
            frame: Input frame (BGR numpy array)
            conf: Confidence threshold (optional)
            
        Returns:
            YOLO Results object
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        confidence = conf if conf is not None else self.confidence
        
        # Run inference
        results = self.model(
            frame,
            conf=confidence,
            device=self.device,
            verbose=False  # Suppress output
        )
        
        return results[0]  # Return first result (single frame)
    
    def get_detections(self, results):
        """
        Extract detection data from YOLO results.
        
        Args:
            results: YOLO Results object
            
        Returns:
            list: List of detection dictionaries with keys:
                  - bbox: [x1, y1, x2, y2]
                  - confidence: float
                  - class_id: int
                  - class_name: str
        """
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections
        
        boxes = results.boxes
        
        for i in range(len(boxes)):
            # Get bounding box coordinates
            bbox = boxes.xyxy[i].cpu().numpy().astype(int)
            
            # Get confidence score
            conf = float(boxes.conf[i].cpu().numpy())
            
            # Get class ID and name
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = self.class_names[class_id]
            
            detections.append({
                'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                'confidence': conf,
                'class_id': class_id,
                'class_name': class_name
            })
        
        return detections
    
    def get_class_names(self):
        """
        Get the list of class names.
        
        Returns:
            dict: Class ID to name mapping
        """
        return self.class_names


# Test the module
if __name__ == "__main__":
    import cv2
    import numpy as np
    
    print("Testing Object Detection Module...")
    
    detector = ObjectDetector()
    
    if detector.load_model():
        # Create a test image or use camera
        print("\nClass names:", list(detector.get_class_names().values())[:10], "...")
        
        # Test with a blank image
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            results = detector.detect(test_frame)
            detections = detector.get_detections(results)
            print(f"\nDetections on blank image: {len(detections)}")
            print("Detection module test complete!")
        except Exception as e:
            print(f"Detection test error: {e}")
    else:
        print("Failed to load model")
