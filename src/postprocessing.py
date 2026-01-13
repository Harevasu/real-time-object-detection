"""
Post-processing Module
Handles filtering, NMS, and detection result processing.
"""

import sys
import numpy as np

sys.path.append('..')
from config import settings


class PostProcessor:
    """
    Handles post-processing of detection results.
    
    Note: YOLO applies NMS internally, but this module provides
    additional filtering and processing utilities.
    """
    
    def __init__(self, confidence_threshold=None, nms_threshold=None, max_detections=None):
        """
        Initialize the post-processor.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections to keep
        """
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self.nms_threshold = nms_threshold or settings.NMS_THRESHOLD
        self.max_detections = max_detections or settings.MAX_DETECTIONS
    
    def filter_by_confidence(self, detections, threshold=None):
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detection dictionaries
            threshold: Confidence threshold (optional)
            
        Returns:
            Filtered list of detections
        """
        thresh = threshold if threshold is not None else self.confidence_threshold
        return [d for d in detections if d['confidence'] >= thresh]
    
    def filter_by_classes(self, detections, allowed_classes):
        """
        Filter detections to only include specific classes.
        
        Args:
            detections: List of detection dictionaries
            allowed_classes: List of allowed class names or IDs
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for d in detections:
            if d['class_name'] in allowed_classes or d['class_id'] in allowed_classes:
                filtered.append(d)
        return filtered
    
    def filter_by_size(self, detections, min_area=None, max_area=None):
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of detection dictionaries
            min_area: Minimum box area in pixels
            max_area: Maximum box area in pixels
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            area = (x2 - x1) * (y2 - y1)
            
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
                
            filtered.append(d)
        return filtered
    
    def limit_detections(self, detections, max_count=None):
        """
        Limit the number of detections (keep highest confidence).
        
        Args:
            detections: List of detection dictionaries
            max_count: Maximum number to keep
            
        Returns:
            Limited list of detections
        """
        max_n = max_count if max_count is not None else self.max_detections
        
        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        return sorted_detections[:max_n]
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value (0-1)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def apply_nms(self, detections, iou_threshold=None):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Note: YOLO applies NMS internally, this is for additional filtering.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        thresh = iou_threshold if iou_threshold is not None else self.nms_threshold
        
        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            
            # Remove overlapping boxes
            sorted_dets = [
                d for d in sorted_dets 
                if self.calculate_iou(best['bbox'], d['bbox']) < thresh
            ]
        
        return keep
    
    def group_by_class(self, detections):
        """
        Group detections by class.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary mapping class_name to list of detections
        """
        grouped = {}
        for d in detections:
            class_name = d['class_name']
            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(d)
        return grouped
    
    def count_objects(self, detections):
        """
        Count objects by class.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary mapping class_name to count
        """
        counts = {}
        for d in detections:
            class_name = d['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def process(self, detections, allowed_classes=None, min_area=None):
        """
        Apply full post-processing pipeline.
        
        Args:
            detections: List of detection dictionaries
            allowed_classes: List of classes to keep (optional)
            min_area: Minimum box area (optional)
            
        Returns:
            Processed list of detections
        """
        result = detections
        
        # Filter by confidence
        result = self.filter_by_confidence(result)
        
        # Filter by classes if specified
        if allowed_classes:
            result = self.filter_by_classes(result, allowed_classes)
        
        # Filter by size if specified
        if min_area:
            result = self.filter_by_size(result, min_area=min_area)
        
        # Limit number of detections
        result = self.limit_detections(result)
        
        return result


# Test the module
if __name__ == "__main__":
    print("Testing Post-processing Module...")
    
    # Create test detections
    test_detections = [
        {'bbox': [100, 100, 200, 200], 'confidence': 0.95, 'class_id': 0, 'class_name': 'person'},
        {'bbox': [105, 105, 205, 205], 'confidence': 0.85, 'class_id': 0, 'class_name': 'person'},
        {'bbox': [300, 300, 400, 400], 'confidence': 0.75, 'class_id': 2, 'class_name': 'car'},
        {'bbox': [500, 500, 550, 550], 'confidence': 0.45, 'class_id': 16, 'class_name': 'dog'},
    ]
    
    processor = PostProcessor(confidence_threshold=0.5)
    
    # Test filtering
    filtered = processor.filter_by_confidence(test_detections)
    print(f"After confidence filter: {len(filtered)} detections")
    
    # Test NMS
    nms_result = processor.apply_nms(filtered, iou_threshold=0.5)
    print(f"After NMS: {len(nms_result)} detections")
    
    # Test counting
    counts = processor.count_objects(filtered)
    print(f"Object counts: {counts}")
    
    # Test full pipeline
    processed = processor.process(test_detections)
    print(f"After full pipeline: {len(processed)} detections")
    
    print("Post-processing module test complete!")
