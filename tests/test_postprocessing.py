"""
Unit tests for Post-processing Module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocessing import PostProcessor


class TestPostProcessorInit:
    """Tests for PostProcessor initialization."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        pp = PostProcessor()
        assert pp.confidence_threshold == 0.5
        assert pp.nms_threshold == 0.4
        assert pp.max_detections == 100
    
    def test_init_custom_confidence(self):
        """Test initialization with custom confidence threshold."""
        pp = PostProcessor(confidence_threshold=0.7)
        assert pp.confidence_threshold == 0.7
    
    def test_init_custom_nms(self):
        """Test initialization with custom NMS threshold."""
        pp = PostProcessor(nms_threshold=0.5)
        assert pp.nms_threshold == 0.5
    
    def test_init_custom_max_detections(self):
        """Test initialization with custom max detections."""
        pp = PostProcessor(max_detections=50)
        assert pp.max_detections == 50
    
    def test_init_all_custom(self):
        """Test initialization with all custom values."""
        pp = PostProcessor(
            confidence_threshold=0.8,
            nms_threshold=0.3,
            max_detections=25
        )
        assert pp.confidence_threshold == 0.8
        assert pp.nms_threshold == 0.3
        assert pp.max_detections == 25


class TestPostProcessorConfidenceFilter:
    """Tests for confidence filtering."""
    
    def test_filter_by_confidence_default(self, sample_detections):
        """Test filtering with default threshold."""
        pp = PostProcessor(confidence_threshold=0.5)
        filtered = pp.filter_by_confidence(sample_detections)
        
        # Should keep person (0.95), car (0.85), cell phone (0.72)
        # Should remove dog (0.45)
        assert len(filtered) == 3
        for d in filtered:
            assert d['confidence'] >= 0.5
    
    def test_filter_by_confidence_custom(self, sample_detections):
        """Test filtering with custom threshold."""
        pp = PostProcessor()
        filtered = pp.filter_by_confidence(sample_detections, threshold=0.8)
        
        # Should keep only person (0.95) and car (0.85)
        assert len(filtered) == 2
        for d in filtered:
            assert d['confidence'] >= 0.8
    
    def test_filter_by_confidence_high_threshold(self, sample_detections):
        """Test filtering with very high threshold."""
        pp = PostProcessor()
        filtered = pp.filter_by_confidence(sample_detections, threshold=0.99)
        
        # Should keep nothing
        assert len(filtered) == 0
    
    def test_filter_by_confidence_low_threshold(self, sample_detections):
        """Test filtering with very low threshold."""
        pp = PostProcessor()
        filtered = pp.filter_by_confidence(sample_detections, threshold=0.1)
        
        # Should keep all
        assert len(filtered) == 4
    
    def test_filter_by_confidence_empty_list(self, empty_detections):
        """Test filtering empty detection list."""
        pp = PostProcessor()
        filtered = pp.filter_by_confidence(empty_detections)
        
        assert len(filtered) == 0
        assert isinstance(filtered, list)


class TestPostProcessorClassFilter:
    """Tests for class-based filtering."""
    
    def test_filter_by_classes_names(self, sample_detections):
        """Test filtering by class names."""
        pp = PostProcessor()
        filtered = pp.filter_by_classes(sample_detections, ['person', 'car'])
        
        assert len(filtered) == 2
        for d in filtered:
            assert d['class_name'] in ['person', 'car']
    
    def test_filter_by_classes_ids(self, sample_detections):
        """Test filtering by class IDs."""
        pp = PostProcessor()
        filtered = pp.filter_by_classes(sample_detections, [0, 2])  # person, car
        
        assert len(filtered) == 2
    
    def test_filter_by_classes_single_class(self, sample_detections):
        """Test filtering for a single class."""
        pp = PostProcessor()
        filtered = pp.filter_by_classes(sample_detections, ['dog'])
        
        assert len(filtered) == 1
        assert filtered[0]['class_name'] == 'dog'
    
    def test_filter_by_classes_nonexistent(self, sample_detections):
        """Test filtering for nonexistent class."""
        pp = PostProcessor()
        filtered = pp.filter_by_classes(sample_detections, ['airplane'])
        
        assert len(filtered) == 0
    
    def test_filter_by_classes_empty_allowed(self, sample_detections):
        """Test filtering with empty allowed classes."""
        pp = PostProcessor()
        filtered = pp.filter_by_classes(sample_detections, [])
        
        assert len(filtered) == 0


class TestPostProcessorSizeFilter:
    """Tests for size-based filtering."""
    
    def test_filter_by_size_min_area(self, sample_detections):
        """Test filtering by minimum area."""
        pp = PostProcessor()
        filtered = pp.filter_by_size(sample_detections, min_area=10000)
        
        # Calculate areas and verify
        for d in filtered:
            x1, y1, x2, y2 = d['bbox']
            area = (x2 - x1) * (y2 - y1)
            assert area >= 10000
    
    def test_filter_by_size_max_area(self, sample_detections):
        """Test filtering by maximum area."""
        pp = PostProcessor()
        filtered = pp.filter_by_size(sample_detections, max_area=30000)
        
        for d in filtered:
            x1, y1, x2, y2 = d['bbox']
            area = (x2 - x1) * (y2 - y1)
            assert area <= 30000
    
    def test_filter_by_size_range(self, sample_detections):
        """Test filtering by area range."""
        pp = PostProcessor()
        filtered = pp.filter_by_size(sample_detections, min_area=5000, max_area=50000)
        
        for d in filtered:
            x1, y1, x2, y2 = d['bbox']
            area = (x2 - x1) * (y2 - y1)
            assert 5000 <= area <= 50000


class TestPostProcessorNMS:
    """Tests for Non-Maximum Suppression."""
    
    def test_apply_nms_overlapping(self, overlapping_detections):
        """Test NMS on overlapping detections."""
        pp = PostProcessor(nms_threshold=0.4)
        result = pp.apply_nms(overlapping_detections)
        
        # Should keep highest confidence and suppress overlapping
        assert len(result) < len(overlapping_detections)
    
    def test_apply_nms_non_overlapping(self, sample_detections):
        """Test NMS on non-overlapping detections."""
        pp = PostProcessor(nms_threshold=0.4)
        result = pp.apply_nms(sample_detections)
        
        # Non-overlapping boxes should all be kept
        assert len(result) == len(sample_detections)
    
    def test_apply_nms_high_threshold(self, overlapping_detections):
        """Test NMS with high threshold (less suppression)."""
        pp = PostProcessor(nms_threshold=0.9)
        result = pp.apply_nms(overlapping_detections)
        
        # High threshold means less suppression
        assert len(result) >= 1
    
    def test_apply_nms_low_threshold(self, overlapping_detections):
        """Test NMS with low threshold (more suppression)."""
        pp = PostProcessor(nms_threshold=0.1)
        result = pp.apply_nms(overlapping_detections)
        
        # Low threshold means more suppression
        assert len(result) == 1
    
    def test_apply_nms_empty(self, empty_detections):
        """Test NMS on empty list."""
        pp = PostProcessor()
        result = pp.apply_nms(empty_detections)
        
        assert len(result) == 0


class TestPostProcessorGrouping:
    """Tests for grouping and counting functions."""
    
    def test_group_by_class(self, sample_detections):
        """Test grouping detections by class."""
        pp = PostProcessor()
        grouped = pp.group_by_class(sample_detections)
        
        assert 'person' in grouped
        assert 'car' in grouped
        assert len(grouped['person']) == 1
    
    def test_count_objects(self, sample_detections):
        """Test counting objects by class."""
        pp = PostProcessor()
        counts = pp.count_objects(sample_detections)
        
        assert counts['person'] == 1
        assert counts['car'] == 1
        assert counts['dog'] == 1
    
    def test_count_objects_empty(self, empty_detections):
        """Test counting with no detections."""
        pp = PostProcessor()
        counts = pp.count_objects(empty_detections)
        
        assert len(counts) == 0


class TestPostProcessorLimitDetections:
    """Tests for limiting detections."""
    
    def test_limit_detections(self, sample_detections):
        """Test limiting number of detections."""
        pp = PostProcessor(max_detections=2)
        limited = pp.limit_detections(sample_detections)
        
        assert len(limited) == 2
    
    def test_limit_detections_more_than_available(self, sample_detections):
        """Test limiting when fewer detections available."""
        pp = PostProcessor(max_detections=100)
        limited = pp.limit_detections(sample_detections)
        
        assert len(limited) == len(sample_detections)
    
    def test_limit_detections_one(self, sample_detections):
        """Test limiting to one detection."""
        pp = PostProcessor(max_detections=1)
        limited = pp.limit_detections(sample_detections)
        
        assert len(limited) == 1


class TestPostProcessorPipeline:
    """Tests for full processing pipeline."""
    
    def test_process_pipeline(self, sample_detections):
        """Test full processing pipeline."""
        pp = PostProcessor(confidence_threshold=0.5)
        processed = pp.process(sample_detections)
        
        # Should filter out low confidence detections
        for d in processed:
            assert d['confidence'] >= 0.5
    
    def test_process_with_class_filter(self, sample_detections):
        """Test processing with class filter."""
        pp = PostProcessor()
        processed = pp.process(sample_detections, allowed_classes=['person', 'car'])
        
        for d in processed:
            assert d['class_name'] in ['person', 'car']


class TestPostProcessorEdgeCases:
    """Edge case tests for PostProcessor."""
    
    def test_filter_none_threshold(self, sample_detections):
        """Test filtering with None threshold (should use default)."""
        pp = PostProcessor(confidence_threshold=0.5)
        filtered = pp.filter_by_confidence(sample_detections, threshold=None)
        
        for d in filtered:
            assert d['confidence'] >= 0.5
    
    def test_process_single_detection(self):
        """Test processing a single detection."""
        pp = PostProcessor()
        single = [{
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.9,
            'bbox': [100, 100, 200, 300]
        }]
        
        filtered = pp.filter_by_confidence(single)
        assert len(filtered) == 1
    
    def test_iou_calculation(self):
        """Test IoU calculation between two boxes."""
        pp = PostProcessor()
        
        box1 = [0, 0, 100, 100]  # Area: 10000
        box2 = [50, 50, 150, 150]  # Area: 10000, Overlap: 2500
        
        iou = pp.calculate_iou(box1, box2)
        
        # IoU = 2500 / (10000 + 10000 - 2500) = 2500 / 17500 ≈ 0.143
        assert 0.14 <= iou <= 0.15
    
    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        pp = PostProcessor()
        
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        
        iou = pp.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_iou_complete_overlap(self):
        """Test IoU with complete overlap."""
        pp = PostProcessor()
        
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        
        iou = pp.calculate_iou(box1, box2)
        assert iou == 1.0
