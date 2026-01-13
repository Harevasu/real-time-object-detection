"""
Image Preprocessing Module
Handles frame preprocessing before detection.
"""

import cv2
import numpy as np
import sys

sys.path.append('..')
from config import settings


class Preprocessor:
    """
    Handles image preprocessing operations.
    
    Note: YOLO models handle most preprocessing internally,
    but this module provides additional utilities if needed.
    """
    
    def __init__(self, target_size=(640, 640)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target size for resizing (width, height)
        """
        self.target_size = target_size
    
    def resize(self, frame, size=None):
        """
        Resize frame to target size.
        
        Args:
            frame: Input frame (numpy array)
            size: Target size (width, height), uses default if None
            
        Returns:
            Resized frame
        """
        if size is None:
            size = self.target_size
        return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    
    def bgr_to_rgb(self, frame):
        """
        Convert BGR to RGB color space.
        
        Args:
            frame: BGR frame
            
        Returns:
            RGB frame
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def rgb_to_bgr(self, frame):
        """
        Convert RGB to BGR color space.
        
        Args:
            frame: RGB frame
            
        Returns:
            BGR frame
        """
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def normalize(self, frame):
        """
        Normalize pixel values to 0-1 range.
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame (float32)
        """
        return frame.astype(np.float32) / 255.0
    
    def denormalize(self, frame):
        """
        Denormalize pixel values back to 0-255 range.
        
        Args:
            frame: Normalized frame
            
        Returns:
            Denormalized frame (uint8)
        """
        return (frame * 255).astype(np.uint8)
    
    def apply_gaussian_blur(self, frame, kernel_size=(5, 5)):
        """
        Apply Gaussian blur for noise reduction.
        
        Args:
            frame: Input frame
            kernel_size: Blur kernel size
            
        Returns:
            Blurred frame
        """
        return cv2.GaussianBlur(frame, kernel_size, 0)
    
    def enhance_contrast(self, frame):
        """
        Enhance image contrast using CLAHE.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Contrast-enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def preprocess(self, frame, apply_blur=False, enhance=False):
        """
        Apply full preprocessing pipeline.
        
        Args:
            frame: Input frame (BGR)
            apply_blur: Whether to apply noise reduction
            enhance: Whether to enhance contrast
            
        Returns:
            Preprocessed frame
        """
        result = frame.copy()
        
        if apply_blur:
            result = self.apply_gaussian_blur(result)
            
        if enhance:
            result = self.enhance_contrast(result)
            
        return result


# Test the module
if __name__ == "__main__":
    print("Testing Preprocessing Module...")
    
    # Create a test image
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    preprocessor = Preprocessor()
    
    # Test resize
    resized = preprocessor.resize(test_frame, (320, 240))
    print(f"Original shape: {test_frame.shape}")
    print(f"Resized shape: {resized.shape}")
    
    # Test color conversion
    rgb = preprocessor.bgr_to_rgb(test_frame)
    print(f"RGB conversion: {rgb.shape}")
    
    # Test normalization
    normalized = preprocessor.normalize(test_frame)
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # Test full pipeline
    processed = preprocessor.preprocess(test_frame, apply_blur=True, enhance=True)
    print(f"Processed shape: {processed.shape}")
    
    print("Preprocessing module test complete!")
