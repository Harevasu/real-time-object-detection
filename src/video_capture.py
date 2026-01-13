"""
Video Capture Module
Handles webcam/camera access and frame capture.
"""

import cv2
import sys

# Add parent directory to path for imports
sys.path.append('..')
from config import settings


class VideoCapture:
    """
    Handles video capture from webcam or external camera.
    
    Attributes:
        cap: OpenCV VideoCapture object
        frame_width: Width of captured frames
        frame_height: Height of captured frames
    """
    
    def __init__(self, camera_index=None, width=None, height=None):
        """
        Initialize the video capture.
        
        Args:
            camera_index: Camera device index (default from settings)
            width: Frame width (default from settings)
            height: Frame height (default from settings)
        """
        self.camera_index = camera_index if camera_index is not None else settings.CAMERA_INDEX
        self.frame_width = width if width is not None else settings.FRAME_WIDTH
        self.frame_height = height if height is not None else settings.FRAME_HEIGHT
        self.cap = None
        
    def start(self):
        """
        Start the video capture.
        
        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        # Use DirectShow backend on Windows for better compatibility
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            # Try without DirectShow if it fails
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)
        
        # Get actual properties (may differ from requested)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
        return True
    
    def read_frame(self):
        """
        Read a single frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
    
    def is_opened(self):
        """
        Check if the camera is opened.
        
        Returns:
            bool: True if camera is opened
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


# Test the module
if __name__ == "__main__":
    print("Testing Video Capture Module...")
    
    with VideoCapture() as camera:
        if camera.is_opened():
            # Capture a few frames
            for i in range(5):
                ret, frame = camera.read_frame()
                if ret:
                    print(f"Frame {i+1}: {frame.shape}")
                else:
                    print(f"Frame {i+1}: Failed to capture")
            
            # Show one frame
            ret, frame = camera.read_frame()
            if ret:
                cv2.imshow("Test Frame", frame)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Failed to open camera")
