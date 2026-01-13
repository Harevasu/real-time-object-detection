"""
Video Capture Module
Handles webcam/camera access, video file input, and frame capture.
"""

import cv2
import sys
import os

# Add parent directory to path for imports
sys.path.append('..')
from config import settings


class VideoCapture:
    """
    Handles video capture from webcam, external camera, or video file.
    
    Attributes:
        cap: OpenCV VideoCapture object
        frame_width: Width of captured frames
        frame_height: Height of captured frames
        source: Video source (camera index or file path)
        is_file: Whether source is a video file
    """
    
    def __init__(self, camera_index=None, width=None, height=None, video_path=None):
        """
        Initialize the video capture.
        
        Args:
            camera_index: Camera device index (default from settings)
            width: Frame width (default from settings)
            height: Frame height (default from settings)
            video_path: Path to video file (overrides camera_index if provided)
        """
        self.video_path = video_path
        self.is_file = video_path is not None
        
        if self.is_file:
            self.source = video_path
        else:
            self.source = camera_index if camera_index is not None else settings.CAMERA_INDEX
        
        self.frame_width = width if width is not None else settings.FRAME_WIDTH
        self.frame_height = height if height is not None else settings.FRAME_HEIGHT
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = settings.TARGET_FPS
        
    def start(self):
        """
        Start the video capture.
        
        Returns:
            bool: True if source opened successfully, False otherwise
        """
        if self.is_file:
            return self._start_video_file()
        else:
            return self._start_camera()
    
    def _start_camera(self):
        """Start capture from camera."""
        # Use DirectShow backend on Windows for better compatibility
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            # Try without DirectShow if it fails
            self.cap = cv2.VideoCapture(self.source)
            
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.source}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)
        
        # Get actual properties (may differ from requested)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = actual_fps
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
        return True
    
    def _start_video_file(self):
        """Start capture from video file."""
        if not os.path.exists(self.source):
            print(f"Error: Video file not found: {self.source}")
            return False
        
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video file: {self.source}")
            return False
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"Video file loaded: {os.path.basename(self.source)}")
        print(f"Resolution: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        print(f"Duration: {duration:.1f} seconds ({self.total_frames} frames)")
        return True
    
    def read_frame(self):
        """
        Read a single frame from the source.
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret and self.is_file:
            self.current_frame += 1
        
        return ret, frame
    
    def get_progress(self):
        """
        Get current progress for video files.
        
        Returns:
            tuple: (current_frame, total_frames, percentage)
        """
        if not self.is_file or self.total_frames == 0:
            return 0, 0, 0.0
        percentage = (self.current_frame / self.total_frames) * 100
        return self.current_frame, self.total_frames, percentage
    
    def seek(self, frame_number):
        """
        Seek to a specific frame (video files only).
        
        Args:
            frame_number: Target frame number
            
        Returns:
            bool: True if successful
        """
        if not self.is_file or self.cap is None:
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        return True
    
    def restart(self):
        """
        Restart video from beginning (video files only).
        
        Returns:
            bool: True if successful
        """
        return self.seek(0)
    
    def get_fps(self):
        """Get the FPS of the video source."""
        return self.fps
    
    def get_frame_size(self):
        """Get the frame dimensions."""
        return self.frame_width, self.frame_height
    
    def is_opened(self):
        """
        Check if the camera is opened.
        
        Returns:
            bool: True if camera is opened
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release the video source resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            source_type = "Video file" if self.is_file else "Camera"
            print(f"{source_type} released")
    
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
