"""
Video Recorder Module
Handles saving detection output to video files.
"""

import cv2
import os
import sys
from datetime import datetime

sys.path.append('..')
from config import settings


class VideoRecorder:
    """
    Handles recording video output with detection overlays.
    
    Attributes:
        writer: OpenCV VideoWriter object
        output_path: Path to output video file
        is_recording: Whether currently recording
    """
    
    def __init__(self, output_path=None, fps=None, frame_size=None, codec='mp4v'):
        """
        Initialize the video recorder.
        
        Args:
            output_path: Path for output video file (auto-generated if None)
            fps: Frames per second (default from settings)
            frame_size: (width, height) tuple (default from settings)
            codec: FourCC codec code (default: 'mp4v' for .mp4)
        """
        self.output_path = output_path
        self.fps = fps if fps is not None else settings.TARGET_FPS
        self.frame_size = frame_size if frame_size is not None else (settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
        self.codec = codec
        self.writer = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
    
    def _generate_output_path(self, extension='.mp4'):
        """Generate a unique output file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_output_{timestamp}{extension}"
        return os.path.join(settings.OUTPUTS_DIR, filename)
    
    def start(self, output_path=None, fps=None, frame_size=None):
        """
        Start video recording.
        
        Args:
            output_path: Override output path
            fps: Override FPS
            frame_size: Override frame size (width, height)
            
        Returns:
            bool: True if recording started successfully
        """
        if self.is_recording:
            print("Warning: Already recording. Stop current recording first.")
            return False
        
        # Use provided values or defaults
        self.output_path = output_path or self.output_path or self._generate_output_path()
        self.fps = fps or self.fps
        self.frame_size = frame_size or self.frame_size
        
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Determine codec based on file extension
        ext = os.path.splitext(self.output_path)[1].lower()
        if ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == '.mkv':
            fourcc = cv2.VideoWriter_fourcc(*'X264')
        else:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
        
        # Create VideoWriter
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        
        if not self.writer.isOpened():
            print(f"Error: Could not create video writer for: {self.output_path}")
            return False
        
        self.is_recording = True
        self.frame_count = 0
        self.start_time = datetime.now()
        
        print(f"Recording started: {self.output_path}")
        print(f"Format: {self.frame_size[0]}x{self.frame_size[1]} @ {self.fps} FPS")
        return True
    
    def write_frame(self, frame):
        """
        Write a frame to the video file.
        
        Args:
            frame: BGR frame to write (numpy array)
            
        Returns:
            bool: True if frame written successfully
        """
        if not self.is_recording or self.writer is None:
            return False
        
        # Ensure frame matches expected size
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        self.frame_count += 1
        return True
    
    def stop(self):
        """
        Stop video recording and finalize the file.
        
        Returns:
            dict: Recording statistics
        """
        if not self.is_recording:
            return None
        
        self.writer.release()
        self.writer = None
        self.is_recording = False
        
        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        actual_fps = self.frame_count / duration if duration > 0 else 0
        file_size = os.path.getsize(self.output_path) if os.path.exists(self.output_path) else 0
        
        stats = {
            'output_path': self.output_path,
            'frame_count': self.frame_count,
            'duration': duration,
            'actual_fps': actual_fps,
            'file_size': file_size,
            'file_size_mb': file_size / (1024 * 1024)
        }
        
        print(f"\nRecording stopped!")
        print(f"Output: {self.output_path}")
        print(f"Frames: {self.frame_count}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Actual FPS: {actual_fps:.1f}")
        print(f"File size: {stats['file_size_mb']:.2f} MB")
        
        return stats
    
    def toggle(self, frame_size=None):
        """
        Toggle recording on/off.
        
        Args:
            frame_size: Frame size for new recording
            
        Returns:
            bool: New recording state
        """
        if self.is_recording:
            self.stop()
            return False
        else:
            self.start(frame_size=frame_size)
            return True
    
    def get_status(self):
        """
        Get current recording status.
        
        Returns:
            dict: Recording status information
        """
        if not self.is_recording:
            return {'is_recording': False}
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'is_recording': True,
            'output_path': self.output_path,
            'frame_count': self.frame_count,
            'elapsed_seconds': elapsed,
            'current_fps': self.frame_count / elapsed if elapsed > 0 else 0
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_recording:
            self.stop()
        return False


# Test the module
if __name__ == "__main__":
    import numpy as np
    
    print("Testing Video Recorder Module...")
    
    # Create test frames
    recorder = VideoRecorder(fps=30, frame_size=(640, 480))
    
    if recorder.start():
        # Write some test frames
        for i in range(90):  # 3 seconds at 30 FPS
            # Create a test frame with changing content
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            color = (i * 2 % 255, (255 - i * 2) % 255, 128)
            cv2.rectangle(frame, (100, 100), (540, 380), color, -1)
            cv2.putText(frame, f"Frame {i+1}", (250, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            recorder.write_frame(frame)
            
            # Print progress every second
            if (i + 1) % 30 == 0:
                status = recorder.get_status()
                print(f"Recording... {status['elapsed_seconds']:.1f}s, {status['frame_count']} frames")
        
        stats = recorder.stop()
        print(f"\nTest complete! Output: {stats['output_path']}")
    else:
        print("Failed to start recording")
