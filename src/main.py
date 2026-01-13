"""
Real-Time Object Detection - Main Application
Entry point that orchestrates all modules.
"""

import cv2
import sys
import argparse
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.video_capture import VideoCapture
from src.video_recorder import VideoRecorder
from src.preprocessing import Preprocessor
from src.detector import ObjectDetector
from src.postprocessing import PostProcessor
from src.visualization import Visualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-Time Object Detection using YOLOv8'
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--camera', '-c',
        type=int,
        default=None,
        help='Camera index (default: 0)'
    )
    input_group.add_argument(
        '--video', '-v',
        type=str,
        default=None,
        help='Path to input video file'
    )
    
    # Model settings
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=settings.MODEL_NAME,
        help='YOLO model name (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--confidence', '-conf',
        type=float,
        default=settings.CONFIDENCE_THRESHOLD,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=settings.DEVICE,
        choices=['cpu', 'cuda:0', 'cuda:1'],
        help='Device for inference (default: cpu)'
    )
    
    # Resolution
    parser.add_argument(
        '--width',
        type=int,
        default=settings.FRAME_WIDTH,
        help='Frame width (default: 1280)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=settings.FRAME_HEIGHT,
        help='Frame height (default: 720)'
    )
    
    # Display options
    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Hide FPS counter'
    )
    parser.add_argument(
        '--counts',
        action='store_true',
        help='Show object counts'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable display window (useful for headless processing)'
    )
    
    # Recording options
    parser.add_argument(
        '--record', '-r',
        action='store_true',
        help='Start recording immediately'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output video file path (for recording)'
    )
    
    # Video file options
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Loop video file playback'
    )
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Determine input source
    is_video_file = args.video is not None
    camera_index = args.camera if args.camera is not None else settings.CAMERA_INDEX
    
    print("=" * 50)
    print("  Real-Time Object Detection")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence Threshold: {args.confidence}")
    
    if is_video_file:
        print(f"Input: Video file - {args.video}")
    else:
        print(f"Input: Camera index {camera_index}")
        print(f"Resolution: {args.width}x{args.height}")
    
    print("=" * 50)
    print("Controls:")
    print("  'q' / ESC  - Quit")
    print("  's'        - Save screenshot")
    print("  'r'        - Toggle recording")
    print("  'p'        - Pause/Resume (video files)")
    print("  SPACE      - Pause/Resume (video files)")
    print("=" * 50)
    
    # Initialize video capture
    if is_video_file:
        capture = VideoCapture(video_path=args.video)
    else:
        capture = VideoCapture(
            camera_index=camera_index,
            width=args.width,
            height=args.height
        )
    
    preprocessor = Preprocessor()
    
    detector = ObjectDetector(
        model_name=args.model,
        device=args.device,
        confidence=args.confidence
    )
    
    postprocessor = PostProcessor(
        confidence_threshold=args.confidence
    )
    
    visualizer = Visualizer()
    
    # Initialize recorder (not started yet)
    recorder = VideoRecorder()
    
    # Load the detection model
    if not detector.load_model():
        print("Error: Failed to load model. Exiting.")
        return 1
    
    # Start video capture
    if not capture.start():
        print("Error: Failed to start video capture. Exiting.")
        return 1
    
    # Get actual frame size for recording
    frame_width, frame_height = capture.get_frame_size()
    fps = capture.get_fps()
    
    # Start recording if requested
    if args.record:
        recorder.start(
            output_path=args.output,
            fps=fps,
            frame_size=(frame_width, frame_height)
        )
    
    try:
        frame_count = 0
        paused = False
        
        while True:
            # Handle pause for video files
            if paused and is_video_file:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p') or key == ord(' '):
                    paused = False
                    print("Resumed")
                elif key == ord('q') or key == 27:
                    print("\nExiting...")
                    break
                continue
            
            # 1. Capture frame
            ret, frame = capture.read_frame()
            
            if not ret:
                if is_video_file:
                    if args.loop:
                        print("Video ended. Looping...")
                        capture.restart()
                        continue
                    else:
                        print("Video ended.")
                        break
                else:
                    print("Error: Failed to read frame")
                    break
            
            # 1.5. Flip frame horizontally (mirror mode) - only for camera
            if not is_video_file and settings.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)  # 1 = horizontal flip
            
            # 2. Preprocess (optional - YOLO handles most preprocessing)
            # processed_frame = preprocessor.preprocess(frame)
            
            # 3. Run detection
            results = detector.detect(frame)
            
            # 4. Extract detections
            detections = detector.get_detections(results)
            
            # 5. Post-process
            filtered_detections = postprocessor.process(detections)
            
            # 6. Visualize
            output_frame = visualizer.visualize(
                frame,
                filtered_detections,
                show_fps=not args.no_fps,
                show_counts=args.counts
            )
            
            # Add recording indicator
            if recorder.is_recording:
                status = recorder.get_status()
                rec_text = f"REC {status['elapsed_seconds']:.1f}s"
                cv2.circle(output_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(output_frame, rec_text, (50, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add progress bar for video files
            if is_video_file:
                current, total, pct = capture.get_progress()
                bar_width = int(frame_width * 0.8)
                bar_x = int(frame_width * 0.1)
                bar_y = frame_height - 20
                progress_width = int(bar_width * pct / 100)
                
                cv2.rectangle(output_frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + 10), (100, 100, 100), -1)
                cv2.rectangle(output_frame, (bar_x, bar_y), 
                            (bar_x + progress_width, bar_y + 10), (0, 255, 0), -1)
                cv2.putText(output_frame, f"{pct:.1f}%", (bar_x + bar_width + 10, bar_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 7. Record frame if recording
            if recorder.is_recording:
                recorder.write_frame(output_frame)
            
            # 8. Display (unless disabled)
            if not args.no_display:
                cv2.imshow(settings.WINDOW_NAME, output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExiting...")
                break
            elif key == ord('s'):  # Save screenshot
                frame_count += 1
                filename = os.path.join(
                    settings.OUTPUTS_DIR,
                    f"screenshot_{frame_count}.jpg"
                )
                cv2.imwrite(filename, output_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):  # Toggle recording
                if recorder.is_recording:
                    recorder.stop()
                else:
                    recorder.start(
                        fps=fps,
                        frame_size=(frame_width, frame_height)
                    )
            elif key == ord('p') or key == ord(' '):  # Pause/Resume
                if is_video_file:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if recorder.is_recording:
            recorder.stop()
        capture.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
