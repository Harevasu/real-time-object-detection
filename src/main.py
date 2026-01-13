"""
Real-Time Object Detection - Main Application
Entry point that orchestrates all modules.
"""

import cv2
import sys
import argparse
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.video_capture import VideoCapture
from src.preprocessing import Preprocessor
from src.detector import ObjectDetector
from src.postprocessing import PostProcessor
from src.visualization import Visualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-Time Object Detection using YOLOv8'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=settings.CAMERA_INDEX,
        help='Camera index (default: 0)'
    )
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
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 50)
    print("  Real-Time Object Detection")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"Camera Index: {args.camera}")
    print(f"Resolution: {args.width}x{args.height}")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("=" * 50)
    
    # Initialize modules
    camera = VideoCapture(
        camera_index=args.camera,
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
    
    # Load the detection model
    if not detector.load_model():
        print("Error: Failed to load model. Exiting.")
        return 1
    
    # Start camera
    if not camera.start():
        print("Error: Failed to start camera. Exiting.")
        return 1
    
    try:
        frame_count = 0
        
        while True:
            # 1. Capture frame
            ret, frame = camera.read_frame()
            
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # 1.5. Flip frame horizontally (mirror mode)
            if settings.FLIP_HORIZONTAL:
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
            
            # 7. Display
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
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
