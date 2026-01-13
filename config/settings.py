"""
Configuration settings for Real-Time Object Detection System.
Modify these values to customize detection behavior.
"""

# =============================================================================
# MODEL SETTINGS
# =============================================================================
MODEL_NAME = "yolov8m.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
DEVICE = "cpu"  # Options: "cpu", "cuda:0" (for GPU) - Set to cpu until CUDA PyTorch installed

# =============================================================================
# DETECTION SETTINGS
# =============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score (0.0 - 1.0)
NMS_THRESHOLD = 0.4  # Non-Maximum Suppression threshold (0.0 - 1.0)
MAX_DETECTIONS = 100  # Maximum number of detections per frame

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAMERA_INDEX = 0  # Default webcam (try 1, 2 if 0 doesn't work)
FRAME_WIDTH = 1280  # Capture width
FRAME_HEIGHT = 720  # Capture height
TARGET_FPS = 30  # Target frame rate
FLIP_HORIZONTAL = True  # Flip camera left-right (mirror mode)

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
BOX_COLOR = (0, 255, 0)  # Bounding box color (BGR) - Green
TEXT_COLOR = (255, 255, 255)  # Label text color (BGR) - White
TEXT_BG_COLOR = (0, 255, 0)  # Label background color (BGR) - Green
BOX_THICKNESS = 2  # Bounding box line thickness
FONT_SCALE = 0.6  # Label font size
FONT_THICKNESS = 2  # Label font thickness
SHOW_FPS = True  # Display FPS counter
SHOW_CONFIDENCE = True  # Display confidence scores

# =============================================================================
# WINDOW SETTINGS
# =============================================================================
WINDOW_NAME = "Real-Time Object Detection"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# =============================================================================
# PATHS
# =============================================================================
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
