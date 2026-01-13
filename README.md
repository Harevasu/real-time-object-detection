# Real-Time Object Detection

A real-time object detection system using OpenCV and YOLOv8 (CNN-based deep learning model).

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-red)

## 🔍 Overview

This project captures live video from a webcam, processes frames through a YOLO object detection model, and displays detected objects with bounding boxes and labels in real-time.

## ✨ Features

- **Real-time detection** from webcam/camera
- **80 object classes** (COCO dataset)
- **Modular architecture** - easy to extend
- **FPS counter** display
- **Screenshot capture** functionality
- **Configurable settings** (confidence, resolution, etc.)

## 🛠️ Installation

### 1. Clone the repository

```bash
cd d:\Project\real-time-object-detection
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Usage

```bash
python src/main.py
```

### Command Line Options

```bash
python src/main.py --help

Options:
  --camera, -c      Camera index (default: 0)
  --model, -m       YOLO model name (default: yolov8n.pt)
  --confidence      Confidence threshold (default: 0.5)
  --device, -d      Device: cpu, cuda:0 (default: cpu)
  --width           Frame width (default: 1280)
  --height          Frame height (default: 720)
  --no-fps          Hide FPS counter
  --counts          Show object counts
```

### Examples

```bash
# Use different camera
python src/main.py --camera 1

# Higher confidence threshold
python src/main.py --confidence 0.7

# Use GPU (if available)
python src/main.py --device cuda:0

# Lower resolution for better FPS
python src/main.py --width 640 --height 480

# Show object counts
python src/main.py --counts
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `ESC` | Quit application |
| `s` | Save screenshot |

## 📁 Project Structure

```
real-time-object-detection/
│
├── src/
│   ├── main.py              # Entry point
│   ├── video_capture.py     # Camera handling
│   ├── preprocessing.py     # Image preprocessing
│   ├── detector.py          # YOLO model wrapper
│   ├── postprocessing.py    # Detection filtering
│   └── visualization.py     # Drawing utilities
│
├── config/
│   └── settings.py          # Configuration
│
├── models/                  # Model weights (auto-downloaded)
├── outputs/                 # Screenshots
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
# Detection settings
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
MODEL_NAME = "yolov8n.pt"

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
```

## 🎯 Detectable Objects (80 Classes)

The model can detect:

- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, boat, train
- **Animals**: dog, cat, bird, horse, cow, sheep, elephant, bear, zebra, giraffe
- **Objects**: cell phone, laptop, keyboard, mouse, tv, chair, couch, bed
- **And more...**

## 📊 Performance

| Model | Size | Speed (CPU) | Speed (GPU) |
|-------|------|-------------|-------------|
| yolov8n | 6 MB | ~30ms | ~5ms |
| yolov8s | 22 MB | ~60ms | ~8ms |
| yolov8m | 52 MB | ~120ms | ~12ms |

## 🐛 Troubleshooting

### Camera not detected
- Try different camera index: `--camera 1`
- Check if camera is being used by another application
- Verify camera drivers are installed

### Low FPS
- Use smaller model: `yolov8n.pt`
- Reduce resolution: `--width 640 --height 480`
- Use GPU if available: `--device cuda:0`

### Model download fails
- Check internet connection
- Manually download from [Ultralytics](https://github.com/ultralytics/assets/releases)

## 🔮 Future Enhancements

- [ ] Video file input support
- [ ] Multi-camera support
- [ ] Object tracking
- [ ] Custom model training
- [ ] Web interface
- [ ] Edge device deployment

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [COCO Dataset](https://cocodataset.org/)
