"""Shared data and reporting utilities for reproducible experiments."""

import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VOC_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

COCO_TO_VOC = {
    0: 14, 1: 1, 2: 6, 3: 13, 4: 0, 5: 5, 6: 18, 8: 3,
    14: 2, 15: 7, 16: 11, 17: 12, 18: 16,
    19: 9, 39: 4, 56: 8, 57: 17, 58: 15, 60: 10, 62: 19,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def environment_info(device):
    import torch
    import ultralytics
    details = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        details["gpu"] = torch.cuda.get_device_name(0)
    return details


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def metrics_dict(metrics):
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
