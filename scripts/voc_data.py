"""Build leakage-free VOC splits and a COCO-compatible baseline validation view."""

import argparse
import os
import shutil
from pathlib import Path

import yaml

from scripts.common import COCO_TO_VOC, VOC_NAMES, write_json

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


def voc_root_from_config(data):
    from ultralytics.data.utils import check_det_dataset
    return Path(check_det_dataset(data, autodownload=True)["path"]).resolve()


def create_voc_splits(data="VOC.yaml", output="datasets/voc_splits"):
    root = voc_root_from_config(data)
    for folder in ("train2007", "val2007", "train2012", "val2012", "test2007"):
        if not (root / "images" / folder).is_dir():
            raise FileNotFoundError(root / "images" / folder)
    target = Path(output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    config = target / "voc_reproducible.yaml"
    config.write_text(yaml.safe_dump({
        "path": str(root),
        "train": ["images/train2007", "images/val2007", "images/train2012"],
        "val": "images/val2012",
        "test": "images/test2007",
        "names": dict(enumerate(VOC_NAMES)),
    }, sort_keys=False), encoding="utf-8")
    return config


def remap_voc_label(line, voc_to_coco):
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Expected YOLO box with 5 columns: {line!r}")
    class_id = int(fields[0])
    return " ".join([str(voc_to_coco[class_id]), *fields[1:]])


def create_coco_view(data="VOC.yaml", output="datasets/voc_coco_eval"):
    root = voc_root_from_config(data)
    target = Path(output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    voc_to_coco = {voc: coco for coco, voc in COCO_TO_VOC.items()}
    if len(voc_to_coco) != 20:
        raise ValueError("VOC mapping must cover 20 classes uniquely")
    images = root / "images" / "test2007"
    labels = root / "labels" / "test2007"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"VOC2007 test images/labels missing under {root}")
    image_files = sorted(images.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No VOC2007 test images found in {images}")
    target_images = target / "images" / "test2007"
    target_labels = target / "labels" / "test2007"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    for image in image_files:
        label = labels / (image.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        converted = [
            remap_voc_label(line, voc_to_coco)
            for line in label.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        (target_labels / label.name).write_text(
            "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
        )
        linked = target_images / image.name
        if not linked.exists():
            try:
                linked.symlink_to(image.resolve())
            except OSError:
                try:
                    os.link(image, linked)
                except OSError:
                    shutil.copyfile(image, linked)
    config = target / "coco_voc_test.yaml"
    config.write_text(yaml.safe_dump({
        "path": str(target), "train": "images/test2007",
        "val": "images/test2007", "test": "images/test2007",
        "names": dict(enumerate(COCO_NAMES)),
    }, sort_keys=False), encoding="utf-8")
    write_json(target / "mapping.json", {
        "voc_to_coco": voc_to_coco, "image_count": len(image_files), "source": str(root),
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="VOC.yaml")
    parser.add_argument("--output", default="datasets/voc_coco_eval")
    args = parser.parse_args()
    print(create_coco_view(args.data, args.output))


if __name__ == "__main__":
    main()
