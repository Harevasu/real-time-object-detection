"""Fine-tune YOLOv8; download PASCAL VOC when needed."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from scripts.common import environment_info, seed_everything, write_json
from scripts.voc_data import create_voc_splits


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default="VOC.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", default="runs/voc_train")
    parser.add_argument("--name", default="yolov8n_voc")
    pre, _ = parser.parse_known_args()
    if Path(pre.config).is_file():
        defaults = yaml.safe_load(Path(pre.config).read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            raise ValueError("Training config must be a YAML mapping")
        parser.set_defaults(**defaults)
    return parser.parse_args()


def main():
    from ultralytics import YOLO
    args = arguments()
    seed_everything(args.seed)
    train_data = str(create_voc_splits(args.data)) if args.data == "VOC.yaml" else args.data
    kwargs = {
        "data": train_data, "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch, "device": args.device, "workers": args.workers,
        "seed": args.seed, "patience": args.patience, "project": args.project,
        "name": args.name, "deterministic": True, "amp": True,
    }
    result = YOLO(args.model).train(**kwargs)
    output = Path(result.save_dir)
    write_json(output / "experiment.json", {
        "parameters": kwargs, "base_model": args.model, "dataset_source": args.data,
        "split": "VOC2007 trainval + VOC2012 train / VOC2012 val / VOC2007 test",
        "environment": environment_info(args.device),
    })
    print(f"Best checkpoint: {output / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
