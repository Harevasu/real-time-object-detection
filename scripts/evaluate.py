"""Compare pretrained COCO model and fine-tuned VOC model on VOC2007 test."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import COCO_TO_VOC, VOC_NAMES, environment_info, metrics_dict, seed_everything, write_json
from scripts.voc_data import COCO_NAMES, create_coco_view, create_voc_splits


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="yolov8n.pt")
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--data", default="VOC.yaml")
    parser.add_argument("--baseline-data", default="datasets/voc_coco_eval")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/evaluation.json")
    return parser.parse_args()


def validate(model_path, data, args, expected_names, classes=None):
    from ultralytics import YOLO
    model = YOLO(model_path)
    if tuple(model.names[i] for i in range(len(model.names))) != expected_names:
        raise ValueError(f"Class names do not match expected dataset for {model_path}")
    started = time.perf_counter()
    metrics = model.val(
        data=str(data), split="test", imgsz=args.imgsz, batch=args.batch,
        device=args.device, classes=classes, conf=0.001, iou=0.7,
        plots=False, verbose=False, project="runs/voc_eval", name="comparison",
        exist_ok=True,
    )
    return {
        "model": str(model_path), "metrics": metrics_dict(metrics),
        "evaluation_wall_seconds": time.perf_counter() - started,
    }


def main():
    args = arguments()
    seed_everything(args.seed)
    baseline_yaml = create_coco_view(args.data, args.baseline_data)
    split_yaml = create_voc_splits(args.data)
    baseline = validate(args.baseline, baseline_yaml, args, COCO_NAMES, sorted(COCO_TO_VOC))
    fine_tuned = validate(args.finetuned, split_yaml, args, VOC_NAMES)
    report = {
        "dataset": args.data, "held_out_split": "VOC2007 test", "imgsz": args.imgsz,
        "batch": args.batch, "seed": args.seed, "baseline": baseline,
        "finetuned": fine_tuned, "environment": environment_info(args.device),
        "methodology": "Both on VOC2007 test; baseline 80-class COCO IDs mapped from VOC ground truth and predictions filtered to the same 20 semantic classes.",
    }
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
