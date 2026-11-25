"""Benchmark warmed YOLO prediction latency (not end-to-end camera FPS)."""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import environment_info, seed_everything, write_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/benchmark.json")
    return parser.parse_args()


def synchronize(device):
    import torch
    if str(device).startswith(("0", "1", "cuda")) and torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(model_name, args):
    import numpy as np
    from ultralytics import YOLO
    model = YOLO(model_name)
    frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)

    def predict():
        model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)

    for _ in range(args.warmup):
        predict()
    synchronize(args.device)
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        predict()
        synchronize(args.device)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {
        "model": str(model_name), "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
        "predict_fps_upper_bound": 1000 / statistics.mean(times),
        "samples": len(times),
    }


def main():
    args = arguments()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    seed_everything(args.seed)
    report = {
        "methodology": "Single synthetic BGR image, batch=1; warmed model.predict() plus preprocessing/postprocessing and GPU synchronization. No camera, visualization or video writing.",
        "imgsz": args.imgsz, "warmup": args.warmup, "iterations": args.iterations,
        "environment": environment_info(args.device),
        "models": [bench(model, args) for model in args.models],
    }
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
