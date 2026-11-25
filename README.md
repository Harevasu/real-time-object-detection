# Real-Time Object Detection — YOLOv8 and PASCAL VOC

This project combines a webcam/video object-detection application with a reproducible YOLOv8 fine-tuning, evaluation, and inference-benchmark workflow.

## Features

- Webcam and video-file inference with detection overlays, screenshots and video recording
- Configurable pretrained YOLOv8 weights or locally trained `.pt` checkpoints
- PASCAL VOC fine-tuning with separate training/validation/test splits
- Pretrained COCO-model baseline versus a VOC fine-tuned checkpoint on matching VOC2007 test images
- JSON accuracy and model-prediction timing reports

## Setup

Requires Python and a supported PyTorch installation. Install the project's dependencies using `pip install -r requirements.txt`. For GPU training, install a CUDA-enabled PyTorch build appropriate for your driver and GPU; verify with `torch.cuda.is_available()`.

The default inference checkpoint is **`yolov8n.pt`** in `config/settings.py`. This is a pretrained 80-class COCO detector. Fine-tuned VOC checkpoints contain 20 classes. The app reads class names from the selected checkpoint.

## Dataset and split protocol

Source: [PASCAL Visual Object Classes (VOC)](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/) via [Ultralytics VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml). The public dataset contains 20 classes. The first run downloads and prepares data into Ultralytics' configured dataset directory (typically `datasets/VOC`, which may be outside this repository); confirm the actual path from Ultralytics output.

**Important:** The stock Ultralytics VOC.yaml assigns VOC2007 test to the validation split. This project generates a *different* config under `datasets/voc_splits/` to protect the held-out test set:

| Role | Image groups |
|---|---|
| Training | VOC2007 train + val, VOC2012 train |
| Validation / early stopping | VOC2012 val |
| Held-out final evaluation | VOC2007 test (4,952 images) |

The training split and validation split do not include VOC2007 test. Custom data YAML paths are accepted via `--data`, but the custom config must provide an appropriate held-out test split.

## Reproducible experiments

Run from the repository root. The scripts take overrides so dataset YAML and checkpoints never need to be hard-coded.

### Train

```bash
python scripts/train.py --model yolov8n.pt --data VOC.yaml --epochs 50 --imgsz 640 --batch 4 --device 0 --seed 42
```

Default training parameters are stored in `config/training.yaml` (edit the file or override flags). The trained checkpoint is saved to `runs/voc_train/<run-name>/weights/best.pt`; the run also records `experiment.json`. To train on CPU set `--device cpu`; CPU training will be substantially slower. Start with batch size 4 for a small-VRAM GPU and lower it if out-of-memory errors occur.

### Compare accuracy on the same held-out images

```bash
python scripts/evaluate.py --baseline yolov8n.pt --finetuned runs/voc_train/yolov8n_voc/weights/best.pt --data VOC.yaml --device 0 --imgsz 640 --batch 4
```

The baseline is an 80-class COCO checkpoint: to compare fairly against VOC's 20 classes, `scripts/voc_data.py` makes a generated COCO-ID ground-truth view of **the same VOC2007 test images**. The baseline predictions are restricted to the corresponding 20 COCO categories. The fine-tuned model is evaluated with its original 20 VOC IDs. Accuracy results are written to `results/evaluation.json` and include mAP50, mAP50-95, mean precision and mean recall.

### Benchmark warmed model prediction

```bash
python scripts/benchmark.py --models yolov8n.pt runs/voc_train/yolov8n_voc/weights/best.pt --device 0 --imgsz 640 --warmup 10 --iterations 100
```

The output `results/benchmark.json` contains mean, median and p95 per-frame **model.predict call** latency, with synchronization for CUDA. It includes Ultralytics/PyTorch versions, device information and image size. The input is a synthetic 640×640 BGR frame. The timing includes YOLO's preprocessing/postprocessing and prediction-call overhead, but **not** webcam capture, video decoding, plotting, rendering or video encoding. The computed prediction-only FPS is **not application FPS**.

## Run the app

```bash
python src/main.py
python src/main.py --video path/to/video.mp4
python src/main.py --model yolov8n.pt --device cpu
python src/main.py --model runs/voc_train/yolov8n_voc/weights/best.pt --device cuda:0
python src/main.py --model models/best.pt --video input.mp4 --record
python src/main.py --help
```

`--model` accepts one of the standard pretrained YOLOv8 model names, a valid filesystem checkpoint path, or a filename inside `models/`. An unknown local checkpoint fails with a useful error instead of triggering an unintended download. `--imgsz` sets YOLO inference resolution independently of webcam capture width/height.

Press `q` or Escape to quit, `s` for a screenshot, `r` to toggle recording, and `p` or Space to pause video-file playback.

## Measured results

**Not measured yet.** Dataset download, 50-epoch training, VOC2007 test evaluation, and GPU runtime experiments must be run in a suitable local environment. We do not claim a baseline mAP, an accuracy improvement, CPU/GPU latency, real-time FPS or a verified test count without the corresponding execution logs.

| Metric | Pretrained COCO YOLOv8n | Fine-tuned VOC YOLOv8n |
|---|---|---|
| mAP50 | Pending experiment | Pending experiment |
| mAP50–95 | Pending experiment | Pending experiment |
| Mean precision | Pending experiment | Pending experiment |
| Mean recall | Pending experiment | Pending experiment |
| Mean model.predict latency | Pending experiment | Pending experiment |
| Application FPS | Not benchmarked | Not benchmarked |

After running the scripts, copy the actual values from the generated JSON reports here, along with the hardware, dependency versions, dataset path and exact commands. Do not substitute generic model speed figures.

## Tests

Use `pytest tests/ -v`. Focused experiment tests in `tests/test_experiments.py` test class mapping, missing/valid checkpoint selection, split isolation, label conversion and metric serialization without requiring downloaded VOC images or pretrained weights. Existing detector integration tests may download `yolov8n.pt` on first execution. The suite's full test count and pass status remain unverified until executed with all dependencies.

## Repository hygiene

Dataset files, model weights, training runs, evaluation artifacts and benchmarks are intentionally ignored by Git. Only source code, configuration and reproducibility instructions belong in version control.

## Credits

[PASCAL VOC](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/) · [Ultralytics YOLO](https://docs.ultralytics.com/) · [OpenCV](https://opencv.org/)

Tracking, multi-camera expansion, and unrelated UI changes are intentionally outside this phase.
