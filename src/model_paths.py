"""Resolve a pretrained YOLOv8 weight name or a real local checkpoint."""

from pathlib import Path

from config import settings

PRETRAINED = {f"yolov8{size}.pt" for size in "nsmlx"}


def resolve_model_reference(model, models_dir=None):
    if not model or not str(model).strip():
        raise ValueError("A model name or checkpoint path is required")
    reference = Path(model).expanduser()
    if reference.is_file():
        return str(reference.resolve())
    if not reference.is_absolute():
        local = Path(models_dir or settings.MODELS_DIR) / reference
        if local.is_file():
            return str(local.resolve())
    if str(model) in PRETRAINED:
        return str(model)
    raise FileNotFoundError(
        f"Checkpoint not found: {model}. Use an existing .pt path or a standard YOLOv8 name."
    )
