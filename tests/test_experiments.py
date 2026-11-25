"""Unit tests that do not download the dataset or model weights."""

from pathlib import Path

import pytest
import yaml

from scripts.common import COCO_TO_VOC, VOC_NAMES, metrics_dict
from scripts import voc_data
from src.model_paths import resolve_model_reference


def test_class_mapping_is_bijective():
    assert len(VOC_NAMES) == 20
    assert len(COCO_TO_VOC) == 20
    assert set(COCO_TO_VOC.values()) == set(range(20))
    assert COCO_TO_VOC[0] == 14
    assert COCO_TO_VOC[58] == 15


def test_checkpoint_resolution(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"unit-test")
    assert resolve_model_reference(str(checkpoint)) == str(checkpoint.resolve())
    assert resolve_model_reference("best.pt", models_dir=tmp_path) == str(checkpoint.resolve())
    assert resolve_model_reference("yolov8n.pt", models_dir=tmp_path) == "yolov8n.pt"
    with pytest.raises(FileNotFoundError):
        resolve_model_reference("missing.pt", models_dir=tmp_path)


def test_reproducible_splits_do_not_validate_on_test(tmp_path, monkeypatch):
    root = tmp_path / "VOC"
    for folder in ("train2007", "val2007", "train2012", "val2012", "test2007"):
        (root / "images" / folder).mkdir(parents=True)
    monkeypatch.setattr(voc_data, "voc_root_from_config", lambda _: root)
    path = voc_data.create_voc_splits(output=tmp_path / "splits")
    config = yaml.safe_load(path.read_text())
    assert config["val"] == "images/val2012"
    assert config["test"] == "images/test2007"
    assert "images/test2007" not in config["train"]


def test_coco_view_remaps_labels(tmp_path, monkeypatch):
    root = tmp_path / "VOC"
    (root / "images" / "test2007").mkdir(parents=True)
    (root / "labels" / "test2007").mkdir(parents=True)
    (root / "images" / "test2007" / "x.jpg").write_bytes(b"fake-image")
    (root / "labels" / "test2007" / "x.txt").write_text(
        "14 0.5 0.5 0.2 0.2\n19 0.4 0.4 0.1 0.1\n"
    )
    monkeypatch.setattr(voc_data, "voc_root_from_config", lambda _: root)
    path = voc_data.create_coco_view(output=tmp_path / "baseline")
    config = yaml.safe_load(path.read_text())
    assert config["test"] == "images/test2007"
    mapped = (tmp_path / "baseline" / "labels" / "test2007" / "x.txt").read_text()
    assert mapped.splitlines()[0].startswith("0 ")
    assert mapped.splitlines()[1].startswith("62 ")
    assert len(config["names"]) == 80


def test_metrics_shape():
    class Metrics:
        class box:
            mp, mr, map50, map = 0.5, 0.6, 0.7, 0.4
    assert metrics_dict(Metrics()) == {
        "precision": 0.5, "recall": 0.6, "map50": 0.7, "map50_95": 0.4,
    }
