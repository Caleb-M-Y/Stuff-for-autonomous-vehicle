from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError("Config file must parse to a dictionary.")
    return data


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_class_name(name: str) -> str:
    raw = name.strip().lower().replace("-", "_").replace(" ", "_")
    out = []
    for ch in raw:
        if ch.isalnum() or ch == "_":
            out.append(ch)
    clean = "".join(out)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def class_name_map(train_names: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, class_name in enumerate(train_names):
        key = normalize_class_name(class_name)
        if key in mapping:
            raise ValueError(f"Duplicate normalized class name in train_names: {class_name}")
        mapping[key] = index
    return mapping


def make_data_yaml(config: dict[str, Any], output_path: str | Path) -> Path:
    classes = config["classes"]["train_names"]
    paths = config["paths"]

    data = {
        "path": ".",
        "train": str(Path(paths["train_images_dir"]).resolve()),
        "val": str(Path(paths["val_images_dir"]).resolve()),
        "names": classes,
    }

    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, sort_keys=False)
    return out_path


def write_hailo_labels_json(
    labels: list[str],
    output_path: str | Path,
    detection_threshold: float,
    max_boxes: int,
) -> Path:
    payload = {
        "detection_threshold": float(detection_threshold),
        "max_boxes": int(max_boxes),
        "labels": labels,
    }
    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return out_path


def list_image_paths(images_dir: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Images directory not found: {root}")

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths: list[Path] = []
    if recursive:
        for pattern in patterns:
            paths.extend(root.rglob(pattern))
    else:
        for pattern in patterns:
            paths.extend(root.glob(pattern))

    unique_sorted = sorted({p.resolve() for p in paths})
    return unique_sorted
