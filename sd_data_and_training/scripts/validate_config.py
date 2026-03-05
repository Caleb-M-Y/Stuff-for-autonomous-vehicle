from __future__ import annotations

import argparse
from pathlib import Path

from common import class_name_map, load_config


REQUIRED_PATH_KEYS = [
    "repo_root",
    "train_images_dir",
    "val_images_dir",
    "train_labels_dir",
    "val_labels_dir",
    "data_yaml_out",
    "train_project_dir",
    "export_out_dir",
    "hailo_labels_json_out",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate training config before a long run.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)

    paths_cfg = config.get("paths", {})
    classes_cfg = config.get("classes", {})

    missing = [key for key in REQUIRED_PATH_KEYS if key not in paths_cfg]
    if missing:
        raise KeyError(f"Missing required paths keys: {missing}")

    train_names = classes_cfg.get("train_names", [])
    if not train_names:
        raise ValueError("classes.train_names must not be empty")

    class_name_map(train_names)

    display_names = classes_cfg.get("hailo_display_names", [])
    if display_names and len(display_names) != len(train_names):
        raise ValueError("classes.hailo_display_names length must equal classes.train_names length")

    repo_root = Path(paths_cfg["repo_root"])
    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root does not exist: {repo_root}")

    print("Config validation passed.")
    print(f"Classes: {len(train_names)}")
    print(f"Repo root: {repo_root}")


if __name__ == "__main__":
    main()
