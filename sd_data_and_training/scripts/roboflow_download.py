from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset version in YOLO format.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    rf_cfg = config.get("roboflow", {})

    api_env = rf_cfg.get("api_key_env", "ROBOFLOW_API_KEY")
    api_key = os.getenv(api_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_env}")

    workspace = rf_cfg.get("workspace")
    project = rf_cfg.get("project")
    version = int(rf_cfg.get("version", 0))
    model_format = rf_cfg.get("format", "yolov8")
    download_dir = Path(rf_cfg.get("download_dir", "./datasets/roboflow"))

    if not workspace or not project or version <= 0:
        raise ValueError("Set roboflow.workspace, roboflow.project, and roboflow.version in config.")

    ensure_dir(download_dir)

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(model_format=model_format, location=str(download_dir))
    )

    print("Roboflow download complete.")
    print(f"Dataset location: {dataset.location}")


if __name__ == "__main__":
    main()
