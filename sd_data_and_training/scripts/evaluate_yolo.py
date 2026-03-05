from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import load_config, make_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint on val/test split.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights.")
    parser.add_argument("--data-yaml", type=str, default=None, help="Optional existing data.yaml override.")
    parser.add_argument("--split", type=str, default=None, help="Split override: val or test.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size override.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. '0' or 'cpu'.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    eval_cfg = config.get("evaluation", {})
    train_cfg = config.get("training", {})

    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        data_yaml = make_data_yaml(config, paths_cfg["data_yaml_out"])

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    split = args.split or eval_cfg.get("split", "val")
    conf = float(args.conf if args.conf is not None else eval_cfg.get("conf", 0.25))
    iou = float(args.iou if args.iou is not None else eval_cfg.get("iou", 0.60))
    imgsz = int(args.imgsz if args.imgsz is not None else train_cfg.get("imgsz", 640))
    device = str(args.device if args.device is not None else train_cfg.get("device", "0"))

    print(f"Evaluating: {weights}")
    print(f"Data YAML: {data_yaml}")
    print(f"Split: {split}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
    )

    print("Evaluation complete.")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
