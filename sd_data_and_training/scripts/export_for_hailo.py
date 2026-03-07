from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from common import ensure_dir, load_config, write_hailo_labels_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX and write Hailo labels JSON.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained best.pt.")
    parser.add_argument("--imgsz", type=int, default=None, help="Export image size override.")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset override.")
    parser.add_argument("--out-dir", type=str, default=None, help="ONNX output directory override.")
    parser.add_argument("--labels-json-out", type=str, default=None, help="Hailo labels JSON path override.")
    simplify_group = parser.add_mutually_exclusive_group()
    simplify_group.add_argument("--simplify", dest="simplify", action="store_true", help="Enable ONNX graph simplify.")
    simplify_group.add_argument("--no-simplify", dest="simplify", action="store_false", help="Disable ONNX graph simplify.")
    parser.set_defaults(simplify=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    train_cfg = config.get("training", {})
    hailo_cfg = config.get("hailo", {})
    classes_cfg = config["classes"]

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    imgsz = int(args.imgsz if args.imgsz is not None else train_cfg.get("imgsz", 640))
    opset = int(args.opset if args.opset is not None else hailo_cfg.get("export_opset", 11))
    simplify = bool(args.simplify) if args.simplify is not None else bool(hailo_cfg.get("export_simplify", False))
    out_dir = ensure_dir(args.out_dir or paths_cfg["export_out_dir"])

    print(f"Exporting ONNX from: {weights}")
    print(f"Export params: imgsz={imgsz}, opset={opset}, simplify={simplify}")
    model = YOLO(str(weights))
    export_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,
            nms=False,
        )
    )

    copied_onnx = out_dir / export_path.name
    if export_path.resolve() != copied_onnx.resolve():
        shutil.copy2(export_path, copied_onnx)
    else:
        copied_onnx = export_path

    train_names = list(classes_cfg.get("train_names", []))
    display_names = list(classes_cfg.get("hailo_display_names", []))
    if not display_names:
        display_names = [name.replace("_", " ") for name in train_names]

    if len(display_names) != len(train_names):
        raise ValueError("classes.hailo_display_names must match classes.train_names length.")

    labels_json_out = args.labels_json_out or paths_cfg["hailo_labels_json_out"]
    labels_json_path = write_hailo_labels_json(
        labels=display_names,
        output_path=labels_json_out,
        detection_threshold=float(hailo_cfg.get("detection_threshold", 0.5)),
        max_boxes=int(hailo_cfg.get("max_boxes", 200)),
    )

    print("Export complete.")
    print(f"ONNX: {copied_onnx}")
    print(f"Hailo labels JSON: {labels_json_path}")


if __name__ == "__main__":
    main()
