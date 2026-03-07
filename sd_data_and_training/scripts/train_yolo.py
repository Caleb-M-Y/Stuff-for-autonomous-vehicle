from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import ensure_dir, load_config, make_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO model using config-defined dataset paths.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--data-yaml", type=str, default=None, help="Optional existing data.yaml override.")
    parser.add_argument("--model", type=str, default=None, help="Base model or checkpoint override.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from a checkpoint instead of starting a new train call.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint path for --resume. Defaults to <project>/<run-name>/weights/last.pt.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size override.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader worker count override.")
    parser.add_argument("--patience", type=int, default=None, help="Early stop patience override.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. '0' or 'cpu'.")
    parser.add_argument("--run-name", type=str, default=None, help="Training run name override.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    train_cfg = config.get("training", {})

    train_project_dir = Path(paths_cfg["train_project_dir"])
    ensure_dir(train_project_dir)

    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        data_yaml = make_data_yaml(config, paths_cfg["data_yaml_out"])

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model_path = args.model or train_cfg.get("model", "yolov8s.pt")
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 120))
    batch = int(args.batch if args.batch is not None else train_cfg.get("batch", 16))
    imgsz = int(args.imgsz if args.imgsz is not None else train_cfg.get("imgsz", 640))
    workers = int(args.workers if args.workers is not None else train_cfg.get("workers", 8))
    patience = int(args.patience if args.patience is not None else train_cfg.get("patience", 30))
    device = str(args.device if args.device is not None else train_cfg.get("device", "0"))
    run_name = args.run_name or train_cfg.get("run_name", "ball_bucket_vnext")
    run_dir = train_project_dir / run_name

    if args.resume:
        checkpoint_path = Path(args.resume_from).expanduser() if args.resume_from else run_dir / "weights" / "last.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"Run output: {run_dir}")
        model = YOLO(str(checkpoint_path))
        model.train(resume=True)

        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"
        print("Resume complete.")
        print(f"best.pt: {best_path}")
        print(f"last.pt: {last_path}")
        return

    print(f"Training model: {model_path}")
    print(f"Data YAML: {data_yaml}")
    print(f"Run output: {run_dir}")

    model = YOLO(model_path)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        patience=patience,
        device=device,
        project=str(train_project_dir),
        name=run_name,
        exist_ok=True,
    )

    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    print("Training complete.")
    print(f"best.pt: {best_path}")
    print(f"last.pt: {last_path}")


if __name__ == "__main__":
    main()
