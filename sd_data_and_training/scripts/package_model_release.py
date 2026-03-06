from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import ensure_dir, load_config


def _git_short_sha(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def _default_release_id(repo_root: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"ballbucket-{timestamp}-{_git_short_sha(repo_root)}"


def _read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    return rows[-1] if rows else {}


def _metric(metrics: dict[str, str], key: str) -> str:
    value = metrics.get(key)
    if value is None or value == "":
        return "n/a"
    return value


def _write_model_info(
    output_path: Path,
    release_id: str,
    git_commit: str,
    run_name: str,
    onnx_name: str,
    labels_name: str,
    class_names: list[str],
    metrics: dict[str, str],
    notes: str,
) -> None:
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"release_id: {release_id}",
        f"created_utc: {created_utc}",
        f"git_commit: {git_commit}",
        f"source_run_name: {run_name}",
        "",
        f"onnx_file: {onnx_name}",
        f"labels_json_file: {labels_name}",
        "",
        "classes:",
    ]

    for name in class_names:
        lines.append(f"- {name}")

    lines.extend(
        [
            "",
            "validation_metrics:",
            f"epoch: {_metric(metrics, 'epoch')}",
            f"precision_B: {_metric(metrics, 'metrics/precision(B)')}",
            f"recall_B: {_metric(metrics, 'metrics/recall(B)')}",
            f"mAP50_B: {_metric(metrics, 'metrics/mAP50(B)')}",
            f"mAP50_95_B: {_metric(metrics, 'metrics/mAP50-95(B)')}",
            "",
            f"notes: {notes or 'n/a'}",
            "",
        ]
    )

    with output_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def _zip_release(release_dir: Path, zip_path: Path) -> Path:
    ensure_dir(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(release_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(release_dir.parent)
                zf.write(file_path, arcname)

    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a versioned model release bundle for robot deployment.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model. Defaults to export_out_dir/best.onnx.")
    parser.add_argument("--labels-json", type=str, default=None, help="Path to labels JSON. Defaults to config path.")
    parser.add_argument("--results-csv", type=str, default=None, help="Path to training results.csv for metadata.")
    parser.add_argument("--release-id", type=str, default=None, help="Release ID override.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output root for release bundles.")
    parser.add_argument("--notes", type=str, default="", help="Optional notes to include in model_info.txt.")
    parser.add_argument("--skip-zip", action="store_true", help="Skip creating a .zip archive.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg: dict[str, Any] = config["paths"]
    train_cfg: dict[str, Any] = config.get("training", {})
    classes_cfg: dict[str, Any] = config.get("classes", {})

    repo_root = Path(paths_cfg.get("repo_root", ".")).resolve()
    release_id = args.release_id or _default_release_id(repo_root)
    out_root = Path(args.out_dir) if args.out_dir else repo_root / "sd_data_and_training" / "artifacts" / "model_releases"
    out_root = ensure_dir(out_root)

    onnx_path = Path(args.onnx) if args.onnx else Path(paths_cfg["export_out_dir"]) / "best.onnx"
    labels_json_path = Path(args.labels_json) if args.labels_json else Path(paths_cfg["hailo_labels_json_out"])

    run_name = str(train_cfg.get("run_name", "unknown_run"))
    default_results_csv = Path(paths_cfg["train_project_dir"]) / run_name / "results.csv"
    results_csv_path = Path(args.results_csv) if args.results_csv else default_results_csv

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not labels_json_path.exists():
        raise FileNotFoundError(f"Labels JSON not found: {labels_json_path}")

    release_dir = ensure_dir(out_root / release_id)
    copied_onnx = release_dir / onnx_path.name
    copied_labels = release_dir / labels_json_path.name
    shutil.copy2(onnx_path, copied_onnx)
    shutil.copy2(labels_json_path, copied_labels)

    metrics = _read_last_metrics(results_csv_path)
    class_names = list(classes_cfg.get("train_names", []))
    info_path = release_dir / "model_info.txt"
    _write_model_info(
        output_path=info_path,
        release_id=release_id,
        git_commit=_git_short_sha(repo_root),
        run_name=run_name,
        onnx_name=copied_onnx.name,
        labels_name=copied_labels.name,
        class_names=class_names,
        metrics=metrics,
        notes=args.notes,
    )

    print(f"Release folder: {release_dir}")
    print(f"ONNX: {copied_onnx}")
    print(f"Labels JSON: {copied_labels}")
    print(f"Model info: {info_path}")

    if not args.skip_zip:
        zip_path = _zip_release(release_dir, out_root / f"{release_id}.zip")
        print(f"Release zip: {zip_path}")


if __name__ == "__main__":
    main()