from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2

from common import ensure_dir, list_image_paths, load_config


def _parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes

    with label_path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            boxes.append((cls, x, y, w, h))
    return boxes


def _xywhn_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = int((x - (w / 2.0)) * width)
    y1 = int((y - (h / 2.0)) * height)
    x2 = int((x + (w / 2.0)) * width)
    y2 = int((y + (h / 2.0)) * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sampled visual review set from YOLO labels.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--images-dir", type=str, default=None, help="Images root override.")
    parser.add_argument("--labels-dir", type=str, default=None, help="Labels root override.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output review directory override.")
    parser.add_argument("--sample-n", type=int, default=400, help="Number of non-empty label images to sample.")
    parser.add_argument("--empty-sample-n", type=int, default=120, help="Number of empty-label images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    class_names: list[str] = list(config["classes"]["train_names"])

    images_dir = Path(args.images_dir or paths_cfg["unlabeled_images_dir"]).resolve()
    labels_dir = Path(args.labels_dir or paths_cfg["autolabel_labels_out_dir"]).resolve()
    out_dir = Path(args.out_dir or (labels_dir.parent / "review_subset")).resolve()

    ensure_dir(out_dir)
    ensure_dir(out_dir / "non_empty")
    ensure_dir(out_dir / "empty")

    image_paths = list_image_paths(images_dir, recursive=True)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    non_empty: list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]] = []
    empty: list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]] = []

    for image_path in image_paths:
        rel = image_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        boxes = _parse_label_file(label_path)
        row = (image_path, label_path, boxes)
        if boxes:
            non_empty.append(row)
        else:
            empty.append(row)

    rng = random.Random(args.seed)
    non_empty_pick = rng.sample(non_empty, min(args.sample_n, len(non_empty))) if non_empty else []
    empty_pick = rng.sample(empty, min(args.empty_sample_n, len(empty))) if empty else []

    manifest_path = out_dir / "review_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "group",
                "image_rel",
                "image_abs",
                "label_abs",
                "label_exists",
                "box_count",
                "preview_abs",
            ]
        )

        for group_name, picks in (("non_empty", non_empty_pick), ("empty", empty_pick)):
            for image_path, label_path, boxes in picks:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                h, w = image.shape[:2]
                for cls, x, y, bw, bh in boxes:
                    x1, y1, x2, y2 = _xywhn_to_xyxy(x, y, bw, bh, w, h)
                    name = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        name,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if group_name == "empty":
                    cv2.putText(
                        image,
                        "EMPTY LABEL",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                rel = image_path.relative_to(images_dir)
                preview_path = out_dir / group_name / rel
                ensure_dir(preview_path.parent)
                cv2.imwrite(str(preview_path), image)

                writer.writerow(
                    [
                        group_name,
                        str(rel).replace("\\", "/"),
                        str(image_path),
                        str(label_path),
                        str(label_path.exists()).lower(),
                        len(boxes),
                        str(preview_path),
                    ]
                )

    print("Review set complete.")
    print(f"Images scanned: {len(image_paths)}")
    print(f"Non-empty labels: {len(non_empty)}")
    print(f"Empty labels: {len(empty)}")
    print(f"Sampled non-empty: {len(non_empty_pick)}")
    print(f"Sampled empty: {len(empty_pick)}")
    print(f"Manifest: {manifest_path}")
    print(f"Preview root: {out_dir}")


if __name__ == "__main__":
    main()
