from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import class_name_map, ensure_dir, load_config, normalize_class_name


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _coco_bbox_to_yolo_xywhn(bbox: list[float], image_w: int, image_h: int) -> tuple[float, float, float, float] | None:
    if image_w <= 0 or image_h <= 0:
        return None
    if len(bbox) < 4:
        return None

    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if w <= 0.0 or h <= 0.0:
        return None

    x1 = _clip(x, 0.0, float(image_w))
    y1 = _clip(y, 0.0, float(image_h))
    x2 = _clip(x + w, 0.0, float(image_w))
    y2 = _clip(y + h, 0.0, float(image_h))

    w_clipped = x2 - x1
    h_clipped = y2 - y1
    if w_clipped <= 0.0 or h_clipped <= 0.0:
        return None

    cx = (x1 + x2) / (2.0 * image_w)
    cy = (y1 + y2) / (2.0 * image_h)
    wn = w_clipped / image_w
    hn = h_clipped / image_h

    return (
        _clip(cx, 0.0, 1.0),
        _clip(cy, 0.0, 1.0),
        _clip(wn, 0.0, 1.0),
        _clip(hn, 0.0, 1.0),
    )


def convert_split(
    split_dir: Path,
    split_key: str,
    target_class_map: dict[str, int],
    train_names: list[str],
    clear_existing: bool,
    write_empty_labels: bool,
) -> dict[str, Any]:
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found for split '{split_key}': {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as fp:
        coco = json.load(fp)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        if "id" not in image:
            continue
        image_by_id[int(image["id"])] = image

    category_name_by_id: dict[int, str] = {}
    for category in categories:
        if "id" not in category:
            continue
        category_name_by_id[int(category["id"])] = str(category.get("name", ""))

    lines_by_image_id: dict[int, list[str]] = defaultdict(list)

    skipped_unknown_category: Counter[str] = Counter()
    kept_by_class_idx: Counter[int] = Counter()

    skipped_missing_image_ref = 0
    skipped_bad_bbox = 0
    skipped_iscrowd = 0
    kept_annotations = 0

    for ann in annotations:
        if int(ann.get("iscrowd", 0)) == 1:
            skipped_iscrowd += 1
            continue

        if "image_id" not in ann or "category_id" not in ann:
            skipped_bad_bbox += 1
            continue

        image_id = int(ann["image_id"])
        image_info = image_by_id.get(image_id)
        if image_info is None:
            skipped_missing_image_ref += 1
            continue

        category_id = int(ann["category_id"])
        raw_category_name = category_name_by_id.get(category_id, str(category_id))
        normalized_category_name = normalize_class_name(raw_category_name)

        class_idx = target_class_map.get(normalized_category_name)
        if class_idx is None:
            skipped_unknown_category[normalized_category_name or str(category_id)] += 1
            continue

        image_w = int(image_info.get("width", 0))
        image_h = int(image_info.get("height", 0))
        bbox = ann.get("bbox", [])

        converted = _coco_bbox_to_yolo_xywhn(bbox, image_w, image_h)
        if converted is None:
            skipped_bad_bbox += 1
            continue

        x, y, w, h = converted
        lines_by_image_id[image_id].append(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        kept_annotations += 1
        kept_by_class_idx[class_idx] += 1

    cleared_existing = 0
    labels_written = 0
    empty_labels_written = 0
    missing_image_files = 0

    for image in images:
        if "id" not in image:
            continue

        image_rel = Path(str(image.get("file_name", "")))
        if not image_rel.name:
            continue

        image_path = split_dir / image_rel
        if not image_path.exists():
            # Some exports store only basename while others include subfolders.
            image_path = split_dir / image_rel.name

        if not image_path.exists():
            missing_image_files += 1
            continue

        label_path = image_path.with_suffix(".txt")
        ensure_dir(label_path.parent)

        if clear_existing and label_path.exists():
            label_path.unlink()
            cleared_existing += 1

        image_id = int(image["id"])
        lines = lines_by_image_id.get(image_id, [])

        if lines:
            with label_path.open("w", encoding="utf-8") as fp:
                fp.write("\n".join(lines) + "\n")
            labels_written += 1
        elif write_empty_labels:
            label_path.write_text("", encoding="utf-8")
            empty_labels_written += 1

    kept_by_class = {train_names[idx]: count for idx, count in sorted(kept_by_class_idx.items())}

    return {
        "split": split_key,
        "annotation_path": str(annotation_path),
        "images_total": len(images),
        "annotations_total": len(annotations),
        "kept_annotations": kept_annotations,
        "kept_by_class": kept_by_class,
        "labels_written": labels_written,
        "empty_labels_written": empty_labels_written,
        "cleared_existing": cleared_existing,
        "missing_image_files": missing_image_files,
        "skipped_missing_image_ref": skipped_missing_image_ref,
        "skipped_bad_bbox": skipped_bad_bbox,
        "skipped_iscrowd": skipped_iscrowd,
        "skipped_unknown_category": dict(sorted(skipped_unknown_category.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Roboflow COCO split exports into YOLO txt labels using config class order."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Folder containing train/valid/test COCO split folders.")
    parser.add_argument("--train-split", type=str, default="train", help="Train split folder name under dataset-root.")
    parser.add_argument("--val-split", type=str, default="valid", help="Validation split folder name under dataset-root.")
    parser.add_argument("--test-split", type=str, default="test", help="Test split folder name under dataset-root.")
    parser.add_argument("--skip-test", action="store_true", help="Skip converting test split.")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing .txt labels before writing new ones.")
    parser.add_argument(
        "--write-empty-labels",
        action="store_true",
        help="Also create empty label files for images with zero kept annotations.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_names: list[str] = list(config["classes"]["train_names"])
    target_class_map = class_name_map(train_names)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {dataset_root}")

    print("Target class order from config:")
    for idx, name in enumerate(train_names):
        print(f"  {idx}: {name}")

    splits: list[tuple[str, str]] = [
        ("train", args.train_split),
        ("val", args.val_split),
    ]
    if not args.skip_test:
        splits.append(("test", args.test_split))

    summaries: list[dict[str, Any]] = []
    for split_key, split_folder in splits:
        split_dir = dataset_root / split_folder
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found for {split_key}: {split_dir}")

        summary = convert_split(
            split_dir=split_dir,
            split_key=split_key,
            target_class_map=target_class_map,
            train_names=train_names,
            clear_existing=args.clear_existing,
            write_empty_labels=args.write_empty_labels,
        )
        summaries.append(summary)

    print("\nConversion summary:")
    total_kept = 0
    total_annotations = 0
    for summary in summaries:
        total_kept += int(summary["kept_annotations"])
        total_annotations += int(summary["annotations_total"])

        print(
            f"[{summary['split']}] labels_written={summary['labels_written']} "
            f"kept={summary['kept_annotations']}/{summary['annotations_total']} "
            f"missing_images={summary['missing_image_files']}"
        )

        if summary["kept_by_class"]:
            print(f"[{summary['split']}] kept_by_class:")
            for class_name, count in summary["kept_by_class"].items():
                print(f"  {class_name}: {count}")

        if summary["skipped_unknown_category"]:
            print(f"[{summary['split']}] skipped categories not in config class list:")
            for class_name, count in summary["skipped_unknown_category"].items():
                print(f"  {class_name}: {count}")

    print(f"\nDone. Total kept annotations: {total_kept}/{total_annotations}")


if __name__ == "__main__":
    main()
