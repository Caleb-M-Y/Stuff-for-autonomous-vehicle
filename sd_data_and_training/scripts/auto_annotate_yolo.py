from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from common import class_name_map, ensure_dir, list_image_paths, load_config, normalize_class_name


def _model_name_lookup(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate images into YOLO labels with a teacher model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--images-dir", type=str, default=None, help="Input images directory override.")
    parser.add_argument("--labels-out", type=str, default=None, help="Output labels directory override.")
    parser.add_argument("--preview-out", type=str, default=None, help="Optional output preview images dir.")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher weights override.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size override.")
    parser.add_argument("--max-det", type=int, default=None, help="Max detections per image override.")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive image search.")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview rendering.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    classes_cfg = config["classes"]
    auto_cfg = config.get("autolabel", {})

    train_names: list[str] = classes_cfg["train_names"]
    target_map = class_name_map(train_names)

    images_dir = Path(args.images_dir or paths_cfg["unlabeled_images_dir"])
    labels_out = Path(args.labels_out or paths_cfg["autolabel_labels_out_dir"])

    save_preview = bool(auto_cfg.get("save_preview", True)) and not args.no_preview
    preview_out = Path(args.preview_out or paths_cfg.get("autolabel_preview_out_dir", labels_out / "_preview"))

    teacher_model = args.teacher_model or auto_cfg.get("teacher_model", "yolov8s.pt")
    conf = float(args.conf if args.conf is not None else auto_cfg.get("conf", 0.35))
    iou = float(args.iou if args.iou is not None else auto_cfg.get("iou", 0.50))
    imgsz = int(args.imgsz if args.imgsz is not None else auto_cfg.get("imgsz", 640))
    max_det = int(args.max_det if args.max_det is not None else auto_cfg.get("max_det", 60))

    ensure_dir(labels_out)
    if save_preview:
        ensure_dir(preview_out)

    image_paths = list_image_paths(images_dir, recursive=not args.no_recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    print(f"Loading teacher model: {teacher_model}")
    model = YOLO(teacher_model)
    names_lookup = _model_name_lookup(model.names)

    total_boxes = 0
    kept_boxes = 0
    class_counter: Counter[str] = Counter()
    skipped_class_counter: Counter[str] = Counter()
    non_empty_files = 0

    for image_path in tqdm(image_paths, desc="Auto-labeling", unit="img"):
        try:
            rel = image_path.relative_to(images_dir)
        except ValueError:
            rel = Path(image_path.name)

        label_path = labels_out / rel.with_suffix(".txt")
        ensure_dir(label_path.parent)

        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        lines: list[str] = []
        preview_rows: list[tuple[float, float, float, float, str, float]] = []

        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            total_boxes += len(cls_ids)

            for idx, cls_id in enumerate(cls_ids):
                pred_name = names_lookup.get(int(cls_id), str(cls_id))
                norm_name = normalize_class_name(pred_name)
                class_index = target_map.get(norm_name)

                if class_index is None:
                    skipped_class_counter[norm_name] += 1
                    continue

                x, y, w, h = xywhn[idx]
                lines.append(f"{class_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                kept_boxes += 1
                class_counter[train_names[class_index]] += 1

                x1, y1, x2, y2 = xyxy[idx]
                preview_rows.append((x1, y1, x2, y2, train_names[class_index], float(confs[idx])))

        with label_path.open("w", encoding="utf-8") as fp:
            if lines:
                fp.write("\n".join(lines) + "\n")
                non_empty_files += 1

        if save_preview:
            image = cv2.imread(str(image_path))
            if image is not None:
                for x1, y1, x2, y2, class_name, score in preview_rows:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f"{class_name} {score:.2f}",
                        (int(x1), max(16, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                preview_path = preview_out / rel
                ensure_dir(preview_path.parent)
                cv2.imwrite(str(preview_path), image)

    print("Auto-label complete.")
    print(f"Images processed: {len(image_paths)}")
    print(f"Label files with >=1 box: {non_empty_files}")
    print(f"Boxes kept / predicted: {kept_boxes} / {total_boxes}")

    if class_counter:
        print("Kept class counts:")
        for class_name, count in class_counter.most_common():
            print(f"  {class_name}: {count}")

    if skipped_class_counter:
        print("Skipped predicted class names not found in config classes:")
        for class_name, count in skipped_class_counter.most_common():
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
