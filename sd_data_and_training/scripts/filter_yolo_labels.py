from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import ensure_dir, load_config, normalize_class_name


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Detection:
    class_id: int
    x: float
    y: float
    w: float
    h: float
    rest_tokens: list[str]
    raw_line: str

    @property
    def area(self) -> float:
        return self.w * self.h

    def format_line(self, class_id: int | None = None) -> str:
        out_class = self.class_id if class_id is None else class_id
        return " ".join([str(out_class), *self.rest_tokens])


@dataclass
class HueRemapRule:
    src_class: int
    dst_class: int
    hmin: float
    hmax: float
    smin: float
    vmin: float
    raw: str

    def matches(self, h: float, s: float, v: float) -> bool:
        if s < self.smin or v < self.vmin:
            return False
        if self.hmin <= self.hmax:
            return self.hmin <= h <= self.hmax
        # Wrap-around support for red hues, e.g. 170..179 and 0..10 in one rule.
        return h >= self.hmin or h <= self.hmax


def parse_detection(line: str) -> Detection | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None

    return Detection(class_id=class_id, x=x, y=y, w=w, h=h, rest_tokens=parts[1:], raw_line=line.strip())


def class_maps_from_config(config_path: str | None) -> tuple[dict[str, int], dict[int, str]]:
    if not config_path:
        return {}, {}

    config = load_config(config_path)
    names = config["classes"]["train_names"]

    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    for idx, class_name in enumerate(names):
        key = normalize_class_name(class_name)
        name_to_id[key] = idx
        id_to_name[idx] = class_name

    return name_to_id, id_to_name


def resolve_target_classes(
    class_ids: list[int],
    class_names: list[str],
    name_to_id: dict[str, int],
) -> set[int]:
    targets = set(class_ids)

    for name in class_names:
        key = normalize_class_name(name)
        if key not in name_to_id:
            valid = ", ".join(sorted(name_to_id.keys()))
            raise ValueError(f"Unknown class name '{name}'. Normalized='{key}'. Valid class names: {valid}")
        targets.add(name_to_id[key])

    return targets


def resolve_class_token(token: str, name_to_id: dict[str, int]) -> int:
    raw = token.strip()
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        return int(raw)

    key = normalize_class_name(raw)
    if key not in name_to_id:
        valid = ", ".join(sorted(name_to_id.keys())) if name_to_id else "<none loaded>"
        raise ValueError(f"Unknown class token '{token}'. Normalized='{key}'. Valid class names: {valid}")
    return name_to_id[key]


def parse_direct_remaps(raw_rules: list[str], name_to_id: dict[str, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for rule in raw_rules:
        parts = [p.strip() for p in rule.split(":")]
        if len(parts) != 2:
            raise ValueError(f"Invalid --remap rule '{rule}'. Expected SRC:DST")
        src = resolve_class_token(parts[0], name_to_id)
        dst = resolve_class_token(parts[1], name_to_id)
        out[src] = dst
    return out


def parse_hue_remaps(raw_rules: list[str], name_to_id: dict[str, int]) -> list[HueRemapRule]:
    out: list[HueRemapRule] = []
    for raw in raw_rules:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) not in (4, 5, 6):
            raise ValueError(
                "Invalid --hue-remap rule "
                f"'{raw}'. Expected SRC:DST:HMIN:HMAX[:SMIN][:VMIN]"
            )

        src = resolve_class_token(parts[0], name_to_id)
        dst = resolve_class_token(parts[1], name_to_id)
        hmin = float(parts[2])
        hmax = float(parts[3])
        smin = float(parts[4]) if len(parts) >= 5 else 35.0
        vmin = float(parts[5]) if len(parts) >= 6 else 35.0

        if not (0.0 <= hmin <= 179.0 and 0.0 <= hmax <= 179.0):
            raise ValueError(f"Hue bounds must be in [0, 179]. Got {hmin}, {hmax} in '{raw}'")
        if smin < 0.0 or vmin < 0.0:
            raise ValueError(f"SMIN/VMIN must be >= 0 in '{raw}'")

        out.append(
            HueRemapRule(
                src_class=src,
                dst_class=dst,
                hmin=hmin,
                hmax=hmax,
                smin=smin,
                vmin=vmin,
                raw=raw,
            )
        )

    return out


def validate_zone(zone: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = zone
    if not (0.0 <= xmin <= xmax <= 1.0 and 0.0 <= ymin <= ymax <= 1.0):
        raise ValueError("Zone must satisfy 0 <= xmin <= xmax <= 1 and 0 <= ymin <= ymax <= 1")
    return zone


def point_in_zone(x: float, y: float, zone: tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = zone
    return xmin <= x <= xmax and ymin <= y <= ymax


def find_matching_image(images_dir: Path, rel_stem: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = rel_stem.with_suffix(ext)
        image_path = images_dir / candidate
        if image_path.exists():
            return image_path

    parent = images_dir / rel_stem.parent
    if not parent.exists():
        return None

    basename = rel_stem.name
    for candidate in parent.glob(f"{basename}.*"):
        if candidate.suffix.lower() in IMAGE_EXTS:
            return candidate
    return None


def bbox_mean_hsv(image_bgr: Any, det: Detection, cv2_module: Any) -> tuple[float, float, float] | None:
    img_h, img_w = image_bgr.shape[:2]
    x1 = max(0, min(img_w - 1, int(round((det.x - det.w / 2.0) * img_w))))
    y1 = max(0, min(img_h - 1, int(round((det.y - det.h / 2.0) * img_h))))
    x2 = max(1, min(img_w, int(round((det.x + det.w / 2.0) * img_w))))
    y2 = max(1, min(img_h, int(round((det.y + det.h / 2.0) * img_h))))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2_module.cvtColor(roi, cv2_module.COLOR_BGR2HSV)
    mean_h = float(hsv[:, :, 0].mean())
    mean_s = float(hsv[:, :, 1].mean())
    mean_v = float(hsv[:, :, 2].mean())
    return mean_h, mean_s, mean_v


def should_remove(det: Detection, args: argparse.Namespace, target_classes: set[int]) -> tuple[bool, str]:
    reasons: list[str] = []

    if target_classes:
        if det.class_id not in target_classes:
            return False, ""
        reasons.append("class")

    zones: list[tuple[float, float, float, float]] = []
    if args.center_zone is not None:
        zones.append(tuple(args.center_zone))
    if args.exclude_zone:
        zones.extend([tuple(z) for z in args.exclude_zone])

    if zones:
        if not any(point_in_zone(det.x, det.y, zone) for zone in zones):
            return False, ""
        reasons.append("zone")

    if args.area_min is not None:
        if det.area < args.area_min:
            return False, ""
        reasons.append("area_min")

    if args.area_max is not None:
        if det.area > args.area_max:
            return False, ""
        reasons.append("area_max")

    if args.width_min is not None:
        if det.w < args.width_min:
            return False, ""
        reasons.append("width_min")

    if args.width_max is not None:
        if det.w > args.width_max:
            return False, ""
        reasons.append("width_max")

    if args.height_min is not None:
        if det.h < args.height_min:
            return False, ""
        reasons.append("height_min")

    if args.height_max is not None:
        if det.h > args.height_max:
            return False, ""
        reasons.append("height_max")

    return bool(reasons), "+".join(reasons)


def write_label_file(path: Path, lines: list[str]) -> None:
    if not lines:
        path.write_text("", encoding="utf-8")
        return
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter YOLO label files to remove likely false positives by class and geometry."
    )
    parser.add_argument("--labels-dir", type=str, required=True, help="Directory containing YOLO *.txt labels.")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML for class-name lookup.")
    parser.add_argument("--class-ids", type=int, nargs="*", default=[], help="Class IDs to target.")
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="*",
        default=[],
        help="Class names to target (requires --config with classes.train_names).",
    )
    parser.add_argument(
        "--center-zone",
        type=float,
        nargs=4,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Only remove detections whose center (x,y) is inside this normalized zone.",
    )
    parser.add_argument(
        "--exclude-zone",
        type=float,
        nargs=4,
        action="append",
        default=[],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Additional center zones where matched detections will be removed. Repeatable.",
    )
    parser.add_argument("--area-min", type=float, default=None, help="Only remove if box area >= this value.")
    parser.add_argument("--area-max", type=float, default=None, help="Only remove if box area <= this value.")
    parser.add_argument("--width-min", type=float, default=None, help="Only remove if box width >= this value.")
    parser.add_argument("--width-max", type=float, default=None, help="Only remove if box width <= this value.")
    parser.add_argument("--height-min", type=float, default=None, help="Only remove if box height >= this value.")
    parser.add_argument("--height-max", type=float, default=None, help="Only remove if box height <= this value.")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional images root (same relative layout as labels). Required for --hue-remap.",
    )
    parser.add_argument(
        "--remap",
        type=str,
        nargs="*",
        default=[],
        help="Direct class remaps, format SRC:DST. Tokens can be IDs or class names.",
    )
    parser.add_argument(
        "--hue-remap",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Color-aware remap rules, format SRC:DST:HMIN:HMAX[:SMIN][:VMIN]. "
            "Hue uses OpenCV range [0,179]."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Write changes to label files. Defaults to dry-run.")
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Optional backup folder for original label files before writing.",
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        default=None,
        help="Optional CSV path with every removed detection candidate.",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir).resolve()
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels-dir not found: {labels_dir}")

    if args.center_zone is not None:
        validate_zone(tuple(args.center_zone))
    for zone in args.exclude_zone:
        validate_zone(tuple(zone))

    name_to_id, id_to_name = class_maps_from_config(args.config)
    target_classes = resolve_target_classes(args.class_ids, args.class_names, name_to_id)
    direct_remaps = parse_direct_remaps(args.remap, name_to_id)
    hue_remaps = parse_hue_remaps(args.hue_remap, name_to_id)

    images_dir: Path | None = None
    cv2_module = None
    if hue_remaps:
        if not args.images_dir:
            raise ValueError("--images-dir is required when --hue-remap is used")
        images_dir = Path(args.images_dir).resolve()
        if not images_dir.exists():
            raise FileNotFoundError(f"images-dir not found: {images_dir}")
        import cv2 as cv2_import

        cv2_module = cv2_import

    has_remove_filter = any(
        [
            bool(target_classes),
            args.center_zone is not None,
            bool(args.exclude_zone),
            args.area_min is not None,
            args.area_max is not None,
            args.width_min is not None,
            args.width_max is not None,
            args.height_min is not None,
            args.height_max is not None,
        ]
    )

    has_remap = bool(direct_remaps) or bool(hue_remaps)
    if not (has_remove_filter or has_remap):
        raise ValueError(
            "At least one action is required. Use remove filters (e.g. --class-ids/--exclude-zone) "
            "or remaps (e.g. --remap or --hue-remap)."
        )

    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None
    if args.apply and backup_dir is not None:
        ensure_dir(backup_dir)

    report_rows: list[dict[str, str | int | float]] = []
    label_files = sorted(labels_dir.rglob("*.txt"))

    files_changed = 0
    files_with_removals = 0
    files_with_remaps = 0
    detections_seen = 0
    detections_removed = 0
    detections_remapped = 0
    parse_skipped = 0
    missing_images_for_hue = 0

    for label_path in label_files:
        rel_path = label_path.relative_to(labels_dir).as_posix()
        raw_lines = label_path.read_text(encoding="utf-8").splitlines()

        image_bgr = None
        if hue_remaps and images_dir is not None and cv2_module is not None:
            rel_stem = Path(rel_path).with_suffix("")
            image_path = find_matching_image(images_dir, rel_stem)
            if image_path is None:
                missing_images_for_hue += 1
            else:
                image_bgr = cv2_module.imread(str(image_path))
                if image_bgr is None:
                    missing_images_for_hue += 1

        kept_lines: list[str] = []
        removed_here = 0
        remapped_here = 0

        for line_idx, raw in enumerate(raw_lines, start=1):
            stripped = raw.strip()
            if not stripped:
                continue

            det = parse_detection(stripped)
            if det is None:
                parse_skipped += 1
                kept_lines.append(stripped)
                continue

            detections_seen += 1
            remove, remove_reason = should_remove(det, args, target_classes)
            if remove:
                detections_removed += 1
                removed_here += 1
                report_rows.append(
                    {
                        "label_file": rel_path,
                        "line": line_idx,
                        "action": "remove",
                        "from_class_id": det.class_id,
                        "from_class_name": id_to_name.get(det.class_id, str(det.class_id)),
                        "to_class_id": "",
                        "to_class_name": "",
                        "x": det.x,
                        "y": det.y,
                        "w": det.w,
                        "h": det.h,
                        "area": det.area,
                        "h_mean": "",
                        "s_mean": "",
                        "v_mean": "",
                        "reason": remove_reason,
                    }
                )
                continue

            out_class = det.class_id
            remap_reason = ""
            h_mean = ""
            s_mean = ""
            v_mean = ""

            if out_class in direct_remaps:
                dst = direct_remaps[out_class]
                if dst != out_class:
                    remap_reason = f"direct:{out_class}->{dst}"
                    out_class = dst

            if hue_remaps and image_bgr is not None:
                hsv = bbox_mean_hsv(image_bgr, det, cv2_module)
                if hsv is not None:
                    h_val, s_val, v_val = hsv
                    h_mean = round(h_val, 3)
                    s_mean = round(s_val, 3)
                    v_mean = round(v_val, 3)
                    for rule in hue_remaps:
                        if out_class != rule.src_class:
                            continue
                        if rule.matches(h_val, s_val, v_val):
                            if rule.dst_class != out_class:
                                prev_class = out_class
                                out_class = rule.dst_class
                                suffix = f"hue:{rule.raw}:{prev_class}->{out_class}"
                                remap_reason = f"{remap_reason}+{suffix}" if remap_reason else suffix
                            break

            if out_class != det.class_id:
                detections_remapped += 1
                remapped_here += 1
                report_rows.append(
                    {
                        "label_file": rel_path,
                        "line": line_idx,
                        "action": "remap",
                        "from_class_id": det.class_id,
                        "from_class_name": id_to_name.get(det.class_id, str(det.class_id)),
                        "to_class_id": out_class,
                        "to_class_name": id_to_name.get(out_class, str(out_class)),
                        "x": det.x,
                        "y": det.y,
                        "w": det.w,
                        "h": det.h,
                        "area": det.area,
                        "h_mean": h_mean,
                        "s_mean": s_mean,
                        "v_mean": v_mean,
                        "reason": remap_reason or "remap",
                    }
                )

            kept_lines.append(det.format_line(out_class))

        if removed_here > 0:
            files_with_removals += 1
        if remapped_here > 0:
            files_with_remaps += 1

        if removed_here > 0 or remapped_here > 0:
            if args.apply:
                if backup_dir is not None:
                    backup_path = backup_dir / rel_path
                    ensure_dir(backup_path.parent)
                    shutil.copy2(label_path, backup_path)
                write_label_file(label_path, kept_lines)
                files_changed += 1

    if args.report_csv:
        report_path = Path(args.report_csv).resolve()
        ensure_dir(report_path.parent)
        with report_path.open("w", newline="", encoding="utf-8") as fp:
            fieldnames = [
                "label_file",
                "line",
                "action",
                "from_class_id",
                "from_class_name",
                "to_class_id",
                "to_class_name",
                "x",
                "y",
                "w",
                "h",
                "area",
                "h_mean",
                "s_mean",
                "v_mean",
                "reason",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Wrote report CSV: {report_path}")

    print("Filter complete.")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Label files scanned: {len(label_files)}")
    print(f"Files with matched removals: {files_with_removals}")
    print(f"Files with remaps: {files_with_remaps}")
    print(f"Detections scanned: {detections_seen}")
    print(f"Detections matched for removal: {detections_removed}")
    print(f"Detections remapped: {detections_remapped}")
    print(f"Lines skipped (non-YOLO parse): {parse_skipped}")
    if hue_remaps:
        print(f"Label files missing paired images for hue remap: {missing_images_for_hue}")
    if args.apply:
        print(f"Files written: {files_changed}")


if __name__ == "__main__":
    main()
