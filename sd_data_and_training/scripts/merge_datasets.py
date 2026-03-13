from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [
    "blue_ball",
    "blue_bucket",
    "green_ball",
    "green_bucket",
    "red_ball",
    "red_bucket",
    "yellow_ball",
    "yellow_bucket",
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    split: str
    images_dir: Path
    labels_dir: Path
    recursive: bool = True


@dataclass
class SourceStats:
    name: str
    split: str
    scanned_images: int = 0
    copied_pairs: int = 0
    copied_nonempty_labels: int = 0
    missing_labels: int = 0


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    out = []
    for ch in lowered:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def iter_images(images_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        candidates = images_dir.rglob("*")
    else:
        candidates = images_dir.glob("*")
    files = [p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def resolve_label_path(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = safe_relative(image_path, images_root)
    return labels_root / rel.with_suffix(".txt")


def make_unique_stem(source_name: str, rel_image_path: Path, used_stems: set[str]) -> str:
    source_tag = slugify(source_name)
    stem_tag = slugify(rel_image_path.stem)
    candidate = f"{source_tag}__{stem_tag}"
    if candidate not in used_stems:
        used_stems.add(candidate)
        return candidate

    digest = hashlib.sha1(str(rel_image_path).encode("utf-8")).hexdigest()[:8]
    candidate = f"{source_tag}__{stem_tag}__{digest}"
    if candidate not in used_stems:
        used_stems.add(candidate)
        return candidate

    suffix = 2
    while True:
        numbered = f"{candidate}__{suffix}"
        if numbered not in used_stems:
            used_stems.add(numbered)
            return numbered
        suffix += 1


def ensure_split_dirs(root: Path) -> None:
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def default_sources(repo_root: Path) -> list[SourceSpec]:
    datasets_root = repo_root / "RoboFlowDatasets"

    return [
        SourceSpec(
            name="dataset_3_11_train",
            split="train",
            images_dir=datasets_root / "DATA_SET_3-11-2026" / "train" / "images",
            labels_dir=datasets_root / "DATA_SET_3-11-2026" / "train" / "labels",
            recursive=True,
        ),
        SourceSpec(
            name="dataset_3_11_valid",
            split="valid",
            images_dir=datasets_root / "DATA_SET_3-11-2026" / "valid" / "images",
            labels_dir=datasets_root / "DATA_SET_3-11-2026" / "valid" / "labels",
            recursive=True,
        ),
        SourceSpec(
            name="dataset_3_11_test",
            split="test",
            images_dir=datasets_root / "DATA_SET_3-11-2026" / "test" / "images",
            labels_dir=datasets_root / "DATA_SET_3-11-2026" / "test" / "labels",
            recursive=True,
        ),
        SourceSpec(
            name="dataset_3_5_train",
            split="train",
            images_dir=datasets_root / "DATA_SET_3-5-2026" / "train",
            labels_dir=datasets_root / "DATA_SET_3-5-2026" / "train",
            recursive=False,
        ),
        SourceSpec(
            name="dataset_3_12_images_19",
            split="train",
            images_dir=datasets_root / "3-12-data-collection" / "3-12-raw-images" / "images_19",
            labels_dir=datasets_root / "3-12-data-collection" / "3-12-auto-annotations" / "images_19",
            recursive=False,
        ),
        SourceSpec(
            name="dataset_3_12_images_20",
            split="train",
            images_dir=datasets_root / "3-12-data-collection" / "3-12-raw-images" / "images_20",
            labels_dir=datasets_root / "3-12-data-collection" / "3-12-auto-annotations" / "images_20",
            recursive=False,
        ),
        SourceSpec(
            name="dataset_3_12_images_21",
            split="train",
            images_dir=datasets_root / "3-12-data-collection" / "3-12-raw-images" / "images_21",
            labels_dir=datasets_root / "3-12-data-collection" / "3-12-auto-annotations" / "images_21",
            recursive=False,
        ),
    ]


def merge_sources(sources: list[SourceSpec], out_root: Path) -> list[SourceStats]:
    ensure_split_dirs(out_root)
    used_stems_by_split = {"train": set(), "valid": set(), "test": set()}
    all_stats: list[SourceStats] = []

    for source in sources:
        if source.split not in used_stems_by_split:
            raise ValueError(f"Unsupported split '{source.split}' for source '{source.name}'")
        if not source.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {source.images_dir}")
        if not source.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {source.labels_dir}")

        stats = SourceStats(name=source.name, split=source.split)
        used_stems = used_stems_by_split[source.split]

        for image_path in iter_images(source.images_dir, source.recursive):
            stats.scanned_images += 1
            label_path = resolve_label_path(image_path, source.images_dir, source.labels_dir)
            if not label_path.exists():
                stats.missing_labels += 1
                continue

            rel_image = safe_relative(image_path, source.images_dir)
            stem = make_unique_stem(source.name, rel_image, used_stems)

            dest_image = out_root / source.split / "images" / f"{stem}{image_path.suffix.lower()}"
            dest_label = out_root / source.split / "labels" / f"{stem}.txt"

            shutil.copy2(image_path, dest_image)
            shutil.copy2(label_path, dest_label)

            stats.copied_pairs += 1
            if label_path.stat().st_size > 0:
                stats.copied_nonempty_labels += 1

        all_stats.append(stats)

    return all_stats


def write_data_yaml(data_yaml_path: Path, merged_root: Path) -> None:
    lines = [
        "path: .",
        f"train: {merged_root.joinpath('train', 'images').resolve().as_posix()}",
        f"val: {merged_root.joinpath('valid', 'images').resolve().as_posix()}",
        f"test: {merged_root.joinpath('test', 'images').resolve().as_posix()}",
        "names:",
    ]
    for name in CLASS_NAMES:
        lines.append(f"  - {name}")

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_split_counts(merged_root: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        images_dir = merged_root / split / "images"
        labels_dir = merged_root / split / "labels"
        labels = sorted(labels_dir.glob("*.txt"))
        summary[split] = {
            "images": len([p for p in images_dir.glob("*") if p.is_file()]),
            "labels": len(labels),
            "nonempty_labels": len([p for p in labels if p.stat().st_size > 0]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge cleaned YOLO datasets into one master dataset.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repository root. Defaults to two levels above this script.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output merged dataset directory. Defaults to RoboFlowDatasets/DATA_SET_3-13-CY.",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=None,
        help="Path to output data.yaml. Defaults to sd_data_and_training/artifacts/data_3_13_CY.yaml.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional summary report path. Defaults to <out-dir>/merge_report.json.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output directory before merge if it already exists.",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[2]

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo_root / "RoboFlowDatasets" / "DATA_SET_3-13-CY")
    data_yaml_path = (
        Path(args.data_yaml).resolve()
        if args.data_yaml
        else (repo_root / "sd_data_and_training" / "artifacts" / "data_3_13_CY.yaml")
    )
    report_json_path = Path(args.report_json).resolve() if args.report_json else (out_dir / "merge_report.json")

    if out_dir.exists() and args.clear_output:
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    sources = default_sources(repo_root)
    source_stats = merge_sources(sources, out_dir)
    write_data_yaml(data_yaml_path, out_dir)
    split_summary = summarize_split_counts(out_dir)

    payload = {
        "repo_root": str(repo_root),
        "out_dir": str(out_dir),
        "data_yaml": str(data_yaml_path),
        "sources": [asdict(source) for source in sources],
        "source_stats": [asdict(stat) for stat in source_stats],
        "split_summary": split_summary,
        "class_names": CLASS_NAMES,
    }
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Merge complete.")
    print(f"Output dataset: {out_dir}")
    print(f"Data YAML: {data_yaml_path}")
    print(f"Report JSON: {report_json_path}")
    print("Split summary:")
    for split in ("train", "valid", "test"):
        stats = split_summary[split]
        print(
            f"  {split}: images={stats['images']} labels={stats['labels']} nonempty_labels={stats['nonempty_labels']}"
        )


if __name__ == "__main__":
    main()
