from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = [
    "blue ball",
    "blue bucket",
    "green ball",
    "green bucket",
    "red ball",
    "red bucket",
    "yellow ball",
    "yellow bucket",
]


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_model = repo_root / "sd_data_and_training" / "exports" / "3-13-CY.onnx"

    parser = argparse.ArgumentParser(
        description="Quick ONNX inference check for class mapping validation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model),
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="g (3).jpg",
        help="Path to test image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="g (3)_annotated.jpg",
        help="Path to save annotated output image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device, e.g. 'cpu' or '0'.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated image in a window.",
    )
    return parser.parse_args()


def _class_name(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"unknown_{class_id}"


def main() -> None:
    args = _parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    print("Expected class mapping:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id}: {class_name}")

    model = YOLO(str(model_path))
    result = model.predict(
        source=str(image_path),
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        device=str(args.device),
        verbose=False,
    )[0]

    if isinstance(result.names, dict):
        print("\nModel-reported class names:")
        for class_id in sorted(result.names):
            print(f"  {class_id}: {result.names[class_id]}")

    print("\nDetections:")
    if result.boxes is None or len(result.boxes) == 0:
        print("  No detections found.")
    else:
        for det_i, box in enumerate(result.boxes, start=1):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            expected_name = _class_name(class_id)
            model_name = result.names.get(class_id, "<missing>") if isinstance(result.names, dict) else "<n/a>"
            print(
                f"  {det_i}. class_id={class_id} expected='{expected_name}' "
                f"model_name='{model_name}' conf={confidence:.4f}"
            )

    annotated = cv2.imread(str(image_path))
    if annotated is None:
        raise RuntimeError(f"Failed to load image for annotation: {image_path}")

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            class_name = _class_name(class_id)
            label = f"{class_name} {confidence:.2f}"

            # Stable per-class color for easier visual scanning.
            color = (
                50 + (37 * class_id) % 205,
                50 + (83 * class_id) % 205,
                50 + (149 * class_id) % 205,
            )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"\nSaved annotated image: {output_path}")

    if args.show:
        cv2.imshow("3-13-CY ONNX Test", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
