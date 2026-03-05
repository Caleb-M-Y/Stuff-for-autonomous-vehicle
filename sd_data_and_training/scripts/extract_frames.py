from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int,
    max_frames: int,
    prefix: str,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % every_n_frames == 0:
            image_name = f"{prefix}_{video_path.stem}_f{frame_index:06d}.jpg"
            image_path = output_dir / image_name
            cv2.imwrite(str(image_path), frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break

        frame_index += 1

    capture.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from one video or all videos in a folder.")
    parser.add_argument("--video", type=str, default=None, help="Path to one video file.")
    parser.add_argument("--video-dir", type=str, default=None, help="Path to a folder of videos.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write extracted images.")
    parser.add_argument("--every-n-frames", type=int, default=8, help="Save one image every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max images per video (0 means no limit).")
    parser.add_argument("--prefix", type=str, default="capture", help="Prefix for image filenames.")
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        raise ValueError("Provide --video or --video-dir.")

    out_dir = Path(args.out_dir)

    videos: list[Path] = []
    if args.video:
        videos.append(Path(args.video))
    if args.video_dir:
        video_dir = Path(args.video_dir)
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.m4v"):
            videos.extend(video_dir.glob(ext))

    videos = sorted({v.resolve() for v in videos})
    if not videos:
        raise FileNotFoundError("No videos found for extraction.")

    total = 0
    for video_path in videos:
        count = extract_frames(
            video_path=video_path,
            output_dir=out_dir,
            every_n_frames=max(1, args.every_n_frames),
            max_frames=max(0, args.max_frames),
            prefix=args.prefix,
        )
        total += count
        print(f"Saved {count} frames from {video_path.name}")

    print(f"Done. Total extracted frames: {total}")


if __name__ == "__main__":
    main()
