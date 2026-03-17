"""
Autonomous course runner: RealSense + Hailo YOLO + Pico control thread.

High-level architecture:
1) Acquire aligned color/depth frames from Intel RealSense D455.
2) Run YOLO via Hailo GStreamer pipeline on color frames.
3) Forward detections/depth into state_machine handlers to compute control commands.
4) Continuously stream latest command to Pico over serial from a background thread.

Design notes:
- Distance is depth-first (hardware depth when valid) with geometry fallback handled in
    state_machine.py.
- Runtime behavior is tuned primarily through autonomy_tuning.py constants.
- Optional run logging captures terminal output, telemetry CSV, metadata, and annotated video.

Run from repo root or AUTONOMOUS_CODE_SD; ensure HEF and labels JSON paths are valid.
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import threading
from datetime import datetime
from time import sleep
from pathlib import Path

import numpy as np
import cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyrealsense2 as rs
import serial

# Hailo inference (same pattern as camera_test2)
import hailo
import autonomy_tuning as tune

# Local state machine: best detection, distance depth/geometry, mode handlers
from state_machine import (
    handle_pause,
    handle_pick,
    handle_drop,
    handle_fixed_ball,
    handle_fixed_bucket,
    handle_fixed_back,
    handle_swivel_small_left,
    handle_swivel_large_right,
    handle_detect_ball,
    handle_detect_bucket,
)


class _TeeStream:
    """
    Duplicate text writes to multiple streams (console + log file).

    Used when `--save-run` is enabled so normal terminal output is preserved while also
    being archived in run artifacts for later debugging/repro.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        self.flush()

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _jsonify(value):
    """Convert common Python objects to JSON-safe values for metadata snapshots."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return str(value)


EXPECTED_LABELS = [
    "blue ball",
    "blue bucket",
    "green ball",
    "green bucket",
    "red ball",
    "red bucket",
    "yellow ball",
    "yellow bucket",
]
# IMPORTANT: this order is the class-index contract expected by the runtime and should
# match the labels JSON used by hailofilter post-process.


def _sha256_file(file_path: Path) -> str:
    """Hash model/config artifacts so each run records exactly what binaries were used."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _normalize_label(label: str) -> str:
    """Normalize label formatting to compare contracts robustly (underscore/space variants)."""
    return str(label).replace("_", " ").strip().lower()


def _load_hailo_labels(labels_path: Path):
    """Load labels list from Hailo post-process config JSON."""
    with open(labels_path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    labels = cfg.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels JSON must contain a non-empty 'labels' list")
    return [str(label) for label in labels]


def _validate_label_contract(actual_labels, expected_labels, strict=True):
    """Validate runtime labels order against expected class order for safe class decoding."""
    normalized_actual = [_normalize_label(label) for label in actual_labels]
    normalized_expected = [_normalize_label(label) for label in expected_labels]
    if normalized_actual == normalized_expected:
        return

    msg = (
        "Label-order contract mismatch between runtime labels JSON and expected class order. "
        f"expected={normalized_expected} actual={normalized_actual}"
    )
    if strict:
        raise RuntimeError(msg)
    print(f"[WARN] {msg}")

# -----------------------------------------------------------------------------
# Hailo inference engine: YOLO on 640x640 RGB frames via GStreamer
# -----------------------------------------------------------------------------
class HailoInference:
    """
    Run YOLO detection on numpy RGB frames through a Hailo GStreamer graph.

    Input contract:
    - `push_frame()` expects 640x640 RGB uint8 frame.

    Output contract:
    - `get_latest()` returns list of `(label, confidence, bbox)` objects from Hailo ROI API.
    """

    def __init__(self, hef_path: str, labels_json: str):
        # GStreamer must be initialized before pipeline construction.
        Gst.init(None)
        # Single-slot queue keeps only freshest detections; low latency > full history.
        self.detection_queue = __import__("queue").Queue(maxsize=1)
        post_process_so = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
        # Pipeline stages:
        # appsrc -> videoconvert/caps -> hailonet(HEF) -> hailofilter(labels JSON) -> appsink.
        pipeline_str = f"""
            appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ! \
            videoconvert ! video/x-raw,format=RGB,width=640,height=640 ! \
            hailonet hef-path={hef_path} force-writable=true ! \
            hailofilter so-path={post_process_so} config-path={labels_json} qos=false ! \
            queue leaky=no max-size-buffers=3 ! \
            appsink name=sink emit-signals=true max-buffers=1 drop=true
        """
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("source")
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_sample)
        # Keep appsrc caps explicit to match model input tensor dimensions.
        caps = Gst.Caps.from_string("video/x-raw,format=RGB,width=640,height=640,framerate=30/1")
        self.appsrc.set_property("caps", caps)

    def _on_sample(self, sink):
        # Called by appsink when a new inference result is available.
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        roi = hailo.get_roi_from_buffer(buf)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        # Debug aid: reveals unlabeled/mismatched outputs with raw class ID.
        for d in dets:
            lbl = d.get_label()
            # Empty/unknown labels often indicate HEF <-> labels JSON contract mismatch.
            if not lbl or lbl not in EXPECTED_LABELS:
                print(f"\n[DEBUG] Mystery Object Detected! Raw Label: '{lbl}' | Class ID: {d.get_class_id()}\n")

        results = [(d.get_label(), d.get_confidence(), d.get_bbox()) for d in dets]
        # Drop stale queue entry so caller always reads the newest inference packet.
        if self.detection_queue.full():
            try:
                self.detection_queue.get_nowait()
            except Exception:
                pass
        self.detection_queue.put(results)
        return Gst.FlowReturn.OK

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)

    def push_frame(self, rgb_640x640: np.ndarray):
        """Feed a 640x640 RGB numpy frame into the pipeline."""
        data = rgb_640x640.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        self.appsrc.emit("push-buffer", buf)

    def get_latest(self):
        """Non-blocking: return list of (label, conf, bbox) or []."""
        try:
            return self.detection_queue.get_nowait()
        except __import__("queue").Empty:
            return []


# -----------------------------------------------------------------------------
# User data: shared state for state machine and Pico thread
# -----------------------------------------------------------------------------
class UserData:
    """Shared mutable runtime state passed into all state_machine handlers."""

    def __init__(self, serial_port: str = "/dev/ttyACM0"):
        # Serial link to Pico firmware that consumes command messages.
        self.messenger = serial.Serial(port=serial_port, baudrate=115200)
        print(f"Pico messenger: {self.messenger.name}")
        # Latest command packet; send_loop keeps pushing this at fixed rate.
        self.latest_msg = "0.0, 0.0, 0, 0, 0\n".encode("utf-8")

        # Finite-state-machine mode fields consumed by state_machine.py.
        self.mode = "fixed_ball"
        self.arm_state = "idle"
        self.fixed_travel_counter = 0
        self.picker_counter = 0
        self.lap_counter = 0
        self.distance = 0.0
        self.distance_from_depth = False
        self.carrying_ball_color = None
        self.target_bucket_color = None
        self._running = True

    def send_loop(self):
        """Background: send latest_msg to Pico at ~50 Hz; drain feedback."""
        while self._running:
            # Drain inbound bytes to avoid serial buffer growth; feedback not parsed here.
            if self.messenger.in_waiting > 0:
                try:
                    self.messenger.readline()
                except Exception:
                    pass
            try:
                self.messenger.write(self.latest_msg)
            except Exception:
                pass
            # 20 ms cadence ~= 50 Hz command refresh.
            sleep(0.02)

    def stop(self):
        self._running = False


# -----------------------------------------------------------------------------
# Main: RealSense + Hailo + state machine loop
# -----------------------------------------------------------------------------
def main():
    """Program entry point: initialize subsystems, then run the perception-control loop."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Autonomous course with depth + YOLO")
    parser.add_argument(
        "--hef",
        "--hef-path",
        dest="hef",
        default=str(script_dir / "2-25-26.hef"),
        help="Path to .hef model",
    )
    parser.add_argument(
        "--labels",
        "--labels-json",
        dest="labels",
        default=str(script_dir / "ball_bucket.json"),
        help="Hailo labels JSON",
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Pico serial port")
    parser.add_argument(
        "--input",
        default=None,
        help="Compatibility only; ignored because this script always uses RealSense color/depth.",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV live camera window")
    parser.add_argument("--no-depth", action="store_true", help="Disable RealSense depth (geometry only)")
    parser.add_argument("--save-run", action="store_true", help="Save terminal log, telemetry CSV, and metadata")
    parser.add_argument("--run-root", default=str(script_dir / "run_logs"), help="Root folder for run artifacts")
    parser.add_argument("--run-name", default=None, help="Optional run folder name (default: timestamp)")
    parser.add_argument("--no-video-log", action="store_true", help="Disable POV video recording when --save-run is enabled")
    parser.add_argument("--video-fps", type=float, default=15.0, help="FPS for saved POV video")
    parser.add_argument(
        "--allow-label-mismatch",
        action="store_true",
        help="Allow runtime to continue even when labels JSON order does not match expected class order.",
    )
    args = parser.parse_args()

    # Preserve original stdio so we can tee logs and restore cleanly on exit.
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    run_dir = None
    metadata_path = None
    terminal_log_fp = None
    telemetry_fp = None
    telemetry_writer = None
    video_writer = None
    record_video = False
    frame_index = 0

    if args.save_run:
        # Build run artifact folder and wire up logging outputs.
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.run_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        terminal_log_fp = open(run_dir / "terminal.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(orig_stdout, terminal_log_fp)
        sys.stderr = _TeeStream(orig_stderr, terminal_log_fp)

        telemetry_fp = open(run_dir / "telemetry.csv", "w", newline="", encoding="utf-8")
        telemetry_writer = csv.writer(telemetry_fp)
        # Telemetry schema is intentionally stable for downstream analysis scripts.
        telemetry_writer.writerow(
            [
                "timestamp_iso",
                "frame_idx",
                "mode",
                "arm_state",
                "distance_m",
                "distance_src",
                "target_ball_requested",
                "carrying_ball_color",
                "target_bucket_color",
                "detection_count",
                "detections",
                "latest_msg",
            ]
        )

        # Capture tuning constants at runtime so logs can be reproduced later.
        tune_snapshot = {name: _jsonify(getattr(tune, name)) for name in dir(tune) if name.isupper()}
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "args": _jsonify(vars(args)),
            "tuning": tune_snapshot,
        }
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as meta_fp:
            json.dump(metadata, meta_fp, indent=2)

        record_video = not args.no_video_log
        print(f"Run artifact capture enabled: {run_dir}")

    # Resolve model path from CLI/default, supporting both local and repo-level models/.
    hef_path = Path(args.hef)
    if not hef_path.is_absolute():
        hef_candidates = [
            script_dir / hef_path,
            script_dir / "models" / hef_path.name,
            script_dir.parent / "models" / hef_path.name,
        ]
        hef_path = next((candidate for candidate in hef_candidates if candidate.exists()), hef_candidates[0])
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_candidates = [
            script_dir / labels_path,
            script_dir / "models" / labels_path.name,
            script_dir.parent / "models" / labels_path.name,
        ]
        labels_path = next((candidate for candidate in labels_candidates if candidate.exists()), labels_candidates[0])
    if not hef_path.exists():
        print(f"HEF not found: {hef_path}")
        sys.exit(1)
    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        sys.exit(1)

    # Validate class index contract before starting motors/camera loop.
    try:
        runtime_labels = _load_hailo_labels(labels_path)
    except Exception as exc:
        print(f"Failed to load labels JSON ({labels_path}): {exc}")
        sys.exit(1)

    try:
        _validate_label_contract(
            runtime_labels,
            EXPECTED_LABELS,
            strict=not args.allow_label_mismatch,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        print("Refusing to start with mismatched labels. Use --allow-label-mismatch to override.")
        sys.exit(1)

    # Print and store hashes for model/config provenance.
    hef_sha256 = _sha256_file(hef_path)
    labels_sha256 = _sha256_file(labels_path)
    print(f"HEF path: {hef_path}")
    print(f"HEF sha256: {hef_sha256}")
    print(f"Labels path: {labels_path}")
    print(f"Labels sha256: {labels_sha256}")
    print(f"Labels order: {runtime_labels}")

    if metadata_path is not None:
        with open(metadata_path, "r", encoding="utf-8") as meta_fp:
            metadata = json.load(meta_fp)
        metadata["artifacts"] = {
            "hef": {"path": str(hef_path), "sha256": hef_sha256},
            "labels": {
                "path": str(labels_path),
                "sha256": labels_sha256,
                "labels": runtime_labels,
            },
            "expected_labels": EXPECTED_LABELS,
            "label_contract_enforced": not args.allow_label_mismatch,
        }
        with open(metadata_path, "w", encoding="utf-8") as meta_fp:
            json.dump(metadata, meta_fp, indent=2)

    # Hailo YOLO engine setup.
    engine = HailoInference(str(hef_path), str(labels_path))
    engine.start()
    print("Hailo inference started.")

    # RealSense setup: aligned depth means bbox center from color can be sampled in depth.
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_pipeline = rs.pipeline()
    rs_pipeline.start(rs_config)
    align_to_color = rs.align(rs.stream.color)
    print("RealSense D455 started (color + depth).")

    # User state + serial sender thread (control messages emitted continuously).
    user_data = UserData(args.port)
    pico_thread = threading.Thread(target=user_data.send_loop, daemon=True)
    pico_thread.start()

    depth_width, depth_height = 640, 480
    model_height = 640
    use_depth = not args.no_depth
    show_display = not args.no_display
    display_name = "Autonomous Depth View"

    if show_display:
        cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)

    try:
        print("Main loop running. Use Ctrl+C to stop.")
        while True:
            # 1) Capture aligned sensor frames.
            frames = rs_pipeline.wait_for_frames()
            aligned = align_to_color.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame() if use_depth else None
            if not color_frame:
                continue
            color_np = np.asanyarray(color_frame.get_data())
            # 2) Prepare YOLO input tensor shape expected by Hailo graph.
            rgb_640 = cv2.resize(color_np, (640, 640))
            engine.push_frame(rgb_640)
            detections = engine.get_latest()
            frame_index += 1
            unlabeled_count = sum(1 for label, _, _ in detections if not str(label).strip())
            if unlabeled_count and (frame_index % 30) == 0:
                print(
                    "[WARN] Received detections with empty labels. "
                    "This usually indicates model/labels JSON mismatch. "
                    f"unlabeled={unlabeled_count}"
                )

            # 3) Dispatch by current behavior mode; handlers update latest_msg and state.
            if user_data.mode == "pause":
                handle_pause(user_data)
            elif user_data.mode == "pick":
                handle_pick(user_data)
            elif user_data.mode == "drop":
                handle_drop(user_data)
            elif user_data.mode == "fixed_ball":
                handle_fixed_ball(user_data)
            elif user_data.mode == "fixed_bucket":
                handle_fixed_bucket(user_data)
            elif user_data.mode == "fixed_back":
                handle_fixed_back(user_data)
            elif user_data.mode == "swivel_small_left":
                handle_swivel_small_left(user_data)
            elif user_data.mode == "swivel_large_right":
                handle_swivel_large_right(user_data)
            elif user_data.mode == "detect":
                handle_detect_ball(
                    user_data,
                    detections,
                    depth_frame,
                    depth_width,
                    depth_height,
                    model_height,
                )
            elif user_data.mode == "detect_bucket":
                handle_detect_bucket(
                    user_data,
                    detections,
                    depth_frame,
                    depth_width,
                    depth_height,
                    model_height,
                )
            else:
                handle_pause(user_data)

            # Optional console trace during active detection phases.
            if user_data.mode in ("detect", "detect_bucket") and detections:
                src = "depth" if getattr(user_data, "distance_from_depth", False) else "geom"
                print(
                    f"[{user_data.mode}] dist={getattr(user_data, 'distance', 0):.2f}m ({src}) "
                    f"target_ball={getattr(tune, 'TARGET_BALL_COLOR', '') or 'any'} "
                    f"carrying={getattr(user_data, 'carrying_ball_color', None)} "
                    f"target_bucket={getattr(user_data, 'target_bucket_color', None)} "
                    f"msg={user_data.latest_msg.decode().strip()}"
                )

            if telemetry_writer is not None:
                # Compact detection summary for quick timeline scans.
                dist_src = "depth" if getattr(user_data, "distance_from_depth", False) else "geom"
                detection_summary = ";".join(
                    f"{(label if str(label).strip() else '<unlabeled>')}:{conf:.2f}"
                    for label, conf, _ in detections[:12]
                )
                telemetry_writer.writerow(
                    [
                        datetime.now().isoformat(timespec="milliseconds"),
                        frame_index,
                        user_data.mode,
                        user_data.arm_state,
                        f"{getattr(user_data, 'distance', 0.0):.3f}",
                        dist_src,
                        getattr(tune, "TARGET_BALL_COLOR", "") or "any",
                        getattr(user_data, "carrying_ball_color", None),
                        getattr(user_data, "target_bucket_color", None),
                        len(detections),
                        detection_summary,
                        user_data.latest_msg.decode("utf-8", errors="ignore").strip(),
                    ]
                )
                if (frame_index % 15) == 0:
                    # Periodic flush limits data loss on abrupt stop.
                    telemetry_fp.flush()

            if show_display or record_video:
                # Visualization uses original 640x480 camera frame; boxes are remapped to that space.
                frame_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
                for label, conf, bbox in detections:
                    if conf < 0.25:
                        continue
                    display_label = label if str(label).strip() else "<unlabeled>"
                    x1 = int(bbox.xmin() * depth_width)
                    y1 = int(bbox.ymin() * depth_height)
                    x2 = int(bbox.xmax() * depth_width)
                    y2 = int(bbox.ymax() * depth_height)
                    x1 = max(0, min(depth_width - 1, x1))
                    y1 = max(0, min(depth_height - 1, y1))
                    x2 = max(0, min(depth_width - 1, x2))
                    y2 = max(0, min(depth_height - 1, y2))
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame_bgr,
                        f"{display_label} {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                dist_value = getattr(user_data, "distance", 0.0)
                dist_src = "depth" if getattr(user_data, "distance_from_depth", False) else "geom"
                cv2.putText(
                    frame_bgr,
                    f"mode={user_data.mode} arm={user_data.arm_state}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"dist={dist_value:.2f}m src={dist_src}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"ball_win={tune.BALL_PICK_MIN_M:.2f}-{tune.BALL_PICK_MAX_M:.2f}m bucket_win={tune.BUCKET_DROP_MIN_M:.2f}-{tune.BUCKET_DROP_MAX_M:.2f}m",
                    (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"ball seen/close/lost={getattr(user_data, 'ball_seen_streak', 0)}/{getattr(user_data, 'ball_close_streak', 0)}/{getattr(user_data, 'ball_lost_streak', 0)}",
                    (10, 102),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"bucket seen/close/lost={getattr(user_data, 'bucket_seen_streak', 0)}/{getattr(user_data, 'bucket_close_streak', 0)}/{getattr(user_data, 'bucket_lost_streak', 0)}",
                    (10, 128),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"target_ball={getattr(tune, 'TARGET_BALL_COLOR', '') or 'any'} carry={getattr(user_data, 'carrying_ball_color', None)} bucket_target={getattr(user_data, 'target_bucket_color', None)}",
                    (10, 154),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"missing_ball/bucket_color={getattr(user_data, 'ball_target_missing_streak', 0)}/{getattr(user_data, 'bucket_target_missing_streak', 0)}",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    "Press q to quit",
                    (10, 206),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                if record_video:
                    if video_writer is None:
                        # Lazy-open writer only after we know frame dimensions.
                        h, w = frame_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_path = run_dir / "pov_annotated.mp4"
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, args.video_fps, (w, h))
                        if not video_writer.isOpened():
                            print(f"WARNING: could not open video writer at {video_path}")
                            video_writer = None
                        else:
                            print(f"POV video recording: {video_path}")
                    if video_writer is not None:
                        video_writer.write(frame_bgr)

            if show_display:
                cv2.imshow(display_name, frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        # Safe shutdown order: stop command loop, send neutral command, then release devices.
        user_data.stop()
        user_data.latest_msg = "0.0, 0.0, 0, 0, 0\n".encode("utf-8")
        try:
            user_data.messenger.write(user_data.latest_msg)
            sleep(0.1)
            user_data.messenger.close()
        except Exception:
            pass
        engine.stop()
        rs_pipeline.stop()
        if video_writer is not None:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        if telemetry_fp is not None:
            telemetry_fp.close()
        if args.save_run:
            print(f"Run artifacts saved to: {run_dir}")
        print("Done.")
        if args.save_run:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
        if terminal_log_fp is not None:
            terminal_log_fp.close()


if __name__ == "__main__":
    main()
