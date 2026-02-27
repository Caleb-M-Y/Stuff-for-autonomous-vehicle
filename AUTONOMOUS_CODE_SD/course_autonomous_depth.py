"""
Autonomous course runner: depth-first navigation with geometry fallback.
- Uses Intel RealSense D455 for color + depth; runs YOLO (Hailo) on color.
- Distance: hardware depth when valid, else geometry (focal length + bbox height).
- Picks the best ball/bucket detection by confidence (not the last one).
- Centralized state machine in state_machine.py (same flow as course_camera.py).
Run from repo root or AUTONOMOUS_CODE_SD; ensure 2-25-26.hef and labels JSON path are correct.
"""

import os
import sys
import argparse
import threading
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

# -----------------------------------------------------------------------------
# Hailo inference engine: YOLO on 640x640 RGB frames via GStreamer
# -----------------------------------------------------------------------------
class HailoInference:
    """Run YOLO detection on numpy frames; results as list of (label, confidence, bbox)."""

    def __init__(self, hef_path: str, labels_json: str):
        Gst.init(None)
        self.detection_queue = __import__("queue").Queue(maxsize=1)
        post_process_so = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
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
        caps = Gst.Caps.from_string("video/x-raw,format=RGB,width=640,height=640,framerate=30/1")
        self.appsrc.set_property("caps", caps)

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        roi = hailo.get_roi_from_buffer(buf)
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        results = [(d.get_label(), d.get_confidence(), d.get_bbox()) for d in dets]
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
    def __init__(self, serial_port: str = "/dev/ttyACM0"):
        self.messenger = serial.Serial(port=serial_port, baudrate=115200)
        print(f"Pico messenger: {self.messenger.name}")
        self.latest_msg = "0.0, 0.0, 0, 0, 0\n".encode("utf-8")
        self.mode = "fixed_ball"
        self.arm_state = "idle"
        self.fixed_travel_counter = 0
        self.picker_counter = 0
        self.lap_counter = 0
        self.distance = 0.0
        self.distance_from_depth = False
        self._running = True

    def send_loop(self):
        """Background: send latest_msg to Pico at ~50 Hz; drain feedback."""
        while self._running:
            if self.messenger.in_waiting > 0:
                try:
                    self.messenger.readline()
                except Exception:
                    pass
            try:
                self.messenger.write(self.latest_msg)
            except Exception:
                pass
            sleep(0.02)

    def stop(self):
        self._running = False


# -----------------------------------------------------------------------------
# Main: RealSense + Hailo + state machine loop
# -----------------------------------------------------------------------------
def main():
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
    args = parser.parse_args()

    # Resolve model path: allow same dir or repo-level models/
    hef_path = Path(args.hef)
    if not hef_path.is_absolute():
        hef_path = script_dir / hef_path
    if not hef_path.exists():
        alt = script_dir.parent / "models" / hef_path.name
        hef_path = alt if alt.exists() else hef_path
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = script_dir / labels_path
    if not labels_path.exists():
        labels_alt = script_dir.parent / "models" / labels_path.name
        labels_path = labels_alt if labels_alt.exists() else labels_path
    if not hef_path.exists():
        print(f"HEF not found: {hef_path}")
        sys.exit(1)
    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        sys.exit(1)

    # Hailo
    engine = HailoInference(str(hef_path), str(labels_path))
    engine.start()
    print("Hailo inference started.")

    # RealSense: color + depth aligned to color (640x480)
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_pipeline = rs.pipeline()
    rs_pipeline.start(rs_config)
    align_to_color = rs.align(rs.stream.color)
    print("RealSense D455 started (color + depth).")

    # User state and Pico thread
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
            # 1) Capture
            frames = rs_pipeline.wait_for_frames()
            aligned = align_to_color.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame() if use_depth else None
            if not color_frame:
                continue
            color_np = np.asanyarray(color_frame.get_data())
            # 2) Resize to 640x640 for YOLO (stretch)
            rgb_640 = cv2.resize(color_np, (640, 640))
            engine.push_frame(rgb_640)
            detections = engine.get_latest()

            # 3) Dispatch by mode (centralized state machine)
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

            # Optional: print mode and distance when in detect modes
            if user_data.mode in ("detect", "detect_bucket") and detections:
                src = "depth" if getattr(user_data, "distance_from_depth", False) else "geom"
                print(f"[{user_data.mode}] dist={getattr(user_data, 'distance', 0):.2f}m ({src}) msg={user_data.latest_msg.decode().strip()}")

            if show_display:
                frame_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
                for label, conf, bbox in detections:
                    if conf < 0.25:
                        continue
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
                        f"{label} {conf:.2f}",
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
                    "Press q to quit",
                    (10, 154),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(display_name, frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Stopping.")
    finally:
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
        if show_display:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
