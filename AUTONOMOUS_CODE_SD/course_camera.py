from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from time import sleep
import threading
import serial

# Local application-specific imports
import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp

# Centralized state machine (geometry-only fallback; no depth in this pipeline)
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


# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        
        self.messenger = serial.Serial(port='/dev/ttyACM0', baudrate=115200)  # New variable example
        print(f"Messenger initiated at: {self.messenger.name}\n")
        # Shared variable for latest message
        self.latest_msg = "0.0, 0.0, 0, 0, 0\n".encode('utf-8')
        
        # Start Pico update thread
        self.pico_thread = threading.Thread(target=self.send_msg, daemon=True)
        self.pico_thread.start()
        self.vel =0
        
        self.mode = "fixed_ball"
        self.arm_state = "idle"
        self.fixed_travel_counter = 0
        self.picker_counter = 0
        self.lap_counter = 0

    def send_msg(self):
        """Continuously send the latest message to the Pico."""
        while True:
            if self.messenger.inWaiting() > 0:
                # print("pico msg received")
                in_msg = self.messenger.readline().strip().decode("utf-8", "ignore")
                # print(f"RPi recieved: {in_msg}")
            self.messenger.write(self.latest_msg)
            sleep(0.02)


# User-defined callback: runs each frame; uses centralized state_machine (geometry-only, no depth).
def app_callback(pad, info, user_data):
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    (caps_string, frame_width, frame_height) = get_caps_from_pad(pad)
    user_data.frame_width = frame_width
    user_data.frame_height = frame_height

    # Convert Hailo detections to list of (label, confidence, bbox) for state_machine
    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    detections_tuples = [
        (d.get_label(), d.get_confidence(), d.get_bbox())
        for d in hailo_detections
    ]
    # No depth in this pipeline (camera-only); state_machine uses geometry fallback
    depth_frame = None
    model_height = 640

    # Dispatch by mode using centralized handlers (same flow as course_autonomous_depth.py)
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
            detections_tuples,
            depth_frame,
            frame_width,
            frame_height,
            model_height,
        )
    elif user_data.mode == "detect_bucket":
        handle_detect_bucket(
            user_data,
            detections_tuples,
            depth_frame,
            frame_width,
            frame_height,
            model_height,
        )
    else:
        handle_pause(user_data)

    string_to_print += f"Target velocity: {user_data.latest_msg}"
    print(string_to_print)
    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    user_data = user_app_callback_class()  # Create an instance of the user app callback class
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
