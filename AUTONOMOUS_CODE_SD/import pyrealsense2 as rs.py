import argparse
import json
import os
import sys
import threading
import time
import numpy as np
import cv2
import serial
import pyrealsense2 as rs
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams)

# -----------------------------------------------------------------------------
# 1. ARGUMENT PARSING
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Ball-E Robot Logic with RealSense & Hailo")
    parser.add_argument("--hef-path", type=str, required=True, help="Path to .hef model file")
    parser.add_argument("--labels-json", type=str, required=True, help="Path to label map JSON")
    parser.add_argument("--input", type=str, default="rpi", help="Input source (ignored, always uses RealSense)")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 2. ROBOT STATE MACHINE
# -----------------------------------------------------------------------------
class RobotController:
    def __init__(self, labels_map):
        # Hardware: Serial to Pico
        try:
            self.messenger = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=0.1)
            print(f"Connected to Pico at {self.messenger.name}")
        except:
            print("WARNING: Pico not connected (Simulation Mode)")
            self.messenger = None

        self.latest_msg = "0.0, 0.0, 0, 0, 0\n".encode('utf-8')
        
        # Logic Variables
        self.mode = "fixed_ball"
        self.fixed_travel_counter = 0
        self.picker_counter = 0
        self.vel = 0.0
        self.distance = 0.0
        
        # Store dynamic class IDs found from JSON
        self.id_ball = labels_map.get("ball", -1)
        self.id_bucket = labels_map.get("bucket", -1)
        
        if self.id_ball == -1: print("WARNING: 'ball' not found in JSON labels!")
        if self.id_bucket == -1: print("WARNING: 'bucket' not found in JSON labels!")

        # Start Serial Thread
        if self.messenger:
            self.thread = threading.Thread(target=self.send_msg_loop, daemon=True)
            self.thread.start()

    def send_msg_loop(self):
        while True:
            if self.messenger:
                self.messenger.write(self.latest_msg)
            time.sleep(0.02)

    def update_logic(self, detections, depth_frame, frame_width):
        msg_str = "0.0, 0.0, 0, 0, 0"

        # --- STATE: PAUSE/PICK/DROP/FIXED ---
        # (This logic remains identical to your script, condensed for brevity)
        if self.mode == "pick":
            if 0 <= self.picker_counter <= 210:
                msg_str = "0.0, 0.0, 3000, 0, 0"
                self.picker_counter += 1
            elif 210 < self.picker_counter <= 295:
                msg_str = "0.0, 0.0, 0, 3000, 0"
                self.picker_counter += 1
            elif 295 < self.picker_counter <= 470:
                msg_str = "0.0, 0.0, -3000, 0, 0"
                self.picker_counter += 1
            else:
                self.mode = "fixed_back"
                self.picker_counter = 0

        elif self.mode == "drop":
            if 0 <= self.picker_counter <= 50:
                msg_str = "0.0, 0.0, 3000, 0, 0"
                self.picker_counter += 1
            elif 50 < self.picker_counter <= 90:  
                msg_str = "0.0, 0.0, 0, -3000, 0"
                self.picker_counter += 1
            else:    
                msg_str = "0.0, 0.0, 0, 0, 10"
                self.mode = "swivel_large_right"
                self.picker_counter = 0
                self.fixed_travel_counter = 0

        elif self.mode == "fixed_ball":
            if self.fixed_travel_counter < 500:
                msg_str = "0.2, 0.0, 0, 0, 10"
                self.fixed_travel_counter += 1
            else:
                self.mode = "detect"
                self.fixed_travel_counter = 0
                msg_str = "0.0, 0.0, 0, 0, 0"
        
        elif self.mode == "detect":
            # Search for BALL using loaded ID
            target_box = None
            for det in detections:
                if int(det[5]) == self.id_ball: 
                    target_box = det
                    break
            
            if target_box is None:
                self.vel = max(self.vel - 0.05, 0.0)
                msg_str = f"{self.vel:.2f}, 0.0, 0, 0, 0"
            else:
                x1, y1, x2, y2 = target_box[:4]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                dist_raw = depth_frame.get_distance(center_x, center_y)
                if dist_raw > 0: self.distance = dist_raw
                
                self.vel = 0.4
                norm_center = center_x / frame_width
                
                # Logic from your script
                if self.distance >= 2.4:
                    if norm_center < 0.4: msg_str = "0.4, 0.5, 0, 0, 0"
                    elif norm_center > 0.7: msg_str = "0.4, -0.5, 0, 0, 0"
                    else: msg_str = "0.4, 0.0, 0, 0, 0"
                elif 1.258 < self.distance <= 2.4:
                    if norm_center < 0.4: msg_str = "0.2, 0.5, 0, 0, 0"
                    elif norm_center > 0.7: msg_str = "0.2, -0.5, 0, 0, 0"
                    else: msg_str = "0.2, 0.0, 0, 0, 0"
                else:
                    msg_str = "0.0, 0.0, 0, 0, 0"
                    self.mode = "pick"

        # ... (Add other fixed modes: swivel_small_left, fixed_bucket, etc. here) ...

        self.latest_msg = (msg_str + "\n").encode('utf-8')
        print(f"Mode: {self.mode} | Dist: {self.distance:.2f}m | Cmd: {msg_str}")

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_labels(json_path):
    """Loads JSON and returns a dict mapping 'name' -> ID"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Handle different JSON formats
    label_map = {}
    if isinstance(data, dict):
        # Format: {"0": "ball", "1": "bucket"}
        for k, v in data.items():
            label_map[v] = int(k)
    elif isinstance(data, list):
        # Format: ["ball", "bucket"]
        for idx, name in enumerate(data):
            label_map[name] = idx
    return label_map

def get_hailo_inference(hef_path):
    hef = HEF(hef_path)
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    input_vstreams_params = InputVStreamParams.make(network_group)
    output_vstreams_params = OutputVStreamParams.make(network_group)
    pipeline = InferVStreams(network_group, input_vstreams_params, output_vstreams_params)
    return target, pipeline, hef

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------
def run_robot():
    args = parse_arguments()
    
    # Check paths
    if not os.path.exists(args.hef_path):
        print(f"Error: HEF file not found at {args.hef_path}")
        return
    if not os.path.exists(args.labels_json):
        print(f"Error: Labels file not found at {args.labels_json}")
        return

    # Load Labels
    labels_map = load_labels(args.labels_json)
    print(f"Loaded Labels: {labels_map}")

    # Initialize Controller
    bot = RobotController(labels_map)

    # Setup RealSense
    print("Initializing RealSense (RSUSB)...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Setup Hailo
    print(f"Loading HEF: {args.hef_path}")
    target, inference_pipeline, hef = get_hailo_inference(args.hef_path)
    
    input_name = hef.get_input_vstream_infos()[0].name
    output_name = hef.get_output_vstream_infos()[0].name

    with inference_pipeline as infer_pipeline:
        input_stream = infer_pipeline.input_vstreams[input_name]
        output_stream = infer_pipeline.output_vstreams[output_name]
        
        print("System Ready.")

        try:
            while True:
                # 1. Capture
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame: continue

                # 2. Inference
                img_rgb = np.asanyarray(color_frame.get_data())
                img_resized = cv2.resize(img_rgb, (640, 640))
                input_batch = np.expand_dims(img_resized, axis=0)

                input_stream.send(input_batch)
                infer_result = output_stream.recv()

                # 3. Decode
                detections = []
                if infer_result.shape[0] > 0:
                    for det in infer_result[0]:
                        ymin, xmin, ymax, xmax, score, class_id = det
                        if score < 0.5: continue
                        detections.append([int(xmin*640), int(ymin*480), int(xmax*640), int(ymax*480), score, class_id])

                # 4. Logic
                bot.update_logic(detections, depth_frame, 640)

                # 5. Display (Optional - comment out for headless speed)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                for x1, y1, x2, y2, s, c in detections:
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('Robot Vision', img_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    run_robot()