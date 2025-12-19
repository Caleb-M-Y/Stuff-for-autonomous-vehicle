import sys
from pathlib import Path
import serial
import pygame
import json
from time import sleep, time
from datetime import datetime
import cv2 as cv
from picamera2 import Picamera2
import numpy as np


# SETUP
# Load configs
params_file_path = str(Path(__file__).parents[1].joinpath("configs.json"))
with open(params_file_path, "r") as file:
    params = json.load(file)
# Init serial port
messenger = serial.Serial(port="/dev/ttyACM0", baudrate=115200)
print(f"Pico is connected to port: {messenger.name}")
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
print(f"Controller: {js.get_name()}")
# Init Pi Camera (for metadata logging example)
try:
    cv.startWindowThread()
    cam = Picamera2()
    cam.configure(
        cam.create_preview_configuration(
            main={"format": "RGB888", "size": (224, 224)},
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / params.get("frame_rate", 24)),
                    int(1_000_000 / params.get("frame_rate", 24)),
                )
            },
        )
    )
    cam.start()
    camera_available = True
    print("Camera initialized for metadata logging")
except Exception as e:
    print(f"Camera not available: {e}")
    camera_available = False
    cam = None

# Init joystick axes values
ax_val_ang = 0.0
ax_val_lin = 0.0
act_lower, act_close = 0, 0
# Flags
is_stopped = False
is_recording = False  # Image collection flag
collected_data = []  # Store (image, metadata) tuples
image_counter = 0

# Robot state tracking (for metadata)
robot_x = 0.0  # Estimated X position (feet)
robot_y = 0.0  # Estimated Y position (feet)
robot_heading = 0.0  # Estimated heading (degrees)
last_odometry_time = time()

print("\n=== Controls ===")
print(f"Left stick: Forward/Backward (max {params['lin_vel_max']} m/s)")
print(f"Right stick: Turn Left/Right (max {params['ang_vel_max']} rad/s)")
print(f"Button {params['lower_button']}: Lower arm")
print(f"Button {params['raise_button']}: Raise arm")
print(f"Button {params['close_button']}: Close claw")
print(f"Button {params['open_button']}: Open claw")
if camera_available:
    print("Button 5: Toggle image collection with metadata")
print("================\n")
print("Robot ready! Press Ctrl+C to stop.\n")


# ============================================================================
# METADATA LOGGING FUNCTIONS
# ============================================================================

def calculate_image_quality(frame):
    """Calculate image quality metrics"""
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    
    # Blur detection using Laplacian variance
    laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
    blur_score = 1.0 / (1.0 + laplacian_var / 1000.0)  # Normalize to 0-1
    
    # Brightness (mean pixel value)
    brightness = np.mean(gray) / 255.0
    
    # Contrast (standard deviation)
    contrast = np.std(gray) / 255.0
    
    return {
        "blur_score": round(blur_score, 3),  # Lower is better
        "brightness": round(brightness, 3),  # 0-1
        "contrast": round(contrast, 3),  # 0-1
    }


def detect_lighting_condition(frame):
    """Detect lighting condition from image"""
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Simple heuristic-based lighting detection
    if mean_brightness > 200:
        if std_brightness > 50:
            return "mixed_glare"  # Very bright with high variance
        else:
            return "sunny"  # Very bright, uniform
    elif mean_brightness < 80:
        return "shadow"  # Dark
    elif std_brightness > 40:
        return "mixed_shadow"  # Medium brightness with high variance
    else:
        return "even"  # Even lighting


def estimate_distance_to_object(frame, object_type="ball"):
    """
    Estimate distance to object based on image analysis
    This is a placeholder - in real implementation, use:
    - Stereo vision
    - Size-based estimation (if object size is known)
    - LiDAR/ultrasonic sensors
    - Manual input during data collection
    """
    # Placeholder: return None for manual entry or sensor-based measurement
    # In practice, you might:
    # 1. Use object detection to find bounding box size
    # 2. Estimate distance from known object size
    # 3. Use depth sensor if available
    return None  # Will be filled manually or by sensor


def update_odometry(lin_vel, ang_vel, dt):
    """Update robot position estimate based on velocity commands"""
    global robot_x, robot_y, robot_heading
    
    # Simple dead reckoning (assumes no wheel slippage)
    robot_heading += np.degrees(ang_vel * dt)
    robot_heading = robot_heading % 360  # Wrap to 0-360
    
    # Convert to feet (assuming lin_vel is in m/s)
    lin_vel_ft_s = lin_vel * 3.28084  # m/s to ft/s
    
    robot_x += lin_vel_ft_s * np.cos(np.radians(robot_heading)) * dt
    robot_y += lin_vel_ft_s * np.sin(np.radians(robot_heading)) * dt


def create_metadata(frame, object_type=None, object_color=None, 
                   distance_ft=None, pickup_success=None):
    """Create comprehensive metadata dictionary for an image"""
    timestamp = datetime.now()
    
    # Calculate image quality metrics
    quality = calculate_image_quality(frame)
    
    # Detect lighting condition
    lighting = detect_lighting_condition(frame)
    
    # Estimate distance (if not provided)
    if distance_ft is None:
        distance_ft = estimate_distance_to_object(frame, object_type)
    
    # Create metadata dictionary
    metadata = {
        "image_id": f"image_{image_counter:06d}",
        "timestamp": timestamp.isoformat(),
        "timestamp_unix": time(),
        
        # Object information
        "object_type": object_type,  # "ball", "bucket", or None
        "object_color": object_color,  # "red", "blue", "green", "yellow", or None
        "distance_ft": distance_ft,  # Estimated distance to object
        
        # Robot state
        "robot_x": round(robot_x, 2),  # Estimated X position (feet)
        "robot_y": round(robot_y, 2),  # Estimated Y position (feet)
        "robot_heading": round(robot_heading, 1),  # Heading (degrees)
        "robot_lin_vel": round(act_lin, 3),  # Linear velocity (m/s)
        "robot_ang_vel": round(act_ang, 3),  # Angular velocity (rad/s)
        
        # Arm/claw state
        "arm_direction": act_lower,  # -1: raise, 0: neutral, 1: lower
        "claw_direction": act_close,  # -1: open, 0: neutral, 1: close
        
        # Image quality metrics
        "image_quality": quality,
        "lighting_condition": lighting,
        
        # Success/failure tracking
        "pickup_success": pickup_success,  # True/False/None (for training data)
        
        # Camera information
        "camera_resolution": {"width": frame.shape[1], "height": frame.shape[0]},
    }
    
    return metadata


def save_image_with_metadata(image, metadata, base_dir):
    """Save image and corresponding metadata JSON file"""
    # Create images directory if it doesn't exist
    images_dir = base_dir / "images_with_metadata"
    images_dir.mkdir(exist_ok=True)
    
    # Save image
    image_filename = f"{metadata['image_id']}.jpg"
    image_path = images_dir / image_filename
    cv.imwrite(str(image_path), cv.cvtColor(image, cv.COLOR_RGB2BGR))
    
    # Save metadata JSON
    metadata_filename = f"{metadata['image_id']}_metadata.json"
    metadata_path = images_dir / metadata_filename
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return image_path, metadata_path

# MAIN LOOP
try:
    while not is_stopped:
        # Capture frame if camera available
        frame = None
        if camera_available:
            try:
                frame = cam.capture_array()
                if frame is not None:
                    cv.imshow("camera", frame)
                    if cv.waitKey(1) == ord("q"):
                        print("Quit signal received.")
                        break
            except Exception as e:
                print(f"Camera error: {e}")
        
        # Process gamepad input
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                # Toggle image collection (Button 5)
                if camera_available and js.get_button(5):
                    is_recording = not is_recording
                    if is_recording:
                        print("Image collection with metadata STARTED")
                    else:
                        print("Image collection PAUSED")
                
                # Arm and claw controls
                if js.get_button(params["lower_button"]):
                    act_lower = 1
                elif js.get_button(params["raise_button"]):
                    act_lower = -1
                if js.get_button(params["close_button"]):
                    act_close = 1
                elif js.get_button(params["open_button"]):
                    act_close = -1
            elif e.type == pygame.JOYBUTTONUP:
                if not js.get_button(params["lower_button"]) and not js.get_button(params["raise_button"]):
                    act_lower = 0
                if not js.get_button(params["close_button"]) and not js.get_button(params["open_button"]):
                    act_close = 0
            elif e.type == pygame.JOYAXISMOTION:
                ax_val_ang = round(
                    (js.get_axis(params["ang_joy_axis"])), 1
                )  # keep 1 decimal
                ax_val_lin = round(
                    (js.get_axis(params["lin_joy_axis"])), 1
                )  # keep 1 decimal
        
        # Calculate steering and throttle value
        act_ang = -ax_val_ang * params["ang_vel_max"]  # -1: left most; +1: right most
        act_lin = (
            -ax_val_lin * params["lin_vel_max"]
        )  # -1: max forward, +1: max backward
        
        # Update odometry for metadata
        current_time = time()
        dt = current_time - last_odometry_time
        if dt > 0:
            update_odometry(act_lin, act_ang, dt)
            last_odometry_time = current_time
        
        # Collect image with metadata if recording
        if camera_available and is_recording and frame is not None:
            # Create metadata (you can manually specify object info or leave as None)
            metadata = create_metadata(
                frame,
                object_type=None,  # Set to "ball" or "bucket" if known
                object_color=None,  # Set to "red", "blue", "green", "yellow" if known
                distance_ft=None,  # Set manually or use sensor
                pickup_success=None  # Set to True/False after pickup attempt
            )
            
            # Store for batch saving (or save immediately)
            collected_data.append((frame.copy(), metadata))
            image_counter += 1
            
            # Print progress every 10 images
            if image_counter % 10 == 0:
                print(f"Collected {image_counter} images with metadata...")
        
        msg = f"{act_lin}, {act_ang}, {act_close}, {act_lower}\n".encode("utf-8")
        messenger.write(msg)
        
        # Drain feedback buffer to prevent clogging (non-blocking)
        # This keeps communication smooth even if feedback isn't being used
        feedback_count = 0
        while messenger.in_waiting > 0 and feedback_count < 10:  # Limit to prevent blocking
            try:
                messenger.readline()  # Discard feedback to prevent buffer buildup
            except:
                break
            feedback_count += 1
        
        # 100Hz control loop (10ms period) - matches blocking poll on Pico
        sleep(0.01)

# Take care terminal signal (Ctrl-c)
except KeyboardInterrupt:
    print("\nShutting down...")
    is_stopped = True

finally:
    # Stop robot
    messenger.write(b"0.0,0.0,0,0\n")
    sleep(0.1)
    
    # Save collected images with metadata
    if camera_available and collected_data:
        base_dir = Path(__file__).parents[1]
        print(f"\nSaving {len(collected_data)} images with metadata...")
        
        saved_count = 0
        for image, metadata in collected_data:
            try:
                image_path, metadata_path = save_image_with_metadata(
                    image, metadata, base_dir
                )
                saved_count += 1
            except Exception as e:
                print(f"Error saving image {metadata['image_id']}: {e}")
        
        print(f"Successfully saved {saved_count} images with metadata to images_with_metadata/")
        print(f"Metadata files saved alongside images as *_metadata.json")
    elif camera_available:
        print("\nNo images collected.")
    
    # Cleanup
    pygame.quit()
    messenger.close()
    if camera_available and cam is not None:
        cam.stop()
        cv.destroyAllWindows()
    print("Robot stopped. Goodbye!")
    sys.exit()