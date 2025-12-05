import sys
from pathlib import Path
import json
from time import time
from datetime import datetime
import csv
import serial
import pygame
import cv2 as cv
from picamera2 import Picamera2


# SETUP
# Define paths
bc_dir = Path(__file__).parents[1]
data_dir = bc_dir.joinpath("data")
image_dir = str(data_dir.joinpath(datetime.now().strftime("%Y-%m-%d-%H-%M"), "images"))
Path(image_dir).mkdir(parents=True, exist_ok=True)
label_path = str(Path(image_dir).parent.joinpath("labels.csv"))
# Load configs
params_file_path = str(bc_dir.joinpath("scripts", "configs.json"))
with open(params_file_path, "r") as file:
    params = json.load(file)
# Init serial port
messenger = serial.Serial(port="/dev/ttyACM0", baudrate=115200)
print(f"Pico is connected to port: {messenger.name}")
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# Create data directory
# Init camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": "RGB888", "size": (180, 200, 3)},
        controls={
            "FrameDurationLimits": (
                int(1_000_000 / params["frame_rate"]),
                int(1_000_000 / params["frame_rate"]),
            )
        },
    )
)
cam.start()
for i in reversed(range(3 * params["frame_rate"])):
    frame = cam.capture_array()
    if frame is None:
        print("No frame received. TERMINATE!")
        sys.exit()
    else:
        # cv.imshow("camera", frame)  # debug
        # if cv.waitKey(1) == ord("q"):  # [q]uit
        #     print("Quit signal received.")  # debug
        #     break
        if not i % params["frame_rate"]:
            print(i / params["frame_rate"])  # count down 3, 2, 1 sec
# Init variables
ax_val_st = 0.0  # center steering
ax_val_th = 0.0  # neutral throttle
# Flags, ordered by priority
is_stopped = False
is_paused = True
is_recording = False
# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.0
# variables
mode = "p"
record_counts = 0

# LOOP
try:
    while not is_stopped:
        # Process camera data
        frame = cam.capture_array()  # read image
        if frame is None:
            print("No frame received. TERMINATE!")
            break
        frame_counts += 1
        # cv.imshow("camera", frame)  # debug
        # if cv.waitKey(1) == ord("q"):  # [q]uit
        #     print("Quit signal received.")
        #     break
        # Log frame rate
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        # print(f"frame rate: {frame_rate}")  # debug
        # Process gamepad data
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params["stop_btn"]):  # emergency stop
                    is_stopped = True
                    print("E-STOP PRESSED. TERMINATE")
                    cv.destroyAllWindows()
                    pygame.quit()
                    messenger.close()
                    sys.exit()
                elif js.get_button(params["pause_btn"]):
                    is_paused = not is_paused
                    if is_paused:
                        mode = "p"
                        is_recording = False
                    else:
                        mode = "n"
                    print(f"Paused: {is_paused}")  # debug
                elif js.get_button(params["record_btn"]):
                    if not is_paused:
                        is_recording = not is_recording
                        if is_recording:
                            mode = "r"
                        else:
                            mode = "n"
                        print(f"Recording: {is_recording}")  # debug
            elif e.type == pygame.JOYAXISMOTION:
                ax_val_st = round(
                    (js.get_axis(params["steering_joy_axis"])), 2
                )  # keep 2 decimals
                ax_val_th = round(
                    (js.get_axis(params["throttle_joy_axis"])), 2
                )  # keep 2 decimals
        # Calaculate steering and throttle value
        act_st = ax_val_st  # -1: left most; +1: right most
        act_th = -ax_val_th  # -1: max forward, +1: max backward
        # Encode steering value to dutycycle in nanosecond
        duty_st = params["steering_center"] + int(params["steering_range"] * act_st)
        # Encode throttle value to dutycycle in nanosecond
        if act_th > 0:
            duty_th = params["throttle_neutral"] + int(
                params["throttle_fwd_range"] * min(act_th, params["throttle_limit"])
            )
        elif act_th < 0:
            duty_th = params["throttle_neutral"] + int(
                params["throttle_fwd_range"] * max(act_th, -params["throttle_limit"])
            )
        else:
            duty_th = params["throttle_neutral"]
        # Transmit control signals
        drive_msg = f"{mode}, {duty_st}, {duty_th}\n".encode("utf-8")
        messenger.write(drive_msg)
        # Log data
        action = [act_st, act_th]
        # print(f"action: {action}")  # debug
        if is_recording:
            cv.imwrite(image_dir + "/" + str(frame_counts) + ".jpg", frame)
            label = [str(frame_counts) + ".jpg"] + action
            with open(label_path, "a+", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(label)
            record_counts += 1
            print(f"recorded frame counts: {record_counts}")  # debug
            if record_counts == params["record_cap"]:  # pause recording if max reached
                mode = "p"
                is_paused = True
                is_recording = False

# Take care terminate signal (Ctrl-c)
except KeyboardInterrupt:
    cv.destroyAllWindows()
    pygame.quit()
    messenger.close()
    sys.exit()
finally:
    cv.destroyAllWindows()
    pygame.quit()
    messenger.close()
    sys.exit()
