import sys
from pathlib import Path
import json
from time import time
import serial
import pygame
import cv2 as cv
from picamera2 import Picamera2
import torch
from torchvision.transforms import v2
from cnn_architectures.bear_net import BearNet


# SETUP
# Define paths
bc_dir = Path(__file__).parents[1]
# Load configs
params_file_path = str(bc_dir.joinpath("scripts", "configs.json"))
with open(params_file_path, "r") as file:
    params = json.load(file)
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
# Load model
pilot = BearNet()
model_path = str(bc_dir.joinpath("models", "pilot.pth"))
pilot.load_state_dict(
    torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
)
pilot.eval()
# Init serial port
messenger = serial.Serial(port="/dev/ttyACM0", baudrate=115200)
print(f"Pico is connected to port: {messenger.name}")
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# Init camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={
            "format": "RGB888",
            "size": (224, 224),
        },  # WARN: BGR for autopilot
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
# Flags, ordered by priority
is_stopped = False
is_paused = True
mode = "p"
# Init timer for FPS computing
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.0


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
        # if cv.waitKey(1) == ord("q"):  # debug, [q]uit
        #     print("Quit signal received.")
        #     break
        # Log frame rate
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")  # debug
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params["stop_btn"]):  # emergency stop
                    is_stopped = True
                    print("E-STOP PRESSED. TERMINATE")
                    pygame.quit()
                    messenger.close()
                    sys.exit()
                elif js.get_button(params["pause_btn"]):
                    is_paused = not is_paused
                    if is_paused:
                        mode = "p"
                    else:
                        mode = "a"
        # Predict steer and throttle
        img_tensor = to_tensor(frame[:, :, [2, 1, 0]])  # WARN: autopilot needs BGR
        with torch.no_grad():
            pred_st, pred_th = map(
                float,
                torch.clamp(
                    pilot(img_tensor[None, :]).squeeze(), min=-0.999, max=0.999
                ),
            )
        # print(f"predicted actions: {pred_st}, {pred_th}")  # debug
        # Encode steering value to dutycycle in nanosecond
        duty_st = params["steering_center"] + int(params["steering_range"] * pred_st)
        # Encode throttle value to dutycycle in nanosecond
        if pred_th > 0:
            duty_th = params["throttle_neutral"] + int(
                params["throttle_fwd_range"] * min(pred_th, params["throttle_limit"])
            )
        elif pred_th < 0:
            duty_th = params["throttle_neutral"] + int(
                params["throttle_fwd_range"] * max(pred_th, -params["throttle_limit"])
            )
        else:
            duty_th = params["throttle_neutral"]
        msg = f"{mode}, {duty_st}, {duty_th}\n".encode("utf-8")
        messenger.write(msg)

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
