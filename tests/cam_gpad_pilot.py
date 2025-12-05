import sys
from pathlib import Path
import json
from time import time
import pygame
import torch
from torchvision.transforms import v2
import cv2 as cv
from picamera2 import Picamera2


# SETUP
# Define paths
bc_dir = Path(__file__).parents[1]
# Import BearNet
sys.path.append(str(bc_dir.joinpath("scripts", "cnn_architectures")))
# print(sys.path)  # debug
from bear_net import BearNet

random_pilot = BearNet()
random_pilot.eval()
# Config image transforms
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
# Load configs
params_file_path = str(bc_dir.joinpath("scripts", "configs.json"))
with open(params_file_path, "r") as file:
    params = json.load(file)
# Config image transforms
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# Init camera
cv.startWindowThread()
cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": "RGB888", "size": (224, 224)},  # WARN: BGR for autopilot
        controls={
            "FrameDurationLimits": (
                int(1000_000 / params["frame_rate"]),
                int(1000_000 / params["frame_rate"]),
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
    while True:
        frame = cam.capture_array()  # read image
        if frame is None:
            print("No frame received. TERMINATE!")
            break
        cv.imshow("Camera", frame)
        if cv.waitKey(1) == ord("q"):
            print("Quit signal received.")
            break
        # Log frame rate
        frame_counts += 1
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")  # debug
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params["stop_btn"]):  # emergency stop
                    is_stopped = True
                    print("E-STOP PRESSED. TERMINATE")
                    pygame.quit()
                    sys.exit()
                elif js.get_button(params["pause_btn"]):
                    is_paused = not is_paused
                    if is_paused:
                        mode = "p"
                        print("PAUSE")
                    else:
                        mode = "a"
                        print("AUTO")
        # Predict steer and throttle
        img_tensor = to_tensor(frame[:, :, [2, 1, 0]])
        with torch.no_grad():
            pred_st, pred_th = map(
                float,
                torch.clamp(
                    random_pilot(img_tensor[None, :]).squeeze(), min=-0.999, max=0.999
                ),
            )
        print(f"guessed actions: {pred_st}, {pred_th}")  # debug
except KeyboardInterrupt:  # take care terminate signal (Ctrl-c)
    cv.destroyAllWindows()
    sys.exit()
finally:
    cv.destroyAllWindows()
    sys.exit()
