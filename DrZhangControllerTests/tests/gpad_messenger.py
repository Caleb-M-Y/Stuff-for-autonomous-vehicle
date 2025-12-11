import sys
from pathlib import Path
import serial
import pygame
import json
from time import sleep


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
# Init joystick axes values
ax_val_ang = 0.0
ax_val_lin = 0.0
act_lower, act_close = 0, 0
# Flags
is_stopped = False
is_enabled = True

print("\n=== Controls ===")
print(f"Left stick: Forward/Backward (max {params['lin_vel_max']} m/s)")
print(f"Right stick: Turn Left/Right (max {params['ang_vel_max']} rad/s)")
print(f"Button {params['lower_button']}: Lower arm")
print(f"Button {params['raise_button']}: Raise arm")
print(f"Button {params['close_button']}: Close claw")
print(f"Button {params['open_button']}: Open claw")
print("================\n")
print("Robot ready! Press Ctrl+C to stop.\n")

# MAIN LOOP
try:
    while not is_stopped:
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(params["lower_button"]):
                    act_lower = 1
                elif js.get_button(params["raise_button"]):
                    act_lower = -1
                if js.get_button(params["close_button"]):
                    act_close = 1
                elif js.get_button(params["open_button"]):
                    act_close = -1
            elif e.type == pygame.JOYBUTTONUP:
                # Reset arm/claw when buttons released
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
        # After line 69, add:
        if frame_count % 50 == 0:  # Print every 50 loops
            print(f"Sending: {act_lin:.2f}, {act_ang:.2f}, {act_close}, {act_lower}")
            frame_count += 1  # Add this counter at the top
        
        # Send control message to Pico
        msg = f"{act_lin},{act_ang},{act_close},{act_lower}\n".encode("utf-8")
        messenger.write(msg)
        #messenger.flush()
        
        # 50Hz control loop (20ms period)
        sleep(0.020)

# Take care terminal signal (Ctrl-c)
except KeyboardInterrupt:
    print("\nShutting down...")
    # Stop robot before closing
    messenger.write(b"0.0,0.0,0,0\n")
    sleep(0.1)
    pygame.quit()
    messenger.close()
    print("Robot stopped. Goodbye!")
    sys.exit()
