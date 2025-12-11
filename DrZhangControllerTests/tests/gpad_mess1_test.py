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

print("\n=== TEST MODE - Communication Only ===")
print("Messages will be sent but robot will NOT move")
print(f"Left stick: Forward/Backward (max {params['lin_vel_max']} m/s)")
print(f"Right stick: Turn Left/Right (max {params['ang_vel_max']} rad/s)")
print(f"Button {params['lower_button']}: Lower arm")
print(f"Button {params['raise_button']}: Raise arm")
print(f"Button {params['close_button']}: Close claw")
print(f"Button {params['open_button']}: Open claw")
print("=======================================\n")
print("Ready! Press Ctrl+C to stop.\n")

# MAIN LOOP
try:
    msg_count = 0
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
        
        msg = f"{act_lin}, {act_ang}, {act_close}, {act_lower}\n".encode("utf-8")
        messenger.write(msg)
        msg_count += 1
        
        # Print every 10th message to monitor communication
        if msg_count % 10 == 0:
            print(f"Sent #{msg_count}: lin={act_lin:.2f}, ang={act_ang:.2f}, claw={act_close}, arm={act_lower}")
        
        # Try to read feedback from Pico (non-blocking)
        # Read all available feedback to prevent buffer buildup
        feedback_count = 0
        while messenger.in_waiting > 0 and feedback_count < 10:  # Limit to prevent blocking
            try:
                feedback = messenger.readline().decode('utf-8').strip()
                # Only print every 10th feedback to reduce spam
                if feedback_count == 0:
                    print(f"Feedback: {feedback}")
            except:
                break
            feedback_count += 1
        
        # 100Hz control loop (10ms period) - matches blocking poll on Pico
        sleep(0.01)

# Take care terminal signal (Ctrl-c)
except KeyboardInterrupt:
    print("\nShutting down...")
    messenger.write(b"0.0,0.0,0,0\n")
    sleep(0.1)
    pygame.quit()
    messenger.close()
    print("Test stopped. Goodbye!")
    sys.exit()

