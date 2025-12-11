"""
Rename this script to main.py, then upload to the pico board.
"""

import sys
import select
from diff_drive_cont import DiffDriveController
from armcontroller import ArmController
from machine import freq
from utime import ticks_us, ticks_diff

# SETUP
# Overclock
freq(300_000_000)  # Pico 2 original: 150_000_000
# Instantiate robot
diff_driver = DiffDriveController(
    right_wheel_ids=((3, 2, 4), (21, 20)),
    left_wheel_ids=((7, 6, 8), (11, 10)),
)
arm_controller = ArmController(12, 13, 14)
# Create a poll to receive messages from host machine
cmd_vel_listener = select.poll()
cmd_vel_listener.register(sys.stdin, select.POLLIN)

target_lin_vel, target_ang_vel = 0.0, 0.0
claw_dir, arm_dir = 0, 0
tic = ticks_us()
last_cmd_time = ticks_us()

print("Pico ready!")

# LOOP
while True:
    # NON-BLOCKING poll (0ms timeout)
    event = cmd_vel_listener.poll(0)
    
    if event:
        # Read ALL available messages, keep only the latest
        latest_buffer = None
        for msg, _ in event:
            # Keep reading until buffer is empty
            while msg.any():
                try:
                    line = msg.readline()
                    if line:
                        latest_buffer = line.strip().split(b",")
                except:
                    break
        
        # Process ONLY the latest message
        if latest_buffer and len(latest_buffer) == 4:
            try:
                target_lin_vel = float(latest_buffer[0])
                target_ang_vel = float(latest_buffer[1])
                claw_dir = int(latest_buffer[2])
                arm_dir = int(latest_buffer[3])
                
                last_cmd_time = ticks_us()
            except:
                pass
    
    # Send command to robot
    diff_driver.set_vels(target_lin_vel, target_ang_vel)
    
    # Only update arm/claw if actually moving (reduces servo jitter)
    if claw_dir != 0:
        arm_controller.close_claw(claw_dir)
    if arm_dir != 0:
        arm_controller.lower_claw(arm_dir)
    
    # Safety: stop if no commands for 500ms
    if ticks_diff(ticks_us(), last_cmd_time) > 500000:
        target_lin_vel, target_ang_vel = 0.0, 0.0
        diff_driver.set_vels(0.0, 0.0)
        last_cmd_time = ticks_us()
    
    # Send feedback to host machine (every 50ms = 20Hz)
    toc = ticks_us()
    if ticks_diff(toc, tic) >= 50000:
        meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
        out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
        sys.stdout.write(out_msg)
        tic = toc