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

print("Pico ready!")

# LOOP
while True:
    # POLL with 1ms timeout (very short, nearly non-blocking)
    event = cmd_vel_listener.poll(1)
    
    if event:
        for msg, _ in event:
            # Read multiple messages if available, keep the last one
            buffer = None
            for _ in range(10):  # Read up to 10 messages
                try:
                    line = msg.readline()
                    if line:
                        buffer = line.strip().split(b",")
                    else:
                        break
                except:
                    break
            
            # Process the latest message
            if buffer and len(buffer) == 4:
                try:
                    target_lin_vel = float(buffer[0])
                    target_ang_vel = float(buffer[1])
                    claw_dir = int(buffer[2])
                    arm_dir = int(buffer[3])
                except:
                    pass
    
    # Send command to robot
    diff_driver.set_vels(target_lin_vel, target_ang_vel)
    arm_controller.close_claw(claw_dir)
    arm_controller.lower_claw(arm_dir)
    
    # Send feedback to host machine (every 50ms = 20Hz instead of 10ms)
    toc = ticks_us()
    if ticks_diff(toc, tic) >= 50000:
        meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
        out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
        sys.stdout.write(out_msg)
        tic = toc