"""
Rename this script to main.py, then upload to the pico board.
"""

import sys
import select
from diff_drive_controller import DiffDriveController
from armcontroller import ArmController
from machine import freq
from utime import ticks_us, ticks_diff

# SETUP
# Overclock
freq(300_000_000)  # Pico 2 original: 150_000_000
# Instantiate robot
diff_driver = DiffDriveController(
    right_wheel_ids=((2, 3, 4), (21, 20)),
    left_wheel_ids=((6, 7, 8), (11, 10)),
)
arm_controller = ArmController(12, 13, 14)
# Create a poll to receive messages from host machine
cmd_vel_listener = select.poll()
cmd_vel_listener.register(sys.stdin, select.POLLIN)
#event = cmd_vel_listener.poll()
target_lin_vel, target_ang_vel = 0.0, 0.0
claw_dir, arm_dir = 0, 0
tic = ticks_us()
last_cmd_time = ticks_us()

print("Pico ready!")

# LOOP
while True:
    # poll for new messages (non-blocking, 0ms timeout)
    event = cmd_vel_listener.poll(0)
    if event:  # Only process if there's actually a message
        for msg, _ in event:
            try:
                buffer = msg.readline().strip().split(",")
                if len(buffer) == 4:
                    target_lin_vel = float(buffer[0])
                    target_ang_vel = float(buffer[1])
                    claw_dir = int(buffer[2])
                    arm_dir = int(buffer[3])
                    
                    last_cmd_time = ticks_us()
                    
                    # Update wheels
                    diff_driver.set_vels(target_lin_vel, target_ang_vel)
                    
                    # Only update arm/claw if they're actually moving
                    if claw_dir != 0:
                        arm_controller.close_claw(claw_dir)
                    if arm_dir != 0:
                        arm_controller.lower_claw(arm_dir)
            except (ValueError, IndexError):
                # Skip malformed messages
                pass

    # Safety timeout: stop robot if no commands for 500ms
    if ticks_diff(ticks_us(), last_cmd_time) > 500000:
        diff_driver.set_vels(0.0, 0.0)
        last_cmd_time = ticks_us()  # Reset timer to avoid constant stopping

    # Send feedback every 10ms
    toc = ticks_us()
    if ticks_diff(toc, tic) >= 10000:
        meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
        out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
        sys.stdout.write(out_msg)
        tic = toc
    
            

# buffer = msg.readline().strip().split(",")
#                 # print(f"{diff_driver.lin_vel},{diff_driver.ang_vel}")
#                 if len(buffer) == 4:
#                     target_lin_vel = float(buffer[0])
#                     target_ang_vel = float(buffer[1])
#                     claw_dir = int(buffer[2])
#                     arm_dir = int(buffer[3])
#                     diff_driver.set_vels(target_lin_vel, target_ang_vel)
#                     arm_controller.close_claw(claw_dir)
#                     arm_controller.lower_claw(arm_dir)
#             toc = ticks_us()
#             if toc - tic >= 10000:
#                 meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
#                 out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
#                 #         out_msg = "PICO\n"
#                 sys.stdout.write(out_msg)
#                 tic = ticks_us()