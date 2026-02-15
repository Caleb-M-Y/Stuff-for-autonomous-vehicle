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
    right_wheel_ids=((16, 17, 18), (27, 26)),
    left_wheel_ids=((21, 20, 19), (7, 6)),
)
arm_controller = ArmController(15, 13, 14)
# Create a poll to receive messages from host machine
cmd_vel_listener = select.poll()
cmd_vel_listener.register(sys.stdin, select.POLLIN)

target_lin_vel, target_ang_vel = 0.0, 0.0
claw_dir, arm_dir = 0, 0
tic = ticks_us()

# Print removed - interferes with serial communication
# print("Pico ready!")

# LOOP
while True:
    event = cmd_vel_listener.poll()  # wait until receive message (blocking - prevents buffer clogging)
    for msg, _ in event:  # read message
        buffer = [x.strip() for x in msg.readline().strip().split(",")]  # strip whitespace from each element
        if len(buffer) == 4:
            try:
                target_lin_vel = float(buffer[0])
                target_ang_vel = float(buffer[1])
                claw_dir = int(buffer[2])
                arm_dir = int(buffer[3])
            except ValueError:
                # Silently handle parse errors to avoid serial interference
                pass
    # send command to robot
    diff_driver.set_vels(target_lin_vel, target_ang_vel)
    arm_controller.close_claw(claw_dir)
    arm_controller.lower_claw(arm_dir)
    # send feedback to host machine
    toc = ticks_us()
    if ticks_diff(toc, tic) >= 10000:  # 10ms = 100Hz feedback rate
        meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
        out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
        sys.stdout.write(out_msg)
        tic = toc

# # LOOP
# while True:
#     event = cmd_vel_listener.poll()  # wait until receive message
#     for msg, _ in event:  # read message
#         buffer = msg.readline().strip().split(",")
#         if len(buffer) == 4:
#             target_lin_vel = float(buffer[0])
#             target_ang_vel = float(buffer[1])
#             claw_dir = int(buffer[2])
#             arm_dir = int(buffer[3])
#     # send command to robot
#     diff_driver.set_vels(target_lin_vel, target_ang_vel)
#     arm_controller.close_claw(claw_dir)
#     arm_controller.lower_claw(arm_dir)
#     # send feedback to host machine
#     toc = ticks_us()
#     if ticks_diff(toc, tic) >= 10000:
#         meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
#         out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
#         sys.stdout.write(out_msg)
#         tic = toc