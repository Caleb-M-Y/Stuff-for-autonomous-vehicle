"""
Rename this script to main.py, then upload to the pico board.
"""

import sys
import select
from diff_drive_controller import DiffDriveController
from arm_controller import ArmController
from machine import freq
from utime import ticks_us

# SETUP
# Overclock
freq(300_000_000)  # Pico 2 original: 150_000_000
# Instantiate robot
diff_driver = DiffDriveController(
    right_wheel_ids=((15, 13, 14), (10, 11)),
    left_wheel_ids=((16, 18, 17), (20, 19)),
)
arm_controller = ArmController(2, 3, 4)
# Create a poll to receive messages from host machine
cmd_vel_listener = select.poll()
cmd_vel_listener.register(sys.stdin, select.POLLIN)
event = cmd_vel_listener.poll()
target_lin_vel, target_ang_vel = 0.0, 0.0
claw_dir, arm_dir = 0, 0
tic = ticks_us()

# LOOP
while True:
    for msg, _ in event:
        buffer = msg.readline().strip().split(",")
        # print(f"{diff_driver.lin_vel},{diff_driver.ang_vel}")
        if len(buffer) == 4:
            target_lin_vel = float(buffer[0])
            target_ang_vel = float(buffer[1])
            claw_dir = int(buffer[2])
            arm_dir = int(buffer[3])
            diff_driver.set_vels(target_lin_vel, target_ang_vel)
            arm_controller.close_claw(claw_dir)
            arm_controller.lower_claw(arm_dir)
    toc = ticks_us()
    if toc - tic >= 10000:
        meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
        out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
        #         out_msg = "PICO\n"
        sys.stdout.write(out_msg)
        tic = ticks_us()
