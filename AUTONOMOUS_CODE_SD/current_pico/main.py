
"""
Rename this script to main.py, then upload to the pico board.
"""

import sys
import select
from diff_drive_controller import DiffDriveController
from ac2 import ArmController
from machine import freq
from utime import ticks_us, ticks_diff

# SETUP
# Overclock
freq(240_000_000)  # Pico 2 original: 150_000_000
# Instantiate robot
ddc = DiffDriveController(
    right_ids=((16, 17, 18), (27, 26)), left_ids=((21, 20, 19), (7, 6))
)
arm = ArmController(15, 13, 14)
# Create a poll to receive messages from host machine
cmd_vel_listener = select.poll()
cmd_vel_listener.register(sys.stdin, select.POLLIN)
target_lin_vel, target_ang_vel = 0.0, 0.0
sho_vel, cla_vel, arm_state = 0, 0, 0
tic = ticks_us()
last_cmd_t = ticks_us()
CMD_TIMEOUT_US = 250_000  # deadman timeout
FEEDBACK_PERIOD_US = 10_000

# LOOP
while True:
    events = cmd_vel_listener.poll(0)
    for msg, _ in events:
        line = msg.readline().strip()
        if not line:
            continue
        buffer = [x.strip() for x in line.split(",")]
        if len(buffer) != 5:
            continue
        try:
            target_lin_vel = float(buffer[0])
            target_ang_vel = float(buffer[1])
            sho_vel = int(buffer[2])
            cla_vel = int(buffer[3])
            arm_state = int(buffer[4])
            last_cmd_t = ticks_us()
        except Exception:
            continue

    now = ticks_us()
    if ticks_diff(now, last_cmd_t) > CMD_TIMEOUT_US:
        target_lin_vel = 0.0
        target_ang_vel = 0.0
        sho_vel = 0
        cla_vel = 0
        arm_state = 0

    ddc.set_vels(target_lin_vel, target_ang_vel)
    if arm_state == 10:  # neutral / return posture
        arm.set_neutral()
    else:
        arm.lower_claw(sho_vel)
        arm.close_claw(cla_vel)

    toc = ticks_us()
    if ticks_diff(toc, tic) >= FEEDBACK_PERIOD_US:
        meas_lin_vel, meas_ang_vel = ddc.get_vels()
        out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
        sys.stdout.write(out_msg)
        tic = toc


