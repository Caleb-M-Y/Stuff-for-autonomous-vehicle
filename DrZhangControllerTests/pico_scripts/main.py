"""
Production main.py - No debug output
"""

import sys
import select
from diff_drive_cont import DiffDriveController
from armcontroller import ArmController
from machine import freq
from utime import ticks_us, ticks_diff

# SETUP
freq(300_000_000)

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

# LOOP
while True:
    event = cmd_vel_listener.poll(0)
    
    if event:
        latest_buffer = None
        
        for msg, _ in event:
            try:
                line = msg.readline()
                if line:
                    latest_buffer = line.decode('utf-8').strip().split(",")
            except:
                pass
        
        if latest_buffer and len(latest_buffer) == 4: 
            try: 
                target_lin_vel = float(latest_buffer[0])
                target_ang_vel = float(latest_buffer[1])
                claw_dir = int(latest_buffer[2])
                arm_dir = int(latest_buffer[3])
                
                last_cmd_time = ticks_us()
                
                diff_driver.set_vels(target_lin_vel, target_ang_vel)
                
                if claw_dir != 0:
                    arm_controller.close_claw(claw_dir)
                if arm_dir != 0:
                    arm_controller.lower_claw(arm_dir)
            except:
                pass

    # Safety timeout
    if ticks_diff(ticks_us(), last_cmd_time) > 500000:
        diff_driver.set_vels(0.0, 0.0)
        last_cmd_time = ticks_us()

    # Send feedback
    toc = ticks_us()
    if ticks_diff(toc, tic) >= 50000:
        meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
        sys.stdout.write(f"{meas_lin_vel:.2f},{meas_ang_vel:.2f}\n")
        tic = toc