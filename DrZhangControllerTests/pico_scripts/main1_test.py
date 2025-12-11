"""
Test version - Communication only, no robot control
Rename this script to main.py, then upload to the pico board for testing.
"""

import sys
import select
from machine import freq
from utime import ticks_us, ticks_diff

# SETUP
# Overclock
freq(300_000_000)  # Pico 2 original: 150_000_000

# Create a poll to receive messages from host machine
cmd_vel_listener = select.poll()
cmd_vel_listener.register(sys.stdin, select.POLLIN)

target_lin_vel, target_ang_vel = 0.0, 0.0
claw_dir, arm_dir = 0, 0
tic = ticks_us()
message_count = 0

print("Pico ready! (TEST MODE - No robot control)")

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
                message_count += 1
                # Print received message (every 10th message to avoid spam)
                if message_count % 10 == 0:
                    print(f"Received: lin={target_lin_vel:.2f}, ang={target_ang_vel:.2f}, claw={claw_dir}, arm={arm_dir} (msg #{message_count})")
            except ValueError as e:
                print(f"Parse error: {e}, buffer={buffer}")
    
    # NO robot control - just echo back what we received
    # (In real version, this would be: diff_driver.set_vels(...) and arm_controller calls)
    
    # send feedback to host machine (echo received values as "measured" values)
    toc = ticks_us()
    if ticks_diff(toc, tic) >= 10000:  # 10ms = 100Hz feedback rate
        # Echo back the received values as if they were measured
        meas_lin_vel = target_lin_vel
        meas_ang_vel = target_ang_vel
        out_msg = f"{meas_lin_vel}, {meas_ang_vel}\n"
        sys.stdout.write(out_msg)
        tic = toc

