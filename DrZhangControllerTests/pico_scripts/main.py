"""
Rename this script to main.py, then upload to the pico board.
"""

import sys
print("sys works")
import select
print("select works")
from diff_drive_cont import DiffDriveController
print("dif drive works")
from armcontroller import ArmController
print("arm import works")
from machine import freq
print("machine works")
from utime import ticks_us, ticks_diff
print("utime works")

# SETUP
# Overclock
try:
    freq(300_000_000)
    print("overclocked to 300MHz")
    print("new freq:", freq())
except Exception as e:
    print(f"overclock failed: {e}")

# Instantiate robot
diff_driver = DiffDriveController(
    right_wheel_ids=((3, 2, 4), (21, 20)),
    left_wheel_ids=((7, 6, 8), (11, 10)),
)
print("diff initialization works")
arm_controller = ArmController(12, 13, 14)
print("arm init works")

# Create a poll to receive messages from host machine
cmd_vel_listener = select.poll()
cmd_vel_listener.register(sys.stdin, select.POLLIN)
target_lin_vel, target_ang_vel = 0.0, 0.0
claw_dir, arm_dir = 0, 0
tic = ticks_us()
last_cmd_time = ticks_us()
msg_count = 0
loop_count = 0  # Track total loops

print("Pico ready!")
print("Waiting for messages from Pi...")

# LOOP
while True:
    loop_count += 1
    
    # Print loop status every 1000 iterations to show it's running
    if loop_count % 1000 == 0:
        print(f"[LOOP {loop_count}] Still running, waiting for data...")
    
    # poll for new messages (non-blocking, 0ms timeout)
    event = cmd_vel_listener.poll(0)

    if event:
        print(f"[EVENT DETECTED] Got event at loop {loop_count}")
        latest_buffer = None

        for msg, _ in event:
            try:
                # Read the latest message
                line = msg.readline()
                if line:
                    print(f"[RAW] Received: {line}")
                    # Decode bytes to string, then split
                    latest_buffer = line.decode('utf-8').strip().split(",")
                    print(f"[PARSED] Buffer: {latest_buffer}")
            except Exception as e:
                print(f"Read error: {e}")
    
        # Process the message
        if latest_buffer and len(latest_buffer) == 4: 
            try: 
                target_lin_vel = float(latest_buffer[0])
                target_ang_vel = float(latest_buffer[1])
                claw_dir = int(latest_buffer[2])
                arm_dir = int(latest_buffer[3])

                last_cmd_time = ticks_us()
                msg_count += 1
                
                # Debug output every message for now
                print(f"MSG #{msg_count}: lin={target_lin_vel:.2f}, ang={target_ang_vel:.2f}, claw={claw_dir}, arm={arm_dir}")

                # Update wheels (always, even if 0.0)
                diff_driver.set_vels(target_lin_vel, target_ang_vel)

                # Only update arm/claw if they're actually moving 
                if claw_dir != 0:
                    arm_controller.close_claw(claw_dir)
                if arm_dir != 0:
                    arm_controller.lower_claw(arm_dir)
            except (ValueError, IndexError) as e:
                print(f"Parse error: {e}, buffer={latest_buffer}")
        elif latest_buffer:
            print(f"[ERROR] Buffer wrong length: {len(latest_buffer)} (expected 4)")

    # Safety timeout: stop robot if no commands for 500ms
    if ticks_diff(ticks_us(), last_cmd_time) > 500000:
        diff_driver.set_vels(0.0, 0.0)
        last_cmd_time = ticks_us()

    # Send feedback every 50ms (20Hz)
    toc = ticks_us()
    if ticks_diff(toc, tic) >= 50000:
        meas_lin_vel, meas_ang_vel = diff_driver.get_vels()
        out_msg = f"{meas_lin_vel:.2f},{meas_ang_vel:.2f}\n"
        sys.stdout.write(out_msg)
        tic = toc