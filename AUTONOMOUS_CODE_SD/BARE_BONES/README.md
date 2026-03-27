# Bare Bones Set

This folder contains a stripped autonomous stack for faster debug loops.

## Goal
Keep only these essentials:
- detection
- odom goal update + odom control
- autonomous state transitions (detect/pick/drop + simple transitions)

## Files
- autonomy_tuning.py: minimal knobs
- odom_autonomous_bridge.py: minimal odom pose + goal controller + camera transform
- state_machine.py: minimal behavior logic
- course_autonomous_depth.py: minimal runner

## Notes
- Original project files are untouched.
- This set keeps the same 5-field host->Pico protocol.
- This set keeps Pico feedback parsing as 2-field lin,ang.
- Label order enforcement is intentionally omitted here for fast iteration.

## Quick run idea on Pi
Use the same command style, but run this script from this folder:

python course_autonomous_depth.py --hef-path /home/ball-e/ball-e/models/calebv8m.hef --labels-json /home/ball-e/ball-e/models/ball_bucket.json --port /dev/ttyACM0 --no-display

## Suggested next test focus
1. Verify linear command direction in latest_msg on first detection.
2. Verify arm movement during pick with stronger command magnitudes.
3. Verify target updates stop when arm_state is active.
