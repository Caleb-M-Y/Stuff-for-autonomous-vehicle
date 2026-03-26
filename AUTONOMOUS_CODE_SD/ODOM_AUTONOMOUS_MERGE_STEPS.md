# Odom + Autonomous Merge Steps

This runbook documents the staged merge so you can test one layer at a time.

## Goal Context
Competition objective is to:
1. Start outside the 30ft x 30ft square.
2. Acquire a specified ball first.
3. Pick ball.
4. Find matching bucket at a corner.
5. Drive to bucket and drop.
6. Reorient and repeat for all four ball-bucket matches.

## What Was Implemented In This Pass

### Step A: Host-side odometry bridge (passive integration)
- Added: `odom_autonomous_bridge.py`
- Added odometry constants in `autonomy_tuning.py`.
- Wired Pico feedback parsing and pose integration in `course_autonomous_depth.py`.
- Added odom and feedback fields to telemetry CSV.
- Added odom overlay text to display.

Behavior note:
- This pass does **not** replace state-machine control decisions yet.
- Existing mission logic and handler transitions remain active.

### Step B: Pico merged entrypoint (new file, not replacing current main)
- Added: `current_pico/main_odom_autonomous_imu.py`
- Added: `current_pico/odom_autonomous_inertial_sensor.py`

Behavior note:
- Keeps 5-field command protocol from host.
- Adds IMU fusion for angular feedback.
- Keeps deadman timeout behavior.

### Step C: Ball-detect odom assist (flag-gated with fallback)
- Updated: `state_machine.py`
- Updated: `autonomy_tuning.py`
- Updated: `course_autonomous_depth.py`

Behavior note:
- Odom assist is active only in ball detect mode and only in far-approach region.
- Close-range pick gating still uses the existing vision/depth logic.
- If odom goal or depth is unavailable, control falls back to legacy vision steering automatically.

### Step D: Bucket-detect odom assist (flag-gated with fallback)
- Updated: `state_machine.py`
- Updated: `autonomy_tuning.py`
- Updated: `course_autonomous_depth.py`

Behavior note:
- Odom assist is now also active in bucket detect mode during far-approach region.
- Close-range drop gating still uses the existing vision/depth logic.
- If odom goal or depth is unavailable, bucket control falls back to legacy steering automatically.

### Step E: Color-aware odom goals + fallback counters
- Updated: `state_machine.py`
- Updated: `autonomy_tuning.py`
- Updated: `course_autonomous_depth.py`

Behavior note:
- Odom goal updates now honor mission color objective when color filtering is enabled.
- Odom goal depth now uses a dedicated median-kernel radius setting.
- Telemetry/HUD now include cumulative counters for:
   - goal updates,
   - color rejects,
   - invalid depth rejects,
   - odom goal errors,
   - and ball/bucket fallback usage.

### Step F: Odom hysteresis + per-lap snapshots
- Updated: `state_machine.py`
- Updated: `autonomy_tuning.py`
- Updated: `course_autonomous_depth.py`

Behavior note:
- Odom goal hysteresis can hold the current goal when new candidates are very close,
   reducing steering jitter in cluttered scenes.
- At lap completion, odom counters are snapshotted and optionally reset for cleaner
   per-lap diagnostics.
- Run metadata now includes an `odom_summary` block with snapshot history.

## Bring-Up Procedure

### 1) Host-only validation (no Pico file swap yet)
1. Run autonomous host script as usual.
2. Verify telemetry now includes odom columns:
   - `odom_x_m`, `odom_y_m`, `odom_theta_rad`
   - `feedback_lin_mps`, `feedback_ang_rps`
3. Confirm no behavior regressions in pick/drop cycle.

### 2) Pico merged runtime validation
1. In `current_pico`, upload `main_odom_autonomous_imu.py` as `main.py`.
2. Upload `odom_autonomous_inertial_sensor.py` with it.
3. Keep existing wheel/arm support modules in same folder.
4. Bench test with wheels lifted:
   - Send idle command.
   - Send small turn command.
   - Verify serial feedback updates and remains stable.

### 3) Integrated floor test
1. Run autonomous host with merged Pico runtime.
2. Verify:
   - Robot still obeys mission state transitions.
   - Pose values evolve smoothly while moving.
   - No serial contention or parser drops.

## Next Coding Pass (Planned)
1. Tune IMU fusion alpha and odom controller gains from field logs.
2. Add an automated offline log analyzer script for odom-assist effectiveness.
3. Add optional auto-disable of odom assist when fallback rates exceed threshold.
4. Add lane-safe speed cap profile for narrow hallway validation.

## Notes
- Pylance in desktop Python may warn about MicroPython imports (`machine`, `utime`) in Pico files; this is expected for firmware-targeted scripts.
- Keep one serial owner on host side. Current design retains a single owner in `course_autonomous_depth.py`.
