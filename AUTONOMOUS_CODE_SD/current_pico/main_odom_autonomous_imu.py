"""
Merged Pico entrypoint for odom + autonomous integration.

Upload workflow:
1) Keep this file in source control as a dedicated merged runtime.
2) Rename/copy to main.py only when you want to run this merged mode on Pico.

Protocol contract:
- RX from host: lin_vel, ang_vel, shoulder_cmd, claw_cmd, arm_state\n
- TX to host: meas_lin_vel, fused_ang_vel\n
Notes:
- Maintains 5-field command protocol.
- Uses IMU fusion for angular velocity feedback.
- Keeps deadman timeout safety behavior.
"""

import select
import sys

from ac2 import ArmController
from diff_drive_controller import DiffDriveController
from machine import freq
from odom_autonomous_inertial_sensor import MPU6050
from utime import ticks_diff, ticks_us


# ----------------------------------------------------------------------------
# Runtime constants
# ----------------------------------------------------------------------------
FREQ_HZ = 240_000_000
CMD_TIMEOUT_US = 250_000
FEEDBACK_PERIOD_US = 10_000
ALPHA_IMU = 0.95


# ----------------------------------------------------------------------------
# Hardware setup
# ----------------------------------------------------------------------------
freq(FREQ_HZ)

# Drive pin mapping follows your validated odometry wheel stack configuration.
ddc = DiffDriveController(
    left_ids=((21, 19, 20), (6, 7)),
    right_ids=((16, 18, 17), (26, 27)),
)

# Arm pin mapping follows your validated autonomous ac2 configuration.
arm = ArmController(15, 13, 14)
imu = MPU6050(pow_id=3, scl_id=5, sda_id=4, i2c_addr=0x68)

cmd_listener = select.poll()
cmd_listener.register(sys.stdin, select.POLLIN)


def parse_host_command(line):
    """Parse 5-field host command; return tuple or None when malformed."""
    if not line:
        return None
    parts = [segment.strip() for segment in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        lin_vel = float(parts[0])
        ang_vel = float(parts[1])
        shoulder = int(float(parts[2]))
        claw = int(float(parts[3]))
        arm_state = int(float(parts[4]))
    except Exception:
        return None
    return lin_vel, ang_vel, shoulder, claw, arm_state


target_lin_vel = 0.0
target_ang_vel = 0.0
sho_vel = 0
cla_vel = 0
arm_state = 0

last_cmd_t = ticks_us()
last_feedback_t = ticks_us()


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
while True:
    events = cmd_listener.poll(0)
    for msg, _ in events:
        parsed = parse_host_command(msg.readline().strip())
        if parsed is None:
            continue
        target_lin_vel, target_ang_vel, sho_vel, cla_vel, arm_state = parsed
        last_cmd_t = ticks_us()

    now = ticks_us()
    if ticks_diff(now, last_cmd_t) > CMD_TIMEOUT_US:
        target_lin_vel = 0.0
        target_ang_vel = 0.0
        sho_vel = 0
        cla_vel = 0
        arm_state = 0

    # Apply wheel and arm commands every loop iteration for responsive control.
    ddc.set_vels(target_lin_vel, target_ang_vel)

    if arm_state == 10:
        arm.set_neutral()
    else:
        arm.lower_claw(sho_vel)
        arm.close_claw(cla_vel)

    if ticks_diff(now, last_feedback_t) >= FEEDBACK_PERIOD_US:
        meas_lin_vel, meas_ang_vel = ddc.get_vels()

        # Fuse wheel-based angular estimate with IMU gyro Z for better yaw robustness.
        imu_data = imu.read_data()
        fused_ang_vel = ALPHA_IMU * imu_data["omg_z"] + (1.0 - ALPHA_IMU) * meas_ang_vel

        sys.stdout.write(f"{meas_lin_vel:.3f}, {fused_ang_vel:.3f}\n")
        last_feedback_t = now
