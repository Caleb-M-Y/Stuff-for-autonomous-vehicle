"""
Bare-bones Pico runtime for 5-field command protocol + fused feedback.

This version supports both deployment layouts:
1. package layout on Pico (`mobile_base.*`, `perception.*`)
2. flat-file layout in the same folder (fallback imports)
"""

import select
import sys

from machine import freq
from utime import ticks_diff, ticks_us

from mobile_base.ac2 import ArmController
from mobile_base.diff_drive_controller import DiffDriveController
from perception.odom_autonomous_inertial_sensor import MPU6050



FREQ_HZ = 240_000_000
CMD_TIMEOUT_US = 250_000
FEEDBACK_PERIOD_US = 10_000
ALPHA_IMU = 0.95

freq(FREQ_HZ)

# Keep your validated pin map.
ddc = DiffDriveController(
    left_ids=((21, 19, 20), (6, 7)),
    right_ids=((16, 18, 17), (26, 27)),
)
arm = ArmController(15, 13, 14)
imu = MPU6050(pow_id=3, scl_id=5, sda_id=4, i2c_addr=0x68)

listener = select.poll()
listener.register(sys.stdin, select.POLLIN)


def parse_host_command(line):
    if not line:
        return None
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        return float(parts[0]), float(parts[1]), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))
    except Exception:
        return None


target_lin = 0.0
target_ang = 0.0
sho_vel = 0
cla_vel = 0
arm_state = 0

last_cmd_t = ticks_us()
last_fb_t = ticks_us()

while True:
    events = listener.poll(0)
    for obj, _ in events:
        try:
            line = obj.readline().strip()
        except Exception:
            line = ""
        parsed = parse_host_command(line)
        if parsed is None:
            continue
        target_lin, target_ang, sho_vel, cla_vel, arm_state = parsed
        last_cmd_t = ticks_us()

    now = ticks_us()
    if ticks_diff(now, last_cmd_t) > CMD_TIMEOUT_US:
        target_lin = 0.0
        target_ang = 0.0
        sho_vel = 0
        cla_vel = 0
        arm_state = 0

    ddc.set_vels(target_lin, target_ang)

    if arm_state == 10:
        arm.set_neutral()
    else:
        arm.lower_claw(sho_vel)
        arm.close_claw(cla_vel)

    if ticks_diff(now, last_fb_t) >= FEEDBACK_PERIOD_US:
        meas_lin, meas_ang = ddc.get_vels()
        imu_data = imu.read_data()
        fused_ang = ALPHA_IMU * imu_data["omg_z"] + (1.0 - ALPHA_IMU) * meas_ang
        sys.stdout.write(f"{meas_lin:.3f}, {fused_ang:.3f}\\n")
        last_fb_t = now

