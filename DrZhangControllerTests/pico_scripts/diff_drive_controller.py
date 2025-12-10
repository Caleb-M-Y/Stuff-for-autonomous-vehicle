from regulated_wheel import RegulatedWheel


class DiffDriveController:
    def __init__(
        self, left_wheel_ids: list | tuple, right_wheel_ids: list | tuple
    ) -> None:
        # Configs
        self.left_wheel = RegulatedWheel(*left_wheel_ids)
        self.right_wheel = RegulatedWheel(*right_wheel_ids)
        # Constants
        self.wheel_sep = 0.52

    def get_vels(self):
        self.meas_lin_vel = 0.5 * (
            self.left_wheel.meas_lin_vel + self.right_wheel.meas_lin_vel
        )
        self.meas_ang_vel = (
            self.right_wheel.meas_lin_vel - self.left_wheel.meas_lin_vel
        ) / self.wheel_sep
        return self.meas_lin_vel, self.meas_ang_vel

    def set_vels(self, target_lin_vel, target_ang_vel):
        left_wheel_ref_vel = target_lin_vel - 0.5 * (target_ang_vel * self.wheel_sep)
        right_wheel_ref_vel = target_lin_vel + 0.5 * (target_ang_vel * self.wheel_sep)
        self.left_wheel.set_wheel_velocity(left_wheel_ref_vel)
        self.right_wheel.set_wheel_velocity(right_wheel_ref_vel)


if __name__ == "__main__":
    from utime import sleep
    from machine import freq

    # SETUP
    ddc = DiffDriveController(
        right_wheel_ids=((15, 13, 14), (10, 11)),
        left_wheel_ids=((16, 18, 17), (20, 19)),
    )

    # LOOP
    for i in range(400):
        if 24 < i <= 174:  # step up @ t=0.5 s
            ddc.set_vels(0.5, 0.0)
        elif 174 < i <= 299:  # step down @ t=2s
            ddc.set_vels(0.0, 0.0)
        elif i == 349:
            print("No command given in the past 1 second, cut off.")
        meas_lin_vel, meas_ang_vel = ddc.get_vels()
        # print(
        #     f"target: {ddc.left_wheel.ref_lin_vel} m/s, {ddc.right_wheel.ref_lin_vel} m/s"
        # )
        print(f"Velocity={meas_lin_vel} m/s, {meas_ang_vel} rad/s")
        sleep(0.02)
    ddc.set_vels(0.0, 0.0)
    print("motors stopped")
