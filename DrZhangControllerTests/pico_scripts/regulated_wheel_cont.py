from sentient_wheel_cont import SentientWheel
from machine import Timer


class RegulatedWheel(SentientWheel):
    def __init__(self, driver_ids: list | tuple, encoder_ids: list | tuple) -> None:
        super().__init__(driver_ids, encoder_ids)
        # Constants
        self.k_p = 0.25
        self.k_i = 0.0
        self.k_d = 0.05
        self.reg_freq = 50  # Hz
        # Variables
        self.reg_vel_counter = 0
        self.duty = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        self.error_inte = 0.0  # integral
        self.error_diff = 0.0  # differentiation
        self.ref_lin_vel = 0.0
        # PID controller config
        self.vel_reg_timer = Timer(
            freq=self.reg_freq,
            mode=Timer.PERIODIC,
            callback=self.regulate_velocity,
        )

    def regulate_velocity(self, timer):
        if self.reg_vel_counter > self.reg_freq:
            self.stop()
            self.reset_encoder_counts()
            self.ref_lin_vel = 0.0
            self.prev_error = 0.0
        else:
            self.error = self.ref_lin_vel - self.meas_lin_vel  # ang_vel also works
            self.error_inte += self.error
            self.error_diff = self.error - self.prev_error
            self.prev_error = self.error  # UPDATE previous error
            inc_duty = (
                self.k_p * self.error
                + self.k_i * self.error_inte
                + self.k_d * self.error_diff
            )
            self.duty = self.duty + inc_duty
            if self.duty > 0:
                if self.duty > 1.0:
                    self.duty = 1.0
                self.forward(self.duty)
            else:
                if self.duty < -1.0:
                    self.duty = -1.0
                self.backward(-self.duty)
            self.reg_vel_counter += 1

    def set_wheel_velocity(self, ref_lin_vel):
        self.reg_vel_counter = 0
        if ref_lin_vel is not self.ref_lin_vel:
            self.ref_lin_vel = ref_lin_vel
            self.prev_error = 0.0
            self.error_inte = 0.0


if __name__ == "__main__":
    """ Use following tuning PID"""
    from utime import sleep

    # rw = RegulatedWheel(
    #     driver_ids=(21, 20, 19),
    #     encoder_ids=(7, 6),
    # )  # left wheel
    rw = RegulatedWheel(
        driver_ids=(16, 17, 18),
        encoder_ids=(27, 26),
    )  # right wheel
    for i in range(400):
        if 24 < i <= 174:  # step up @ t=0.5s
            rw.set_wheel_velocity(-0.1)
        elif 174 < i <= 299:  # step down @ t=2s
            rw.set_wheel_velocity(0.0)
        elif i == 349:
            print("No command given in the past 1 second, cut off.")
        print(
            f"Reference velocity={rw.ref_lin_vel} m/s, Measured velocity={rw.meas_lin_vel} m/s"
        )
        sleep(0.02)

    # Terminate
    rw.stop()
    sleep(0.5)
    print("wheel stopped.")


