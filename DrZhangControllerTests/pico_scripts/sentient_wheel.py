from encoded_motor import EncodedMotor
from machine import Timer
from math import pi


class SentientWheel(EncodedMotor):
    def __init__(self, driver_ids: list | tuple, encoder_ids: list | tuple) -> None:
        super().__init__(driver_ids, encoder_ids)
        # Constants
        self.wheel_radius = 0.075  # m
        self.gear_ratio = 102.083
        self.cpr = 64  # CPR = PPR * 4
        self.meas_freq = 100  # Hz
        # Velocity measuring timer
        self.vel_meas_timer = Timer(
            freq=self.meas_freq,
            mode=Timer.PERIODIC,
            callback=self.measure_velocity,
        )
        # Variables
        self.prev_counts = 0
        self.meas_ang_vel = 0.0
        self.meas_lin_vel = 0.0

    def measure_velocity(self, timer):
        curr_counts = self.encoder_counts
        delta_counts = curr_counts - self.prev_counts
        self.prev_counts = curr_counts  # UPDATE previous counts
        counts_per_sec = delta_counts * self.meas_freq  # delta_c / delta_t
        orig_rev_per_sec = counts_per_sec / self.cpr
        orig_rad_per_sec = orig_rev_per_sec * 2 * pi  # original motor shaft velocity
        self.meas_ang_vel = orig_rad_per_sec / self.gear_ratio
        self.meas_lin_vel = self.meas_ang_vel * self.wheel_radius

    def reset_encoder_counts(self):
        super().reset_encoder_counts()
        self.prev_counts = 0


# TEST
if __name__ == "__main__":  # Test only the encoder part
    from time import sleep

    # SETUP
    # sw = SentientWheel(
    #     driver_ids=(16, 18, 17),
    #     encoder_ids=(20, 19),
    # )  # left wheel
    sw = SentientWheel(
        driver_ids=(15, 13, 14),
        encoder_ids=(10, 11),
    )  # right wheel

    # LOOP
    # Forward ramp up and down
    for i in range(100):
        sw.forward((i + 1) / 100)
        print(
            f"Wheel's angular velocity={sw.meas_ang_vel}, linear velocity={sw.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        sw.forward((i + 1) / 100)
        print(
            f"Wheel's angular velocity={sw.meas_ang_vel}, linear velocity={sw.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp down
    # Summarize forward travel distance
    print(
        f"Wheel traveled: {sw.encoder_counts / sw.cpr / sw.gear_ratio * 2 * pi * sw.wheel_radius} m"
    )
    sw.reset_encoder_counts()
    sleep(1)
    # Backward ramp up and down
    for i in range(100):
        sw.backward((i + 1) / 100)
        print(
            f"Wheel's angular velocity={sw.meas_ang_vel}, linear velocity={sw.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        sw.backward((i + 1) / 100)
        print(
            f"Wheel's angular velocity={sw.meas_ang_vel}, linear velocity={sw.meas_lin_vel}"
        )
        sleep(4 / 100)  # 4 seconds to ramp down
    print(
        f"Wheel traveled: {sw.encoder_counts / sw.cpr / sw.gear_ratio * 2 * pi * sw.wheel_radius} m"
    )

    # Terminate
    sw.stop()
    sleep(0.5)
    print("wheel stopped.")
