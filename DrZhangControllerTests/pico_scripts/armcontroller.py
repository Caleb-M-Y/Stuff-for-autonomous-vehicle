from machine import Pin, PWM
from time import sleep

# SHOULDER_NEUTRAL = 1_400_000
CLAW_MAX = 2_600_000
CLAW_MIN = 1_800_000
CLAW_MID = (CLAW_MAX + CLAW_MIN) // 2
CLAW_RANGE = (CLAW_MAX - CLAW_MIN) // 2
R_SHOULDER_MAX = 400_000
R_SHOULDER_MIN = 2_100_000
R_SHOULDER_MID = 1_600_000
L_SHOULDER_MAX = 2_500_000
L_SHOULDER_MIN = 800_000
L_SHOULDER_MID = 1_300_000
# SHOULDER_RANGE = (CLAW_MAX - CLAW_MIN) // 2


class ArmController:
    def __init__(self, claw_id, left_shoulder_id, right_shoulder_id):
        # Config servo pins
        self.claw = PWM(Pin(claw_id))
        self.claw.freq(50)
        self.left_shoulder = PWM(Pin(left_shoulder_id))
        self.left_shoulder.freq(50)
        self.right_shoulder = PWM(Pin(right_shoulder_id))
        self.right_shoulder.freq(50)
        # Variables
        self.claw_pulse_width = CLAW_MIN
        self.left_shoulder_pulse_width = L_SHOULDER_MID
        self.right_shoulder_pulse_width = R_SHOULDER_MID
        # Constants
        self.claw.duty_ns(CLAW_MIN)
        self.left_shoulder.duty_ns(L_SHOULDER_MID)
        self.right_shoulder.duty_ns(R_SHOULDER_MID)
        self.pw_inc = 7_500

    def close_claw(self, dir):  # Close claw
        """
        Open/Close claw
        Args:
            dir: -1/0/1, open/maintain/close
        """
        assert dir == 0 or dir == -1 or dir == 1
        self.claw_pulse_width += self.pw_inc * dir
        if self.claw_pulse_width >= CLAW_MAX:
            self.claw_pulse_width = CLAW_MAX
        elif self.claw_pulse_width <= CLAW_MIN:
            self.claw_pulse_width = CLAW_MIN
        self.claw.duty_ns(self.claw_pulse_width)

    def lower_claw(self, dir=0):  # Lower arm
        """
        Raise/Lower claw
        Args:
            dir: -1/0/1, raise/maintain/lower
        """
        assert dir == 0 or dir == -1 or dir == 1
        self.left_shoulder_pulse_width += self.pw_inc * dir
        self.right_shoulder_pulse_width -= self.pw_inc * dir

        if self.left_shoulder_pulse_width >= L_SHOULDER_MAX:
            self.left_shoulder_pulse_width = L_SHOULDER_MAX
            self.right_shoulder_pulse_width = 2 * R_SHOULDER_MID - R_SHOULDER_MAX

        elif self.left_shoulder_pulse_width <= L_SHOULDER_MIN:
            self.left_shoulder_pulse_width = L_SHOULDER_MIN
            self.right_shoulder_pulse_width = 2 * R_SHOULDER_MID - R_SHOULDER_MIN

        self.left_shoulder.duty_ns(self.left_shoulder_pulse_width)
        self.right_shoulder.duty_ns(self.right_shoulder_pulse_width)


# Example usage
if __name__ == "__main__":
    from utime import sleep

    sleep(1)
    ac = ArmController(15, 13, 14)
    for _ in range(40):
        ac.close_claw(-1)
        sleep(0.1)
    for _ in range(40):
        ac.close_claw(1)
        sleep(0.1)
    for _ in range(40):
        ac.lower_claw(-1)
        sleep(0.1)
    for _ in range(40):
        ac.lower_claw(1)
        sleep(0.1)
