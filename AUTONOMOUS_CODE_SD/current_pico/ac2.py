from machine import Pin, PWM
from time import sleep
import arm_tuning as tune


class ArmController:
    def __init__(self, claw_pin, arm_pin_a, arm_pin_b):
        self.claw_servo = PWM(Pin(claw_pin))
        self.shoulder_servo_a = PWM(Pin(arm_pin_a))
        self.shoulder_servo_b = PWM(Pin(arm_pin_b))
        self.claw_servo.freq(50)
        self.shoulder_servo_a.freq(50) #LEFT
        self.shoulder_servo_b.freq(50) #RIGHT

        # Set initial positions
        self.set_neutral()

    @staticmethod
    def _clamp(value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value

    @staticmethod
    def _sanitize_host_cmd(dc_inc):
        try:
            val = int(dc_inc)
        except Exception:
            val = 0
        return ArmController._clamp(val, -tune.MAX_ABS_HOST_CMD, tune.MAX_ABS_HOST_CMD)
        
    def set_neutral(self):
        self.shoulder_duty_a = tune.SHOULDER_A_NEUTRAL
        self.shoulder_duty_b = tune.SHOULDER_B_NEUTRAL
        self.claw_duty = tune.CLAW_NEUTRAL

        self.shoulder_servo_a.duty_ns(self.shoulder_duty_a)
        self.shoulder_servo_b.duty_ns(self.shoulder_duty_b)
        self.claw_servo.duty_ns(self.claw_duty)
        
# R_SHOULDER_MAX = 400_000
# R_SHOULDER_MIN = 2_100_000
# R_SHOULDER_MID = 1_600_000
# L_SHOULDER_MAX = 2_500_000
# L_SHOULDER_MIN = 800_000
# L_SHOULDER_MID = 1_300_000

    def lower_claw(self, dc_inc=0):  # Lower arm
        base_cmd = self._sanitize_host_cmd(dc_inc)
        if base_cmd > 0:
            scaled = int(base_cmd * tune.SHOULDER_CMD_SCALE * tune.SHOULDER_LOWER_SCALE)
        else:
            scaled = int(base_cmd * tune.SHOULDER_CMD_SCALE * tune.SHOULDER_RAISE_SCALE)

        self.shoulder_duty_a += scaled
        self.shoulder_duty_b -= scaled

        self.shoulder_duty_a = self._clamp(
            self.shoulder_duty_a, tune.SHOULDER_A_MIN, tune.SHOULDER_A_MAX
        )
        self.shoulder_duty_b = self._clamp(
            self.shoulder_duty_b, tune.SHOULDER_B_MIN, tune.SHOULDER_B_MAX
        )

        self.shoulder_servo_a.duty_ns(self.shoulder_duty_a)
        self.shoulder_servo_b.duty_ns(self.shoulder_duty_b)


# CLAW_MAX = 2_600_000
# CLAW_MIN = 1_800_000
# CLAW_MID = (CLAW_MAX + CLAW_MIN) // 2
# CLAW_RANGE = (CLAW_MAX - CLAW_MIN) // 2

    def close_claw(self, dc_inc=0):  # Close claw
        base_cmd = self._sanitize_host_cmd(dc_inc)
        scaled = int(base_cmd * tune.CLAW_CMD_SCALE)
        self.claw_duty += scaled
        self.claw_duty = self._clamp(self.claw_duty, tune.CLAW_MIN, tune.CLAW_MAX)
        self.claw_servo.duty_ns(self.claw_duty)
        

# Example usage
if __name__ == "__main__":
    from utime import sleep

    sleep(1)
    ac = ArmController(15, 13, 14)
    for _ in range(80):
        ac.close_claw(10_000)
        sleep(0.1)
        print(f"Closing claw duty cycle: {ac.claw_duty}")
#     for _ in range(20):
#         ac.close_claw(-20_000)
#         sleep(0.1)
#         print(f"Opening claw duty cycle: {ac.claw_duty}")
# 
#     ac.set_neutral()
#     sleep(1)
#     print("arm set to neutral")
# 
#     for _ in range(20):
#         ac.lower_claw(20_000)
#         sleep(0.1)
#         #print(f"Lowering claw duty cycle: {ac.shoulder_duty}")
#     for _ in range(20):
#         ac.lower_claw(-20_000)
#         sleep(0.1)
        #print(f"Lifting claw duty cycle: {ac.shoulder_duty}")




