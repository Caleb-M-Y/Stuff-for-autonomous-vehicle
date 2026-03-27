from machine import Pin, PWM
import arm_tuning as tune


class ArmController:
    def __init__(self, claw_pin, arm_pin_a, arm_pin_b):
        self.claw_servo = PWM(Pin(claw_pin))
        self.shoulder_servo_a = PWM(Pin(arm_pin_a))
        self.shoulder_servo_b = PWM(Pin(arm_pin_b))
        self.claw_servo.freq(50)
        self.shoulder_servo_a.freq(50)
        self.shoulder_servo_b.freq(50)
        self.set_neutral()

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def _sanitize(cmd):
        try:
            val = int(cmd)
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

    def lower_claw(self, dc_inc=0):
        base = self._sanitize(dc_inc)
        if base > 0:
            scaled = int(base * tune.SHOULDER_CMD_SCALE * tune.SHOULDER_LOWER_SCALE)
        else:
            scaled = int(base * tune.SHOULDER_CMD_SCALE * tune.SHOULDER_RAISE_SCALE)

        self.shoulder_duty_a = self._clamp(self.shoulder_duty_a + scaled, tune.SHOULDER_A_MIN, tune.SHOULDER_A_MAX)
        self.shoulder_duty_b = self._clamp(self.shoulder_duty_b - scaled, tune.SHOULDER_B_MIN, tune.SHOULDER_B_MAX)
        self.shoulder_servo_a.duty_ns(self.shoulder_duty_a)
        self.shoulder_servo_b.duty_ns(self.shoulder_duty_b)

    def close_claw(self, dc_inc=0):
        scaled = int(self._sanitize(dc_inc) * tune.CLAW_CMD_SCALE)
        self.claw_duty = self._clamp(self.claw_duty + scaled, tune.CLAW_MIN, tune.CLAW_MAX)
        self.claw_servo.duty_ns(self.claw_duty)
