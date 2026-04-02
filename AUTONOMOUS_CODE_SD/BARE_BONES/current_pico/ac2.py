from machine import Pin, PWM
import mobile_base.arm_tuning as tune


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

    @staticmethod
    def _slew(current, target, max_step):
        delta = target - current
        if delta > max_step:
            return current + max_step
        if delta < -max_step:
            return current - max_step
        return target

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

        target_a = self._clamp(self.shoulder_duty_a + scaled, tune.SHOULDER_A_MIN, tune.SHOULDER_A_MAX)
        target_b = self._clamp(self.shoulder_duty_b - scaled, tune.SHOULDER_B_MIN, tune.SHOULDER_B_MAX)

        step = int(getattr(tune, "SHOULDER_STEP_LIMIT_NS", 8_000))
        self.shoulder_duty_a = self._slew(self.shoulder_duty_a, target_a, step)
        self.shoulder_duty_b = self._slew(self.shoulder_duty_b, target_b, step)

        self.shoulder_servo_a.duty_ns(self.shoulder_duty_a)
        self.shoulder_servo_b.duty_ns(self.shoulder_duty_b)

    def close_claw(self, dc_inc=0):
        scaled = int(self._sanitize(dc_inc) * tune.CLAW_CMD_SCALE)
        target = self._clamp(self.claw_duty + scaled, tune.CLAW_MIN, tune.CLAW_MAX)

        step = int(getattr(tune, "CLAW_STEP_LIMIT_NS", 10_000))
        self.claw_duty = self._slew(self.claw_duty, target, step)
        self.claw_servo.duty_ns(self.claw_duty)

