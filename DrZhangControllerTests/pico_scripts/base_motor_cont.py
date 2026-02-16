from machine import Pin, PWM


class BaseMotor:
    """DC motor driver: in1_id, in2_id = direction pins; pwm_id = speed (0--1 maps to duty)."""

    def __init__(self, pwm_id, in1_id, in2_id, ) -> None:
        self.pwm_pin = PWM(Pin(pwm_id))
        self.pwm_pin.freq(2000)
        self.in1_pin = Pin(in1_id, Pin.OUT)
        self.in2_pin = Pin(in2_id, Pin.OUT)
        self.disable()  # motors off until first command

    def stop(self):
        self.pwm_pin.duty_u16(0)

    def forward(self, speed=0.0):
        """Drive motor forward (robot forward). speed in [0, 1], clamped."""
        speed = max(0.0, min(1.0, float(speed)))
        self.in1_pin.off()
        self.in2_pin.on()
        self.pwm_pin.duty_u16(int(65535 * speed))

    def backward(self, speed=0.0):
        """Drive motor backward (robot backward). speed in [0, 1], clamped."""
        speed = max(0.0, min(1.0, float(speed)))
        self.in1_pin.on()
        self.in2_pin.off()
        self.pwm_pin.duty_u16(int(65535 * speed))
        
    def disable(self):
        self.in1_pin.off()
        self.in2_pin.off()


# TEST
if __name__ == "__main__":
    from utime import sleep

    # SETUP
    m = BaseMotor(pwm_id=21, in1_id=20, in2_id=19)  # left motor
    #m = BaseMotor(pwm_id=16, in1_id=17, in2_id=18)  # right motor

    # LOOP
    # Forwardly ramp up and down
    for i in range(100):
        m.forward((i + 1) / 100)
        print(f"f, dc: {i}%")
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        m.forward((i + 1) / 100)
        print(f"f, dc: {i}%")
        sleep(4 / 100)  # 4 seconds to ramp down
    # Backwardly ramp up and down
    for i in range(100):
        m.backward((i + 1) / 100)
        print(f"b, dc: {i}%")
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        m.backward((i + 1) / 100)
        print(f"b, dc: {i}%")
        sleep(4 / 100)  # 4 seconds to ramp down

    # Terminate
    m.disable()
    print("motor stopped.")
