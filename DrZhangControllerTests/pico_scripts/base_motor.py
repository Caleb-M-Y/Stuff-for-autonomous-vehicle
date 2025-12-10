from machine import Pin, PWM


class BaseMotor:
    def __init__(self, pwm_id, in1_id, in2_id) -> None:
        self.pwm_pin = PWM(Pin(pwm_id))
        self.pwm_pin.freq(2000)
        self.in1_pin = Pin(in1_id, Pin.OUT)
        self.in2_pin = Pin(in2_id, Pin.OUT)

    def stop(self):
        self.pwm_pin.duty_u16(0)

    def forward(self, speed=0.0):  # map 0~65535 to 0~1
        assert 0 <= speed <= 1  # make sure speed in range [0, 1]
        self.in1_pin.off()
        self.in2_pin.on()
        self.pwm_pin.duty_u16(int(65535 * speed))

    def backward(self, speed=0.0):  # map 0~65535 to 0~1
        assert 0 <= speed <= 1  # make sure speed in range [0, 1]
        self.in1_pin.on()
        self.in2_pin.off()
        self.pwm_pin.duty_u16(int(65535 * speed))


# TEST
if __name__ == "__main__":
    from utime import sleep

    # SETUP
    m = BaseMotor(pwm_id=15, in1_id=13, in2_id=14)  # right motor
    # m = BaseMotor(pwm_id=16, in1_id=18, in2_id=17)  # left motor

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
    m.stop()
    print("motor stopped.")
