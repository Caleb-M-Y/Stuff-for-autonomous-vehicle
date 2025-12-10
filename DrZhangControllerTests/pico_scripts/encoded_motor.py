from base_motor import BaseMotor
from machine import Pin


class EncodedMotor(BaseMotor):
    def __init__(self, driver_ids: list | tuple, encoder_ids: list | tuple) -> None:
        super().__init__(*driver_ids)
        # Pin configuration
        self.enca_pin = Pin(encoder_ids[0], Pin.IN)
        self.encb_pin = Pin(encoder_ids[1], Pin.IN)
        self.enca_pin.irq(
            trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=self.update_counts_a
        )
        self.encb_pin.irq(
            trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=self.update_counts_b
        )
        # Variables
        self.enca_val = self.enca_pin.value()
        self.encb_val = self.encb_pin.value()
        self.encoder_counts = 0
        self.prev_counts = 0
        self.meas_ang_vel = 0.0
        self.meas_lin_vel = 0.0

    def update_counts_a(self, pin):
        self.enca_val = pin.value()
        if self.enca_val == 1:
            if self.encb_val == 0:  # a=1, b=0
                self.encoder_counts += 1
            else:  # a=1, b=1
                self.encoder_counts -= 1
        else:
            if self.encb_val == 0:  # a=0, b=0
                self.encoder_counts -= 1
            else:  # a=0, b=1
                self.encoder_counts += 1

    def update_counts_b(self, pin):
        self.encb_val = pin.value()
        if self.encb_val == 1:
            if self.enca_val == 0:  # b=1, a=0
                self.encoder_counts -= 1
            else:  # b=1, a=1
                self.encoder_counts += 1
        else:
            if self.enca_val == 0:  # b=0, a=0
                self.encoder_counts += 1
            else:  # b=0, a=1
                self.encoder_counts -= 1

    def reset_encoder_counts(self):
        self.encoder_counts = 0


# TEST
if __name__ == "__main__":  # Test only the encoder part
    from utime import sleep

    # SETUP
    # em = EncodedMotor(
    #     driver_ids=(16, 18, 17),
    #     encoder_ids=(20, 19),
    # )  # left motor, encoder's green and yellow on GP19 and GP20
    em = EncodedMotor(
        driver_ids=(15, 13, 14),
        encoder_ids=(10, 11),
    )  # right motor, encoder's green and yellow on GP10 and GP11

    # LOOP
    # Forward ramp up and down
    for i in range(100):
        em.forward((i + 1) / 100)
        print(f"f, dc: {i}%, enc_cnt: {em.encoder_counts}")
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        em.forward((i + 1) / 100)
        print(f"f, dc: {i}%, enc_cnt: {em.encoder_counts}")
        sleep(4 / 100)  # 4 seconds to ramp down
    # Backward ramp up and down
    for i in range(100):
        em.backward((i + 1) / 100)
        print(f"f, dc: {i}%, enc_cnt: {em.encoder_counts}")
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        em.backward((i + 1) / 100)
        print(f"f, dc: {i}%, enc_cnt: {em.encoder_counts}")
        sleep(4 / 100)  # 4 seconds to ramp down

    # Terminate
    em.stop()
    sleep(0.5)
    print("motor stopped.")
