"""
Manual arm calibration helper for Pico.
Upload with current_pico files and run on Pico REPL:
    import arm_calibration_test
"""

from utime import sleep
from ac2 import ArmController


def run():
    arm = ArmController(15, 13, 14)
    print("Neutral:", arm.shoulder_duty_a, arm.shoulder_duty_b, arm.claw_duty)
    sleep(1.0)

    # Lower shoulder
    for _ in range(120):
        arm.lower_claw(3000)
        sleep(0.02)
    print("After lower:", arm.shoulder_duty_a, arm.shoulder_duty_b)
    sleep(1.0)

    # Raise shoulder
    for _ in range(120):
        arm.lower_claw(-3000)
        sleep(0.02)
    print("After raise:", arm.shoulder_duty_a, arm.shoulder_duty_b)
    sleep(1.0)

    # Close claw
    for _ in range(120):
        arm.close_claw(3000)
        sleep(0.02)
    print("After close:", arm.claw_duty)
    sleep(1.0)

    # Open claw
    for _ in range(120):
        arm.close_claw(-3000)
        sleep(0.02)
    print("After open:", arm.claw_duty)


run()
