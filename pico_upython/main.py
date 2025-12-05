"""
Upload this script to Pico as 'main.py'.
"""

import sys
import select
from time import sleep
from machine import Pin, PWM, WDT, freq, reset

# SETUP
# Constants
NEUTRAL_DUTY = 1_500_000  # throttle zero and servo center
# Overclock
freq(200_000_000)  # Pico 2 original: 150_000_000
# Config LED pins
r_led = Pin(19, Pin.OUT)
# r_led.valeu(1)
g_led = Pin(18, Pin.OUT)
# g_led.valeu(1)
b_led = Pin(20, Pin.OUT)
# b_led.valeu(1)
# Config drivetrain pins
steering = PWM(Pin(17))
steering.freq(50)  # ESC only works with 50Hz PWM
steering.duty_ns(NEUTRAL_DUTY)
throttle = PWM(Pin(16))
throttle.freq(50)
throttle.duty_ns(NEUTRAL_DUTY)
# Config USB BUS
listener = select.poll()
listener.register(sys.stdin, select.POLLIN)
event = listener.poll()
# print("Pico listening...")  # uncomment to debug
# Config watchdog timer
wdt = WDT(timeout=500)  # ms
# Variables
mode = 'e'  # default mode not able to move drivetrain

# LOOP
try:
    while True:
        dutycycle_st = NEUTRAL_DUTY
        dutycycle_th = NEUTRAL_DUTY
        for msg, _ in event:
            if msg:
                wdt.feed()
                msg_line = msg.readline().rstrip()
                # print(f"Pico heard: {msg_line}")  # debug
                msg_parts = msg_line.split(",")
                # print(f"Pico heard: {msg_parts}")  # debug
                if len(msg_parts) == 3:
                    # print("Pico heard 2 parts")  # debug
                    try:
                        # Process driving actions
                        dutycycle_st = int(msg_parts[1])
                        dutycycle_th = int(msg_parts[2])
                        # Process LED actions
                        mode = msg_parts[0]
                        if mode is 'n':  # normal
                            r_led.value(1)
                            g_led.value(0)
                            b_led.value(1)
                        elif mode is 'r':  # recording
                            r_led.value(1)
                            g_led.value(1)
                            b_led.value(0)
                        elif mode is 'a':  # autopilot
                            r_led.value(0)
                            g_led.value(1)
                            b_led.value(0)
                        elif mode is 'p':  # pause
                            r_led.value(0)
                            g_led.value(0)
                            b_led.value(1)
                            dutycycle_st = NEUTRAL_DUTY
                            dutycycle_th = NEUTRAL_DUTY
                        else:  # error
                            r_led.value(0)
                            g_led.value(1)
                            b_led.value(1)
                            dutycycle_st = NEUTRAL_DUTY
                            dutycycle_th = NEUTRAL_DUTY
                        # print(f"Pico received command: {mode}, {dutycycle_st}, {dutycycle_th}") # debug
                        steering.duty_ns(dutycycle_st)
                        throttle.duty_ns(dutycycle_th)
                    except ValueError:
                        # print("ValueError!")  # debug
                        reset()
            else:
                throttle.duty_ns(NEUTRAL_DUTY)
                steering.duty_ns(NEUTRAL_DUTY)
                reset()
except Exception as e:
    # print('Pico reset')  # debug
    reset()
finally:
    # print('Pico reset')  # debug
    reset()
