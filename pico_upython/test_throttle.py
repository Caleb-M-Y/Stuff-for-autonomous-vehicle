from machine import Pin, PWM, reset
from time import sleep

# SAFETY CHECK
is_goggled = "n"
while is_goggled is not "y":
    print("Please put on eye protections!\n")
    is_goggled = input("Are goggles on your face? (y/N)")
is_contacted = "y"
while is_contacted is not "n":
    print("Please lift BearCar up and remove everything that is making the contact\n")
    is_contacted = input("Is anything touching any wheel of BearCar? (Y/n)")
print("Hold tight! Unleash the beast!!!")
for i in reversed(range(3)):
    print(i+1)
    sleep(1)

# SETUP
throttle = PWM(Pin(16))
throttle.freq(50)
# TODO: config led for a naive HRI

# LOOP
print("\nFORWARD: ramp up\n")
for dc in range(1500000, 2000000, 10000): # forward up
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
print("\nFORWARD: ramp down\n")
for dc in reversed(range(1500000, 2000000, 10000)): # forward down
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
print("\nREVERSE: ramp up\n")
for dc in reversed(range(1000000, 1500000, 10000)): # reverse up
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
print("\nREVERSE: ramp down\n")
for dc in range(1000000, 1500000, 10000): # reverse down
    throttle.duty_ns(dc)
    print(dc)
    sleep(0.2)
throttle.duty_ns(1500000)
print("STOP")
sleep(1)
throttle.deinit()
reset()

