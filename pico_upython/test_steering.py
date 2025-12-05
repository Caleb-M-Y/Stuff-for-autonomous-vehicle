from machine import Pin, PWM, reset
from time import sleep

# SAFETY CHECK
is_contacted = "y"
while is_contacted is not "n":
    print("Please lift BearCar up and remove everything that is making the contact\n")
    is_contacted = input("Is anything touching any wheel of BearCar? (Y/n)")
print("BearCar is about to turn!!!")

# SETUP
servo = PWM(Pin(17))
servo.freq(50)

# LOOP
print("<<<--- <<-- <- W -> -->> --->>>\n")
for i in range(1000000, 2000000, 10000):
    servo.duty_ns(i)
    print(i)
    sleep(0.2)
print("--->>> -->> -> W <- <<-- <<<---- \n")
for i in reversed(range(1000000, 2000000, 10000)):
    servo.duty_ns(i)
    print(i)
    sleep(0.2)
servo.duty_ns(1500000)
sleep(0.5)
servo.duty_ns(1800000)
sleep(0.5)
servo.duty_ns(1500000)
sleep(0.5)
servo.duty_ns(1200000)
sleep(0.5)
servo.duty_ns(1500000)
sleep(0.5)
servo.deinit()
reset()

