from machine import Pin, reset
from time import sleep


# SETUP
# Config the common anode LED
g = Pin(18, Pin.OUT)
g.value(1)  # 0 = on; 1 = off
r = Pin(19, Pin.OUT)
r.value(1)  # 0 = on; 1 = off
b = Pin(20, Pin.OUT)
b.value(1)  # 0 = on; 1 = off
leds = (g, r, b)

# LOOP
for i in range(len(leds)):
    for _ in range(4):
        leds[i].toggle()
        sleep(0.5)

print("Yellow")
leds[0].value(0)
leds[1].value(0)
leds[2].value(1)
sleep(1)
print("Cyan")
leds[0].value(0)
leds[1].value(1)
leds[2].value(0)
sleep(1)
print("Purple")
leds[0].value(1)
leds[1].value(0)
leds[2].value(0)
sleep(1)
# reset()
reset()
