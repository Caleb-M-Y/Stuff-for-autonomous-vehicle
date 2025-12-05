"""
This script will test communication between Pico and RPi.
Please uncomment line 45 in the 'main.py' on Pico.
"""

from serial import Serial
from time import sleep

# SAFETY CHECK
is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
while is_lifted != 'n':
    print("Please lift BearCart up and remove everything that is making the contact")
    is_lifted = input("Is something contacting any wheel of BearCart? (Y/n)")
print("Hold tight! You are about to unleash the beast!")
print("Verify light order: green -> blue -> purple -> green -> yellow -> red")
for i in reversed(range(3)):
    print(i+1)
    sleep(1)

# SETUP
messenger = Serial(port='/dev/ttyACM0', baudrate=115200)
print(messenger.name)
# Constants
modes = ('n', 'r', 'a', 'n')
dutycycles = (1_350_000, 1_400_000, 1_650_000, 1_600_000)
# Variables
st_dc = 1_500_000
th_dc = 1_500_000

# LOOP
for i in range(100):
    # messenger.write(f"Hello from RPi: {i}\n".encode('utf-8'))  # simple test
    if i < 80:
        st_dc = dutycycles[int(i/20)]
        th_dc = dutycycles[int(i/20)]
        msg = f"{modes[int(i/20)]}, {st_dc}, {th_dc}\n".encode('utf-8')
    else:
        msg = f"p, {dutycycles[-1]}, {dutycycles[-1]}\n".encode('utf-8')
    messenger.write(msg)
    # print(f"[RPi Transmit] {msg}")  # debug
    # Listen to Pico's response
    if messenger.inWaiting() > 0:
        reply = messenger.readline()
        reply = reply.decode('utf-8', 'ignore')
        print(f"[Pico Response] {reply}")
    sleep(0.1)
messenger.write(f"e, {dutycycles[-1]}, {dutycycles[-1]}\n".encode('utf-8'))
# print(f"[RPi Transmit] e, {dutycycles[-1]}, {dutycycles[-1]}\n")  # debug
if messenger.inWaiting() > 0:
    reply = messenger.readline()
    reply = reply.decode('utf-8', 'ignore')
    print(f"[Pico Response] {reply}")  
sleep(0.5)
messenger.close()
