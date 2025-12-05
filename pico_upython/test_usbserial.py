"""
Upload this script to the pico board, then rename it to main.py.
"""
import sys
import select

# SETUP
poller = select.poll()
poller.register(sys.stdin, select.POLLIN)
event = poller.poll()

# LOOP
while True:
    # read data from serial
    for msg, _ in event:
        buffer = msg.readline().rstrip()
        print(buffer)
        # if len(buffer) == 17: # Hello from RPi: 0
        #     print(f"Pico hears: {buffer}")
            # sys.stdout.write(buffer)