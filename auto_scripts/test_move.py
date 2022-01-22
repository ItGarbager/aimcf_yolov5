from time import sleep

import pynput

mouse = pynput.mouse.Controller()

for i in range(100):
    mouse.position = (19.2 * i, 100)
    sleep(0.5)
