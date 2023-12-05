from gpiozero import LED
from time import sleep

forward = LED(17)
backward = LED(18)

while True:
    forward.on()
    backward.off()
    sleep(1000)
    forward.off()
    backward.on()
    sleep(1000)