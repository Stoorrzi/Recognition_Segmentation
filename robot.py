from gpiozero import Motor
from time import sleep

motor = Motor(17, 18)


while True:
    motor.forward()
    sleep(3)
    motor.backward()
    sleep(3)