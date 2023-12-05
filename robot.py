from gpiozero import Motor
from time import sleep

motor = Motor(17, 18)


while True:
    motor.forward(1)
    # print('forward')
    # sleep(3)
    # motor.backward()
    # print('backward')
    # sleep(3)