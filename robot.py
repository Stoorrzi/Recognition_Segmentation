from gpiozero import Motor
from time import sleep

motor = Motor(17, 18)

motor.forward(1)

while True:
    print(motor.is_active)
    sleep(1)
    # motor.backward()
    # print('backward')
    # sleep(3)