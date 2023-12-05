from gpiozero import Motor

motor = Motor(17, 18)


while True:
    motor.forward()
    sleep(1000)
    motor.backward()
    sleep(1000)