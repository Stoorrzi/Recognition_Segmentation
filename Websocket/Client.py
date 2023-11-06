import cv2
import socket
import pickle
import os
import numpy as np
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)

server_ip = "192.168.178.142"
server_port = 6666

img= picam2.capture_array()

while True:
    img= picam2.capture_array()

    cv2.imshow("Img Client", img)

    ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])

    x_as_byte = pickle.dumps(buffer)

    s.sendto((x_as_byte),(server_ip, server_port))

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
