import cv2
import socket
import pickle
import numpy as np
from ultralytics import YOLO


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_ip = "192.168.178.142"
server_port = 6666

s.bind((server_ip, server_port))

while True:
    x = s.recvfrom(1000000)
    clientip = x[1][0]
    data = x[0]

    data = pickle.loads(data)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    model = YOLO("yolov8n.pt")

    # Run inference on the source
    results = model(img, stream=True)  # list of Results objects

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

    cv2.imshow("Webcam", img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
