import cv2
import socket
import pickle
import numpy as np
from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_ip = "192.168.178.142"
server_port = 6666

s.bind((server_ip, server_port))

def predict_image(img):
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Run inference on the source
    results = model(img, stream=True)  # list of Results objects

    for result in results:
        boxes = result.boxes
        x1, y1, x2, y2 = boxes.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

        # put box in cam
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        bbox=boxes.xyxy.tolist()[0]
        print (bbox)
        return bbox

while True:
    x = s.recvfrom(1000000)
    clientip = x[1][0]
    data = x[0]

    data = pickle.loads(data)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)



    # Create a FastSAM model
    model = FastSAM('./weights/FastSAM-x.pt')  # or FastSAM-x.pt

    # Run inference on an image
    everything_results = model(img, stream=True, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    # Prepare a Prompt Process object
    prompt_process = FastSAMPrompt(img, everything_results, device='cpu')

    # Everything prompt
    ann = prompt_process.everything_prompt()

    # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    bbox = predict_image()
    ann = prompt_process.box_prompt(bbox=bbox)
    arr = prompt_process.results[0].masks.xy[0]
    prompt_process.plot(annotations=ann, output='./')

    arr1,arr2=np.split(arr,2,axis=1)

    arr1 = list(arr1.reshape(-1))
    arr2 = list(arr2.reshape(-1))


    center = sum(arr1)/len(arr1), sum(arr2)/len(arr2)
    h, w, c = (cv2.imread(img)).shape
    centerOfImg =  w/2, h/2,
    print (center, centerOfImg)

    if (center[0] > centerOfImg[0]):
        print("Nach oben fahren y+")
    if (center[0] < centerOfImg[0]):
        print("Nach unten fahren y-")
    if (center[1] > centerOfImg[1]):
        print("Nach links fahren x-")
    if (center[1] < centerOfImg[1]):
        print("Nach rechts fahren x+")
    cv2.imshow("Webcam", img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()

