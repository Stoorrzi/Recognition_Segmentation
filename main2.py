from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from ultralytics import YOLO

def predict_image():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Define path to the image file
    source = './Images/IMG_1.jpg'

    # Run inference on the source
    results = model(source)  # list of Results objects

    for result in results:
        boxes = result.boxes

        bbox=boxes.xyxy.tolist()[0]
        print (bbox)
        return bbox


# Define an inference source
source = './Images/IMG_1.jpg'

# Create a FastSAM model
model = FastSAM('./weights/FastSAM-x.pt')  # or FastSAM-x.pt

# Run inference on an image
everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# Prepare a Prompt Process object
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

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
h, w, c = (cv.imread(source)).shape
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

# plt.plot(center[0], center[1], marker='o')
# print(str(center[0]) + ", " + str(center[1]))
# plt.annotate(
#    "center",
#    xy=center, xytext=(-20, 20),
#    textcoords='offset points', ha='right', va='bottom',
#    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
# plt.show()

# from scipy import stats

# # print(type(arr1))

# slope, intercept, r, p, std_err = stats.linregress(arr1, arr2)

# def myfunc(x):
#   return slope * x + intercept

# mymodel = list(map(myfunc, arr1))

# plt.scatter(arr1, arr2)
# plt.plot(arr1, mymodel)
# plt.show() 
