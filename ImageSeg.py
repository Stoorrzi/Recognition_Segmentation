import cv2
from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import time

start_time = time.time()
model = YOLO('yolov8s-seg.pt')

video_path = "Vid_3.mov"
cap = cv2.VideoCapture(video_path)
target_name = ""
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()     

        # Center with bbox
        if (len(results[0].boxes.xyxy) > 0) :

            target_name = results[0].names[int(results[0].boxes.cls[0])]

            


            bbox=results[0].boxes.xyxy.tolist()[0]
            print(bbox)

            p1 = (bbox[2] + bbox[0])/2
            p2 = (bbox[3] + bbox[1])/2
    
            cv2.circle(annotated_frame, (int(p1), int(p2)), 5, (0, 0, 255), 10)
        
        # Center with mask
        if (len(results[0]) > 0):
            arr = results[0].masks.xy[0]
            arr1,arr2=np.split(arr,2,axis=1)

            arr1 = list(arr1.reshape(-1))
            arr2 = list(arr2.reshape(-1))
            center = sum(arr1)/len(arr1), sum(arr2)/len(arr2)
            cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), 10)
        
        # Center with Mask and bbox compaired
        if (len(results[0].boxes.xyxy) > 0 and len(results[0]) > 0 ):
            bbox=results[0].boxes.xyxy.tolist()[0]
            #print(bbox)

            p1 = (bbox[2] + bbox[0])/2
            p2 = (bbox[3] + bbox[1])/2


            arr = results[0].masks.xy[0]
            arr1,arr2=np.split(arr,2,axis=1)

            arr1 = list(arr1.reshape(-1))
            arr2 = list(arr2.reshape(-1))
            center = sum(arr1)/len(arr1), sum(arr2)/len(arr2)

            x1 = (center[0] + p1) / 2
            y1 = (center[1] + p2) / 2
            cv2.circle(annotated_frame, (int(x1), int(y1)), 5, (0, 255, 0), 10)

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))










# model = FastSAM('./weights/FastSAM-s.pt')  # or FastSAM-x.pt

#         # Run inference on an image
#         everything_results = model(annotated_frame, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

#         # Prepare a Prompt Process object
#         prompt_process = FastSAMPrompt(annotated_frame, everything_results, device='cpu')

#         # Everything prompt
#         ann = prompt_process.everything_prompt()

#         # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
#         ann = prompt_process.box_prompt(bbox=results[0].boxes.xyxy.tolist()[0])
        
#         print(prompt_process.results[0].masks[0].xy[0])

#         if (len(prompt_process.results) > 0):
#             arr = prompt_process.results[0].masks[0].xy[0]
#             arr1,arr2=np.split(arr,2,axis=1)

#             arr1 = list(arr1.reshape(-1))
#             arr2 = list(arr2.reshape(-1))
#             center = sum(arr1)/len(arr1), sum(arr2)/len(arr2)
#             cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 5, (255, 255, 0), 10)