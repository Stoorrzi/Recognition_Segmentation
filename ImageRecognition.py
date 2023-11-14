from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

     # Define path to the image file
source = 'Vid.mov'

     # Run inference on the source
results = model(source, stream=True)  # list of Results objects

for result in results:
    boxes = result.boxes

#     for box in boxes:
#         # bounding box
#         x1, y1, x2, y2 = box.xyxy[0]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

 #         # put box in cam
#         cv2.rectangle(source, (x1, y1), (x2, y2), (255, 0, 255), 3)

 #         # object details
#         org = [x1, y1]
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         fontScale = 1
#         color = (255, 0, 0)
#         thickness = 2
    if (len(boxes.xyxy) > 0) :
        bbox=boxes.xyxy.tolist()[0]
        print(bbox)
   
    

#     cv2.imshow("Webcam", source)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()