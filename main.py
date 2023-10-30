import ultralytics
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from IPython.display import display, Image
from PIL import Image as img
import cv2
import numpy as np
import matplotlib.pyplot as plt
ultralytics.checks()

def predict_image():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Define path to the image file
    source = './Images/IMG_2.jpg'

    # Run inference on the source
    results = model(source)  # list of Results objects

    for result in results:
        boxes = result.boxes

        bbox=boxes.xyxy.tolist()[0]
        print (bbox)

        segment_image(bbox)

    # for r in results:
    #     im_array = r.plot()  # plot a BGR numpy array of predictions
    #     im = img.fromarray(im_array[..., ::-1])  # RGB PIL image
    #     im.show()  # show image
    #     im.save('results.jpg')  # save image

def segment_image(bbox):
    sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    image = cv2.cvtColor(cv2.imread('./Images/IMG_2.jpg'), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_box = np.array(bbox)

    masks, a, b = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    print (b)

    # pred = predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_box[None, :],
    #     multimask_output=False,
    #     return_logits=True,
    # )

    
 
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.savefig("res.jpg")
    plt.show()
    

def show_mask(mask, ax, random_color=False):
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    print(mask.shape)
    #print (h, w)
    #print ("jalsdfjlaksjdlkas")
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #print(mask_image)
    ax.imshow(mask_image)
    
    

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    print (pos_points)
    print (neg_points)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))



if __name__ == "__main__":
    predict_image()