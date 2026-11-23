from ultralytics import YOLO

#load a pretrained onject detection model, basicly a library that can be used to detect objects in images and video
#this is premade to help early developers to get started with object detection without having to train their own models from scratch
model = YOLO("yolov8n.pt")

#Detect objects in an image
#There are many parameters that can be used to customize the detection process, such as the source of the image, the device to use for inference, and whether to save the results to a file or not.
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",  #image source
    device="cpu",  #use cpu for inference
    save=True,  #save the results to a file
)

#print the results of the detection process, which includes the detected objects, their confidence scores, and their bounding boxes.
result = results[0]  #get the first result from the list of results
print(f"objects detected: {len(result.boxes)}")
print(f"Result saved to : {result.save_dir}")  #print the directory where the results were saved