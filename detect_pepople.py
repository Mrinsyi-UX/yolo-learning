from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained object detection model

results = model.predict(
    source="bus.jpg",  # image source
    classes=[0],  # only detect people (class 0)
    conf=0.25,  # confidence threshold
    device="cpu",  # use cpu for inference
    save=True,  # save the results to a file
    name="people_detected",  # name of the output file
)

result = results[0]  # get the first result from the list of results
print(f"People detected: {len(result.boxes)}")
print(f"Result saved to: {result.save_dir}")  # print the directory where the results were saved