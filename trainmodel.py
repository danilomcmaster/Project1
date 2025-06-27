### TRAINING MODEL FOR CARD RECOGNITION - DO NOT RUN ###
### See "https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/4" for model and training data source ###

from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
model.train(data="Project1/PlayingCards/data.yaml", epochs=10, imgsz=416)

model = YOLO("runs/detect/train/weights/best.pt")
results = model("PlayingCards/test/images/001771721_jpg.rf.687025a63ae5c9e58f2454ab1e41eaa9.jpg")
results[0].show()