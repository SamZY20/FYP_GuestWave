from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

model.train(data="train\config.yaml", epochs=3,
            save_period=1)  # train the model
