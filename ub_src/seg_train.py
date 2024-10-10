from ultralytics import YOLO

model = YOLO("yolo11l-seg.pt")

results = model.train(data="ub-coco8-seg-waterflow.yaml", epochs=300, imgsz=320)