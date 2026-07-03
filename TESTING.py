from ultralytics import YOLO

model = YOLO(r"C:\Users\vivek\Documents\Cricket Ball Analyzer\runs\detect\train26\weights\best.pt", task='detect')
print(model)