from ultralytics import YOLO
import cv2

# Load your trained model (update the path to where best.pt is located)
model = YOLO(r"C:\Users\vivek\Documents\Cricket Ball Analyzer\runs\detect\train26\weights\best.pt")

# Path to the image you want to test
image_path = r"C:\Users\vivek\Downloads\BALL3.jpg"

# Run inference
results = model(image_path, show=True)  # show=True opens a window with detections

# Optionally save the output image with bounding boxes
for result in results:
    annotated_frame = result.plot()
    cv2.imwrite("output.jpg", annotated_frame)

print("Detection complete. Output saved as output.jpg")