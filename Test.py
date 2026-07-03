from roboflow import Roboflow
from ultralytics import YOLO
import multiprocessing

def main():
    rf = Roboflow(api_key="BtNwCe4uvuGwoWjvQnZS")
    project = rf.workspace().project("cricket_ball_detection-3ukl0")
    dataset = project.version(1).download("yolov8")
    model = YOLO("yolov8n.pt")
    dataset_path = dataset.location

    model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=100,
        imgsz=640,
    )

    results = model.predict(source=r"C:\Users\vivek\Downloads\ms-dhoni-practice-1559564708.jpg", show=False,save=True)
    print(results)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
