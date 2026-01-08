from ultralytics import YOLO

def main():
    # YOLOv8 classification model (NOT detection)
    model = YOLO("yolov8n-cls.pt")  # or "yolov8s-cls.pt" if you want larger

    model.train(
        data="dataset/images",  # has train/val/test subfolders with class folders
        epochs=40,
        imgsz=224,
        batch=32,
        project="runs",
        name="wound_cls",
    )

if __name__ == "__main__":
    main()
