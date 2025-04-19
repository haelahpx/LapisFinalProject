import torch
from ultralytics import YOLO

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Training will be done in", torch.cuda.get_device_name(0))
    else:
        print("Training will be done in CPU")

    # model = YOLO("yolov8n.pt")
    model = YOLO("last.pt")
    # model = YOLO("best.pt")

    model.train(
        # data="datasets/_HD_RGB_CarAndPlateDetection/data.yaml",
        data="datasets/_HD_GRY_CarAndPlateDetection/data.yaml",
        epochs=100,
        imgsz=720,
        batch=10,
        project="vec-YOLO8",
        name="vec-YOLO8-GRY-v4",
        exist_ok=True,
        amp=True  # Enable automatic mixed precision (Tensor Cores)
    )

if __name__ == "__main__":
    main()
