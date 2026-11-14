"""
OPTIONAL - Fine-tune YOLOv8 on your own (e.g. Indian traffic) dataset.   (Owner: Member 1)
This is the easiest way to add NOVELTY to the project / research paper:
the COCO model does not know auto-rickshaws or e-rickshaws.

1. Label images in YOLO format (tools: Roboflow, CVAT, LabelImg) with classes such as
   car, motorcycle, bus, truck, auto-rickshaw, bicycle
2. Create a data.yaml:
       path: datasets/indian_traffic
       train: images/train
       val: images/val
       names: {0: car, 1: motorcycle, 2: bus, 3: truck, 4: auto-rickshaw, 5: bicycle}
3. Train:   python train.py --data datasets/indian_traffic/data.yaml --epochs 50
4. Use it:  python main.py --source traffic.mp4 --weights runs/detect/train/weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz,
                        batch=a.batch, device=a.device)


if __name__ == "__main__":
    main()
