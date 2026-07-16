"""Fine-tune a YOLO11 model on a custom dataset.

Replaces the 2020 Keras/TF1 training pipeline (frozen-then-unfrozen stages,
anchor k-means, manual data generators). Ultralytics handles augmentation,
anchor-free detection, LR scheduling, early stopping, and checkpointing.

Dataset must be in YOLO format (see convert_annotations.py to migrate the old
annotation file) with a data.yaml describing train/val paths and class names.
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="model_data/data.yaml",
                        help="Path to the dataset data.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base weights to fine-tune (n/s/m/l/x variants)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                        help="cuda device id, 'mps' (Apple Silicon), or 'cpu'; auto if omitted")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=10,
    )
    print(f"Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
