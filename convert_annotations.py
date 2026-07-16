"""Convert the 2020 annotation format to modern YOLO format.

Old format (one line per image):
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,1

New format: one .txt per image with normalized "class cx cy w h" rows, plus a
data.yaml. Images are split into train/val and referenced in place.
"""

import argparse
import random
from pathlib import Path

from PIL import Image


def convert_line(line, labels_dir):
    parts = line.split()
    image_path = Path(parts[0])
    if not image_path.exists():
        print(f"skip (missing image): {image_path}")
        return None

    with Image.open(image_path) as img:
        iw, ih = img.size

    rows = []
    for box in parts[1:]:
        x1, y1, x2, y2, cls = map(int, box.split(","))
        cx = (x1 + x2) / 2 / iw
        cy = (y1 + y2) / 2 / ih
        w = (x2 - x1) / iw
        h = (y2 - y1) / ih
        rows.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path = labels_dir / (image_path.stem + ".txt")
    label_path.write_text("\n".join(rows) + "\n")
    return str(image_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True,
                        help="Old-format annotation file (e.g. model_data/new_data_1750_2.txt)")
    parser.add_argument("--classes", default="model_data/car_class_2.txt",
                        help="Class names file, one name per line")
    parser.add_argument("--out", default="dataset", help="Output dataset directory")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.out)
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    lines = [ln.strip() for ln in Path(args.annotations).read_text().splitlines() if ln.strip()]
    image_paths = [p for ln in lines if (p := convert_line(ln, labels_dir))]

    random.seed(10101)
    random.shuffle(image_paths)
    num_val = max(1, int(len(image_paths) * args.val_split))
    (out / "val.txt").write_text("\n".join(image_paths[:num_val]) + "\n")
    (out / "train.txt").write_text("\n".join(image_paths[num_val:]) + "\n")

    class_names = [c.strip() for c in Path(args.classes).read_text().splitlines() if c.strip()]
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n"
    )
    print(f"Converted {len(image_paths)} images "
          f"({len(image_paths) - num_val} train / {num_val} val) -> {out}/data.yaml")
    print("Note: YOLO expects label .txt files next to images or in a parallel "
          "'labels' directory; adjust paths in data.yaml if your images live elsewhere.")


if __name__ == "__main__":
    main()
