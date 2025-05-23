""" Conversion from RGB to Grayscale. """
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def convert_to_bw() -> None:
    if len(sys.argv) != 3:
        raise ValueError("Specify input and output paths!")
    path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    img_paths = list(path.glob("*"))

    for img_path in tqdm(img_paths, total=len(img_paths)):
        img = Image.open(img_path)
        bw_img = img.convert("L")
        bw_img.save(out / Path(f"{img_path.stem}_bw{img_path.suffix}"))


if __name__ == "__main__":
    convert_to_bw()
