"""
preprocess.py
Preprocesses a dataset by adding JPEG artifacts, scaling, and noise.

Usage:
    python preprocess.py <input_images_dir> <output_images_dir> <quality> <scale> <noise>

    input_images_dir (str): The directory where the input images are stored.
    output_images_dir (str): The directory where the output images are stored.
    quality (int): The JPEG quality of the artifacts.
    scale (float): The scale of the scaling.
    noise (int): The intensity of the noise.

Example:
    python preprocess.py dataset/voc/images dataset/voc/images_preprocessed 50 0.75 10
"""

import os
from PIL import Image
import numpy as np

def add_jpeg_artifacts(image: Image, quality: int) -> Image:
    from io import BytesIO
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format = "JPEG", quality = quality)
    return Image.open(buffer)

def add_colored_noise(image: Image, intensity: int) -> Image:
    intensity = intensity / 100
    np_img = np.array(image)
    noise = np.random.randint(0, 256, np_img.shape).astype(np.uint8)
    return Image.fromarray((np_img * (1 - intensity) + noise * intensity).astype(np.uint8))

def scale_image(image: Image, resolution: tuple) -> Image:
    return image.resize(resolution, Image.BICUBIC)

def main(input_images_dir: str, output_images_dir: str, quality: int, scale: float, noise: int):
    for filename in os.listdir(input_images_dir):
        if os.path.isdir(os.path.join(input_images_dir, filename)):
            continue
        ext = filename.split(".")[-1]
        if ext not in ["png", "jpg", "jpeg"]:
            continue
        img = Image.open(os.path.join(input_images_dir, filename))
        resolution = (img.width, img.height)
        img = add_jpeg_artifacts(img, quality)
        img = scale_image(img, (int(resolution[0] * scale), int(resolution[1] * scale)))
        img = add_colored_noise(img, noise)
        img = scale_image(img, resolution)
        img.save(
            os.path.join(output_images_dir, filename),
            ext.upper(),
            quality = 100
        )

if __name__ == "__main__":
    import sys;
    try:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    except Exception:
        import traceback
        traceback.print_exc()
        print(__doc__)
        sys.exit(1)