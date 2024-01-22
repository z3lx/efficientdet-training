"""
convert.py
Converts a dataset from YOLO annotation format to VOC annotation format.

Usage:
    python convert.py <input_images_dir> <input_annotations_dir> <output_images_dir> <output_annotations_dir>

    input_images_dir (str): The directory where the input images are stored.
    input_annotations_dir (str): The directory where the input annotations are stored.
    output_images_dir (str): The directory where the output images are stored.
    output_annotations_dir (str): The directory where the output annotations are stored.

Example:
    python convert.py dataset/yolo/images dataset/yolo/labels dataset/voc/images dataset/voc/labels
"""

import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from imgann import Convertor

def main(input_images_dir: str, input_annotations_dir: str, output_images_dir: str, output_annotations_dir: str):
    # Cleanup
    shutil.rmtree(output_images_dir, ignore_errors = True)
    shutil.rmtree(output_annotations_dir, ignore_errors = True)
    os.makedirs(output_images_dir, exist_ok = True)
    os.makedirs(output_annotations_dir, exist_ok = True)

    # Checks
    if not os.path.isdir(input_images_dir):
        raise Exception(f"Directory {input_images_dir} does not exist.")
    if not os.path.isdir(input_annotations_dir):
        raise Exception(f"Directory {input_annotations_dir} does not exist.")

    # Convert images
    for filename in os.listdir(input_images_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(input_images_dir, filename))
            img = img.convert("RGB")
            img.save(
                os.path.join(output_images_dir, filename.split(".")[0] + ".jpeg"),
                "JPEG",
                quality = 100
            )
        else:
            if filename.filename.endswith(".jpeg"):
                shutil.copy(os.path.join(input_images_dir, filename), output_images_dir)

    # Convert labels
    Convertor.yolo2voc(
        dataset_dir = os.path.join(input_images_dir, ".."),
        yolo_ann_dir = os.path.join(input_annotations_dir, ".."),
        save_dir = output_annotations_dir
    )

    # Fix VOC annotations
    for filename in os.listdir(output_annotations_dir):
        if not (filename.endswith(".xml")):
            continue
        filepath = os.path.join(output_annotations_dir, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        filename_tag = root.find("filename")
        filename_tag.text = filename_tag.text.replace(".png", ".jpeg")
        root.find("path").text = filename_tag.text
        for name_elem in root.iter("name"):
            try:
                value = float(name_elem.text)
                name_elem.text = str(int(value))
            except ValueError:
                pass

        tree.write(filepath)

if __name__ == "__main__":
    import sys;
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except Exception:
        import traceback
        traceback.print_exc()
        print(__doc__)
        sys.exit(1)