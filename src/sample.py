"""
sample.py
Displays a number of sample images from a given directory, with annotations.

Usage:
    python sample.py <annotation_type> <num_samples> <images_dir> <labels_dir>

    annotation_type (str): The type of annotation to be used. One of ["coco", "voc", "csv", "yolo"].
    num_samples (int): The number of sample images to display.
    images_dir (str): The directory where the images are stored.
    labels_dir (str): The directory where the labels are stored.

Example:
    python sample.py voc 10 dataset/voc/images dataset/voc/labels
"""

from imgann import Sample

def main(annotation_type: str, num_samples: int, images_dir: str, labels_dir: str):
    Sample.show_samples(
        data_path = images_dir,
        ann_path = labels_dir,
        num_of_samples = num_samples,
        ann_type = annotation_type,
        seed = 0,
        image_shape = [512, 512]
    )

if __name__ == "__main__":
    import sys;
    try:
        main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
    except Exception:
        import traceback
        traceback.print_exc()
        print(__doc__)
        sys.exit(1)