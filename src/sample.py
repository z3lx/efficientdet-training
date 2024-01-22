from imgann import Sample
import os

images_dir = os.path.join("voc", "images")
labels_dir = os.path.join("voc", "labels")

venv_dir = os.environ["VIRTUAL_ENV"]
os.chdir(os.path.join(venv_dir, "..", "dataset"))

Sample.show_samples(
    data_path = images_dir,
    ann_path = labels_dir,
    num_of_samples = 5,
    ann_type = "voc",
    seed = 0,
    image_shape = [512, 512]
)