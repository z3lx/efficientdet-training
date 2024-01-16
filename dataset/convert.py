from imgann import Convertor
from PIL import Image
import os
import shutil

venv_dir = os.environ['VIRTUAL_ENV']
os.chdir(venv_dir + '/../dataset')

images_dir = './yolo/images'
labels_dir = './yolo/labels'
images_output_dir = './voc/images'
labels_output_dir = './voc/labels'

for filename in os.listdir(images_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = Image.open(os.path.join(images_dir, filename))
        img = img.convert('RGB')
        img.save(os.path.join(images_output_dir, filename.split('.')[0] + '.jpeg'), 'JPEG', quality=100)
    else:
        if filename.filename.endswith(".jpeg"):
            shutil.copy(os.path.join(images_dir, filename), images_output_dir)

Convertor.yolo2voc(
    dataset_dir=os.path.join(images_dir, '..'),
    yolo_ann_dir=os.path.join(labels_dir, '..'),
    save_dir=labels_output_dir
)