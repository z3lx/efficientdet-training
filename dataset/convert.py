from imgann import Convertor
from PIL import Image
import os
import shutil

venv_dir = os.environ['VIRTUAL_ENV']
os.chdir(venv_dir + '/../dataset')

images_dir = 'yolo/images'
labels_dir = 'yolo/labels'
images_output_dir = 'voc/images'
labels_output_dir = 'voc/labels'

# Checks
if not os.path.isdir(images_dir):
    raise Exception(f'Directory {images_dir} does not exist.')
if not os.path.isdir(labels_dir):
    raise Exception(f'Directory {labels_dir} does not exist.')

# Cleanup
shutil.rmtree(images_output_dir, ignore_errors=True)
shutil.rmtree(labels_output_dir, ignore_errors=True)
os.makedirs(images_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)

# Convert images
for filename in os.listdir(images_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img = Image.open(os.path.join(images_dir, filename))
        img = img.convert('RGB')
        img.save(os.path.join(images_output_dir, filename.split('.')[0] + '.jpeg'), 'JPEG', quality=100)
    else:
        if filename.filename.endswith('.jpeg'):
            shutil.copy(os.path.join(images_dir, filename), images_output_dir)

# Convert labels
Convertor.yolo2voc(
    dataset_dir = images_dir + '/..',
    yolo_ann_dir = labels_dir + '/..',
    save_dir=labels_output_dir
)

# Fix labels
import xml.etree.ElementTree as ET
for filename in os.listdir(labels_output_dir):
    if (filename.endswith('.xml')):
        filepath = os.path.join(labels_output_dir, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        filename_tag = root.find('filename')
        filename_tag.text = filename_tag.text.replace('.png', '.jpeg')
        root.find('path').text = filename_tag.text
        for name_elem in root.iter('name'):
            name_elem.text = 'note'
            #try:
            #    double_value = float(name_elem.text)
            #    name_elem.text = str(int(double_value))
            #except ValueError:
            #    pass

        tree.write(filepath)