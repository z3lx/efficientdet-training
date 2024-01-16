from imgann import Convertor
import os

venv_dir = os.environ['VIRTUAL_ENV']
os.chdir(venv_dir + '/../dataset')

Convertor.yolo2csv(dataset_dir='./yolo/',
                   yolo_ann_dir='./yolo/',
                   save_dir='./csv/dataset.csv')