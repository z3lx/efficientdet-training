from imgann import Sample
import os

venv_dir = os.environ['VIRTUAL_ENV']
os.chdir(venv_dir + '/../dataset')

Sample.show_samples(data_path='./yolo/images',
                    ann_path='./csv/dataset.csv',
                    num_of_samples=5,
                    ann_type='csv',
                    seed=0,
                    image_shape=[512, 512])