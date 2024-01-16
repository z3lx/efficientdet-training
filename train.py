import os

#from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
tf.get_logger().setLevel('INFO')
from absl import logging
logging.set_verbosity(logging.INFO)

# Set model architecture
spec = model_spec.get('efficientdet_lite0')

venv_dir = os.environ['VIRTUAL_ENV']
os.chdir(venv_dir + '/..')

# Load dataset
train_data = object_detector.DataLoader.from_pascal_voc(
    annotations_dir = 'dataset/voc/labels',
    images_dir = 'dataset/voc/images',
    label_map = ['note']
)

# Train model
model = object_detector.create(
    train_data = train_data,
    model_spec = spec,
    batch_size = 16,
    train_whole_model = True,
    validation_data = train_data,
    epochs = 50
)

# Export model
model.export(
    export_dir = 'export',
    export_format = [ExportFormat.SAVED_MODEL, ExportFormat.LABEL]
)