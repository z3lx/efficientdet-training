import os
import tensorflow as tf
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
#from tflite_model_maker.config import QuantizationConfig
from utils import create_indexed_subdirectory
from utils import get_project_root

tf.get_logger().setLevel("INFO")
from absl import logging
logging.set_verbosity(logging.INFO)

#print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
#tf.debugging.set_log_device_placement(True)

# Paths
dataset_annotation_dir = "dataset/voc/labels"
dataset_image_dir = "dataset/voc/images"

# Create directories
runs_dir = get_project_root("runs")
run_dir = create_indexed_subdirectory(runs_dir, "train")
export_dir = os.path.join(run_dir, "export")
os.mkdir(export_dir)

# Set model architecture
spec = model_spec.get(
    "efficientdet_lite0",
    model_dir = run_dir,
    tf_random_seed = 111111
)

# Load dataset
train_data = object_detector.DataLoader.from_pascal_voc(
    annotations_dir = dataset_annotation_dir,
    images_dir = dataset_image_dir,
    label_map = ["0"]
)

# Train model
model = object_detector.create(
    train_data = train_data,
    model_spec = spec,
    batch_size = 32,
    train_whole_model = True,
    validation_data = train_data,
    epochs = 1
)

# Export model
model.export(
    export_dir = export_dir,
    export_format = [
        ExportFormat.TFLITE,
        ExportFormat.SAVED_MODEL,
        ExportFormat.LABEL
    ]
)