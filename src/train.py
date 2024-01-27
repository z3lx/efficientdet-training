"""
train.py
Trains an EfficientDet model on a dataset.

Usage:
    python train.py <dataset_annotation_dir> <dataset_image_dir> <debug>

    dataset_annotation_dir (str): The directory where the dataset annotations are stored.
    dataset_image_dir (str): The directory where the dataset images are stored.
    debug (bool): Whether to run in debug mode.

Example:
    python train.py dataset/voc/labels dataset/voc/images True
"""

import os
import time
import tensorflow as tf
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
#from tflite_model_maker.config import QuantizationConfig
from utils import create_indexed_subdirectory
from utils import get_project_root

def main(dataset_annotation_dir:str, dataset_image_dir: str, debug: bool = False):
    start_time = time.time()

    from absl import logging
    if debug:
        tf.get_logger().setLevel("DEBUG")
        logging.set_verbosity(logging.DEBUG)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
        tf.debugging.set_log_device_placement(True)
    else:
        tf.get_logger().setLevel("INFO")
        logging.set_verbosity(logging.INFO)

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
        epochs = 50
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training took {:.2f} hours".format(elapsed_time / 3600))
    print("Model exported to {}".format(export_dir))

if __name__ == "__main__":
    import sys;
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3].lower == "true")
    except Exception:
        import traceback
        traceback.print_exc()
        print(__doc__)
        sys.exit(1)
