import os
from PIL import Image

import cv2
import tensorflow as tf

from utils import create_subdirectory, chdir_to_project_root, run_odt_and_draw_results

MODEL_PATH = "exports/export_0/model.tflite"
INPUT_IMG_PATH = "dataset/voc/images/000000000000.jpeg"
DETECTION_THRESHOLD = 0.8
RUNS_DIR = "runs"

# Create directory
chdir_to_project_root()
predict_dir = create_subdirectory(RUNS_DIR, "predict")

# Resize image
img = Image.open(INPUT_IMG_PATH)
img.thumbnail((320, 320), Image.LANCZOS)
tmp_img_path = os.path.join(predict_dir, "tmp.png")
img.save(tmp_img_path, "PNG")

# Load model
interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
interpreter.allocate_tensors()

output_img = run_odt_and_draw_results(
    tmp_img_path,
    interpreter,
    threshold = DETECTION_THRESHOLD
)

output_img_path = os.path.join(predict_dir, "output.png")
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_img_path, output_img)
os.remove(tmp_img_path)
