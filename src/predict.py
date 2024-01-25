"""
predict.py
Predicts bounding boxes on images or videos.

Usage:
    python predict.py <model_path> <input_path> <threshold> <num_threads>

    model_path (str): The path to the model.
    input_path (str): The path to the input image or video.
    threshold (float): The minimum confidence threshold.
    num_threads (int): The number of threads to use.

Example:
    python predict.py runs/train_0/export/model.tflite dataset/voc/images_preprocessed/000000.jpeg 0.5 4
"""

import colorsys
import os
import cv2
import tensorflow as tf
import numpy as np
from utils import create_indexed_subdirectory
from utils import get_project_root

def infer(interpreter: tf.lite.Interpreter, image: np.ndarray, threshold: float) -> list:
    # Preprocess the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(image)
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]
    input_size = (input_height, input_width)
    tensor = tf.image.resize(tensor, input_size)
    tensor = tensor[tf.newaxis, :]
    tensor = tf.cast(tensor, dtype=tf.uint8)

    # Run inference
    signature_fn = interpreter.get_signature_runner()
    output = signature_fn(images = tensor)

    # Postprocess the output
    count = int(np.squeeze(output["output_0"]))
    scores = np.squeeze(output["output_1"])
    classes = np.squeeze(output["output_2"])
    boxes = np.squeeze(output["output_3"])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": classes[i],
                "score": scores[i]
            }
            results.append(result)
    return results
 
def draw_results(image: np.ndarray, results:list) -> None:
    for obj in results:
        # Convert bounding box from relative to absolute coordinates
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * image.shape[1])
        xmax = int(xmax * image.shape[1])
        ymin = int(ymin * image.shape[0])
        ymax = int(ymax * image.shape[0])

        # Get class id
        class_id = int(obj["class_id"])

        # Generate color for the object class
        hue = (class_id * 0.618033988749895) % 1.0
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1, 1))
        color = rgb_color[::-1]

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        # Draw label
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "id {}: {:.0f}%".format(class_id, obj["score"] * 100)
        cv2.putText(image, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_image(interpreter: tf.lite.Interpreter, input_path: str, output_path: str, threshold: float) -> None:
    mat = cv2.imread(input_path)
    results = infer(interpreter, mat, threshold)
    draw_results(mat, results)
    cv2.imwrite(output_path, mat)

def process_video(interpreter: tf.lite.Interpreter, input_path: str, output_path: str, threshold: float) -> None:
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while(cap.isOpened()):
        ret, mat = cap.read()
        if ret:
            results = infer(interpreter, mat, threshold)
            draw_results(mat, results)
            out.write(mat)
        else:
            break

    cap.release()
    out.release()

def main(model_path: str, input_path: str, threshold: float, num_threads: int) -> None:
    # Create directory
    runs_dir = get_project_root("runs")
    predict_dir = create_indexed_subdirectory(runs_dir, "predict")
    output_path = os.path.join(predict_dir, os.path.basename(input_path))

    # Load model
    interpreter = tf.lite.Interpreter(model_path = model_path, num_threads = num_threads)
    interpreter.allocate_tensors()

    # Check file extension
    ext = input_path.split(".")[-1]
    if ext in ["jpg", "png", "jpeg"]:
        process_image(interpreter, input_path, output_path, threshold)
    elif ext in ["mp4", "avi"]:
        process_video(interpreter, input_path, output_path, threshold)
    else:
        raise Exception(f"Unsupported file type: {ext}")

if __name__ == "__main__":
    import sys
    try:
        main(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    except Exception:
        import traceback
        traceback.print_exc()
        print(__doc__)
        sys.exit(1)