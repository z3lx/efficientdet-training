import colorsys
import os

import cv2
import numpy as np
import tensorflow as tf

def create_subdirectory(base_dir: str, prefix: str) -> str:
    """Create a subdirectory with a given prefix in the given base directory"""
    os.makedirs(base_dir, exist_ok=True)
    sub_dir = None
    for i in range(len(os.listdir(base_dir)) + 1):
        sub_dir = os.path.join(base_dir, f"{prefix}_{i}")
        if not os.path.exists(sub_dir) or not os.listdir(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
            break
    return sub_dir

def chdir_to_project_root():
    """Change the current working directory to the project root"""
    venv_dir = os.environ["VIRTUAL_ENV"]
    os.chdir(os.path.join(venv_dir, ".."))

def preprocess_image(mat, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(mat)
    #img = tf.cast(img, dtype=tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""

    signature_fn = interpreter.get_signature_runner()

    # Feed the input image to the model
    output = signature_fn(images=image)

    # Get all outputs from the model
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

def get_color(index):
    """Generate a color for a given object class"""
    hue = (index * 0.618033988749895) % 1.0
    rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1, 1))
    return rgb_color

def run_odt_and_draw_results(mat, interpreter, threshold = 0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        mat,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)

    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj["class_id"])

        # Draw the bounding box and label on the image
        color = get_color(class_id)
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "id {}: {:.0f}%".format(class_id, obj["score"] * 100)
        cv2.putText(original_image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    original_uint8 = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
    return original_uint8
