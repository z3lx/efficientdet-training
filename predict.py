import os
import cv2
import tensorflow as tf
from utils import create_subdirectory
from utils import chdir_to_project_root
from utils import run_odt_and_draw_results

DETECTION_THRESHOLD = 0.5
RUNS_DIR = "runs"

def process_image(interpreter, input_path, output_path):
    mat = cv2.imread(input_path)
    output_img = run_odt_and_draw_results(
        mat,
        interpreter,
        threshold = DETECTION_THRESHOLD
    )

    cv2.imwrite(output_path, output_img)

def process_video(interpreter, input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            output_frame = run_odt_and_draw_results(
                frame,
                interpreter,
                threshold = DETECTION_THRESHOLD
            )
            out.write(output_frame)
        else:
            break

    cap.release()
    out.release()

def main(model_path, input_path, num_threads = 8):
    # Create directory
    chdir_to_project_root()
    predict_dir = create_subdirectory(RUNS_DIR, "predict")
    output_path = os.path.join(predict_dir, os.path.basename(input_path))

    # Load model
    interpreter = tf.lite.Interpreter(model_path = model_path, num_threads = num_threads)
    interpreter.allocate_tensors()

    # Check file extension
    ext = os.path.splitext(input_path)[1]
    if ext in [".jpg", ".png", ".jpeg"]:
        process_image(interpreter, input_path, output_path)
    elif ext in [".mp4", ".avi"]:
        process_video(interpreter, input_path, output_path)
    else:
        raise Exception(f"Unsupported file type: {ext}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])