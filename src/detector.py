import os
import cv2


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2023mar.onnx"
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}"
    )


detector = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=(320, 320),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000
)


def detect_faces(frame):

    height, width, _ = frame.shape

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    return faces