import os
import cv2
import numpy as np
import onnxruntime as ort

from src.alignment import align_face


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "w600k_r50.onnx"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Face embedding model not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD ARCFACE MODEL
# ============================================================

print("[INFO] Loading ArcFace model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("[INFO] ArcFace model loaded.")


# ============================================================
# MODEL INPUT NAME
# ============================================================

INPUT_NAME = session.get_inputs()[0].name


# ============================================================
# GET EMBEDDING FROM ALIGNED FACE
# ============================================================

def get_embedding_from_aligned_face(
    aligned_face
):
    """
    Convert an already aligned 112x112 face
    into a normalized 512-dimensional ArcFace embedding.
    """

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if aligned_face is None:

        raise ValueError(
            "Aligned face is None."
        )


    if aligned_face.size == 0:

        raise ValueError(
            "Aligned face is empty."
        )


    # --------------------------------------------------------
    # Make sure image is 112x112
    # --------------------------------------------------------

    if aligned_face.shape[:2] != (112, 112):

        aligned_face = cv2.resize(
            aligned_face,
            (112, 112)
        )


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    face_image = cv2.cvtColor(
        aligned_face,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # uint8 → float32
    # --------------------------------------------------------

    face_image = face_image.astype(
        np.float32
    )


    # --------------------------------------------------------
    # ArcFace normalization
    #
    # 0   → -1
    # 127 → approximately 0
    # 255 → +1
    # --------------------------------------------------------

    face_image = (
        face_image / 127.5
    ) - 1.0


    # --------------------------------------------------------
    # HWC → CHW
    # --------------------------------------------------------

    face_image = np.transpose(
        face_image,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    face_image = np.expand_dims(
        face_image,
        axis=0
    )


    # --------------------------------------------------------
    # Run ArcFace
    # --------------------------------------------------------

    embedding = session.run(
        None,
        {
            INPUT_NAME: face_image
        }
    )[0]


    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    embedding = embedding[0]


    # --------------------------------------------------------
    # L2 normalization
    # --------------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        raise ValueError(
            "Embedding has zero norm."
        )


    embedding = embedding / norm


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return embedding.astype(
        np.float32
    )


# ============================================================
# GET EMBEDDING FROM RAW FACE + YUNET
# ============================================================

def get_embedding(
    frame,
    face
):
    """
    Generate an ArcFace embedding directly from:

        frame = full OpenCV image

        face = YuNet detected face

    YuNet landmarks are used to align the face
    before ArcFace processing.
    """

    # --------------------------------------------------------
    # Align face using YuNet landmarks
    # --------------------------------------------------------

    aligned_face = align_face(
        frame,
        face
    )


    # --------------------------------------------------------
    # Generate embedding
    # --------------------------------------------------------

    embedding = get_embedding_from_aligned_face(
        aligned_face
    )


    return embedding
