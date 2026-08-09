import cv2
import numpy as np


# ============================================================
# ARCFACE STANDARD 5-POINT LANDMARKS
# ============================================================
#
# These are the standard reference positions for a
# 112 x 112 ArcFace aligned face.
#
# Order:
# 1. Left eye
# 2. Right eye
# 3. Nose
# 4. Left mouth corner
# 5. Right mouth corner
#
# ============================================================

ARCFACE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32
)


# ============================================================
# ALIGN FACE
# ============================================================

def align_face(frame, face):

    # --------------------------------------------------------
    # Get YuNet landmarks
    # --------------------------------------------------------

    landmarks = np.array(
        [
            [face[4], face[5]],      # left eye
            [face[6], face[7]],      # right eye
            [face[8], face[9]],      # nose
            [face[10], face[11]],    # left mouth
            [face[12], face[13]],    # right mouth
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Estimate similarity transformation
    # --------------------------------------------------------

    transform, _ = cv2.estimateAffinePartial2D(
        landmarks,
        ARCFACE_LANDMARKS,
        method=cv2.LMEDS
    )


    if transform is None:

        raise ValueError(
            "Could not calculate face alignment transform."
        )


    # --------------------------------------------------------
    # Warp image
    # --------------------------------------------------------

    aligned_face = cv2.warpAffine(
        frame,
        transform,
        (112, 112),
        borderValue=0
    )


    return aligned_face
