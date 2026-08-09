import cv2

from src.detector import detect_faces
from src.alignment import align_face


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = "data/faces/Nabil/001.jpg"


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)


if image is None:

    print(
        f"[ERROR] Could not load {IMAGE_PATH}"
    )

    exit()


print(
    f"[INFO] Image loaded: {image.shape}"
)


# ============================================================
# DETECT FACE
# ============================================================

faces = detect_faces(
    image
)


if faces is None:

    print(
        "[ERROR] No face detected."
    )

    exit()


print(
    f"[INFO] Faces detected: {len(faces)}"
)


# ============================================================
# CHECK MULTIPLE FACES
# ============================================================

if len(faces) != 1:

    print(
        "[ERROR] Expected exactly one face."
    )

    exit()


# ============================================================
# ALIGN FACE
# ============================================================

face = faces[0]


print(
    "[INFO] Aligning face..."
)


aligned_face = align_face(
    image,
    face
)


print(
    f"[INFO] Aligned face shape: "
    f"{aligned_face.shape}"
)


# ============================================================
# SAVE ALIGNED FACE
# ============================================================

output_path = "aligned_test.jpg"


success = cv2.imwrite(
    output_path,
    aligned_face
)


if success:

    print(
        f"[SAVED] {output_path}"
    )

else:

    print(
        "[ERROR] Could not save aligned face."
    )


# ============================================================
# DISPLAY
# ============================================================

cv2.imshow(
    "Aligned Face",
    aligned_face
)


print(
    "[INFO] Press any key to close."
)


cv2.waitKey(0)

cv2.destroyAllWindows()
