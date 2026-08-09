import cv2

from src.detector import detect_faces
from src.embedding import get_embedding


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = "data/faces/Nabil/001.jpg"


# ============================================================
# LOAD IMAGE
# ============================================================

print("[INFO] Loading image...")

image = cv2.imread(
    IMAGE_PATH
)


if image is None:

    print(
        f"[ERROR] Could not load image: {IMAGE_PATH}"
    )

    exit()


print(
    f"[INFO] Image shape: {image.shape}"
)


# ============================================================
# DETECT FACE
# ============================================================

print("[INFO] Detecting face...")

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
# REQUIRE ONE FACE
# ============================================================

if len(faces) != 1:

    print(
        "[ERROR] Expected exactly one face."
    )

    exit()


# ============================================================
# GET FACE
# ============================================================

face = faces[0]


# ============================================================
# GENERATE EMBEDDING
# ============================================================

print(
    "[INFO] Generating aligned ArcFace embedding..."
)

embedding = get_embedding(
    image,
    face
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("==========================================")
print("       EMBEDDING GENERATED")
print("==========================================")

print(
    f"Embedding shape: {embedding.shape}"
)

print(
    f"Embedding length: {len(embedding)}"
)

print(
    f"Embedding dtype: {embedding.dtype}"
)

print(
    f"Embedding norm: "
    f"{__import__('numpy').linalg.norm(embedding):.6f}"
)

print()

print("First 10 values:")

print(
    embedding[:10]
)

print()
print("==========================================")
