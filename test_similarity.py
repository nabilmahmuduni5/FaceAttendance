import os
import cv2
import numpy as np

from src.embedding import get_embedding


# ============================================================
# SETTINGS
# ============================================================

IMAGE_1 = "data/faces/Nabil/001.jpg"
IMAGE_2 = "data/faces/Nabil/002.jpg"


# ============================================================
# COSINE SIMILARITY FUNCTION
# ============================================================

def cosine_similarity(
    embedding1,
    embedding2
):

    # --------------------------------------------------------
    # Because our embeddings are already L2 normalized,
    # cosine similarity is simply the dot product.
    # --------------------------------------------------------

    similarity = np.dot(
        embedding1,
        embedding2
    )

    return float(similarity)


# ============================================================
# CHECK IMAGE 1
# ============================================================

if not os.path.exists(IMAGE_1):

    print(
        f"[ERROR] Image not found: {IMAGE_1}"
    )

    exit()


# ============================================================
# CHECK IMAGE 2
# ============================================================

if not os.path.exists(IMAGE_2):

    print(
        f"[ERROR] Image not found: {IMAGE_2}"
    )

    exit()


# ============================================================
# LOAD IMAGES
# ============================================================

print("[INFO] Loading images...")

image1 = cv2.imread(
    IMAGE_1
)

image2 = cv2.imread(
    IMAGE_2
)


if image1 is None:

    print(
        f"[ERROR] Could not read {IMAGE_1}"
    )

    exit()


if image2 is None:

    print(
        f"[ERROR] Could not read {IMAGE_2}"
    )

    exit()


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("[INFO] Generating embedding for image 1...")

embedding1 = get_embedding(
    image1
)


print("[INFO] Generating embedding for image 2...")

embedding2 = get_embedding(
    image2
)


# ============================================================
# CHECK EMBEDDINGS
# ============================================================

print()
print(
    f"Embedding 1 shape: {embedding1.shape}"
)

print(
    f"Embedding 2 shape: {embedding2.shape}"
)


# ============================================================
# CALCULATE COSINE SIMILARITY
# ============================================================

similarity = cosine_similarity(
    embedding1,
    embedding2
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("==========================================")
print("         FACE SIMILARITY TEST")
print("==========================================")

print(
    f"Image 1: {IMAGE_1}"
)

print(
    f"Image 2: {IMAGE_2}"
)

print()

print(
    f"Cosine similarity: {similarity:.6f}"
)

print(
    f"Similarity percentage: {similarity * 100:.2f}%"
)

print()
print("==========================================")
