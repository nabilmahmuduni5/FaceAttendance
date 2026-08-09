import os
import cv2
import numpy as np

from src.detector import detect_faces
from src.embedding import get_embedding


# ============================================================
# SETTINGS
# ============================================================

PERSON_NAME = "Nabil"

INPUT_FOLDER = os.path.join(
    "data",
    "faces",
    PERSON_NAME
)

OUTPUT_FOLDER = "data/embeddings"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    f"{PERSON_NAME}.npy"
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# CHECK INPUT FOLDER
# ============================================================

if not os.path.exists(INPUT_FOLDER):

    print(
        f"[ERROR] Input folder not found: {INPUT_FOLDER}"
    )

    exit()


# ============================================================
# GET IMAGE FILES
# ============================================================

image_files = []

for filename in os.listdir(INPUT_FOLDER):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        image_files.append(filename)


# Sort images

image_files.sort()


# ============================================================
# CHECK IMAGES
# ============================================================

if len(image_files) == 0:

    print(
        "[ERROR] No images found."
    )

    exit()


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print()
print("==========================================")
print("      FACE EMBEDDING GENERATION")
print("==========================================")
print()

print(
    f"[INFO] Person: {PERSON_NAME}"
)

print(
    f"[INFO] Input folder: {INPUT_FOLDER}"
)

print(
    f"[INFO] Images found: {len(image_files)}"
)

print()


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

embeddings = []

successful = 0
failed = 0


for index, filename in enumerate(
    image_files,
    start=1
):

    print(
        f"[{index}/{len(image_files)}] "
        f"Processing {filename}..."
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            "    [ERROR] Could not load image."
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # Detect face using YuNet
    # --------------------------------------------------------

    faces = detect_faces(
        image
    )


    if faces is None:

        print(
            "    [ERROR] No face detected."
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # Require exactly one face
    # --------------------------------------------------------

    if len(faces) != 1:

        print(
            f"    [ERROR] Expected 1 face, "
            f"found {len(faces)}."
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # Get detected face
    # --------------------------------------------------------

    face = faces[0]


    # --------------------------------------------------------
    # Generate aligned ArcFace embedding
    # --------------------------------------------------------

    try:

        embedding = get_embedding(
            image,
            face
        )

    except Exception as e:

        print(
            f"    [ERROR] Embedding failed: {e}"
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # Store embedding
    # --------------------------------------------------------

    embeddings.append(
        embedding
    )

    successful += 1

    print(
        "    [OK] Aligned embedding generated"
    )


# ============================================================
# CHECK RESULTS
# ============================================================

print()
print("==========================================")
print("             RESULTS")
print("==========================================")

print(
    f"Successful images: {successful}"
)

print(
    f"Failed images: {failed}"
)


if successful == 0:

    print()
    print(
        "[ERROR] No embeddings were generated."
    )

    exit()


# ============================================================
# CONVERT TO NUMPY ARRAY
# ============================================================

embeddings = np.array(
    embeddings,
    dtype=np.float32
)


print(
    f"Embedding array shape: {embeddings.shape}"
)

print(
    f"Embedding dimension: {embeddings.shape[1]}"
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    OUTPUT_FILE,
    embeddings
)


print()
print(
    f"[SAVED] {OUTPUT_FILE}"
)


# ============================================================
# VERIFY SAVED FILE
# ============================================================

loaded_embeddings = np.load(
    OUTPUT_FILE
)


print()
print(
    f"[VERIFY] Loaded shape: "
    f"{loaded_embeddings.shape}"
)

print(
    f"[VERIFY] Data type: "
    f"{loaded_embeddings.dtype}"
)


# ============================================================
# CHECK EMBEDDING NORMS
# ============================================================

norms = np.linalg.norm(
    loaded_embeddings,
    axis=1
)


print(
    f"[VERIFY] Minimum norm: "
    f"{norms.min():.6f}"
)

print(
    f"[VERIFY] Maximum norm: "
    f"{norms.max():.6f}"
)


# ============================================================
# FINISHED
# ============================================================

print()
print("==========================================")
print("    ALIGNED EMBEDDING GENERATION COMPLETE")
print("==========================================")
