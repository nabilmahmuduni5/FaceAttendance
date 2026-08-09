import os
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

PERSON_NAME = "Nabil"

EMBEDDING_FILE = os.path.join(
    "data",
    "embeddings",
    f"{PERSON_NAME}.npy"
)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(EMBEDDING_FILE):

    print(
        f"[ERROR] Embedding file not found: "
        f"{EMBEDDING_FILE}"
    )

    exit()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embeddings = np.load(
    EMBEDDING_FILE
)


print()
print("==========================================")
print("      SAME-PERSON SIMILARITY TEST")
print("==========================================")

print(
    f"[INFO] Person: {PERSON_NAME}"
)

print(
    f"[INFO] Embeddings: {len(embeddings)}"
)

print(
    f"[INFO] Embedding shape: {embeddings.shape}"
)


# ============================================================
# REFERENCE EMBEDDING
# ============================================================

reference = embeddings[0]


print()
print(
    "=========================================="
)

print(
    "Reference: 001.jpg"
)

print(
    "=========================================="
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarities = []


for i in range(
    1,
    len(embeddings)
):

    current = embeddings[i]


    # --------------------------------------------------------
    # Cosine similarity
    #
    # Since embeddings are already L2 normalized,
    # dot product = cosine similarity.
    # --------------------------------------------------------

    similarity = np.dot(
        reference,
        current
    )


    similarities.append(
        similarity
    )


    print(
        f"001.jpg vs "
        f"{i + 1:03d}.jpg → "
        f"{similarity:.6f} "
        f"({similarity * 100:.2f}%)"
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

similarities = np.array(
    similarities
)


# ============================================================
# STATISTICS
# ============================================================

minimum = similarities.min()

maximum = similarities.max()

average = similarities.mean()

median = np.median(
    similarities
)


# ============================================================
# RESULTS
# ============================================================

print()
print("==========================================")
print("             FINAL RESULTS")
print("==========================================")

print(
    f"Comparisons: {len(similarities)}"
)

print(
    f"Minimum similarity: "
    f"{minimum:.6f} "
    f"({minimum * 100:.2f}%)"
)

print(
    f"Maximum similarity: "
    f"{maximum:.6f} "
    f"({maximum * 100:.2f}%)"
)

print(
    f"Average similarity: "
    f"{average:.6f} "
    f"({average * 100:.2f}%)"
)

print(
    f"Median similarity: "
    f"{median:.6f} "
    f"({median * 100:.2f}%)"
)

print()
print("==========================================")
