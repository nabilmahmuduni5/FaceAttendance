import cv2


def calculate_mse(image1, image2):

    image1 = cv2.resize(image1, (100, 100))
    image2 = cv2.resize(image2, (100, 100))

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    gray1 = gray1.astype("float")
    gray2 = gray2.astype("float")

    difference = (gray1 - gray2) ** 2

    mse = difference.mean()

    return mse


def is_duplicate_face(
    current_face,
    previous_face,
    threshold=100.0
):

    mse = calculate_mse(
        current_face,
        previous_face
    )

    print(
        f"[INFO] Face MSE: {mse:.2f}"
    )

    return mse < threshold