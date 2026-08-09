import os
import cv2


def create_person_folder(person_name):
    folder_path = os.path.join(
        "data",
        "faces",
        person_name
    )

    os.makedirs(folder_path, exist_ok=True)

    return folder_path


def save_face(face_image, person_name, image_number):

    folder_path = create_person_folder(person_name)

    filename = f"{image_number:03d}.jpg"

    file_path = os.path.join(
        folder_path,
        filename
    )

    success = cv2.imwrite(
        file_path,
        face_image
    )

    return success, file_path
