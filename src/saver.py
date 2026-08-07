import os
import cv2

def save_face(face_image, person_name, image_number):

    folder = f"data/faces/{person_name}"

    os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/{image_number}.jpg"

    cv2.imwrite(filename, face_image)