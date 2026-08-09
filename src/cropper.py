import cv2


def crop_face(frame, face):

    x, y, w, h = face[:4].astype(int)

    # Make sure coordinates are inside the image
    x = max(0, x)
    y = max(0, y)

    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)

    cropped = frame[y:y2, x:x2]

    return cropped