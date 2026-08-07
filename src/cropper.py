def crop_face(frame, face):

    x, y, w, h = face[:4].astype(int)

    cropped = frame[y:y+h, x:x+w]

    return cropped