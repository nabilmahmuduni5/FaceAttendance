import cv2
import time
import sys

from src.camera import start_camera, get_frame
from src.detector import detect_faces
from src.cropper import crop_face
from src.saver import save_face
from src.utils import is_duplicate_face


# ============================================================
# SETTINGS
# ============================================================

TARGET_IMAGES = 30

# Default DroidCam URL
DEFAULT_CAMERA_SOURCE = "http://192.168.0.101:4747/video"

MIN_FACE_WIDTH = 100
MIN_FACE_HEIGHT = 100

# Prevent capturing too quickly
CAPTURE_DELAY = 0.5

# Duplicate threshold
# Lower = stricter similarity check
DUPLICATE_THRESHOLD = 100.0


# ============================================================
# GET PERSON NAME
# ============================================================

person_name = input(
    "Enter person's name: "
).strip()


if not person_name:

    print("[ERROR] Person name cannot be empty.")
    exit()


print()
print(f"[INFO] Starting enrollment for: {person_name}")
print(f"[INFO] Target images: {TARGET_IMAGES}")
print()


# ============================================================
# CAMERA SOURCE
# ============================================================

# You can run:
#
# python3 enrollment.py
#
# It will use the default DroidCam URL.
#
# Or:
#
# python3 enrollment.py http://YOUR_PHONE_IP:4747/video
#
# This allows you to change the phone IP without editing code.

camera_source = (
    sys.argv[1]
    if len(sys.argv) > 1
    else DEFAULT_CAMERA_SOURCE
)


print(
    f"[INFO] Camera source: {camera_source}"
)


# ============================================================
# START CAMERA
# ============================================================

try:

    camera = start_camera(
        source=camera_source
    )

except Exception as e:

    print(
        f"[ERROR] Camera error: {e}"
    )

    exit()


print("[INFO] Camera started.")
print()


# ============================================================
# VARIABLES
# ============================================================

image_count = 0

last_capture_time = 0

# Stores the last successfully saved face
previous_face = None


# ============================================================
# INSTRUCTIONS
# ============================================================

print("==========================================")
print("          FACE ENROLLMENT")
print("==========================================")
print()
print("Instructions:")
print("  SPACE → Capture face")
print("  Q     → Quit")
print()
print("Only ONE person should be visible.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Get latest camera frame
    # --------------------------------------------------------

    frame = get_frame(camera)


    if frame is None:

        continue


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = detect_faces(frame)


    # --------------------------------------------------------
    # Draw detected faces
    # --------------------------------------------------------

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            confidence = face[-1]


            # Draw bounding box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # Display YuNet confidence

            label = (
                f"Face: "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------------
    # Display person name
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Person: {person_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Images: {image_count}/{TARGET_IMAGES}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Display instructions
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "SPACE: Capture | Q: Quit",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Show camera
    # --------------------------------------------------------

    cv2.imshow(
        "Face Enrollment",
        frame
    )


    # --------------------------------------------------------
    # Keyboard input
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # CAPTURE
    # ========================================================

    if key == ord(" "):

        current_time = time.time()


        # ----------------------------------------------------
        # Prevent rapid captures
        # ----------------------------------------------------

        if (
            current_time - last_capture_time
            < CAPTURE_DELAY
        ):

            continue


        # ----------------------------------------------------
        # Check whether face exists
        # ----------------------------------------------------

        if faces is None:

            print(
                "[WARNING] No face detected."
            )

            continue


        # ----------------------------------------------------
        # Check multiple faces
        # ----------------------------------------------------

        if len(faces) > 1:

            print(
                "[WARNING] Multiple faces detected."
            )

            continue


        # ----------------------------------------------------
        # Get the single face
        # ----------------------------------------------------

        face = faces[0]


        x, y, w, h = face[:4].astype(int)


        # ----------------------------------------------------
        # Check face size
        # ----------------------------------------------------

        if (
            w < MIN_FACE_WIDTH
            or h < MIN_FACE_HEIGHT
        ):

            print(
                "[WARNING] Face is too small."
            )

            print(
                "[INFO] Move closer to the camera."
            )

            continue


        # ----------------------------------------------------
        # Crop face
        # ----------------------------------------------------

        face_image = crop_face(
            frame,
            face
        )


        # ----------------------------------------------------
        # Validate crop
        # ----------------------------------------------------

        if face_image is None:

            print(
                "[WARNING] Could not crop face."
            )

            continue


        if face_image.size == 0:

            print(
                "[WARNING] Empty face crop."
            )

            continue


        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        if previous_face is not None:

            duplicate = is_duplicate_face(
                face_image,
                previous_face,
                threshold=DUPLICATE_THRESHOLD
            )


            if duplicate:

                print(
                    "[WARNING] Face is too similar "
                    "to the previous image."
                )

                print(
                    "[INFO] Move your head slightly "
                    "or change your position."
                )

                continue


        # ----------------------------------------------------
        # Save face
        # ----------------------------------------------------

        image_number = image_count + 1


        success, file_path = save_face(
            face_image,
            person_name,
            image_number
        )


        # ----------------------------------------------------
        # Save successful
        # ----------------------------------------------------

        if success:

            image_count += 1

            last_capture_time = current_time

            # Remember this image for the next
            # duplicate comparison
            previous_face = face_image.copy()


            print(
                f"[SAVED] "
                f"{file_path}"
            )


        else:

            print(
                "[ERROR] Failed to save image."
            )


    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q"):

        print()
        print("[INFO] Enrollment stopped.")

        break


    # ========================================================
    # FINISHED
    # ========================================================

    if image_count >= TARGET_IMAGES:

        print()
        print("==========================================")
        print("       ENROLLMENT COMPLETE")
        print("==========================================")
        print()

        print(
            f"Person: {person_name}"
        )

        print(
            f"Images: {image_count}"
        )

        print(
            f"Folder: data/faces/{person_name}"
        )

        print()

        break


# ============================================================
# CLEANUP
# ============================================================

camera.stop()

cv2.destroyAllWindows()

