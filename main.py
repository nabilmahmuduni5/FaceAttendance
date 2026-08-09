import cv2
import sys

from src.camera import start_camera, get_frame
from src.detector import detect_faces
from src.saver import create_person_folder


def main():

    # --------------------------------------------------
    # 1. Get person's name
    # --------------------------------------------------

    person_name = input("Enter person's name: ").strip()

    if not person_name:
        print("[ERROR] Name cannot be empty.")
        return

    # --------------------------------------------------
    # 2. Automatically create person's face folder
    # --------------------------------------------------

    try:
        folder_path = create_person_folder(person_name)
    except Exception as e:
        print(f"[ERROR] Could not create person folder: {e}")
        return

    print(f"[INFO] Person: {person_name}")
    print(f"[INFO] Face folder: {folder_path}")

    # --------------------------------------------------
    # 3. Camera source
    # --------------------------------------------------

    # Pass camera source from command line:
    # python main.py 0
    #
    # Or use your phone camera:
    # python main.py http://YOUR_PHONE_IP:4747/mjpegfeed

    video_source = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "http://192.168.0.101:4747/video"  
    )

    # --------------------------------------------------
    # 4. Start camera
    # --------------------------------------------------

    try:
        cam = start_camera(source=video_source)

    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("[INFO] Camera started.")
    print("[INFO] Press 'q' in the window to exit.")

    # --------------------------------------------------
    # 5. Main camera loop
    # --------------------------------------------------

    while True:

        frame = get_frame(cam)

        if frame is None:
            continue

        # --------------------------------------------------
        # 6. Detect faces using YuNet
        # --------------------------------------------------

        faces = detect_faces(frame)

        # --------------------------------------------------
        # 7. Draw detected faces
        # --------------------------------------------------

        if faces is not None:

            for face in faces:

                x, y, w, h = map(int, face[:4])

                confidence = face[14]

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Display confidence
                label = f"{confidence * 100:.1f}%"

                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # --------------------------------------------------
        # 8. Display camera
        # --------------------------------------------------

        cv2.imshow("Face Attendance", frame)

        # --------------------------------------------------
        # 9. Press Q to quit
        # --------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # 10. Release camera
    # --------------------------------------------------

    if hasattr(cam, "stop"):
        cam.stop()

    cv2.destroyAllWindows()


# ------------------------------------------------------
# Program entry point
# ------------------------------------------------------

if __name__ == "__main__":
    main()