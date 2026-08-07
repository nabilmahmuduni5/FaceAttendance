import cv2
import sys
from src.camera import start_camera, get_frame
from src.detector import detect_faces

def main():
    # Pass phone URL if using phone camera, or 0 if using webcam
    video_source = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.0.102:4747/mjpegfeed"
    
    try:
        cam = start_camera(source=video_source)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("[INFO] Press 'q' in the window to exit.")

    while True:
        frame = get_frame(cam)
        if frame is None:
            continue

        faces = detect_faces(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                confidence = face[14]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{confidence * 100:.1f}%"
                cv2.putText(frame, label, (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Face Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if hasattr(cam, 'stop'):
        cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
