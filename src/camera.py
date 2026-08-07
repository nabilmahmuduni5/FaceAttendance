import cv2
import threading
import time

class ThreadedCamera:
    def __init__(self, source=0):
        # source can be phone IP URL like "http://192.168.0.102:4747/mjpegfeed" or webcam index 0
        self.stream_url = source
        print(f"[INFO] Connecting to camera source: {self.stream_url}")
        self.cap = cv2.VideoCapture(self.stream_url)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video source {self.stream_url}")
            
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed = grabbed
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.cap.release()

_camera_instance = None

def start_camera(source=0):
    global _camera_instance
    _camera_instance = ThreadedCamera(source=source)
    return _camera_instance

def get_frame(cap=None):
    global _camera_instance
    if _camera_instance:
        return _camera_instance.read()
    return None
