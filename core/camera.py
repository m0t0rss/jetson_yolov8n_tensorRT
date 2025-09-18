import cv2
from threading import Thread

class CameraStream:
    def __init__(self, gst_pipeline):
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Не вдалося відкрити камеру!")

        self.ret, self.frame = self.cap.read()
        self.stopped = False

        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()


    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

def video_cam():
    # GStreamer pipeline під Raspberry Pi Cam Module 2
    gst_pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=640, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )
    return CameraStream(gst_pipeline)
