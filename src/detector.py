import os
import shutil
import tempfile

import cv2


class Detector:
    """目标检测模块，封装 Haar Cascade 和 DNN 两种检测方式。"""

    def __init__(self, method="haar", model_path=None):
        self.method = method
        self.net = None
        self.class_names = []
        self._temp_dir = None

        if method == "haar":
            self._temp_dir = tempfile.mkdtemp(prefix="cv_cascades_")
            self.face_cascade = self._load_cascade("haarcascade_frontalface_alt.xml")
            self.profile_cascade = self._load_cascade("haarcascade_profileface.xml")
        elif method == "dnn" and model_path:
            self._load_dnn(model_path)

    def _load_cascade(self, filename):
        src = cv2.data.haarcascades + filename
        dst = os.path.join(self._temp_dir, filename)
        shutil.copy2(src, dst)
        cascade = cv2.CascadeClassifier(dst)
        if cascade.empty():
            raise RuntimeError(f"无法加载级联分类器: {filename}")
        return cascade

    def __del__(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _load_dnn(self, model_path):
        self.net = cv2.dnn.readNetFromCaffe(
            model_path + ".prototxt", model_path + ".caffemodel"
        )
        self.class_names = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ]

    def detect(self, frame, confidence_threshold=0.5):
        if self.method == "haar":
            return self._detect_haar(frame)
        elif self.method == "dnn":
            return self._detect_dnn(frame, confidence_threshold)
        return []

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in faces:
            results.append({
                "bbox": (x, y, w, h),
                "label": "face",
                "confidence": 1.0,
            })
        return results

    def _detect_dnn(self, frame, confidence_threshold):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < confidence_threshold:
                continue
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")
            results.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "label": self.class_names[class_id] if class_id < len(self.class_names) else "unknown",
                "confidence": float(confidence),
            })
        return results
