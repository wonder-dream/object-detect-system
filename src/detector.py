"""目标检测模块 —— 统一封装两种检测方式。

  - vj:   Viola-Jones 论文自实现 (积分图 + Haar 特征 + 级联分类器)
  - haar: OpenCV 内置 Haar Cascade (用于对照)

两种方式输出统一格式的结果列表，GUI 无需关心底层实现差异。
"""

import cv2
import numpy as np

from .cascade import CascadeDetector


class Detector:
    """目标检测器统一接口。

    对外暴露单一的 detect(frame) 方法，内部根据 method 分发到
    不同的检测实现。支持运行时切换方法 (修改 .method 属性)。

    Usage:
        d = Detector(method="vj")        # Viola-Jones 论文复现
        d = Detector(method="haar")      # OpenCV 内置 Haar (对照)
        results = d.detect(frame)        # 统一调用
    """

    def __init__(
        self,
        method: str = "vj",
        cascade_xml: str | None = None,
    ):
        """初始化检测器。

        Args:
            method: 检测方式 ("vj" | "haar")
            cascade_xml: Viola-Jones 自定义 cascade XML 路径
        """
        self.method = method

        # ── Viola-Jones 自实现 ──
        self.vj_detector: CascadeDetector | None = None
        if method == "vj":
            # 默认使用 OpenCV 自带的正面人脸级联文件
            xml_path = cascade_xml or (
                cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            )
            self.vj_detector = CascadeDetector(xml_path)

        # ── OpenCV 内置 Haar Cascade (对照) ──
        self.face_cascade: cv2.CascadeClassifier | None = None
        self.profile_cascade: cv2.CascadeClassifier | None = None
        if method == "haar":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            )
            # 侧脸级联 (备用)
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml"
            )

    # ── 统一检测接口 ─────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """对单帧执行目标检测。

        Args:
            frame: BGR 彩色图像 (H×W×3 numpy uint8 数组)

        Returns:
            list[dict]: 检测结果列表，每项包含:
              - "bbox":       (x, y, w, h) 边界框像素坐标
              - "label":      str 类别名称 (如 "face")
              - "confidence": float 置信度 (0.0 ~ 1.0)
        """
        if self.method == "vj":
            return self._detect_vj(frame)
        elif self.method == "haar":
            return self._detect_haar(frame)
        return []

    # ── Viola-Jones 自实现 ───────────────────────────────

    def _detect_vj(self, frame: np.ndarray) -> list[dict]:
        """使用论文自实现的 Viola-Jones 检测器。

        处理管线:
          1. BGR → Gray (灰度转换)
          2. 直方图均衡化 (CascadeDetector 内部自动处理)
          3. 多尺度滑动窗口 + 级联评估 (Numba JIT 加速)
          4. NMS (非极大值抑制) 合并重叠框

        Args:
            frame: BGR 彩色图像

        Returns:
            统一格式的检测结果列表
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 多尺度级联检测 (scale_factor=1.25 对应论文推荐值)
        boxes = self.vj_detector.detect(
            gray,
            scale_factor=1.25,       # 每级放大 1.25x
            min_size=30,             # 最小检测窗口 30px
            overlap_threshold=0.3,   # NMS IoU 阈值
        )

        # 转换为统一格式
        results: list[dict] = []
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            results.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "label": "face",
                "confidence": 0.95,  # 级联无连续置信度，用固定高值表示通过
            })
        return results

    # ── OpenCV 内置 Haar Cascade (对照) ───────────────────

    def _detect_haar(self, frame: np.ndarray) -> list[dict]:
        """使用 OpenCV 内置 Haar Cascade 检测（对照参考）。

        处理管线:
          1. BGR → Gray
          2. 直方图均衡化 (增强对比度)
          3. detectMultiScale (OpenCV 内部优化的级联检测)

        Args:
            frame: BGR 彩色图像

        Returns:
            统一格式的检测结果列表
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 直方图均衡化

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,        # 缩放步长
            minNeighbors=5,          # 邻接框数阈值 (越大越严格)
            minSize=(30, 30),        # 最小检测尺寸
        )

        results: list[dict] = []
        for (x, y, w, h) in faces:
            results.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "label": "face",
                "confidence": 1.0,   # OpenCV 级联不返回置信度
            })
        return results


