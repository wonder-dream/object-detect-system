"""摄像头采集模块 —— 多线程视频帧读取。

摄像头读取 (cap.read()) 是阻塞操作——它会等待摄像头硬件
准备好下一帧才返回，通常耗时 10~30ms。

多线程方案：
GUI 和采集完全解耦，各自按自己的节奏跑，互不等待。
get_frame() 返回 frame.copy() 而非 frame 本身。
原因：在 GUI 线程处理当前帧时（检测+绘制）,
采集线程可能又读到新帧并覆盖 self.frame。

如果返回引用 → 绘制到一半图像内容被替换 → 画面撕裂 / 颜色错乱。
返回副本 → 两个线程各持一份独立数据 → 安全。
"""

import threading
import cv2

class Camera:
    """摄像头采集器。
    设计约束:
      1. 采集线程永远只写 self.frame
      2. GUI 线程只通过 get_frame() 读取副本
      3. Lock 保证self.frame 的读写原子性

    线程安全保证:
      - self.frame 的赋值在with self.lock 中
      - self.frame 的访问在with self.lock 中
      - 一次只有一方持有锁 不会出现"读一半被写覆盖"
    """

    def __init__(self, source=0, width=640, height=480):
        """初始化参数 —— 不立即打开设备。
        延迟到 start() 才打开摄像头
        Args:
            source: 摄像头编号 (0=第一个摄像头)
                    也支持: 视频文件路径 ("video.mp4")
                           RTSP 流 ("rtsp://...")
            width:  期望宽度 (摄像头可能不支持，OpenCV 取最近值)
            height: 期望高度
        """
        self.source = source           # 视频源标识
        self.width = width             # 期望画面宽度 (px)
        self.height = height           # 期望画面高度 (px)

        # ── 运行时状态 ──
        self.cap = None                # cv2.VideoCapture 对象
        self.frame = None              # 最新帧缓存 (BGR, numpy uint8)
        self.running = False           # 控制采集循环的开关
        self.lock = threading.Lock()   # 互斥锁: 保护 frame 的读写
        self._thread = None            # 采集线程句柄

    def start(self):
        """打开摄像头并启动后台采集线程。
        线程设为 daemon=True 的含义:
          主程序退出时，daemon 线程会被 Python 自动杀死,
          不需要手动 join。

        如果摄像头打不开 (cap.isOpened() == False):
          采集循环中cap.read() 返回 (False, None),
          不会崩溃，只是 frame 保持为 None。
        """
        self.cap = cv2.VideoCapture(self.source)

        # 设置分辨率 — OpenCV 会取硬件支持的最接近值
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True

        # daemon=True → 主程序退出时自动终止
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """安全停止采集并释放资源。

        步骤:
          1. 设置 running=False → 采集循环退出
          2. join(timeout=1.0) → 等待线程结束
          3. cap.release() → 释放摄像头硬件
        """
        self.running = False

        # 等待采集线程自然退出
        if self._thread:
            self._thread.join(timeout=1.0)

        # 归还摄像头给操作系统
        if self.cap:
            self.cap.release()

    def _capture_loop(self):
        """后台死循环: 不断从摄像头读帧 → 更新 self.frame。

        GUI 每 30ms 只取一次，中间帧不需要
        旧帧被覆盖 = 隐式丢帧，不积压
        """
        while self.running:
            ret, frame = self.cap.read()  # 阻塞等待摄像头
            if ret:
                with self.lock:
                    self.frame = frame     # 原子替换最新帧

    def get_frame(self):
        """获取最新帧的线程安全副本。
        Returns:
            numpy.ndarray (BGR 格式) 或 None (尚未捕获到帧)
        """
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()   # 深拷贝，释放锁后也安全
