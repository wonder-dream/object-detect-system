"""主界面模块 —— Tkinter 实时视频显示与交互控制。
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk


class MainWindow:
    """实时目标检测系统主窗口。

    核心流程:
      1. 每 30ms 触发 update_frame()
      2. 从 Camera 获取最新帧
      3. 调用 Detector 执行检测
      4. 在帧上绘制检测框和标签
      5. 转换为 PIL/Tkinter 格式并更新显示
    """

    def __init__(self, camera, detector):
        """初始化窗口、控件和定时器。

        Args:
            camera: Camera 实例 (提供视频帧)
            detector: Detector 实例 (执行目标检测)
        """
        self.camera = camera
        self.detector = detector

        # ── Tkinter 根窗口 ──
        self.root = tk.Tk()
        self.root.title("实时目标检测系统")
        self.root.configure(bg="#1e1e1e")  # 深色主题背景

        # ── FPS 计数器 ──
        self.fps = 0                   # 上次统计的 FPS
        self._fps_counter = 0          # 当前秒内累计帧数
        self._fps_timer_id = None      # FPS 定时器 ID (用于取消)

        self._build_ui()    # 构建界面控件
        self._bind_events() # 绑定窗口关闭事件

    # ── 界面构建 ───────────────────────────────────────

    def _build_ui(self):
        """构建完整的 UI 布局。"""
        # 主容器 (水平排列：左=视频，右=控制)
        main_frame = ttk.Frame(self.root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── 左侧: 视频显示区域 ──
        video_frame = ttk.LabelFrame(main_frame, text="实时画面", padding=4)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, background="#2d2d2d")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ── 右侧: 控制面板 ──
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=8)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        # 检测方式选择 (单选按钮组)
        ttk.Label(control_frame, text="检测方式").pack(anchor=tk.W, pady=(0, 2))
        self.method_var = tk.StringVar(value=self.detector.method)
        ttk.Radiobutton(
            control_frame, text="Viola-Jones（论文复现）",
            variable=self.method_var, value="vj",
            command=self._on_method_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            control_frame, text="Haar Cascade（OpenCV内置）",
            variable=self.method_var, value="haar",
            command=self._on_method_change,
        ).pack(anchor=tk.W)

        # 显示设置 (复选框组)
        ttk.Label(control_frame, text="显示设置").pack(anchor=tk.W, pady=(12, 2))
        self.show_label_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示标签",
                        variable=self.show_label_var).pack(anchor=tk.W)
        self.show_conf_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示置信度",
                        variable=self.show_conf_var).pack(anchor=tk.W)

        # ── 底部: 状态栏 ──
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT)
        self.fps_label = ttk.Label(status_frame, text="FPS: --")
        self.fps_label.pack(side=tk.RIGHT)

    # ── 事件处理 ───────────────────────────────────────

    def _bind_events(self):
        """绑定窗口关闭协议。"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _on_method_change(self):
        """检测方式切换回调：懒初始化对应检测器。

        如果切换到尚未初始化的模式，则即时创建对应的检测器实例。
        这样避免了启动时加载所有检测器的开销。
        """
        method = self.method_var.get()
        self.detector.method = method

        if method == "vj" and self.detector.vj_detector is None:
            # 懒初始化 Viola-Jones 检测器
            from src.cascade import CascadeDetector
            import cv2
            xml_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            )
            self.detector.vj_detector = CascadeDetector(xml_path)

        elif method == "haar" and self.detector.face_cascade is None:
            # 懒初始化 OpenCV Haar 检测器
            import cv2
            self.detector.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            )

    def on_close(self):
        """窗口关闭时清理资源。"""
        self.camera.stop()                        # 停止摄像头
        if self._fps_timer_id:
            self.root.after_cancel(self._fps_timer_id)  # 取消 FPS 定时器
        self.root.destroy()                       # 销毁窗口

    # ── 主循环 ─────────────────────────────────────────

    def update_frame(self):
        """核心帧更新回调 (每 ~30ms 触发一次)。

        完整管线:
          获取帧 → 检测 → 绘制结果 → 格式转换 → 显示
        """
        # 1. 获取最新帧
        frame = self.camera.get_frame()
        if frame is None:
            self.video_label.after(30, self.update_frame)
            return

        # 2. 执行目标检测
        results = self.detector.detect(frame)

        # 3. 在帧上绘制检测框
        frame = self._draw_detections(frame, results)

        # 4. BGR → RGB (OpenCV 用 BGR, Tkinter/PIL 用 RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # 5. 更新 Label 显示的图片 (保持引用防止 GC)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # 6. FPS 累计
        self._fps_counter += 1

        # 7. 调度下一次更新
        self.video_label.after(30, self.update_frame)

    def _draw_detections(self, frame, results):
        """在帧上绘制检测框和标签。

        不同检测方式用不同颜色区分:
          - VJ 自实现: 蓝色  #(255,100,0) in BGR
          - OpenCV:     绿色  #(0,255,0)

        Args:
            frame: BGR 图像数组
            results: 检测结果列表 [{"bbox":(x,y,w,h), "label":str, "confidence":float}]

        Returns:
            绘制后的帧 (原地修改)
        """
        color_map = {
            "vj": (255, 100, 0),    # 蓝色 (BGR 格式)
            "haar": (0, 255, 0),     # 绿色
        }
        color = color_map.get(self.detector.method, (0, 255, 0))

        for r in results:
            x, y, w, h = r["bbox"]
            label = r.get("label", "")
            conf = r.get("confidence", 0)

            # 绘制矩形框 (线宽 2px)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 绘制标签文字
            if self.show_label_var.get():
                text = label
                if self.show_conf_var.get() and conf < 1.0:
                    text = f"{label} {conf:.2f}"
                # 文字位置: 框的上方 8px
                cv2.putText(frame, text, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    # ── FPS 统计 ───────────────────────────────────────

    def start_fps_timer(self):
        """启动每秒 FPS 统计定时器。"""
        self._fps_counter = 0

        def update_fps():
            """每秒回调: 显示 FPS 并重置计数器。"""
            self.fps_label.config(text=f"FPS: {self._fps_counter}")
            self._fps_counter = 0
            self._fps_timer_id = self.root.after(1000, update_fps)

        self._fps_timer_id = self.root.after(1000, update_fps)
        self.status_label.config(text="运行中")

    def run(self):
        """启动应用: 摄像头 → 定时器 → 帧循环 → Tkinter 主循环。"""
        self.camera.start()         # 打开摄像头
        self.start_fps_timer()      # 启动 FPS 计数器
        self.update_frame()         # 启动帧更新循环
        self.root.mainloop()        # 进入 Tkinter 事件循环 (阻塞)
