import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk


class MainWindow:
    """目标检测系统主界面。"""

    def __init__(self, camera, detector):
        self.camera = camera
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("实时目标检测系统")
        self.root.configure(bg="#1e1e1e")

        self.fps = 0
        self._fps_counter = 0
        self._fps_timer_id = None

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧视频区域
        video_frame = ttk.LabelFrame(main_frame, text="实时画面", padding=4)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, background="#2d2d2d")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 右侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=8)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        # 检测方式
        ttk.Label(control_frame, text="检测方式").pack(anchor=tk.W, pady=(0, 2))
        self.method_var = tk.StringVar(value="haar")
        ttk.Radiobutton(control_frame, text="Haar Cascade（人脸）",
                        variable=self.method_var, value="haar",
                        command=self._on_method_change).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="DNN（通用目标）",
                        variable=self.method_var, value="dnn",
                        command=self._on_method_change).pack(anchor=tk.W)

        # 置信度阈值
        ttk.Label(control_frame, text="置信度阈值").pack(anchor=tk.W, pady=(12, 2))
        self.confidence_var = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.confidence_var,
                  orient=tk.HORIZONTAL, length=180).pack(fill=tk.X)

        # 显示设置
        ttk.Label(control_frame, text="显示设置").pack(anchor=tk.W, pady=(12, 2))
        self.show_label_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示标签",
                        variable=self.show_label_var).pack(anchor=tk.W)
        self.show_conf_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="显示置信度",
                        variable=self.show_conf_var).pack(anchor=tk.W)

        # 状态栏
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT)
        self.fps_label = ttk.Label(status_frame, text="FPS: --")
        self.fps_label.pack(side=tk.RIGHT)

    def _bind_events(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _on_method_change(self):
        method = self.method_var.get()
        self.detector.method = method
        if method == "dnn" and self.detector.net is None:
            self.status_label.config(text="DNN 模型未加载，请配置模型路径")

    def on_close(self):
        self.camera.stop()
        if self._fps_timer_id:
            self.root.after_cancel(self._fps_timer_id)
        self.root.destroy()

    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            self.video_label.after(30, self.update_frame)
            return

        results = self.detector.detect(
            frame, confidence_threshold=self.confidence_var.get()
        )

        frame = self._draw_detections(frame, results)

        # 转换为 Tkinter 可显示的格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # FPS 计数
        self._fps_counter += 1

        self.video_label.after(30, self.update_frame)

    def _draw_detections(self, frame, results):
        for r in results:
            x, y, w, h = r["bbox"]
            label = r.get("label", "")
            conf = r.get("confidence", 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self.show_label_var.get():
                text = label
                if self.show_conf_var.get() and conf < 1.0:
                    text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    def start_fps_timer(self):
        self._fps_counter = 0

        def update_fps():
            self.fps_label.config(text=f"FPS: {self._fps_counter}")
            self._fps_counter = 0
            self._fps_timer_id = self.root.after(1000, update_fps)

        self._fps_timer_id = self.root.after(1000, update_fps)
        self.status_label.config(text="运行中")

    def run(self):
        self.camera.start()
        self.start_fps_timer()
        self.update_frame()
        self.root.mainloop()
