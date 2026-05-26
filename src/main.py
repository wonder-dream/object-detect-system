"""程序入口 —— 命令行参数解析与启动。

用法示例:
  python -m src.main                          # 默认 VJ 检测
  python -m src.main --method haar            # OpenCV 内置 Haar
  python -m src.main --source 0 --width 1280 --height 720
  python -m src.main --cascade-xml path/to/custom.xml
"""

import argparse

from src.camera import Camera
from src.detector import Detector
from src.gui import MainWindow


def main():
    """解析命令行参数，组装 Camera + Detector + GUI 并启动。"""

    parser = argparse.ArgumentParser(description="实时目标检测系统")

    # ── 摄像头参数 ──
    parser.add_argument(
        "--source", type=int, default=0,
        help="摄像头编号 (默认 0)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="画面宽度 (像素)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="画面高度 (像素)",
    )

    # ── 检测方式 ──
    parser.add_argument(
        "--method", choices=["vj", "haar"], default="vj",
        help="检测方式: vj=Viola-Jones论文自实现, haar=OpenCV内置",
    )

    # ── 自定义级联文件 ──
    parser.add_argument(
        "--cascade-xml", type=str, default=None,
        help="Viola-Jones 自定义 cascade XML 路径",
    )

    args = parser.parse_args()

    # 1. 创建摄像头采集器
    camera = Camera(source=args.source, width=args.width, height=args.height)

    # 2. 创建目标检测器
    detector = Detector(
        method=args.method,
        cascade_xml=args.cascade_xml,
    )

    # 3. 创建主窗口并启动事件循环
    window = MainWindow(camera, detector)
    window.run()


if __name__ == "__main__":
    main()
