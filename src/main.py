import argparse

from src.camera import Camera
from src.detector import Detector
from src.gui import MainWindow


def main():
    parser = argparse.ArgumentParser(description="实时目标检测系统")
    parser.add_argument("--source", type=int, default=0, help="摄像头编号（默认 0）")
    parser.add_argument("--width", type=int, default=640, help="画面宽度")
    parser.add_argument("--height", type=int, default=480, help="画面高度")
    parser.add_argument("--method", choices=["haar", "dnn"], default="haar",
                        help="检测方式: haar（人脸）或 dnn（通用目标）")
    parser.add_argument("--model", type=str, default=None,
                        help="DNN 模型路径前缀（不含扩展名）")
    args = parser.parse_args()

    camera = Camera(source=args.source, width=args.width, height=args.height)
    detector = Detector(method=args.method, model_path=args.model)

    window = MainWindow(camera, detector)
    window.run()


if __name__ == "__main__":
    main()
