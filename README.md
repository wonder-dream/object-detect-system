# 实时目标检测系统 — Viola-Jones 论文复现

基于 **"Robust Real-Time Face Detection"** (Paul Viola & Michael Jones, 2004)
的算法复现。核心推理管线（积分图、Haar 特征求值、级联分类器、多尺度检测、NMS）
完全从零实现，使用 **Numba JIT** 编译为机器码以达到实时性能。

级联参数（特征选择、阈值、权重）来自 OpenCV 预训练的
`haarcascade_frontalface_alt.xml`，推理引擎是自实现的。


## 算法管线

```
输入图像 (BGR)
  │
  ├─ cvtColor -> Gray                          [OpenCV 工具]
  ├─ equalizeHist                             [OpenCV 工具]
  │
  ▼
┌─────────────────────────────────────────────────────┐
│            自实现 Viola-Jones 推理引擎              │
│                                                     │
│  1. compute_integral_image(gray)     ← integral_image │
│     compute_squared_integral_image   ← integral_image │
│                                                     │
│  2. 尺度金字塔 (scale 1.0 -> 1.25 -> 1.56 -> ...)    │
│     for each scale:                                  │
│       for each 滑动窗口:                              │
│         │                                            │
│         ▼                                            │
│  3. _cascade_classify_window() ← cascade (Numba JIT) │
│     ├─ get_window_stats()      ← integral_image      │
│     ├─ for 22 stages:                               │
│     │    for each weak classifier:                   │
│     │      ├─ get_rect_sum()   ← integral_image      │
│     │      │   (D-B-C+A, O(1))                      │
│     │      ├─ 方差归一化                              │
│     │      └─ 阈值判决 -> 累加 leaf value             │
│     │    if stage_sum < threshold: reject            │
│     └─ pass all stages -> accept                     │
│                                                     │
│  4. non_max_suppression()         ← cascade          │
│     (IoU-based greedy NMS)                          │
└─────────────────────────────────────────────────────┘
  │
  ▼
检测框列表 [(x, y, w, h), ...]
```


## 自实现清单

| 论文章节 | 算法组件 | 文件 |
|---------|---------|------|
| 3.1 | 积分图 + O(1) 矩形求和 (D-B-C+A) | `src/integral_image.py` |
| 3.1 | 平方积分图 + 方差归一化 | `src/integral_image.py` |
| 3.2 | Haar 特征表示 (FeatureRect, HaarFeature) | `src/haar_features.py` |
| 3.2 | 特征模板生成 (162k 组合) | `src/haar_features.py` |
| 4 | 弱分类器 / 级联阶段数据结构 | `src/cascade.py` |
| 4 | OpenCV 级联 XML 解析 -> 扁平化数组 | `src/cascade.py` |
| 4 | 级联逐级评估 (Numba JIT) | `src/cascade.py` |
| 5.1 | 尺度金字塔 + 滑动窗口 | `src/cascade.py` |
| 5.2 | 非极大值抑制 (NMS) | `src/cascade.py` |

**不包含的部分**：AdaBoost 训练（需要约 5000 正样本，非本次范围）。


## 快速开始

```bash
# 安装依赖
pip install -e .

# 默认 Viola-Jones 自实现
python -m src.main

# 切换到 OpenCV 内置检测（对照）
python -m src.main --method haar

# 自定义参数
python -m src.main --source 0 --width 1280 --height 720
python -m src.main --cascade-xml path/to/custom.xml
```


## 项目结构

```
src/
├── integral_image.py   # 积分图 + 平方积分图 + O(1) 矩形求和 + 方差归一化
├── haar_features.py    # Haar 特征定义 + 模板生成 + 特征值求值
├── cascade.py          # XML 解析 -> 扁平化 -> Numba JIT 级联评估 -> NMS
├── detector.py         # 统一接口 (vj / haar 双模式)
├── camera.py           # 多线程摄像头采集
├── gui.py              # Tkinter 界面: 实时画面 + 控制面板 + FPS
└── main.py             # 命令行入口
```


## GUI 界面

```
┌────────────────────────────────────┐
│  ┌──────────────┐  ┌───────────┐  │
│  │              │  │ 检测方式   │  │
│  │   实时画面    │  │ o VJ 复现  │  │
│  │   (蓝框=VJ)  │  │ o OpenCV   │  │
│  │   (绿框=Haar)│  │           │  │
│  │              │  │ 显示设置   │  │
│  │              │  │ [x] 标签   │  │
│  │              │  │ [x] 置信度 │  │
│  └──────────────┘  └───────────┘  │
│  状态: 运行中              FPS: 15 │
└────────────────────────────────────┘
```


## 依赖

| 包 | 用途 |
|---|------|
| `opencv-python` | 图像读写、色彩转换、直方图均衡、GUI 对照 |
| `numpy` | 积分图数组、NMS 向量化计算、扁平化参数 |
| `numba` | JIT 编译级联评估为机器码 (约 15 FPS) |
| `pillow` | Tkinter 图像格式转换 |
