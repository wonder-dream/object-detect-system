"""级联分类器模块

级联结构是 Viola-Jones 实现实时检测的关键创新：
  - 将多个 AdaBoost 强分类器串联成检测管道
  - 早期阶段使用极少特征快速过滤非人脸窗口 (~50% 在第一阶段被过滤)
  - 后期阶段使用更多特征对候选窗口精细判断
  - 论文标准: 38 级，共 6060 个 Haar 特征

本模块实现:
  1. OpenCV 级联 XML 解析 → JIT 友好的扁平化数据结构
  2. Numba JIT 加速的级联评估
  3. 多尺度滑动窗口检测 + 非极大值抑制 (NMS)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import cv2
import numpy as np
from numba import jit, prange

from .integral_image import (
    compute_integral_image,
    compute_squared_integral_image,
    get_rect_sum,
    get_window_stats,
)


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class WeakClassifier:
    """弱分类器 —— 基于单个 Haar 特征的决策树桩。

    论文公式:
        h_j(x) = α_j · sign(f_j(x) − θ_j)
    其中 f_j 为 Haar 特征值, θ_j 为阈值, α_j 为权重（编码在 leaf 值中）。
    """
    feature_idx: int      # 使用的 Haar 特征索引
    threshold: float      # 判决阈值 θ_j
    left_val: float       # 特征值 < 阈值时的输出 (实质是带符号的权重)
    right_val: float      # 特征值 ≥ 阈值时的输出


@dataclass
class StageClassifier:
    """一级强分类器 —— 由若干弱分类器加权投票组成。

    论文公式:
        H(x) = sign( Σ α_t h_t(x) − stage_threshold )
    通过调整 stage_threshold 可平衡检出率与误检率。
    """
    threshold: float                # 该级阈值，累积和低于此值则拒绝
    weak_classifiers: list[WeakClassifier] = field(default_factory=list)


@dataclass
class CascadeParams:
    """扁平化的级联参数为 Numba JIT 优化。

    所有数据以 NumPy 数组形式存储，消除 Python 对象开销，

    Attributes:
        win_w, win_h: 训练窗口尺寸（24×24）

        feature_rects:     shape=(total_rects, 5), 每行 [x, y, w, h, weight]
        feature_offsets:   shape=(n_features,), 第 i 个特征在 feature_rects 中的起始行
        feature_n_rects:   shape=(n_features,), 第 i 个特征包含几个矩形

        wc_feature_idx:    shape=(total_wc,), 每个弱分类器使用的特征索引
        wc_thresholds:     shape=(total_wc,), 每个弱分类器的阈值
        wc_left_vals:      shape=(total_wc,), 特征值 < 阈值时的输出
        wc_right_vals:     shape=(total_wc,), 特征值 ≥ 阈值时的输出

        stage_wc_start:    shape=(n_stages,), 每级弱分类器的起始索引
        stage_wc_count:    shape=(n_stages,), 每级弱分类器的数量
        stage_thresholds:  shape=(n_stages,), 每级的累积和阈值
    """
    win_w: int
    win_h: int

    feature_rects: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    feature_offsets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    feature_n_rects: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    wc_feature_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    wc_thresholds: np.ndarray = field(default_factory=lambda: np.empty(0))
    wc_left_vals: np.ndarray = field(default_factory=lambda: np.empty(0))
    wc_right_vals: np.ndarray = field(default_factory=lambda: np.empty(0))

    stage_wc_start: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    stage_wc_count: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    stage_thresholds: np.ndarray = field(default_factory=lambda: np.empty(0))


# ══════════════════════════════════════════════════════════════
# XML 解析
# ══════════════════════════════════════════════════════════════

def _find_cascade_root(xml_root: ET.Element) -> ET.Element:
    """在 XML 树中定位级联根节点。

    OpenCV cascade XML 格式:
        <opencv_storage>
          <cascade>           ← 目标节点
            <height>20</height>
            <width>20</width>
            <stages>...</stages>
            <features>...</features>
          </cascade>
        </opencv_storage>
    """
    for child in xml_root:
        tag = child.tag.lower()
        if 'cascade' in tag or 'haarcascade' in tag:
            return child
    raise ValueError("Cannot find cascade root in XML")


def parse_cascade_xml(xml_path: str) -> tuple[list[StageClassifier], tuple[int, int]]:
    """解析 OpenCV Haar Cascade XML 文件。
    提取级联的所有阶段、弱分类器、Haar 特征定义。
    XML 结构概要:
        <opencv_storage>
          <cascade_name>
            <size>24 24</size>
            <stages>
              <_>
                <stageThreshold>...</stageThreshold>
                <weakClassifiers>
                  <_>
                    <internalNodes> left right feat_idx threshold </internalNodes>
                    <leafValues>    left_val right_val              </leafValues>
                  </_>
                </weakClassifiers>
              </_>
            </stages>
            <features>
              <_>
                <rects>
                  <_> x y w h weight </_>
                </rects>
                <tilted>0</tilted>
              </_>
            </features>
          </cascade_name>
        </opencv_storage>

    Args:
        xml_path: OpenCV cascade XML 文件路径

    Returns:
        (stages, win_size): 级联阶段列表和训练窗口尺寸
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cascade_root = _find_cascade_root(root)

    # ── 窗口尺寸 ──
    size_elem = cascade_root.find('size')
    if size_elem is not None and size_elem.text is not None:
        size_parts = size_elem.text.strip().split()
        win_w, win_h = int(size_parts[0]), int(size_parts[1])
    else:
        w_elem = cascade_root.find('width')
        h_elem = cascade_root.find('height')
        if w_elem is None or h_elem is None:
            raise ValueError("Cannot find window size (width/height or size) in XML")
        win_w = int(w_elem.text.strip())
        win_h = int(h_elem.text.strip())

    # ── 解析特征 ──
    features_elem = cascade_root.find('features')
    if features_elem is None:
        raise ValueError("Cannot find <features> in XML")

    raw_features: list[list[tuple[float, float, float, float, float]]] = []
    for feat_elem in features_elem:
        rects_elem = feat_elem.find('rects')
        if rects_elem is None:
            raw_features.append([])
            continue
        rects: list[tuple[float, float, float, float, float]] = []
        for rect_elem in rects_elem:
            if rect_elem.text is None:
                continue
            vals = [float(v) for v in rect_elem.text.strip().split()]
            if len(vals) == 5:
                # vals: x, y, w, h, weight
                rects.append((vals[0], vals[1], vals[2], vals[3], vals[4]))
        raw_features.append(rects)

    # ── 解析阶段 ──
    stages_elem = cascade_root.find('stages')
    if stages_elem is None:
        raise ValueError("Cannot find <stages> in XML")

    stages: list[StageClassifier] = []
    for stage_elem in stages_elem:
        th_elem = stage_elem.find('stageThreshold')
        if th_elem is None or th_elem.text is None:
            continue
        stage_threshold = float(th_elem.text.strip())

        wc_container = stage_elem.find('weakClassifiers')
        if wc_container is None:
            continue

        wcs: list[WeakClassifier] = []
        for wc_elem in wc_container:
            internal = wc_elem.find('internalNodes')
            leaf = wc_elem.find('leafValues')
            if internal is None or leaf is None:
                continue
            if internal.text is None or leaf.text is None:
                continue

            internal_vals = [float(v) for v in internal.text.strip().split()]
            leaf_vals = [float(v) for v in leaf.text.strip().split()]

            if len(internal_vals) < 4 or len(leaf_vals) < 2:
                continue

            # internalNodes 格式: [left_idx, right_idx, feat_idx, threshold]
            # 这是 AdaBoost 训练出的决策树桩，每个弱分类器只含一个分裂节点
            # - left_idx=0  → 特征值 < threshold 时取 leafValues[0]
            # - right_idx=-1 → 特征值 >= threshold 时取 leafValues[1]（右分支用负数编码）
            # 两个 leaf value 可正可负，是 AdaBoost 训练的权重 alpha
            feature_idx = int(internal_vals[2])
            threshold = internal_vals[3]
            left_val = leaf_vals[0]
            right_val = leaf_vals[1]

            wcs.append(WeakClassifier(
                feature_idx=feature_idx,
                threshold=threshold,
                left_val=left_val,
                right_val=right_val,
            ))

        stages.append(StageClassifier(
            threshold=stage_threshold,
            weak_classifiers=wcs,
        ))

    print(f"[Cascade] 解析完成: {len(stages)} 级, {len(raw_features)} 个特征, "
          f"窗口 {win_w}×{win_h}")

    return stages, raw_features, (win_w, win_h)


# 扁平化转换 (Python 对象 → NumPy 数组, 供 Numba JIT 使用)

def flatten_cascade(
    stages: list[StageClassifier],
    raw_features: list[list[tuple[float, float, float, float, float]]],
    win_size: tuple[int, int],
) -> CascadeParams:
    """将级联参数从 Python 对象转换为 Numba 友好的扁平化数组。

    这一步是性能关键：Numba JIT 无法处理 Python 对象列表，
    但可以高效处理 NumPy 数组。扁平化后的数据可直接传入
    @jit 编译的检测函数。
    Args:
        stages: StageClassifier 列表
        raw_features: 从 XML 解析的原始特征数据
        win_size: (w, h) 训练窗口尺寸

    Returns:
        CascadeParams 扁平化参数
    """
    # ── 特征矩形展平 ──
    all_rects: list[list[float]] = []
    feature_offsets: list[int] = []
    feature_n_rects: list[int] = []

    for feat_rects in raw_features:
        feature_offsets.append(len(all_rects))
        feature_n_rects.append(len(feat_rects))
        for r in feat_rects:
            all_rects.append([r[0], r[1], r[2], r[3], r[4]])

    # ── 弱分类器展平 ──
    wc_feature_idx: list[int] = []
    wc_thresholds: list[float] = []
    wc_left_vals: list[float] = []
    wc_right_vals: list[float] = []
    stage_wc_start: list[int] = []
    stage_wc_count: list[int] = []
    stage_thresholds: list[float] = []

    for stage in stages:
        stage_wc_start.append(len(wc_feature_idx))
        stage_wc_count.append(len(stage.weak_classifiers))
        stage_thresholds.append(stage.threshold)

        for wc in stage.weak_classifiers:
            wc_feature_idx.append(wc.feature_idx)
            wc_thresholds.append(wc.threshold)
            wc_left_vals.append(wc.left_val)
            wc_right_vals.append(wc.right_val)

    n_stages = len(stages)
    total_wc = len(wc_feature_idx)
    total_rects = len(all_rects)

    print(f"[Cascade] 扁平化: {n_stages} stage(s), {total_wc} weak classifier(s), "
          f"{total_rects} rect(s)")

    return CascadeParams(
        win_w=win_size[0],
        win_h=win_size[1],
        feature_rects=np.array(all_rects, dtype=np.float64),
        feature_offsets=np.array(feature_offsets, dtype=np.int32),
        feature_n_rects=np.array(feature_n_rects, dtype=np.int32),
        wc_feature_idx=np.array(wc_feature_idx, dtype=np.int32),
        wc_thresholds=np.array(wc_thresholds, dtype=np.float64),
        wc_left_vals=np.array(wc_left_vals, dtype=np.float64),
        wc_right_vals=np.array(wc_right_vals, dtype=np.float64),
        stage_wc_start=np.array(stage_wc_start, dtype=np.int32),
        stage_wc_count=np.array(stage_wc_count, dtype=np.int32),
        stage_thresholds=np.array(stage_thresholds, dtype=np.float64),
    )



# Numba JIT 加速: 级联评估

@jit(nopython=True, cache=True, nogil=True)
def _cascade_classify_window(
    # 级联参数
    feature_rects: np.ndarray,
    feature_offsets: np.ndarray,
    feature_n_rects: np.ndarray,
    wc_feature_idx: np.ndarray,
    wc_thresholds: np.ndarray,
    wc_left_vals: np.ndarray,
    wc_right_vals: np.ndarray,
    stage_wc_start: np.ndarray,
    stage_wc_count: np.ndarray,
    stage_thresholds: np.ndarray,
    # 积分图 + 归一化
    ii: np.ndarray,
    sq_ii: np.ndarray,
    x: int, y: int,
    win_w: int, win_h: int,
    scale: float,
) -> bool:
    """级联评估 + 方差归一化 (Numba JIT 编译)。

    论文 Algorithm 1 — Attentional Cascade:
        for each stage s:
            stage_sum = Σ weak_classifier_output
            if stage_sum < stage_threshold[s]: return REJECT
        return ACCEPT

    特征缩放: 论文核心思路是 "缩放特征而非图像"。
    XML 中的特征坐标基于 base_window (如 20×20)，
    通过 scale = win_w / base_w 将矩形映射到当前检测窗口。

    Args:
        (级联参数): 扁平化数组
        ii, sq_ii: 积分图与平方积分图
        x, y: 窗口左上角坐标
        win_w, win_h: 当前检测窗口尺寸
        scale: 特征缩放因子 = win_w / base_w

    Returns:
        True 如果窗口通过所有阶段
    """
    area = float(win_w * win_h)
    inv_area = 1.0 / area

    # Step 1: 用积分图的 O(1) 统计函数计算窗口标准差
    mean, variance, std_dev = get_window_stats(ii, sq_ii, x, y, win_w, win_h)

    n_stages = stage_wc_start.shape[0]  # 级联共有几级

    # Step 2: 逐级评估 — 任何一级不通过就立即拒绝（级联的核心加速机制）
    for stage_idx in range(n_stages):
        stage_sum = 0.0  # 该级所有弱分类器的加权和
        wc_start = stage_wc_start[stage_idx]
        wc_end = wc_start + stage_wc_count[stage_idx]

        # Step 3: 评估该级的每个弱分类器（决策树桩）
        for wc_idx in range(wc_start, wc_end):
            feat_idx = wc_feature_idx[wc_idx]
            threshold = wc_thresholds[wc_idx]

            # 3a. 计算 Haar 特征值 = SUM(weight_i × rect_sum_i)
            feat_val = 0.0
            rect_start = feature_offsets[feat_idx]
            rect_end = rect_start + feature_n_rects[feat_idx]

            for rect_idx in range(rect_start, rect_end):
                rx, ry = feature_rects[rect_idx, 0], feature_rects[rect_idx, 1]
                rw, rh = feature_rects[rect_idx, 2], feature_rects[rect_idx, 3]
                rwgt = feature_rects[rect_idx, 4]

                # 矩形坐标 × scale → 映射到当前尺度的图像位置
                sx = x + int(rx * scale)
                sy = y + int(ry * scale)
                sw = int(rw * scale)
                sh = int(rh * scale)

                # 积分图 O(1) 矩形求和
                feat_val += rwgt * get_rect_sum(ii, sx, sy, sw, sh)

            # 3b. 方差归一化：消除光照差异
            norm_val = feat_val * inv_area  # inv_area = 1/(窗口面积)
            if std_dev > 0.0:
                norm_val /= std_dev

            # 3c. 弱分类器判决：特征值 vs 阈值，输出对应 leaf value
            if norm_val < threshold:
                stage_sum += wc_left_vals[wc_idx]
            else:
                stage_sum += wc_right_vals[wc_idx]

        # Step 4: 该级判决 — 累积和不够阈值 → 非人脸，立即拒绝
        if stage_sum < stage_thresholds[stage_idx]:
            return False

    # 通过全部阶段 → 判定为人脸
    return True


# Numba JIT 加速: 多尺度滑动窗口检测

@jit(nopython=True, cache=True, nogil=True)
def _detect_multiscale(
    feature_rects: np.ndarray,
    feature_offsets: np.ndarray,
    feature_n_rects: np.ndarray,
    wc_feature_idx: np.ndarray,
    wc_thresholds: np.ndarray,
    wc_left_vals: np.ndarray,
    wc_right_vals: np.ndarray,
    stage_wc_start: np.ndarray,
    stage_wc_count: np.ndarray,
    stage_thresholds: np.ndarray,
    gray: np.ndarray,
    base_w: int,
    base_h: int,
    scale_factor: float,
    min_size: int,
    max_results: int,
) -> np.ndarray:
    """多尺度滑动窗口检测 (Numba JIT 编译)。

    论文核心思想: 缩放特征矩形而非重采样图像。
    因为积分图使得任意尺寸的矩形求和都是 O(1)，
    所以按 scale_factor 逐步放大检测窗口，每次在原图上
    滑动放大后的窗口即可等价于多尺度检测。

    Args:
        (级联参数): 扁平化参数数组
        gray: 灰度图像
        base_w, base_h: 基础检测窗口尺寸 (来自 XML 的 width/height)
        scale_factor: 缩放因子 (论文推荐 1.25)
        min_size: 最小检测窗口边长
        max_results: 最大返回结果数

    Returns:
        shape=(N, 4) 数组, 每行 [x, y, w, h]
    """
    img_h, img_w = gray.shape

    # 一次性计算积分图 + 平方积分图，后续所有尺度的窗口共享
    ii = compute_integral_image(gray)
    sq_ii = compute_squared_integral_image(gray)

    # 预分配结果数组（Numba 要求固定大小）
    results = np.zeros((max_results, 4), dtype=np.int32)
    n_results = 0

    # ── 尺度金字塔 ──
    # 从基础窗口 (如 20x20) 开始，每次 ×1.25 放大
    # 论文思路：缩放矩形坐标，而非重采样图像
    cur_scale = 1.0

    while True:
        win_w = int(base_w * cur_scale)
        win_h = int(base_h * cur_scale)

        if win_w > img_w or win_h > img_h:
            break  # 窗口比图像还大，停止
        if win_w < min_size or win_h < min_size:
            cur_scale *= scale_factor
            continue  # 窗口太小，跳到下一级

        # 步长 = 1（最小尺度）→ scale（大尺度）
        # 大窗口冗余度高，跳步加速
        step = max(1, int(cur_scale * 1.0))

        for y in range(0, img_h - win_h, step):
            for x in range(0, img_w - win_w, step):
                if n_results >= max_results:
                    break

                if _cascade_classify_window(
                    feature_rects, feature_offsets, feature_n_rects,
                    wc_feature_idx, wc_thresholds,
                    wc_left_vals, wc_right_vals,
                    stage_wc_start, stage_wc_count, stage_thresholds,
                    ii, sq_ii,
                    x, y, win_w, win_h,
                    cur_scale,  # 特征缩放因子
                ):
                    results[n_results, 0] = x
                    results[n_results, 1] = y
                    results[n_results, 2] = win_w
                    results[n_results, 3] = win_h
                    n_results += 1

            if n_results >= max_results:
                break

        if n_results >= max_results:
            break

        cur_scale *= scale_factor

    return results[:n_results]


# 非极大值抑制 (NMS)
def non_max_suppression(
    boxes: np.ndarray,
    overlap_threshold: float = 0.3,
) -> np.ndarray:
    """非极大值抑制 —— 合并重叠检测框。

    滑动窗口检测会在同一个人脸附近产生多个略有偏移的检测框。
    NMS 按面积（越大越可信）排序后，贪心保留不重叠的框。

    Args:
        boxes: shape=(N, 4), [x, y, w, h]
        overlap_threshold: IoU 阈值, 超过则抑制

    Returns:
        shape=(M, 4) 合并后的检测框
    """
    if len(boxes) == 0:
        return boxes

    # 转为 [x1, y1, x2, y2] 便于交集计算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    # 按面积降序 — 大框更可信（经过了更多级联阶段的验证）
    order = np.argsort(-areas)
    keep: list[int] = []

    while len(order) > 0:
        i = order[0]           # 当前面积最大的框
        keep.append(i)          # 保留它

        if len(order) == 1:
            break

        # 计算剩余框与当前框的 IoU（交集/并集），IoU 大的视为重复
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)  # 交集宽度
        h = np.maximum(0.0, yy2 - yy1)  # 交集高度
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-8)  # 防止除零

        # IoU > 阈值 → 重复检测，丢弃
        inds = np.where(iou <= overlap_threshold)[0]
        order = order[inds + 1]           # +1 因为 order[0] 已被取出

    return boxes[keep]


# 公开 API: CascadeDetector
class CascadeDetector:
    """级联检测器。

    完整实现论文 "Robust Real-Time Face Detection" 的检测管线:
      1. 灰度化
      2. 直方图均衡化 (增强对比度)
      3. 积分图计算
      4. 多尺度滑动窗口 + 级联评估
      5. 非极大值抑制 (NMS)

    Usage:
        detector = CascadeDetector("haarcascade_frontalface_alt.xml")
        faces = detector.detect(gray_image)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    """

    def __init__(self, xml_path: str):
        # 解析 XML
        stages, raw_features, win_size = parse_cascade_xml(xml_path)

        # 扁平化 → Numba JIT 友好
        self.params = flatten_cascade(stages, raw_features, win_size)

        self._stages = stages  # 保留 Python 对象引用（调试用）
        self._loaded = True

    @property
    def n_stages(self) -> int:
        """级联的级数。"""
        return len(self.params.stage_thresholds)

    @property
    def n_features(self) -> int:
        """Haar 特征总数。"""
        return len(self.params.feature_offsets)

    @property
    def n_weak_classifiers(self) -> int:
        """弱分类器总数。"""
        return len(self.params.wc_feature_idx)

    def summary(self) -> str:
        """返回级联结构摘要。"""
        lines = [
            f"Viola-Jones Cascade Detector",
            f"  训练窗口: {self.params.win_w}×{self.params.win_h}",
            f"  级联级数: {self.n_stages}",
            f"  弱分类器: {self.n_weak_classifiers}",
            f"  Haar 特征: {self.n_features}",
            f"  矩形总数: {self.params.feature_rects.shape[0]}",
        ]
        for i, stage in enumerate(self._stages):
            lines.append(
                f"  Stage {i:2d}: {len(stage.weak_classifiers):4d} weak classifiers, "
                f"threshold={stage.threshold:.4f}"
            )
        return "\n".join(lines)

    def detect(
        self,
        gray: np.ndarray,
        scale_factor: float = 1.25,
        min_size: int = 30,
        overlap_threshold: float = 0.3,
        max_results: int = 5000,
    ) -> np.ndarray:
        """检测图像中的人脸。

        Args:
            gray: 灰度图像，uint8
            scale_factor: 缩放因子 (默认 1.25, 论文推荐)
            min_size: 最小检测窗口边长 (默认 30px)
            overlap_threshold: NMS IoU 阈值 (默认 0.3)
            max_results: 最大候选框数 (防止内存溢出)

        Returns:
            shape=(N, 4) 数组, 每行 [x, y, w, h]
        """
        # 预处理: 直方图均衡化增强对比度
        gray_eq = cv2.equalizeHist(gray)

        # 多尺度检测 (Numba JIT 编译)
        raw_boxes = _detect_multiscale(
            self.params.feature_rects,
            self.params.feature_offsets,
            self.params.feature_n_rects,
            self.params.wc_feature_idx,
            self.params.wc_thresholds,
            self.params.wc_left_vals,
            self.params.wc_right_vals,
            self.params.stage_wc_start,
            self.params.stage_wc_count,
            self.params.stage_thresholds,
            gray_eq,
            self.params.win_w,
            self.params.win_h,
            scale_factor,
            min_size,
            max_results,
        )

        # NMS 合并重叠框
        if len(raw_boxes) == 0:
            return raw_boxes

        filtered = non_max_suppression(raw_boxes, overlap_threshold)
        return filtered
