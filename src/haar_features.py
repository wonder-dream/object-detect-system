"""Haar-like 特征模块 

Haar 特征是用一组相邻矩形的像素和之差来描述图像的局部纹理。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
特征值公式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  feature_value = Σ weight_i × sum(rect_i)

  weight 正负交替，且按面积配比，使得:
    Σ weight_i × area_i = 0
  即对均匀图像（所有像素相同），特征值为 0。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
特征总数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在一个 24×24 窗口中：
  - 二矩形水平 : ~ 43,200 个
  - 二矩形垂直 : ~ 43,200 个
  - 三矩形水平 : ~ 27,600 个
  - 三矩形垂直 : ~ 27,600 个
  - 四矩形     : ~ 20,736 个
  ────────────────────────
  总计         : ~162,336 个

实际检测只用到 XML 中 AdaBoost 选出的少量特征。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class FeatureRect:
    """Haar 特征中的一个带权矩形。
    Attributes:
        x, y:   矩形左上角在检测窗口内的坐标 (整数，0 ~ 窗口尺寸-1)
        w, h:   矩形的宽度和高度 (像素)
        weight: 权重。正数表示"白色区域"(贡献为正)，
                负数表示"黑色区域"(贡献为负)。
                权重已按面积归一化，确保对均匀区域特征值为 0。
    """
    x: int
    y: int
    w: int
    h: int
    weight: float


@dataclass(slots=True)
class HaarFeature:
    """单个 Haar-like 特征 —— 由若干 FeatureRect 加权组成。
    Attributes:
        rects:        组成该特征的矩形列表
        feature_type: 类型标识字符串:
                      "2h" = 二矩形水平, "2v" = 二矩形垂直,
                      "3h" = 三矩形水平, "3v" = 三矩形垂直,
                      "4"  = 四矩形
    """
    rects: list[FeatureRect] = field(default_factory=list)
    feature_type: str = ""

    def __repr__(self) -> str:
        return f"HaarFeature(type={self.feature_type}, n_rects={len(self.rects)})"


# 特征模板生成
# 核心策略：在目标窗口内枚举所有合法的位置、尺寸组合。
# 以二矩形水平 (2h) 为例:
#   - 基础矩形尺寸: (w, h)，其中 1≤w≤window_w, 1≤h≤window_h
#   - 需要 2 个并排矩形 → 总宽度 2w ≤ window_w → w ≤ window_w/2
#   - 左上角位置: x ∈ [0, window_w-2w], y ∈ [0, window_h-h]

def generate_features(
    window_w: int = 24,
    window_h: int = 24,
) -> list[HaarFeature]:
    """生成窗口内所有可能的 Haar-like 特征模板。

    按"先定位置和尺寸，再判断类型是否合法"的顺序枚举。
    每个类型的循环结构相同，只是约束条件不同。

    ── 时间复杂度 ──
    四层嵌套 (x, y, w, h)，每层 O(window_dim)，总计 O(W²H²)。
    对于 24×24 约 162k 个特征

    Args:
        window_w, window_h: 检测窗口尺寸 24*24

    Returns:
        HaarFeature 列表，包含所有合法特征模板
    """
    features: list[HaarFeature] = []

    for y in range(window_h):
        for x in range(window_w):
            for h in range(1, window_h - y + 1):
                for w in range(1, window_w - x + 1):
                    if x + 2 * w <= window_w:                   # 两个矩形不能出界
                        r1 = FeatureRect(x, y, w, h, 1.0)       # 左白
                        r2 = FeatureRect(x + w, y, w, h, -1.0)  # 右黑
                        features.append(HaarFeature([r1, r2], "2h"))


    for y in range(window_h):
        for x in range(window_w):
            for h in range(1, window_h - y + 1):
                if y + 2 * h <= window_h:                       # 两个矩形不能出界
                    for w in range(1, window_w - x + 1):
                        r1 = FeatureRect(x, y, w, h, 1.0)       # 上白
                        r2 = FeatureRect(x, y + h, w, h, -1.0)  # 下黑
                        features.append(HaarFeature([r1, r2], "2v"))


    for y in range(window_h):
        for x in range(window_w):
            for h in range(1, window_h - y + 1):
                for w in range(1, window_w - x + 1):
                    if x + 3 * w <= window_w:                    # 三个矩形不能出界
                        r1 = FeatureRect(x, y, w, h, 1.0)
                        r2 = FeatureRect(x + w, y, w, h, -2.0)
                        r3 = FeatureRect(x + 2 * w, y, w, h, 1.0)
                        features.append(HaarFeature([r1, r2, r3], "3h"))

    # ── 三矩形垂直 (3v): 白-黑-白，各 w×h ──
    for y in range(window_h):
        for x in range(window_w):
            for h in range(1, window_h - y + 1):
                if y + 3 * h <= window_h:
                    for w in range(1, window_w - x + 1):
                        r1 = FeatureRect(x, y, w, h, 1.0)
                        r2 = FeatureRect(x, y + h, w, h, -2.0)
                        r3 = FeatureRect(x, y + 2 * h, w, h, 1.0)
                        features.append(HaarFeature([r1, r2, r3], "3v"))


    for y in range(window_h):
        for x in range(window_w):
            for h in range(1, window_h - y + 1):
                if y + 2 * h <= window_h:
                    for w in range(1, window_w - x + 1):
                        if x + 2 * w <= window_w:
                            r1 = FeatureRect(x, y, w, h, 1.0)          # 左上 白
                            r2 = FeatureRect(x + w, y, w, h, -1.0)     # 右上 黑
                            r3 = FeatureRect(x, y + h, w, h, -1.0)     # 左下 黑
                            r4 = FeatureRect(x + w, y + h, w, h, 1.0)  # 右下 白
                            features.append(HaarFeature([r1, r2, r3, r4], "4"))

    return features


def generate_features_fast(
    window_w: int = 24,
    window_h: int = 24,
) -> list[HaarFeature]:
    """快速生成特征模板 —— 优化循环顺序减少重复判断。
    与 generate_features 生成完全相同的特征集合，但通过把 (w,h)
    提到最外层来减少内层循环的边界检查次数。

    循环结构:
      for w, h (矩形尺寸):       ← 外层: 先定尺寸
        for x, y (左上角):       ← 内层: 遍历所有合法位置
          尝试 5 种类型，每种只做一个边界判断

    每个 (w,h,x,y) 组合一次判断所有 5 种类型，减少了大量重复的循环初始化开销。

    Args:
        window_w, window_h: 窗口尺寸

    Returns:
        HaarFeature 列表 (与 generate_features 输出等价)
    """
    features: list[HaarFeature] = []

    # 外层循环: 枚举矩形的基础尺寸 (w, h)
    for w in range(1, window_w + 1):
        for h in range(1, window_h + 1):
            # 内层循环: 枚举矩形左上角位置 (x, y)
            for x in range(window_w - w + 1):
                for y in range(window_h - h + 1):

                    # 二矩形水平: 需要 2w 宽度
                    if x + 2 * w <= window_w:
                        features.append(HaarFeature([
                            FeatureRect(x, y, w, h, 1.0),
                            FeatureRect(x + w, y, w, h, -1.0),
                        ], "2h"))

                    # 二矩形垂直: 需要 2h 高度
                    if y + 2 * h <= window_h:
                        features.append(HaarFeature([
                            FeatureRect(x, y, w, h, 1.0),
                            FeatureRect(x, y + h, w, h, -1.0),
                        ], "2v"))

                    # 三矩形水平: 需要 3w 宽度
                    if x + 3 * w <= window_w:
                        features.append(HaarFeature([
                            FeatureRect(x, y, w, h, 1.0),
                            FeatureRect(x + w, y, w, h, -2.0),
                            FeatureRect(x + 2 * w, y, w, h, 1.0),
                        ], "3h"))

                    # 三矩形垂直: 需要 3h 高度
                    if y + 3 * h <= window_h:
                        features.append(HaarFeature([
                            FeatureRect(x, y, w, h, 1.0),
                            FeatureRect(x, y + h, w, h, -2.0),
                            FeatureRect(x, y + 2 * h, w, h, 1.0),
                        ], "3v"))

                    # 四矩形: 需要 2w×2h
                    if x + 2 * w <= window_w and y + 2 * h <= window_h:
                        features.append(HaarFeature([
                            FeatureRect(x, y, w, h, 1.0),
                            FeatureRect(x + w, y, w, h, -1.0),
                            FeatureRect(x, y + h, w, h, -1.0),
                            FeatureRect(x + w, y + h, w, h, 1.0),
                        ], "4"))

    return features


# ════════════════════════════════════════════════════════════
# 特征值求值
# ════════════════════════════════════════════════════════════

def evaluate_feature(
    feature: HaarFeature,
    ii: "np.ndarray",
    x: int, y: int,
    scale: float = 1.0,
) -> float:
    """在积分图上 O(1) 计算单个 Haar 特征值（支持多尺度）。

    ── 核心原理 ──
    缩放特征而非图像策略的具体实现。
    在原图积分图上，缩放矩形坐标 → 等价于缩放了窗口

    ── 计算过程 ──
    对特征中的每个 FeatureRect:
      1. 坐标和尺寸乘以 scale → 映射到原图
      2. 在积分图上 O(1) 求矩形和
      3. 乘以 weight 累加

    Args:
        feature: 已定义好的 HaarFeature（矩形坐标基于检测窗口）
        ii:      积分图 (H+1, W+1)
        x, y:    检测窗口在原始图像中的左上角像素坐标
        scale:   当前缩放因子 (= 检测窗口尺寸 / 基础窗口尺寸)

    Returns:
        特征值 (float) = Σ weight_i × sum(scaled_rect_i)
    """
    import numpy as np
    from .integral_image import get_rect_sum

    total = 0.0
    for rect in feature.rects:
        # 1. 将基础窗口内的坐标缩放到原图坐标系
        rx = x + int(rect.x * scale)     # 原图 X 坐标
        ry = y + int(rect.y * scale)     # 原图 Y 坐标
        rw = int(rect.w * scale)         # 缩放后宽度
        rh = int(rect.h * scale)         # 缩放后高度

        # 2. 积分图 O(1) 求矩形和, 3. 乘以权重累加
        total += rect.weight * get_rect_sum(ii, rx, ry, rw, rh)

    return total
