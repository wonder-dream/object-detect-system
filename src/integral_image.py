"""积分图 (Integral Image) 模块

Haar 特征需要反复计算图像上任意矩形区域的像素和。如果每次都对矩形
内所有像素求和，一个 24×24 窗口在 640×480 图像上滑动，计算量不可接受。
积分图的核心思想：预处理一次 (O(W×H))，之后任意矩形的求和只需 O(1)。

II(x, y) = 原图中 (0,0) 到 (x,y) 矩形区域内所有像素值之和。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
矩形求和 (O(1), 仅 4 次查表 + 3 次加减)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
平方积分图: 用于方差归一化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True, nogil=True)
def compute_integral_image(img: np.ndarray) -> np.ndarray:
    """计算积分图 (Numba JIT 加速)。
    Args:
        img: 灰度图像 (h, w), uint8 或 float64

    Returns:
        积分图 (h+1, w+1), float64。第 0 行/列为哨兵 0，
        使 get_rect_sum 无需处理边界条件。
    """
    h, w = img.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)

    for y in range(h):
        row_sum = 0.0                     # 每行开始，行前缀和归零
        for x in range(w):
            row_sum += img[y, x]          # 当前行的前缀和（隐含消去了重复区域）
            # ii[y, x+1]: 上一行同列的积分值（上方区域已累积完毕）
            # + row_sum:  本行到 x 为止的前缀和
            ii[y + 1, x + 1] = ii[y, x + 1] + row_sum
    return ii


@jit(nopython=True, cache=True, nogil=True)
def compute_squared_integral_image(img: np.ndarray) -> np.ndarray:
    """计算平方积分图 —— 用于窗口方差归一化。

    与普通积分图算法区别：累加的是像素值平方 val²。
    同一张脸在明亮/昏暗环境下，Haar 特征值可能差几倍。
    除以窗口标准差 σ 后，特征值变得与光照无关，
    训练好的阈值 θ 才能在不同光照条件下通用。

    Args:
        img: 灰度图像 (h, w)

    Returns:
        平方积分图 (h+1, w+1), float64
    """
    h, w = img.shape
    sq_ii = np.zeros((h + 1, w + 1), dtype=np.float64)

    for y in range(h):
        row_sum = 0.0
        for x in range(w):
            val = img[y, x]
            row_sum += val * val            # val² 代替 val
            sq_ii[y + 1, x + 1] = sq_ii[y, x + 1] + row_sum
    return sq_ii


@jit(nopython=True, cache=True, nogil=True)
def get_rect_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """O(1) 求矩形区域像素和 —— 积分图的核心价值所在。
    Args:
        ii: 积分图 (H+1, W+1)
        x, y: 矩形左上角在原图中的坐标 (0-based)
        w, h: 矩形宽高 (像素数)

    Returns:
        矩形区域内像素值之和 (float64)
    """
    return ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]


@jit(nopython=True, cache=True, nogil=True)
def get_window_stats(
    ii: np.ndarray,
    sq_ii: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> tuple:
    """O(1) 计算窗口的均值、方差、标准差。
    设 N = w × h (像素总数)

    1. pixel_sum = get_rect_sum(ii,    x, y, w, h)    → Σp
    2. sq_sum    = get_rect_sum(sq_ii, x, y, w, h)    → Σ(p²)

    3. mean     = pixel_sum / N                         → E[X]
    4. variance = sq_sum/N - mean²                      → E[X²]−E[X]²
    5. std_dev  = √max(variance, 1.0)                   → σ (≥1.0)

    ── 在 Viola-Jones 级联中的用途 ──

    每个检测窗口先算 σ，然后对所有 Haar 特征值做归一化:
      norm_feature = (Σ weight_i × rect_sum_i) / (N × σ)
    这使特征值对光照不敏感，保证从 XML 加载的阈值 θ 能正常工作。

    Args:
        ii, sq_ii: 积分图与平方积分图
        x, y, w, h: 窗口在图像中的位置和尺寸

    Returns:
        (mean, variance, std_dev) 三元组
    """
    area = float(w * h)
    pixel_sum = get_rect_sum(ii, x, y, w, h)        # Σp，来自普通积分图
    sq_sum = get_rect_sum(sq_ii, x, y, w, h)         # Σ(p²)，来自平方积分图

    mean = pixel_sum / area                           # 像素均值
    # 方差 = 平方的期望 - 期望的平方
    variance = sq_sum / area - mean * mean
    if variance < 1.0:
        variance = 1.0                                # 截断，防除零
    std_dev = np.sqrt(variance)

    return mean, variance, std_dev
