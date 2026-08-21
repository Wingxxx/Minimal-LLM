"""梯度检查：用有限差分（数值梯度）验证手写 backward（解析梯度）。

原理：
    解析梯度（手写 backward）和数值梯度（有限差分）是两条独立路径，
    若两者最大误差 < 1e-4，则证明 backward 实现正确。

    数值梯度公式（中心差分）：
        dL/dθ ≈ [L(θ + ε) − L(θ − ε)] / (2ε)
    其中标量 L = Σ(y · dy)，dy 为固定上游梯度。
"""

import os
import sys

import numpy as np

# 将项目根目录加入模块搜索路径，使 `import model.layers` 在任意目录下均可生效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.layers import Linear, LayerNorm, GELU


def finite_diff(param, layer, x, dy):
    """用中心差分计算 param 的数值梯度，形状与 param 一致。

    Args:
        param: 被微调的对象（可为层参数，也可为输入 x），函数内临时修改后会恢复
        layer: 被检查的层实例
        x:     前向输入
        dy:    固定上游梯度（标量 L = Σ(y·dy)，dL/dparam 即所求数值梯度）

    Returns:
        grad: 数值梯度，形状与 param 相同
    """
    eps = 1e-6                        # 中心差分的微调步长 ε
    grad = np.zeros_like(param)       # 梯度容器：与 param 同形状（形状铁律）

    # 逐个元素独立微调：每个元素对输出的影响不同，必须分别测量
    for idx in np.ndindex(param.shape):
        old = param[idx]

        # L(θ + ε)：该元素加 ε 后前向，输出与 dy 点积得标量
        param[idx] = old + eps
        lp = np.sum(layer.forward(x) * dy)

        # L(θ − ε)：该元素减 ε 后前向
        param[idx] = old - eps
        lm = np.sum(layer.forward(x) * dy)

        # 中心差分公式：[L(θ+ε) − L(θ−ε)] / (2ε)
        grad[idx] = (lp - lm) / (2 * eps)

        param[idx] = old              # 恢复原值，避免污染下一轮循环

    return grad


def check_derivative(analytic, numeric, name):
    """对比解析梯度与数值梯度，最大误差 < 1e-4 视为通过。

    Args:
        analytic: 解析梯度（backward 计算）
        numeric:  数值梯度（finite_diff 计算）
        name:    梯度名（如 'dW'），用于打印与报错提示

    Raises:
        AssertionError: 最大误差 >= 1e-4，说明 backward 实现有误
    """
    err = np.max(np.abs(analytic - numeric))
    print(f"  {name}: 解析 vs 数值 最大误差 = {err:.2e}")
    assert err < 1e-4, f"{name} 梯度检查失败: {err}"


def test_linear():
    """Linear 的 dW / db / dx 三组梯度检查。"""
    print("== Linear ==")
    np.random.seed(0)                       # 固定随机种子，保证结果可复现
    layer = Linear(4, 6)                    # 4 输入 → 6 输出
    x = np.random.randn(2, 3, 4)            # 输入 [B=2, T=3, d=4]
    dy = np.random.randn(2, 3, 6)           # 上游梯度 [B=2, T=3, out=6]

    layer.zero_grad()                       # 清账：backward 用 += 累加，先归零
    layer.forward(x)                        # 前向：缓存输入 x
    dx = layer.backward(dy)                 # 反向：dW/db 存入缓存，dx 为返回值

    # 三组对比：参数 W、b 用 finite_diff 微调；输入 x 同样按"参数"方式微调
    check_derivative(layer.dW, finite_diff(layer.W, layer, x, dy), "dW")
    check_derivative(layer.db, finite_diff(layer.b, layer, x, dy), "db")
    check_derivative(dx, finite_diff(x, layer, x, dy), "dx")


def test_layernorm():
    """LayerNorm 的 dgamma / dbeta / dx 三组梯度检查。"""
    print("== LayerNorm ==")
    np.random.seed(1)                       # 与 Linear 不同的种子，避免重复数据
    layer = LayerNorm(4)                    # 特征维 d=4
    x = np.random.randn(2, 3, 4)            # 输入 [B=2, T=3, d=4]
    dy = np.random.randn(2, 3, 4)           # 上游梯度，形状与输入一致

    layer.zero_grad()                       # 清账
    layer.forward(x)                        # 前向：缓存 mean/var/xhat
    dx = layer.backward(dy)                 # 反向：dgamma/dbeta 入缓存，dx 为返回值

    check_derivative(layer.dgamma, finite_diff(layer.gamma, layer, x, dy), "dgamma")
    check_derivative(layer.dbeta, finite_diff(layer.beta, layer, x, dy), "dbeta")
    check_derivative(dx, finite_diff(x, layer, x, dy), "dx")


def test_gelu():
    """GELU 无参数，只检查 dx（输入梯度）。"""
    print("== GELU ==")
    np.random.seed(2)
    layer = GELU()
    x = np.random.randn(2, 3, 4)            # 输入 [B=2, T=3, d=4]
    dy = np.random.randn(2, 3, 4)           # 上游梯度，形状与输入一致

    layer.forward(x)                         # 前向：缓存输入 x（GELU 无参数，无需 zero_grad）
    dx = layer.backward(dy)                  # 反向：返回输入梯度 dx

    # GELU 没有可学习参数，只检查 dx 一条
    check_derivative(dx, finite_diff(x, layer, x, dy), "dx")


if __name__ == '__main__':
    # 启动按钮：仅直接运行本文件时执行；依次跑三个梯度检查，任一 assert 失败即报错中断
    test_linear()
    test_layernorm()
    test_gelu()
    print("\n梯度检查全部通过：解析梯度与数值梯度一致，backward 实现正确。")
