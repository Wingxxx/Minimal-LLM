"""MHA 测试：梯度检查 + 形状测试（KV Cache 一致性见 test_kv_cache.py）。

原理（与 test_layers.py 相同）：
    解析梯度（backward）与数值梯度（中心差分）是两条独立路径，
    若两者最大误差 < 1e-4，则证明 backward 实现正确。
    标量损失 L = Σ(y · dy)，dy 为固定上游梯度，dL/dθ 即所求梯度。

    数值梯度公式（中心差分）：
        dL/dθ ≈ [L(θ + ε) − L(θ − ε)] / (2ε)

测试内容：
    1. 梯度检查：输入 x 与 4 个投影层的 dW、db（共 9 组）
    2. 形状测试：forward 输出与 backward 返回的 dx 均与输入同形 [B,T,d]
    3. KV Cache 一致性：直接调用 test_kv_cache.py 中的测试
"""

import os
import sys

import numpy as np

# 将项目根目录加入模块搜索路径，使 `import model.attention` 在任意目录下均可生效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.attention import MultiHeadAttention
import test_kv_cache

EPS, TOL = 1e-6, 1e-4


def numeric_grad(f, param):
    """用中心差分计算 param 的数值梯度，形状与 param 一致。

    Args:
        f:     无参闭包，调用后返回标量损失 L = Σ(forward(x) · dy)
        param: 被微调的 numpy 数组；函数内临时 ±ε，结束后恢复原值
    """
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        old = param[idx]
        param[idx] = old + EPS          # L(θ + ε)
        lp = f()
        param[idx] = old - EPS          # L(θ − ε)
        lm = f()
        grad[idx] = (lp - lm) / (2 * EPS)  # 中心差分公式
        param[idx] = old                # 恢复原值，避免污染后续计算
    return grad


def check_derivative(analytic, numeric, name):
    """对比解析梯度与数值梯度，最大误差 < 1e-4 视为通过。"""
    err = np.max(np.abs(analytic - numeric))
    print(f"  {name}: 解析 vs 数值 最大误差 = {err:.2e}")
    assert err < TOL, f"{name} 梯度检查失败: {err}"


def test_mha_grad():
    """MHA backward 的 dx / dW / db 共 9 组梯度检查。"""
    print("== MHA 梯度检查 ==")
    np.random.seed(0)
    mha = MultiHeadAttention(d_model=8, n_head=2)   # d=8, 2 头, d_k=4
    x = np.random.randn(2, 3, 8) * 0.5              # 输入 [B=2, T=3, d=8]
    dy = np.random.randn(2, 3, 8)                   # 固定上游梯度

    def loss():
        # 标量损失 L：闭包读取 x 与当前权重（numeric_grad 微调的对象）
        return np.sum(mha.forward(x) * dy)

    # 输入 x 的梯度：backward 返回的 dx vs 数值梯度
    mha.zero_grad(); mha.forward(x); dx = mha.backward(dy)
    check_derivative(dx, numeric_grad(loss, x), "dx")

    # 4 个投影层的 dW / db：每组先清账 → 前向 → 反向，再与数值梯度对比
    for lname in ["Wq", "Wk", "Wv", "Wo"]:
        layer = getattr(mha, lname)
        mha.zero_grad(); mha.forward(x); mha.backward(dy)
        check_derivative(layer.dW, numeric_grad(loss, layer.W), f"d{lname}.W")
        check_derivative(layer.db, numeric_grad(loss, layer.b), f"d{lname}.b")


def test_shapes():
    """形状测试：forward 输出与 backward 返回的 dx 都与输入同形。"""
    print("== MHA 形状测试 ==")
    np.random.seed(1)
    for d_model, n_head in [(4, 1), (8, 2), (8, 4)]:
        mha = MultiHeadAttention(d_model, n_head)
        x = np.random.randn(2, 5, d_model)          # 输入 [B=2, T=5, d]
        y = mha.forward(x)
        assert y.shape == x.shape, f"d={d_model},n_head={n_head}: y 形状 {y.shape} != {x.shape}"
        dx = mha.backward(np.random.randn(2, 5, d_model))
        assert dx.shape == x.shape, f"d={d_model},n_head={n_head}: dx 形状 {dx.shape} != {x.shape}"
    print("  forward/backward 输入输出同形 [B,T,d]: PASS")


def test_gqa_kv_consistency():
    """GQA（n_head=4, n_kv_head=2）：KV Cache 一致性 + 梯度检查。

    验证 K/V 分组共享（组内两个 Q 头共用一个 KV）在缓存与反向传播下均正确。
    """
    print("== GQA（n_kv_head=2）测试 ==")
    np.random.seed(2)
    mha = MultiHeadAttention(d_model=16, n_head=4, n_kv_head=2)  # d_k=4，缓存只存 2 份 KV
    x_full = np.random.randn(1, 5, 16) * 0.5

    # 1. KV Cache 一致性：增量（带缓存）vs 全量（无缓存）
    y_full = mha.forward(x_full)
    cache = [np.zeros((1, 2, 0, 4)), np.zeros((1, 2, 0, 4))]   # n_kv_head=2 → 只存 2 份
    outs = [mha.forward(x_full[:, t:t + 1, :], cache) for t in range(5)]
    y_step = np.concatenate(outs, axis=1)
    err_cache = np.abs(y_full - y_step).max()
    assert err_cache < 1e-9, f"GQA KV Cache 不一致: {err_cache}"
    print(f"  KV Cache 一致性: 全量 vs 增量 最大误差 = {err_cache:.2e}")

    # 2. 梯度检查：dx（覆盖 _expand_kv 反向的「复制求和」分支，reps=2）
    dy = np.random.randn(1, 5, 16)
    mha.zero_grad(); mha.forward(x_full); dx = mha.backward(dy)
    err_grad = np.abs(dx - numeric_grad(lambda: np.sum(mha.forward(x_full) * dy), x_full)).max()
    assert err_grad < TOL, f"GQA 梯度检查失败: {err_grad}"
    print(f"  dx 梯度检查: 解析 vs 数值 最大误差 = {err_grad:.2e}")


if __name__ == '__main__':
    test_mha_grad()
    test_shapes()
    test_kv_cache.test_kv_cache_consistency()
    test_gqa_kv_consistency()
    print("\nALL ATTENTION TESTS PASS")
