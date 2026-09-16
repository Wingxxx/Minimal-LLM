# -*- coding: utf-8 -*-
"""加权主损失单测：GPT.loss(logits, targets, weights) 与对应 backward 的加权口径

覆盖对象：GPT.loss / GPT.backward（仅加权主损失部分）。
口径：
  加权交叉熵 loss = -Σ(w·log p) / Σw，分母是逐位置权重之和（非位置数 N）；
  weights[i] 与 targets[i] 一一对齐（预测第 i 个目标 token 的重要度），weight=0
  表示忽略该位置（分子分母均不计，其梯度贡献为 0）；weights=None 等价全 1 权重，
  此时 Σw = N，退化为旧实现的 -Σlogp/N（须逐位一致，零回归）。
  反向 dlogits = (probs - one-hot)·w / Σw。
验证方式：
  Step 1 与手算参考（独立重算 log_softmax 后加权求和）比对数值；整体缩放不变性锁定
  分母为 Σw；Step 2 用小模型 + 有限差分（中心差分）校验 lm_head.W / lm_head.b /
  tok_emb 的参数梯度，并用「改动被 mask 位置的 logits / target 不改变任何参数梯度」
  锁定 mask 语义。用例自包含、固定随机种子可复现；不依赖任何测试框架，可直接
  `python test/test_weighted_loss.py` 运行（自带 ✓/✗ 汇总器），任一失败时进程以非零码退出。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import numpy as onp

from model.gpt import GPT

_EPS = 1e-5        # 有限差分步长（中心差分）
_RTOL = 1e-4       # 梯度核对相对误差阈值（实测余量充足）


def _tiny_gpt():
    """构造可复现的小模型（vocab=7 / d=8 / 2 头 / 1 层 / ctx=8），CPU 秒级。"""
    onp.random.seed(20260911)
    return GPT(vocab_size=7, d_model=8, n_head=2, n_layer=1, ctx_len=8)


def _sample_logits_targets():
    """构造可复现的随机 logits[B,T,V] 与 targets[B,T]（V=7）。"""
    onp.random.seed(4242)
    logits = onp.random.randn(2, 4, 7).astype(onp.float64)
    targets = onp.random.randint(0, 7, size=(2, 4)).astype(onp.int64)
    return logits, targets


def _log_softmax(logits):
    """数值稳定 log_softmax（独立重算，不依赖被测实现）。"""
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - onp.log(onp.exp(logits - m).sum(axis=-1, keepdims=True))


def _ref_weighted_loss(logits, targets, weights):
    """手算参考：-Σ(w·log p[target]) / Σw，log p 由独立 log_softmax 得到。"""
    B, T, V = logits.shape
    logp = _log_softmax(logits).reshape(-1, V)     # [B*T, V]
    t = targets.reshape(-1)                        # [B*T]
    w = weights.reshape(-1)                        # [B*T]
    picked = logp[onp.arange(len(t)), t]           # 各位置真实 token 的 log 概率
    return -(w * picked).sum() / w.sum()


def _snapshot_grads(gpt):
    """快照全模型梯度为 {名字: 副本}（按 _named_grads 同名字同顺序）。"""
    return {name: onp.asarray(g).copy() for name, g in gpt._named_grads()}


def _numeric_grad(gpt, idx, targets, weights, param, eps=_EPS):
    """中心差分逐元素数值梯度：对 param 每项做 ±eps 扰动后重算 loss。

    每次扰动都重跑真实 forward 以取得该参数下的 logits，再算加权损失；
    与解析梯度（forward + loss + backward）不同，此处不触发反向传播。
    """
    num = onp.zeros_like(onp.asarray(param))
    flat = param.reshape(-1)
    nflat = num.reshape(-1)
    for k in range(flat.size):
        orig = flat[k]
        flat[k] = orig + eps
        lp = float(gpt.loss(gpt.forward(idx), targets, weights))
        flat[k] = orig - eps
        lm = float(gpt.loss(gpt.forward(idx), targets, weights))
        flat[k] = orig
        nflat[k] = (lp - lm) / (2 * eps)
    return num


# ── Step 1：加权损失数值口径 ──

def test_default_equals_all_ones():
    """weights=None 与显式全 1 权重逐位一致，且等于旧实现 -Σlogp/N 口径。"""
    gpt = _tiny_gpt()
    logits, targets = _sample_logits_targets()
    ones = onp.ones(targets.shape, dtype=onp.float64)
    l_none = float(gpt.loss(logits, targets))
    l_ones = float(gpt.loss(logits, targets, weights=ones))
    l_old = float(_ref_weighted_loss(logits, targets, ones))
    assert abs(l_none - l_ones) == 0.0, f"None 与全 1 应逐位一致：{l_none} vs {l_ones}"
    assert abs(l_none - l_old) < 1e-12, f"应与旧 mean 口径 -Σlogp/N 一致：{l_none} vs {l_old}"


def test_default_backward_equals_all_ones():
    """weights=None 与全 1 权重的参数梯度逐位一致（零回归：None 即旧均值口径的回传）。"""
    gpt = _tiny_gpt()
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    targets = onp.array([[5, 6, 0, 3]], dtype=onp.int64)
    logits = onp.asarray(gpt.forward(idx)).copy()
    gpt.loss(logits, targets)
    gpt.zero_grad()
    gpt.backward()
    g_none = _snapshot_grads(gpt)
    gpt.loss(logits, targets, weights=onp.ones(targets.shape, dtype=onp.float64))
    gpt.zero_grad()
    gpt.backward()
    g_ones = _snapshot_grads(gpt)
    for name in g_none:
        assert onp.array_equal(g_none[name], g_ones[name]), f"{name} 在 None 与全 1 下梯度应逐位一致"


def test_mask_and_nonuniform_match_reference():
    """含 0（mask）与非均匀权重的加权损失与手算 -Σ(w·logp)/Σw 一致。"""
    gpt = _tiny_gpt()
    logits, targets = _sample_logits_targets()
    weights = onp.array([[1.0, 0.0, 2.0, 0.0],
                         [1.5, 1.0, 0.0, 2.0]], dtype=onp.float64)
    L = float(gpt.loss(logits, targets, weights=weights))
    R = float(_ref_weighted_loss(logits, targets, weights))
    assert abs(L - R) < 1e-12, f"加权损失口径不符：实现 {L} vs 手算 {R}"


def test_uniform_scaling_invariant():
    """权重整体缩放不改变损失（分子分母同乘同一因子），锁定分母为 Σw 而非 N。"""
    gpt = _tiny_gpt()
    logits, targets = _sample_logits_targets()
    l1 = float(gpt.loss(logits, targets, weights=onp.full(targets.shape, 1.0)))
    l2 = float(gpt.loss(logits, targets, weights=onp.full(targets.shape, 2.0)))
    assert abs(l1 - l2) < 1e-12, f"整体缩放应不变（若分母为 N 则 l2≈2·l1）：{l1} vs {l2}"


# ── Step 2：反向加权口径 + 有限差分校验 ──

def test_finite_diff_parameter_grads():
    """有限差分校验 lm_head.W / lm_head.b / tok_emb 的参数梯度（含 mask 与非均匀权重）。"""
    gpt = _tiny_gpt()
    idx = onp.array([[1, 2, 3]], dtype=onp.int64)
    targets = onp.array([[4, 5, 0]], dtype=onp.int64)
    weights = onp.array([[1.0, 0.0, 2.0]], dtype=onp.float64)   # 位置 1 被 mask
    # 解析梯度：真实 forward -> 加权 loss -> backward
    gpt.loss(gpt.forward(idx), targets, weights)
    gpt.zero_grad()
    gpt.backward()
    ana = {"lm_head.W": onp.asarray(gpt.lm_head.dW).copy(),
           "lm_head.b": onp.asarray(gpt.lm_head.db).copy(),
           "tok_emb": onp.asarray(gpt.d_tok_emb).copy()}
    params = {"lm_head.W": gpt.lm_head.W, "lm_head.b": gpt.lm_head.b, "tok_emb": gpt.tok_emb}
    for name in ("lm_head.W", "lm_head.b", "tok_emb"):
        num = _numeric_grad(gpt, idx, targets, weights, params[name])
        rel = onp.abs(num - ana[name]) / (onp.abs(num) + onp.abs(ana[name]) + 1e-8)
        assert rel.max() < _RTOL, f"{name} 数值/解析梯度相对误差过大：{rel.max()}"


def test_masked_position_contributes_zero_gradient():
    """被 mask（权重 0）位置：改动其 logits 或 target 均不改变任何参数梯度与损失。"""
    gpt = _tiny_gpt()
    idx = onp.array([[1, 2, 3]], dtype=onp.int64)
    targets = onp.array([[4, 5, 0]], dtype=onp.int64)
    weights = onp.array([[1.0, 0.0, 2.0]], dtype=onp.float64)   # 位置 1 被 mask
    logits0 = onp.asarray(gpt.forward(idx)).copy()               # 缓存真实激活 + 基准 logits
    L0 = float(gpt.loss(logits0, targets, weights))
    gpt.zero_grad()
    gpt.backward()
    base = _snapshot_grads(gpt)
    # 对照 A：只改动被 mask 位置的 logits（该位置 dlogits 恒为 0，不应影响任何参数梯度）
    logits1 = logits0.copy()
    logits1[0, 1, :] += 0.7
    L1 = float(gpt.loss(logits1, targets, weights))
    gpt.zero_grad()
    gpt.backward()
    gA = _snapshot_grads(gpt)
    # 对照 B：只改动被 mask 位置的 target（分子分母均不计，不应影响任何参数梯度）
    targets2 = targets.copy()
    targets2[0, 1] = 6
    L2 = float(gpt.loss(logits0, targets2, weights))
    gpt.zero_grad()
    gpt.backward()
    gB = _snapshot_grads(gpt)
    assert L1 == L0 and L2 == L0, f"mask 位置改动不应改变损失：{L0} / {L1} / {L2}"
    for name in base:
        assert onp.array_equal(base[name], gA[name]), f"mask 位置 logits 改动影响了 {name} 梯度"
        assert onp.array_equal(base[name], gB[name]), f"mask 位置 target 改动影响了 {name} 梯度"


def _run(fn):
    """执行单项测试：断言失败或被测接口缺失均记为 ✗，返回是否通过。"""
    try:
        fn()
        print(f"✓ {fn.__name__}")
        return True
    except Exception as e:                       # 含 AssertionError 与接口缺失（TypeError 等）
        print(f"✗ {fn.__name__}: {type(e).__name__}: {e}")
        return False


TESTS = (
    test_default_equals_all_ones,
    test_default_backward_equals_all_ones,
    test_mask_and_nonuniform_match_reference,
    test_uniform_scaling_invariant,
    test_finite_diff_parameter_grads,
    test_masked_position_contributes_zero_gradient,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部加权主损失单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
