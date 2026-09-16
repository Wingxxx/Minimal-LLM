# -*- coding: utf-8 -*-
"""辅助头单测：GPT 的平仄头（tone）+ 韵部头（rhyme）及其掩码加权交叉熵。

覆盖对象：GPT.__init__ / GPT.forward / GPT.aux_loss / GPT.backward /
_named_params / _named_grads / zero_grad 中与辅助头相关的部分。

口径：
  辅助头复用末端隐藏态 h = ln_f(x)，各自做一次线性投影：
    tone_logits[B,T,2]  = tone_head(h)
    rhyme_logits[B,T,W] = rhyme_head(h)（W = n_rhyme）
  标签对齐口径为「第 i 个位置的隐藏态（predict 第 i+1 个 token）的属性」，
  由调用方按此口径生成；本函数只做按位置 mask 加权的交叉熵：
    L = -Σ(mask·log p[target]) / Σmask
  分母为 mask 之和（非位置数）；mask=0 的位置分子分母均不计。
  辅助损失不改变主 LM 分支的数值：三路梯度在 h 处相加后才进入 ln_f，且
  ln_f.backward 全程只被调用一次。

验证方式：
  Step 1 零回归：禁用辅助头时 _named_params/_named_grads 名字列表与旧版逐项相等、
    forward 输出与旧版逐位相等（同 seed），且不构造任何辅助 Linear。
  Step 2 启用行为：参数名/形状/顺序、forward 缓存形状、掩码加权损失与独立手算
    参考比对、全零 mask 与禁用态退化、仅平仄头与仅韵部头的单头配置、启用头时
    缺省 mask 的断言保护、含辅助头的 checkpoint 往返一致性。
  Step 3 数值梯度：小模型中心差分逐参数校验 tone_head/rhyme_head/ln_f/mlp/tok_emb
    的总损失（主损失 + 辅助损失）梯度，阈值 1e-6，并内置判别力对照。
  Step 4 结构：ln_f.backward 在全流程中恰好调用一次。
  用例自包含、固定随机种子可复现；不依赖任何测试框架，可直接
  `python test/test_aux_heads.py` 运行（自带 ✓/✗ 汇总器），任一失败时进程以非零码退出。
"""
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import numpy as onp

from model.gpt import GPT

_EPS = 1e-5        # 有限差分步长（中心差分）
_RTOL = 1e-6       # 数值/解析梯度相对误差阈值
_SEED = 20260911   # 固定随机种子，保证可复现


def _tiny_gpt(n_rhyme=0, use_tone=False):
    """构造可复现的小模型（vocab=7 / d=8 / 2 头 / 1 层 / ctx=8），CPU 秒级。"""
    onp.random.seed(_SEED)
    return GPT(vocab_size=7, d_model=8, n_head=2, n_layer=1, ctx_len=8,
               n_rhyme=n_rhyme, use_tone=use_tone)


def _legacy_names(n_layer=1):
    """旧版（禁用辅助头）参数名列表，顺序与旧实现逐项一致。"""
    names = ["tok_emb", "pos_emb"]
    for i in range(n_layer):
        names += [f"blocks.{i}.ln1.gamma", f"blocks.{i}.ln1.beta"]
        for w in ("Wq", "Wk", "Wv", "Wo"):
            names += [f"blocks.{i}.attn.{w}.W", f"blocks.{i}.attn.{w}.b"]
        names += [f"blocks.{i}.ln2.gamma", f"blocks.{i}.ln2.beta"]
        names += [f"blocks.{i}.mlp.fc1.W", f"blocks.{i}.mlp.fc1.b",
                  f"blocks.{i}.mlp.fc2.W", f"blocks.{i}.mlp.fc2.b"]
    names += ["ln_f.gamma", "ln_f.beta", "lm_head.W", "lm_head.b"]
    return names


def _log_softmax(logits):
    """数值稳定 log_softmax（独立重算，不依赖被测实现）。"""
    m = logits.max(axis=-1, keepdims=True)
    return logits - m - onp.log(onp.exp(logits - m).sum(axis=-1, keepdims=True))


def _ref_masked_ce(logits, target, mask):
    """手算参考：-Σ(mask·log p[target]) / Σmask，log p 由独立 log_softmax 得到。"""
    C = logits.shape[-1]
    logp = _log_softmax(logits).reshape(-1, C)
    t = target.reshape(-1)
    m = mask.reshape(-1)
    picked = logp[onp.arange(len(t)), t]
    return -(m * picked).sum() / m.sum()


# ── Step 1：零回归 ──

def test_zero_regression_named_params_exact():
    """禁用辅助头：参数/梯度名列表与旧版逐项相等，且不构造辅助 Linear。"""
    g = _tiny_gpt()
    exp = _legacy_names(1)
    got_p = [n for n, _ in g._named_params()]
    got_g = [n for n, _ in g._named_grads()]
    assert got_p == exp, f"参数名列表偏离旧版：\n{got_p}\n{exp}"
    assert got_g == exp, f"梯度名列表偏离旧版：\n{got_g}\n{exp}"
    assert g.tone_head is None and g.rhyme_head is None, "禁用时不应构造辅助头"


def test_zero_regression_forward_bitwise():
    """同 seed 下「默认参数」与「显式禁用」两模型 forward 逐位精确相等。"""
    onp.random.seed(7)
    g1 = GPT(7, 8, 2, 1, 8)
    onp.random.seed(7)
    g2 = GPT(7, 8, 2, 1, 8, n_rhyme=0, use_tone=False)
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    o1 = onp.asarray(g1.forward(idx))
    o2 = onp.asarray(g2.forward(idx))
    assert onp.array_equal(o1, o2), "禁用辅助头时 forward 输出应逐位相等"


# ── Step 2：启用行为 ──

def test_aux_heads_do_not_perturb_lm_path():
    """辅助头不回灌 LM 分支：共享参数同值时两者 logits 逐位相等。"""
    onp.random.seed(11)
    a = GPT(7, 8, 2, 1, 8)                       # 禁用
    onp.random.seed(11)
    b = GPT(7, 8, 2, 1, 8, n_rhyme=107, use_tone=True)   # 启用
    shared = a.dump_params()
    for name, p in b._named_params():
        if name in shared:                        # 只拷共享参数，辅助头保持自身初值
            p[...] = onp.asarray(shared[name])
    idx = onp.array([[1, 2, 3, 4, 5]], dtype=onp.int64)
    oa = onp.asarray(a.forward(idx))
    ob = onp.asarray(b.forward(idx))
    assert onp.array_equal(oa, ob), "启用辅助头后 LM 分支 logits 被扰动"


def test_enabled_param_names_and_shapes():
    """启用时参数名在 lm_head 后按序追加，形状为 (d,2)/(2,)/(d,W)/(W,)。"""
    g = _tiny_gpt(n_rhyme=107, use_tone=True)
    names = [n for n, _ in g._named_params()]
    exp_prefix = _legacy_names(1)
    assert names[:len(exp_prefix)] == exp_prefix, "旧参数名前缀顺序改变"
    assert names[len(exp_prefix):] == ["tone_head.W", "tone_head.b",
                                       "rhyme_head.W", "rhyme_head.b"], names
    assert [n for n, _ in g._named_grads()] == names, "梯度名与参数名不一致"
    d = dict(g._named_params())
    assert d["tone_head.W"].shape == (8, 2), d["tone_head.W"].shape
    assert d["tone_head.b"].shape == (2,), d["tone_head.b"].shape
    assert d["rhyme_head.W"].shape == (8, 107), d["rhyme_head.W"].shape
    assert d["rhyme_head.b"].shape == (107,), d["rhyme_head.b"].shape


def test_forward_aux_logits_shape():
    """启用时 forward 缓存 (B,T,2)/(B,T,W)；禁用时两者为 None。"""
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    g = _tiny_gpt(n_rhyme=107, use_tone=True)
    g.forward(idx)
    assert onp.asarray(g.tone_logits).shape == (1, 4, 2), onp.asarray(g.tone_logits).shape
    assert onp.asarray(g.rhyme_logits).shape == (1, 4, 107), onp.asarray(g.rhyme_logits).shape
    g2 = _tiny_gpt()
    g2.forward(idx)
    assert g2.tone_logits is None and g2.rhyme_logits is None, "禁用时辅助 logits 应为 None"


def test_aux_loss_masked_reference():
    """含部分 mask=0 的辅助损失与手算参考一致（容差 1e-10）。"""
    g = _tiny_gpt(n_rhyme=5, use_tone=True)
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    g.forward(idx)
    tone_t = onp.array([[0, 1, 0, 1]], dtype=onp.int64)
    tone_m = onp.array([[1.0, 0.0, 1.0, 1.0]])
    rhyme_t = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    rhyme_m = onp.array([[1.0, 1.0, 0.0, 0.0]])
    lam_t, lam_r = 0.3, 0.5
    L = float(g.aux_loss(tone_t, tone_m, rhyme_t, rhyme_m, lam_t, lam_r))
    ref = (lam_t * _ref_masked_ce(onp.asarray(g.tone_logits), tone_t, tone_m)
           + lam_r * _ref_masked_ce(onp.asarray(g.rhyme_logits), rhyme_t, rhyme_m))
    assert abs(L - ref) < 1e-10, f"辅助损失口径不符：实现 {L} vs 手算 {ref}"
    # 被 mask 位置的 logits 梯度应为 0
    td = onp.asarray(g.tone_dlogits)
    rd = onp.asarray(g.rhyme_dlogits)
    assert onp.all(td[0, 1, :] == 0.0), "tone mask=0 位置梯度非零"
    assert onp.all(rd[0, 2:, :] == 0.0), "rhyme mask=0 位置梯度非零"


def test_aux_loss_all_zero_mask():
    """mask 全 0：损失返回 0，辅助头梯度全 0，经 h 回传不产生 NaN。"""
    g = _tiny_gpt(n_rhyme=5, use_tone=True)
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    logits = onp.asarray(g.forward(idx)).copy()
    lm_t = onp.array([[2, 3, 4, 5]], dtype=onp.int64)
    g.loss(logits, lm_t)
    B, T = idx.shape
    zero_m = onp.zeros((B, T), dtype=onp.float64)
    L = float(g.aux_loss(onp.zeros((B, T), dtype=onp.int64), zero_m,
                         onp.zeros((B, T), dtype=onp.int64), zero_m))
    assert L == 0.0, f"全 0 mask 损失应为 0，实为 {L}"
    g.zero_grad()
    g.backward()
    for name in ("tone_head.W", "tone_head.b", "rhyme_head.W", "rhyme_head.b"):
        grad = onp.asarray(dict(g._named_grads())[name])
        assert onp.all(grad == 0.0), f"全 0 mask 时 {name} 梯度非零"
    assert onp.all(onp.isfinite(onp.asarray(g.d_tok_emb))), "经 h 回传出现 NaN"


def test_aux_loss_disabled_returns_zero():
    """两头皆关时 aux_loss 返回标量 0（不依赖任何标签）。"""
    g = _tiny_gpt()
    L = g.aux_loss()
    assert float(L) == 0.0, f"禁用时应返回标量 0，实为 {L}"


def test_tone_head_only_config():
    """仅启用平仄头（n_rhyme=0, use_tone=True）：前向、参数名、损失与反向梯度。"""
    g = _tiny_gpt(n_rhyme=0, use_tone=True)
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    g.forward(idx)
    assert onp.asarray(g.tone_logits).shape == (1, 4, 2), onp.asarray(g.tone_logits).shape
    assert g.rhyme_logits is None, "仅平仄头时 rhyme_logits 应为 None"
    names = [n for n, _ in g._named_params()]
    assert "tone_head.W" in names and "tone_head.b" in names, "缺平仄头参数"
    assert not any(n.startswith("rhyme_head.") for n in names), "仅平仄头时不应含韵部头参数"
    lm_t = onp.array([[2, 3, 4, 5]], dtype=onp.int64)
    tone_t = onp.array([[0, 1, 0, 1]], dtype=onp.int64)
    tone_m = onp.array([[1.0, 1.0, 0.0, 1.0]])
    g.loss(onp.asarray(g.logits), lm_t)
    L = float(g.aux_loss(tone_target=tone_t, tone_mask=tone_m))
    assert onp.isfinite(L), f"仅平仄头损失应为有限值，实为 {L}"
    g.zero_grad()
    g.backward()
    grads = dict(g._named_grads())
    gw = onp.asarray(grads["tone_head.W"])
    assert gw.shape == g.tone_head.W.shape, f"平仄头 W 梯度形状 {gw.shape}"
    assert onp.all(onp.isfinite(gw)), "平仄头 W 梯度应全为有限值"


def test_rhyme_head_only_config():
    """仅启用韵部头（n_rhyme=107, use_tone=False）——M1 里程碑的实际配置。"""
    g = _tiny_gpt(n_rhyme=107, use_tone=False)
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    g.forward(idx)
    assert onp.asarray(g.rhyme_logits).shape == (1, 4, 107), onp.asarray(g.rhyme_logits).shape
    assert g.tone_logits is None, "仅韵部头时 tone_logits 应为 None"
    names = [n for n, _ in g._named_params()]
    assert "rhyme_head.W" in names and "rhyme_head.b" in names, "缺韵部头参数"
    assert not any(n.startswith("tone_head.") for n in names), "仅韵部头时不应含平仄头参数"
    lm_t = onp.array([[2, 3, 4, 5]], dtype=onp.int64)
    rhyme_t = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    rhyme_m = onp.array([[1.0, 1.0, 0.0, 1.0]])
    g.loss(onp.asarray(g.logits), lm_t)
    L = float(g.aux_loss(rhyme_target=rhyme_t, rhyme_pos_mask=rhyme_m))
    assert onp.isfinite(L), f"仅韵部头损失应为有限值，实为 {L}"
    g.zero_grad()
    g.backward()
    grads = dict(g._named_grads())
    gw = onp.asarray(grads["rhyme_head.W"])
    assert gw.shape == g.rhyme_head.W.shape, f"韵部头 W 梯度形状 {gw.shape}"
    assert onp.all(onp.isfinite(gw)), "韵部头 W 梯度应全为有限值"


def test_aux_checkpoint_roundtrip():
    """含辅助头的 checkpoint 往返：dump_params 逐名逐元素一致，名字含两头。"""
    g1 = _tiny_gpt(n_rhyme=3, use_tone=True)
    g2 = _tiny_gpt(n_rhyme=3, use_tone=True)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "aux_heads_ckpt.npz")
    try:
        g1.save(path)
        g2.load(path)
        d1 = g1.dump_params()
        d2 = g2.dump_params()
        assert list(d1.keys()) == list(d2.keys()), "往返后参数名列表不一致"
        assert any(n.startswith("tone_head.") for n in d1), "名字列表应含平仄头参数"
        assert any(n.startswith("rhyme_head.") for n in d1), "名字列表应含韵部头参数"
        for name in d1:
            assert onp.array_equal(onp.asarray(d1[name]), onp.asarray(d2[name])), \
                f"参数 {name} 往返后不一致"
    finally:
        if os.path.exists(path):
            os.remove(path)
        os.rmdir(tmpdir)


def test_aux_loss_requires_mask_when_enabled():
    """启用头但缺省 mask 时应抛 AssertionError（锁定 I1 不被回归）。"""
    g = _tiny_gpt(n_rhyme=3, use_tone=True)
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    g.forward(idx)
    B, T = idx.shape
    tone_t = onp.zeros((B, T), dtype=onp.int64)
    rhyme_t = onp.zeros((B, T), dtype=onp.int64)
    ones_m = onp.ones((B, T), dtype=onp.float64)
    # 启用平仄头但缺 tone_mask
    try:
        g.aux_loss(tone_target=tone_t, rhyme_target=rhyme_t, rhyme_pos_mask=ones_m)
    except AssertionError:
        pass
    else:
        raise AssertionError("启用平仄头但缺 tone_mask 时应抛 AssertionError")
    # 启用韵部头但缺 rhyme_pos_mask
    try:
        g.aux_loss(tone_target=tone_t, tone_mask=ones_m, rhyme_target=rhyme_t)
    except AssertionError:
        pass
    else:
        raise AssertionError("启用韵部头但缺 rhyme_pos_mask 时应抛 AssertionError")


# ── Step 3：数值梯度 ──

def _total_loss(g, args):
    """总损失 = 主损失 + 辅助损失（用于有限差分）。"""
    idx, lm_t, tone_t, tone_m, rhyme_t, rhyme_m, lam_t, lam_r = args
    logits = onp.asarray(g.forward(idx))
    return float(g.loss(logits, lm_t)) + float(
        g.aux_loss(tone_t, tone_m, rhyme_t, rhyme_m, lam_t, lam_r))


def _numeric_grad(g, args, param, eps=_EPS):
    """中心差分逐元素数值梯度：对 param 每项 ±eps 扰动后重算总损失。"""
    num = onp.zeros_like(onp.asarray(param))
    flat = param.reshape(-1)
    nflat = num.reshape(-1)
    for k in range(flat.size):
        orig = flat[k]
        flat[k] = orig + eps
        lp = _total_loss(g, args)
        flat[k] = orig - eps
        lm = _total_loss(g, args)
        flat[k] = orig
        nflat[k] = (lp - lm) / (2 * eps)
    return num


def test_numerical_grad_full_stack():
    """有限差分校验辅助头、ln_f、mlp、tok_emb 的总损失梯度（阈值 1e-6）。"""
    g = GPT(7, 8, 2, 1, 8, n_rhyme=3, use_tone=True)
    idx = onp.array([[1, 2, 3]], dtype=onp.int64)
    lm_t = onp.array([[2, 3, 4]], dtype=onp.int64)
    tone_t = onp.array([[0, 1, 0]], dtype=onp.int64)
    tone_m = onp.array([[1.0, 1.0, 0.0]])
    rhyme_t = onp.array([[1, 2, 0]], dtype=onp.int64)
    rhyme_m = onp.array([[1.0, 0.0, 1.0]])
    lam_t, lam_r = 0.3, 0.5
    args = (idx, lm_t, tone_t, tone_m, rhyme_t, rhyme_m, lam_t, lam_r)
    # 解析梯度：真实 forward -> 主损失 + 辅助损失 -> backward
    logits = onp.asarray(g.forward(idx))
    g.loss(logits, lm_t)
    g.aux_loss(tone_t, tone_m, rhyme_t, rhyme_m, lam_t, lam_r)
    g.zero_grad()
    g.backward()
    ana = {n: onp.asarray(a).copy() for n, a in g._named_grads()}
    params = {"tone_head.W": g.tone_head.W, "tone_head.b": g.tone_head.b,
              "rhyme_head.W": g.rhyme_head.W, "rhyme_head.b": g.rhyme_head.b,
              "ln_f.gamma": g.ln_f.gamma, "ln_f.beta": g.ln_f.beta,
              "blocks.0.mlp.fc1.W": g.blocks[0].mlp.fc1.W, "tok_emb": g.tok_emb}
    maxrel = 0.0
    for name, p in params.items():
        num = _numeric_grad(g, args, p)
        a = ana[name]
        rel = onp.abs(num - a) / (onp.abs(num) + onp.abs(a) + 1e-8)
        m = float(rel.max())
        maxrel = max(maxrel, m)
        assert m < _RTOL, f"{name} 数值/解析梯度相对误差过大：{m:.3e}"
    # 判别力对照：解析梯度人为放大 1.5 倍后相对误差应显著超出阈值
    p0 = g.blocks[0].mlp.fc1.W
    num0 = _numeric_grad(g, args, p0)
    a0 = ana["blocks.0.mlp.fc1.W"]
    bad = float((onp.abs(num0 - a0 * 1.5) / (onp.abs(num0) + onp.abs(a0 * 1.5) + 1e-8)).max())
    assert bad > _RTOL, "数值梯度校验对错误梯度缺乏判别力"
    print(f"  最大相对误差 = {maxrel:.3e}（判别力对照 {bad:.3e}）")


# ── Step 4：结构约束 ──

def test_ln_f_backward_called_once():
    """主损失 + 辅助损失全流程中 ln_f.backward 恰好被调用 1 次。"""
    g = GPT(7, 8, 2, 1, 8, n_rhyme=5, use_tone=True)
    idx = onp.array([[1, 2, 3, 4]], dtype=onp.int64)
    lm_t = onp.array([[2, 3, 4, 5]], dtype=onp.int64)
    calls = {"n": 0}
    orig = g.ln_f.backward

    def _wrapped(dy):
        calls["n"] += 1
        return orig(dy)

    g.ln_f.backward = _wrapped
    try:
        logits = onp.asarray(g.forward(idx))
        g.loss(logits, lm_t)
        g.aux_loss(onp.zeros_like(lm_t), onp.ones(lm_t.shape, dtype=onp.float64),
                   onp.zeros_like(lm_t), onp.ones(lm_t.shape, dtype=onp.float64))
        g.zero_grad()
        g.backward()
    finally:
        g.ln_f.backward = orig
    assert calls["n"] == 1, f"ln_f.backward 调用次数应为 1，实为 {calls['n']}"


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
    test_zero_regression_named_params_exact,
    test_zero_regression_forward_bitwise,
    test_aux_heads_do_not_perturb_lm_path,
    test_enabled_param_names_and_shapes,
    test_forward_aux_logits_shape,
    test_aux_loss_masked_reference,
    test_aux_loss_all_zero_mask,
    test_aux_loss_disabled_returns_zero,
    test_tone_head_only_config,
    test_rhyme_head_only_config,
    test_aux_checkpoint_roundtrip,
    test_aux_loss_requires_mask_when_enabled,
    test_numerical_grad_full_stack,
    test_ln_f_backward_called_once,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("ALL AUX HEADS TESTS PASS")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
