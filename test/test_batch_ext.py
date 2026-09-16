# -*- coding: utf-8 -*-
"""逐位置元信息取批单测：train.get_batch_ext / train.derive_masks / train.load_meta

覆盖对象：
  - get_batch_ext(data, meta, batch_size, ctx_len, duizhang_weight)：返回
    7 元组 (x, y, w, tone_t, tone_m, rhyme_t, rhyme_m)，形状均 (B, T)；
  - derive_masks(tone_labels, rhyme_labels, weights)：派生辅助头掩码；
  - load_meta(path)：载入 data/meta.npz 为原生 numpy 数组字典。

掩码派生口径（死规则）：
  tone_mask  = (tone_labels != 2)
  rhyme_mask = (weights == 2.0) & (rhyme_labels != 0)

  韵脚掩码**禁止**由 zone_tags == 1 派生：zone_tags 优先级为
  「特殊 token(3) > 对仗区(4) > 韵脚(1) > 半句末标点(2) > 常态(0)」，
  颔联/颈联内的韵脚会被标记为 4 而丢失监督。故仅以「离线原始 weights == 2.0」
  作韵脚判据；且必须先派生掩码、再做 duizhang_weight 融合——融合后的
  w = maximum(weights, (zone_tags == 4) * duizhang_weight) 在 duizhang_weight == 2.0
  时会引入对仗区假韵脚。

对齐口径（死规则）：y/w/tone_t/rhyme_t 一律以 data 绝对位置 (s+1) 起、长度 T 切片；
  x 自 s 起、长度 T。起点 s 满足 s + 1 + T <= len(data)。

用例自包含、固定随机种子可复现；不依赖任何测试框架，可直接
`python test/test_batch_ext.py` 运行（自带 ✓/✗ 汇总器），任一失败时进程以非零码退出。
"""
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import numpy as onp

import train
from model.gpt import GPT

# 全量 data/meta.npz 实测锚点（T3 全量重建口径）
_ANCHOR_TONE = 2370966
_ANCHOR_RHYME = 254856


def _synthetic(n, seed=20260911):
    """构造长度 n 的合成 (data, meta)，取值覆盖全部掩码判定分支。

    取值约定（保证每个分支在样本内可命中）：
      weights     : {1.0, 1.5, 2.0}
      tone_labels : {0, 1, 2}
      rhyme_labels: {0, 1, 2, 3}
      zone_tags   : {0, 1, 4}
    """
    rng = onp.random.RandomState(seed)
    data = rng.randint(0, 50, size=n).astype(onp.int64)
    meta = {
        "weights": rng.choice(onp.array([1.0, 1.5, 2.0]), size=n).astype(onp.float32),
        "tone_labels": rng.randint(0, 3, size=n).astype(onp.int8),
        "rhyme_labels": rng.randint(0, 4, size=n).astype(onp.int16),
        "zone_tags": rng.choice(onp.array([0, 1, 4]), size=n).astype(onp.int8),
    }
    return data, meta


# ── Step 1：返回结构与形状 ──

def test_returns_septuple_with_shapes():
    """get_batch_ext 返回 7 元组，7 个数组形状均为 (B, T)。"""
    n, B, T = 200, 5, 16
    data, meta = _synthetic(n)
    out = train.get_batch_ext(data, meta, B, T)
    assert len(out) == 7, f"应返回 7 元组，实为 {len(out)} 元"
    for i, a in enumerate(out):
        arr = onp.asarray(a)
        assert arr.shape == (B, T), f"第 {i} 个数组形状应为 {(B, T)}，实为 {arr.shape}"


# ── Step 2：对齐硬校验（非同义反复：合成 len(data) == ctx_len + 1 使起点确定为 0）──

def test_alignment_hardcoded():
    """len(data) == ctx_len + 1 时起点恒为 0，逐元素比对手写期望值。"""
    T = 4
    data = onp.array([10, 11, 12, 13, 14], dtype=onp.int64)          # len = T + 1
    meta = {
        "weights": onp.array([1.0, 2.0, 1.5, 1.0, 2.0], dtype=onp.float32),
        "tone_labels": onp.array([2, 0, 1, 2, 0], dtype=onp.int8),
        "rhyme_labels": onp.array([0, 5, 0, 7, 3], dtype=onp.int16),
        "zone_tags": onp.array([4, 1, 4, 0, 1], dtype=onp.int8),
    }
    B = 3
    x, y, w, tone_t, tone_m, rhyme_t, rhyme_m = train.get_batch_ext(
        data, meta, B, T, duizhang_weight=1.0)
    x = onp.asarray(x); y = onp.asarray(y); w = onp.asarray(w)
    tone_t = onp.asarray(tone_t); tone_m = onp.asarray(tone_m)
    rhyme_t = onp.asarray(rhyme_t); rhyme_m = onp.asarray(rhyme_m)
    # 起点 s = 0：x = data[0:T]，y/w/tone_t/rhyme_t = 同一起点 s+1 起、长度 T
    exp_x = onp.tile(data[0:T], (B, 1))
    exp_y = onp.tile(data[1:1 + T], (B, 1))
    exp_tones = onp.tile(meta["tone_labels"][1:1 + T], (B, 1))
    exp_rhymes = onp.tile(meta["rhyme_labels"][1:1 + T], (B, 1))
    assert onp.array_equal(x, exp_x), f"x 错位：{x[0]} vs {exp_x[0]}"
    assert onp.array_equal(y, exp_y), f"y 错位：{y[0]} vs {exp_y[0]}"
    assert onp.array_equal(tone_t, exp_tones), f"tone_t 错位：{tone_t[0]} vs {exp_tones[0]}"
    assert onp.array_equal(rhyme_t, exp_rhymes), f"rhyme_t 错位：{rhyme_t[0]} vs {exp_rhymes[0]}"
    # tone_m = (tone_labels != 2)，取 s+1 起切片
    exp_tone_m = onp.tile(meta["tone_labels"][1:1 + T] != 2, (B, 1))
    assert onp.array_equal(tone_m, exp_tone_m), f"tone_m 错位：{tone_m[0]} vs {exp_tone_m[0]}"
    # w = maximum(weights, (zone_tags == 4) * duizhang_weight)，取 s+1 起切片
    w_raw = meta["weights"][1:1 + T]
    zone = meta["zone_tags"][1:1 + T]
    exp_w = onp.tile(onp.maximum(w_raw, (zone == 4) * 1.0), (B, 1))
    assert onp.allclose(w, exp_w), f"w 错位：{w[0]} vs {exp_w[0]}"
    # rhyme_m = (weights_raw == 2.0) & (rhyme_labels != 0)，用**未融合**原始权重
    exp_rhyme_m = onp.tile((w_raw == 2.0) & (meta["rhyme_labels"][1:1 + T] != 0), (B, 1))
    assert onp.array_equal(rhyme_m, exp_rhyme_m), f"rhyme_m 错位：{rhyme_m[0]} vs {exp_rhyme_m[0]}"


# ── Step 3：tone_mask 派生 ──

def test_tone_mask_derivation():
    """tone_labels == 2（未覆盖/非汉字）→ False；0/1 → True。"""
    tone = onp.array([0, 1, 2, 2, 0], dtype=onp.int8)
    rhyme = onp.zeros(5, dtype=onp.int16)
    weights = onp.ones(5, dtype=onp.float32)
    tone_m, _ = train.derive_masks(tone, rhyme, weights)
    assert onp.array_equal(tone_m, onp.array([True, True, False, False, True])), tone_m
    assert tone_m.dtype == onp.bool_, f"tone_mask 应为 bool，实为 {tone_m.dtype}"


# ── Step 4：rhyme_mask 核心回归（禁止 zone_tags 派生）──

def test_rhyme_mask_ignores_zone_tags():
    """weights==2.0 & zone_tags==4 & rhyme_labels!=0 仍为 True（不被对仗区优先级遮蔽）。"""
    # 位置 0：weight 2.0、zone 4（对仗区，优先级高于韵脚）、rhyme 非 0 → 应判为韵脚
    weights = onp.array([2.0, 2.0, 1.0], dtype=onp.float32)
    rhyme = onp.array([9, 0, 5], dtype=onp.int16)
    tone = onp.zeros(3, dtype=onp.int8)
    _, rhyme_m = train.derive_masks(tone, rhyme, weights)
    assert rhyme_m[0] == True, "weight==2.0 的真韵脚被误判为非韵脚（疑似 zone_tags 派生）"
    # 位置 1：weight==2.0 但 rhyme_labels==0（未覆盖字）→ False
    assert rhyme_m[1] == False, "rhyme_labels==0 的未覆盖韵脚应跳过"
    # 位置 2：rhyme_labels!=0 但 weight==1.0（普通汉字）→ False
    assert rhyme_m[2] == False, "普通汉字不应判为韵脚"
    assert rhyme_m.dtype == onp.bool_, f"rhyme_mask 应为 bool，实为 {rhyme_m.dtype}"


# ── Step 5：对仗区权重融合取最大（不叠乘）──

def test_duizhang_fusion_takes_max():
    """zone_tags==4 处 w == maximum(weights, duizhang_weight)，不叠乘、不压低原权重。"""
    T = 4
    data = onp.array([1, 2, 3, 4, 5], dtype=onp.int64)               # len = T + 1 → s = 0
    meta = {
        "weights": onp.array([1.0, 1.0, 2.0, 1.5, 1.0], dtype=onp.float32),
        "tone_labels": onp.zeros(5, dtype=onp.int8),
        "rhyme_labels": onp.zeros(5, dtype=onp.int16),
        # 取 s+1 起切片 = [zone[1], zone[2], zone[3], zone[4]]
        "zone_tags": onp.array([0, 4, 4, 0, 4], dtype=onp.int8),
    }
    # 切片后 w_raw = [1.0, 2.0, 1.5, 1.0]，zone = [4, 4, 0, 4]
    _, _, w, _, _, _, _ = train.get_batch_ext(data, meta, 1, T, duizhang_weight=1.5)
    w = onp.asarray(w)[0]
    exp = onp.array([max(1.0, 1.5), max(2.0, 1.5), 1.5, max(1.0, 1.5)])   # [1.5, 2.0, 1.5, 1.5]
    assert onp.allclose(w, exp), f"对仗融合取最大失败：{w} vs {exp}"
    # 不叠乘：weight==2.0 处仍为 2.0（若叠乘则 3.0）
    assert abs(float(w[1]) - 2.0) < 1e-6, f"weight==2.0 处被叠乘为 {w[1]}"
    # duizhang_weight=1.0 时不低于原权重
    _, _, w1, _, _, _, _ = train.get_batch_ext(data, meta, 1, T, duizhang_weight=1.0)
    w1 = onp.asarray(w1)[0]
    assert onp.all(w1 >= onp.asarray(meta["weights"])[1:1 + T] - 1e-6), w1


# ── Step 5b：duizhang_weight==2.0 时韵脚掩码的顺序判别 ──

def test_rhyme_mask_uses_unfused_weights_at_dz2():
    """duizhang_weight==2.0 时韵脚掩码仍取自未融合权重，不被融合后的 2.0 误导。

    构造「zone_tags==4 且原始 weights==1.0 且 rhyme_labels!=0」的位置：
      - 先派生掩码后融合（本实现）：rhyme_m 由原始 weights==1.0 派生 → False；
      - 先融合后派生（错误口径）：融合后 w==maximum(1.0, 2.0)==2.0 命中
        weights==2.0 判据 → 误判 True。
    duizhang_weight 取 1.0/1.5 时两种顺序结果相同（融合值低于 2.0，不触发假韵脚），
    唯有 2.0 能区分二者，故本用例专测该取值。len(data) == ctx_len + 1 使起点唯一
    为 0，期望值可手写，避免同义反复。
    """
    T = 4
    data = onp.array([10, 11, 12, 13, 14], dtype=onp.int64)          # len = T + 1 → s = 0
    meta = {
        # 切片 [1:1+T]=[1:5]：切片下标 2（绝对位置 3）同时满足 zone==4、weights==1.0、rhyme!=0
        "weights": onp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=onp.float32),
        "tone_labels": onp.zeros(5, dtype=onp.int8),
        "rhyme_labels": onp.array([0, 0, 0, 5, 0], dtype=onp.int16),
        "zone_tags": onp.array([0, 0, 0, 4, 0], dtype=onp.int8),
    }
    B = 2
    _, _, w, _, _, rhyme_t, rhyme_m = train.get_batch_ext(
        data, meta, B, T, duizhang_weight=2.0)
    w = onp.asarray(w)
    rhyme_t = onp.asarray(rhyme_t)
    rhyme_m = onp.asarray(rhyme_m)
    # 融合后该位置 w == maximum(1.0, 2.0) == 2.0：误取融合后权重将命中 weights==2.0 判据
    assert abs(float(w[0][2]) - 2.0) < 1e-6, f"该位置融合权重应为 2.0，实为 {w[0][2]}"
    assert int(rhyme_t[0][2]) == 5, f"该位置韵部标签应为 5，实为 {rhyme_t[0][2]}"
    # 死规则：韵脚掩码取自未融合 weights(==1.0)，该位置必须为 False
    assert bool(rhyme_m[0][2]) is False, (
        "duizhang_weight==2.0 时韵脚掩码被融合后的 2.0 误导（疑似先融合后派生）")
    assert not bool(rhyme_m.any()), f"本合成数据不含真韵脚，rhyme_m 应全 False，实为 {rhyme_m}"


# ── Step 6：长度不一致断言 ──

def test_length_mismatch_raises():
    """meta 数组与 data 不等长时 get_batch_ext 抛 AssertionError。"""
    data = onp.arange(50, dtype=onp.int64)
    meta = {"weights": onp.ones(49, dtype=onp.float32),
            "tone_labels": onp.zeros(49, dtype=onp.int8),
            "rhyme_labels": onp.zeros(49, dtype=onp.int16),
            "zone_tags": onp.zeros(49, dtype=onp.int8)}
    try:
        train.get_batch_ext(data, meta, 2, 8)
    except AssertionError:
        return
    raise AssertionError("data 与 meta 不等长时应抛 AssertionError")


# ── Step 7：真实数据锚点 ──

def test_real_meta_anchor_counts():
    """全量 data/meta.npz：tone_mask.sum()==2370966 且 rhyme_mask.sum()==254856。"""
    meta = train.load_meta()
    assert len(meta["weights"]) == len(meta["tone_labels"]) == \
        len(meta["rhyme_labels"]) == len(meta["zone_tags"]), "meta 四数组长度应一致"
    tone_m, rhyme_m = train.derive_masks(
        meta["tone_labels"], meta["rhyme_labels"], meta["weights"])
    tsum = int(tone_m.sum())
    rsum = int(rhyme_m.sum())
    assert tsum == _ANCHOR_TONE, f"tone_mask.sum() 应为 {_ANCHOR_TONE}，实为 {tsum}"
    assert rsum == _ANCHOR_RHYME, f"rhyme_mask.sum() 应为 {_ANCHOR_RHYME}，实为 {rsum}"
    print(f"  真实锚点：tone_mask={tsum}，rhyme_mask={rsum}")


# ── Step 8：同 seed 可复现 ──

def test_same_seed_reproducible():
    """同 seed 两次调用返回的 7 元组逐元素相等。"""
    n, B, T = 300, 6, 12
    data, meta = _synthetic(n)
    onp.random.seed(4321)
    a = train.get_batch_ext(data, meta, B, T, duizhang_weight=1.5)
    onp.random.seed(4321)
    b = train.get_batch_ext(data, meta, B, T, duizhang_weight=1.5)
    for i, (x, y) in enumerate(zip(a, b)):
        assert onp.array_equal(onp.asarray(x), onp.asarray(y)), f"第 {i} 个数组不可复现"


# ── Step 9：load_params_partial 轻量单测（跨里程碑按名载参）──

_GPT_KW = dict(d_model=16, n_head=2, n_layer=1, ctx_len=16)


def test_load_params_partial_counts_missing():
    """load_params_partial：缺失数 == 模型有而档内无的参数个数，下划线键不计入。"""
    onp.random.seed(20260911)
    g = GPT(vocab_size=20, **_GPT_KW)
    names = [n for n, _ in g._named_params()]
    dropped = names[:2]                              # 故意抽走 2 个参数键制造缺失
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "partial.npz")
        d = {k: onp.asarray(v) for k, v in g.dump_params().items()}
        for n in dropped:
            d.pop(n)
        # 混入下划线元数据键：本函数只读非 "_" 前缀键，这些键不得被当作模型参数名
        d["_chars"] = onp.asarray(list("床前明月光"))
        d["_opt.t"] = onp.asarray(7)
        d["_sched.peak_lr"] = onp.asarray(3e-4)
        onp.savez(path, **d)
        target = GPT(vocab_size=20, **_GPT_KW)
        missing = train.load_params_partial(path, target)
        assert missing == len(dropped), f"缺失数应为 {len(dropped)}，实为 {missing}"
        src, dst = g.dump_params(), target.dump_params()
        for k in names:                              # 共有参数应逐名一致
            if k in dropped:
                continue
            assert onp.allclose(onp.asarray(src[k]), onp.asarray(dst[k])), f"共有参数 {k} 未载入"


def test_load_params_partial_cross_milestone():
    """跨里程碑：只有旧参数 + rhyme_head.* 的档载入带 tone_head 的模型不抛错，missing==2。"""
    onp.random.seed(20260911)
    src = GPT(vocab_size=20, n_rhyme=107, **_GPT_KW)     # 旧档：有 rhyme_head、无 tone_head
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m1.npz")
        onp.savez(path, **{k: onp.asarray(v) for k, v in src.dump_params().items()})
        target = GPT(vocab_size=20, n_rhyme=107, use_tone=True, **_GPT_KW)
        missing = train.load_params_partial(path, target)  # 严禁 KeyError：新增头保留初始化
        assert missing == 2, f"应仅缺 tone_head.W/b 共 2 个，实为 {missing}"
        s, t = src.dump_params(), target.dump_params()
        for k in s:
            assert onp.allclose(onp.asarray(s[k]), onp.asarray(t[k])), f"共有参数 {k} 未载入"


def test_load_params_partial_shape_mismatch():
    """同名但形状不符 → AssertionError（防静默错位）。"""
    onp.random.seed(20260911)
    g = GPT(vocab_size=20, **_GPT_KW)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "badshape.npz")
        d = {k: onp.asarray(v) for k, v in g.dump_params().items()}
        d["tok_emb"] = onp.zeros((21, 16), dtype=onp.float32)   # 同名，形状 (21,16) != (20,16)
        onp.savez(path, **d)
        try:
            train.load_params_partial(path, GPT(vocab_size=20, **_GPT_KW))
        except AssertionError:
            return
        raise AssertionError("同名参数形状不符时应抛 AssertionError")


def _run(fn):
    """执行单项测试：断言失败或被测接口缺失均记为 ✗，返回是否通过。"""
    try:
        fn()
        print(f"✓ {fn.__name__}")
        return True
    except Exception as e:                       # 含 AssertionError 与接口缺失（AttributeError 等）
        print(f"✗ {fn.__name__}: {type(e).__name__}: {e}")
        return False


TESTS = (
    test_returns_septuple_with_shapes,
    test_alignment_hardcoded,
    test_tone_mask_derivation,
    test_rhyme_mask_ignores_zone_tags,
    test_duizhang_fusion_takes_max,
    test_rhyme_mask_uses_unfused_weights_at_dz2,
    test_length_mismatch_raises,
    test_real_meta_anchor_counts,
    test_same_seed_reproducible,
    test_load_params_partial_counts_missing,
    test_load_params_partial_cross_milestone,
    test_load_params_partial_shape_mismatch,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部 get_batch_ext 单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
