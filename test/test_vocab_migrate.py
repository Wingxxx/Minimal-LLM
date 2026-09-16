# -*- coding: utf-8 -*-
"""扩词表迁移单测：tools/migrate_vocab.migrate 的逐元素继承、新行/新列全零与数值等价

覆盖对象：migrate_vocab.migrate（旧档 model.npz，V0=8196 → 新档，V=V0+5=8201）。
口径：
  新词表 = 旧「剥标签纯文本字符表」(8196) + train.SPECIAL_TOKENS(5)，故 V = 8196 + 5 = 8201。
  继承采用「扩行/扩列 + 新行/新列严格全零」：零初始化下纯文本前向永不索引新 token、
  lm_head 新列恒输出 0，故新模型 logits 前 V0 列与旧模型逐元素精确相等（无容差）。
用例在临时目录内调用 migrate 产出临时新档再断言，不依赖 model-ext.npz 预先存在；
不依赖任何测试框架，可直接 `python test/test_vocab_migrate.py` 运行（自带 ✓/✗ 汇总器），
依赖仅为项目运行时依赖 numpy 与标准库，任一失败时进程以非零码退出。
"""
import os
import shutil
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))

import numpy as onp

import train
from model.gpt import GPT
import migrate_vocab

OLD = os.path.join(BASE, "model.npz")

_OLD_D = None       # 旧档全量字典缓存（惰性加载，避免多用例重复读盘）
_NEW_D = None       # 迁移产物全量字典缓存
_TMP = None         # 托管迁移产物的临时目录
_NEW_PATH = None    # 迁移产物路径
_INFO = None        # migrate 返回的摘要
_OLD_M = None       # 旧模型（vocab=V0）缓存
_NEW_M = None       # 新模型（vocab=V）缓存


def _old():
    """惰性加载旧档 model.npz 为 {键: 数组} 字典（含 _chars 等元数据）。"""
    global _OLD_D
    if _OLD_D is None:
        _OLD_D = dict(onp.load(OLD))
    return _OLD_D


def _v0():
    """旧词表大小：由旧档 tok_emb 行数实测推导。"""
    return int(_old()["tok_emb"].shape[0])


def _migrated():
    """在临时目录内执行一次迁移，返回 (产物路径, migrate 摘要)。"""
    global _TMP, _NEW_PATH, _INFO
    if _NEW_PATH is None:
        _TMP = tempfile.mkdtemp(prefix="vocab_mig_")
        _NEW_PATH = os.path.join(_TMP, "model-ext.npz")
        _INFO = migrate_vocab.migrate(OLD, _NEW_PATH)
    return _NEW_PATH, _INFO


def _new():
    """惰性加载迁移产物为字典。"""
    global _NEW_D
    if _NEW_D is None:
        _NEW_D = dict(onp.load(_migrated()[0]))
    return _NEW_D


def _old_model():
    """构造旧模型（vocab=V0）并载入旧档权重（去掉 "_" 前缀元数据键）。"""
    global _OLD_M
    if _OLD_M is None:
        d = _old()
        onp.random.seed(1234)                       # 固定初始化随机源（权重随后被整体覆盖）
        m = GPT(vocab_size=_v0(), d_model=256, n_head=8, n_layer=6, ctx_len=128)
        m.load_params({k: v for k, v in d.items() if not k.startswith("_")})
        _OLD_M = m
    return _OLD_M


def _new_model():
    """构造新模型（vocab=V）并载入迁移产物。"""
    global _NEW_M
    if _NEW_M is None:
        info = _migrated()[1]
        onp.random.seed(1234)                       # 与旧模型同 seed，保证初始化口径一致
        m = GPT(vocab_size=info["V"], d_model=info["d_model"], n_head=info["n_head"],
                n_layer=info["n_layer"], ctx_len=info["ctx_len"])
        m.load(_migrated()[0])
        _NEW_M = m
    return _NEW_M


# ── Step 1：维度推导 + 词表来源校验 ──

def test_migrate_reports_target_arch():
    """摘要应给出 V0=8196→V=8201，且实测架构 d256/h8/L6/ctx128 与目标一致。"""
    _, info = _migrated()
    assert info["V0"] == 8196, f"V0 应为 8196，实为 {info['V0']}"
    assert info["V"] == 8196 + 5, f"V 应为 8201，实为 {info['V']}"
    arch = (info["d_model"], info["n_head"], info["n_layer"], info["ctx_len"])
    assert arch == (256, 8, 6, 128), f"架构应为 d256/h8/L6/ctx128，实为 {arch}"


def test_vocab_source_matches_corpus():
    """纯文本词表来自 train.load_corpus().chars（8196），且旧档 _chars 与之逐项相等。"""
    chars = train.load_corpus()[1]
    assert len(chars) == 8196, f"纯文本字符表应为 8196，实为 {len(chars)}"
    assert list(_old()["_chars"]) == list(chars), "旧档 _chars 应与语料 chars 逐项（含顺序）一致"
    want = ["<五绝>", "<七绝>", "<五律>", "<七律>", "<杂言>"]
    assert list(train.SPECIAL_TOKENS) == want, f"SPECIAL_TOKENS 值不符：{train.SPECIAL_TOKENS}"


def test_shapes():
    """迁移产物形状：tok_emb(V,d) / lm_head.W(d,V) / lm_head.b(V,) / pos_emb(ctx,d)。"""
    new = _new()
    V, d = _v0() + 5, 256
    assert new["tok_emb"].shape == (V, d), f"tok_emb 应为 {(V, d)}，实为 {new['tok_emb'].shape}"
    assert new["lm_head.W"].shape == (d, V), f"lm_head.W 应为 {(d, V)}，实为 {new['lm_head.W'].shape}"
    assert new["lm_head.b"].shape == (V,), f"lm_head.b 应为 {(V,)}，实为 {new['lm_head.b'].shape}"
    assert new["pos_emb"].shape == (128, d), f"pos_emb 应为 {(128, d)}，实为 {new['pos_emb'].shape}"


# ── Step 1：扩行/扩列继承 + 新行/新列全零 ──

def test_tok_emb_prefix_exact_and_new_rows_zero():
    """tok_emb 前 V0 行与旧档逐元素精确相等，后 5 行严格全零（非随机初始化）。"""
    old, new, V0 = _old(), _new(), _v0()
    assert onp.array_equal(new["tok_emb"][:V0], old["tok_emb"]), "tok_emb 前 V0 行应逐元素相等"
    assert onp.all(new["tok_emb"][V0:] == 0), "tok_emb 新行应严格全零"


def test_lm_head_W_prefix_exact_and_new_cols_zero():
    """lm_head.W 前 V0 列与旧档逐元素精确相等，后 5 列严格全零。"""
    old, new, V0 = _old(), _new(), _v0()
    assert onp.array_equal(new["lm_head.W"][:, :V0], old["lm_head.W"]), "lm_head.W 前 V0 列应逐元素相等"
    assert onp.all(new["lm_head.W"][:, V0:] == 0), "lm_head.W 新列应严格全零"


def test_lm_head_b_prefix_exact_and_new_tail_zero():
    """lm_head.b 前 V0 项与旧档逐元素精确相等，后 5 项严格全零。"""
    old, new, V0 = _old(), _new(), _v0()
    assert onp.array_equal(new["lm_head.b"][:V0], old["lm_head.b"]), "lm_head.b 前 V0 项应逐元素相等"
    assert onp.all(new["lm_head.b"][V0:] == 0), "lm_head.b 新项应严格全零"


def test_shared_params_exact():
    """pos_emb / blocks.* / ln_f.* 与旧档按名逐元素精确相等（直接覆盖）。"""
    old, new = _old(), _new()
    names = ["pos_emb", "ln_f.gamma", "ln_f.beta"] + sorted(k for k in old if k.startswith("blocks."))
    assert len(names) == 3 + 96, f"共享参数名数应 99（pos_emb + 96 个 blocks.* + ln_f.gamma/beta），实为 {len(names)}"
    for name in names:
        assert onp.array_equal(new[name], old[name]), f"{name} 应与旧档逐元素相等"


# ── Step 2：数值等价（纯文本前向，逐元素无容差）──

def test_forward_numerical_equivalence():
    """同权重同输入：新模型 logits 前 V0 列与旧模型逐元素精确相等（纯文本 id 输入）。"""
    old_m, new_m = _old_model(), _new_model()
    chars = train.load_corpus()[1]
    ids = onp.array([[chars.index(c) for c in chars[:16]]], dtype=onp.int64)  # 纯文本 id，不含特殊 token
    V0 = _v0()
    onp.random.seed(1234)                           # 同 seed（前向无随机，显式固定以求可复现）
    old_logits = onp.asarray(old_m.forward(ids))
    new_logits = onp.asarray(new_m.forward(ids))
    assert new_logits.shape == (1, 16, V0 + 5), f"新模型 logits 形状应为 {(1, 16, V0 + 5)}，实为 {new_logits.shape}"
    assert onp.array_equal(old_logits, new_logits[:, :, :V0]), "前 V0 列 logits 应逐元素精确相等"


def test_forward_with_special_token_shape():
    """含特殊 token id（8196）的输入：形状正确且输出有限（不要求与旧模型等价）。"""
    new_m = _new_model()
    ids = onp.array([[8196, 0, 1, 2]], dtype=onp.int64)     # 首列为特殊 token id
    logits = onp.asarray(new_m.forward(ids))
    assert logits.shape == (1, 4, 8201), f"logits 形状应为 (1,4,8201)，实为 {logits.shape}"
    assert onp.all(onp.isfinite(logits)), "含特殊 token 的 logits 应全部有限"


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
    test_migrate_reports_target_arch,
    test_vocab_source_matches_corpus,
    test_shapes,
    test_tok_emb_prefix_exact_and_new_rows_zero,
    test_lm_head_W_prefix_exact_and_new_cols_zero,
    test_lm_head_b_prefix_exact_and_new_tail_zero,
    test_shared_params_exact,
    test_forward_numerical_equivalence,
    test_forward_with_special_token_shape,
)


def _cleanup():
    """清理迁移产物的临时目录（环境洁癖）。"""
    global _TMP
    if _TMP and os.path.isdir(_TMP):
        shutil.rmtree(_TMP, ignore_errors=True)
        _TMP = None


def main():
    try:
        passed = sum(_run(t) for t in TESTS)
        total = len(TESTS)
    finally:
        _cleanup()
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部扩词表迁移单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
