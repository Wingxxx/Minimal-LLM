# -*- coding: utf-8 -*-
"""词表与编码器（含特殊 token）单测：SPECIAL_TOKENS / load_corpus / encode / load_val

覆盖对象：train.SPECIAL_TOKENS / train.load_corpus / train.encode / train.load_val，
以及跨模块锁定：train.SPECIAL_TOKENS 与 build_corpus.POEM_TAGS 逐项一致、
train.load_corpus 产出的 token 流长度与 data/meta.npz 四数组长度一致。

口径：
  词表 = 「剥去诗体特殊 token 后的纯文本字符集」（按编码排序，id 0..8195）
        + 5 个特殊 token 追加在末尾（id 8196..8200），故 V = 8196 + 5 = 8201。
  文本中每个特殊 token（4 字符）在 token 流中计 1 个 token，故
  L = len(text) − 3 × 标签个数，与 build_corpus.build_meta 的数组长度口径一致。
用例读取真实 data/corpus.txt 与 data/meta.npz 作跨模块锁定（秒级），
不触发 build_corpus 的全量重建管线；不依赖任何测试框架，可直接
`python test/test_tokenizer.py` 运行（自带 ✓/✗ 汇总器），依赖仅为项目运行时依赖
numpy 与标准库，任一失败时进程以非零码退出。
"""
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))

import numpy as onp

import train
import build_corpus

CORPUS = os.path.join(BASE, "data", "corpus.txt")
VAL = os.path.join(BASE, "data", "val.txt")
META = os.path.join(BASE, "data", "meta.npz")

_PACK = None        # 真实语料加载结果缓存（惰性加载，失败时由各用例内部捕获）


def _corpus():
    """惰性加载真实语料（含特殊 token 的完整词表与 token 流），结果缓存复用。"""
    global _PACK
    if _PACK is None:
        _PACK = train.load_corpus(CORPUS)
    return _PACK


def _pure_text(text):
    """独立重算纯文本：逐处剥去全部诗体特殊 token（不依赖 train 内部实现）。"""
    for tok in build_corpus.POEM_TAGS:
        text = text.replace(tok, "")
    return text


def _tag_count(text):
    """统计文本中诗体特殊 token 的个数。"""
    return sum(text.count(tok) for tok in build_corpus.POEM_TAGS)


# ── Step 1：特殊 token 编码 + 纯文本字符 id 与旧词表一致 ──

def test_encode_special_token_first():
    """序列首 token 为 <五绝> 的特殊 id；纯文本字符 id 与「去标签纯文本字符集」下标一致。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    s = "<五绝>床前明月光，疑是地上霜。"           # 1 个特殊 token + 12 个纯文本字符
    ids = train.encode(s)                          # 省略 stoi：走默认词表路径
    assert ids.dtype == onp.int64, f"encode 应返回 int64，实为 {ids.dtype}"
    assert len(ids) == 1 + 12, f"token 数应为 13，实为 {len(ids)}"
    assert int(ids[0]) == special_ids[0], f"首 token 应为 <五绝> id={special_ids[0]}，实为 {int(ids[0])}"
    # 纯文本字符 id 与独立重算的旧词表（去标签纯文本字符集按编码排序）逐位一致
    old_chars = sorted(set(_pure_text(text)))
    for k, ch in enumerate(s[len(train.SPECIAL_TOKENS[0]):]):
        want = old_chars.index(ch)
        assert int(ids[1 + k]) == want, f"字符 {ch} id 应为 {want}，实为 {int(ids[1 + k])}"
        assert stoi[ch] == want, f"stoi[{ch}] 应等于旧词表下标 {want}"
    print("  <五绝> 首 token 为特殊 id，纯文本 id 与旧词表一致 ✓")


def test_encode_multiple_specials():
    """多个特殊 token 混排：逐个产出对应 id，特殊 token 之间的字符逐字编码。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    s = "<五绝>床<七律>明"
    ids = train.encode(s, stoi)
    want = [special_ids[0], stoi["床"], special_ids[3], stoi["明"]]
    assert list(ids) == want, f"混排编码应为 {want}，实为 {list(ids)}"
    print("  多特殊 token 混排逐个成 id ✓")


def test_encode_skips_oov():
    """词表外字符（OOV）跳过而非抛错（与验证集编码口径一致）。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    ids = train.encode("床\U0001F600明", stoi)     # 中间为 emoji，必为 OOV
    assert list(ids) == [stoi["床"], stoi["明"]], f"OOV 应跳过，实为 {list(ids)}"
    print("  词表外字符跳过 ✓")


def test_encode_boundaries():
    """边界用例：空文本、孤立 '<'（跳过）、以及无汉字间隔的连续特殊 token。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    empty = train.encode("", stoi)
    assert empty.dtype == onp.int64, f"空文本应返回 int64，实为 {empty.dtype}"
    assert list(empty) == [], f"空文本应得空数组，实为 {list(empty)}"
    assert list(train.encode("<", stoi)) == [], "孤立 '<' 应被跳过（词表外）"
    assert list(train.encode("床<", stoi)) == [stoi["床"]], "尾随孤立 '<' 应被跳过"
    ids = train.encode("<五绝><七绝>", stoi)        # 两标签直接相连，中间无汉字
    assert list(ids) == [special_ids[0], special_ids[1]], (
        f"连续标签应逐个成 id，实为 {list(ids)}")
    print("  空文本/孤立 '<'/连续标签边界 ✓")


# ── Step 3：词表构成（纯文本字符 + 末尾特殊 token）──

def test_vocab_size_and_pure_text():
    """len(stoi)=8201、len(chars)=8196；id 0..8195 对应去标签纯文本字符；< > 不入词表。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    assert len(stoi) == 8196 + 5, f"完整词表应为 8201，实为 {len(stoi)}"
    assert len(chars) == 8196, f"纯文本字符表应为 8196，实为 {len(chars)}"
    assert len(itos) == len(stoi), "itos 应与 stoi 等长"
    assert "<" not in stoi and ">" not in stoi, "< / > 不得作为单字符进入词表"
    old_chars = sorted(set(_pure_text(text)))
    assert list(chars) == old_chars, "chars 应为去标签纯文本字符集（按编码排序）"
    for i, c in enumerate(chars):
        assert stoi[c] == i, f"字符 {c} 的 id 应为 {i}，实为 {stoi[c]}"
        assert itos[i] == c, f"itos[{i}] 应为 {c}，实为 {itos[i]}"
    print(f"  词表 {len(stoi)} = 纯文本 {len(chars)} + 特殊 5 ✓")


def test_special_ids_appended():
    """5 个特殊 token 追加在末尾（id 8196..8200），顺序与 SPECIAL_TOKENS 逐项一致。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    assert special_ids == [8196, 8197, 8198, 8199, 8200], f"特殊 id 应为 8196..8200，实为 {special_ids}"
    for i, tok in enumerate(train.SPECIAL_TOKENS):
        assert stoi[tok] == special_ids[i], f"{tok} 的 id 应为 {special_ids[i]}"
        assert itos[special_ids[i]] == tok, f"itos[{special_ids[i]}] 应为 {tok}"
    print("  特殊 token 追加在末尾且可逆 ✓")


# ── Step 3b：跨模块锁定 + token 流长度一致 ──

def test_special_tokens_cross_module_locked():
    """SPECIAL_TOKENS 值锁定，且与 build_corpus.POEM_TAGS 逐项（含顺序）相等。"""
    want = ("<五绝>", "<七绝>", "<五律>", "<七律>", "<杂言>")
    assert train.SPECIAL_TOKENS == want, f"SPECIAL_TOKENS 值不符：{train.SPECIAL_TOKENS}"
    assert train.SPECIAL_TOKENS == build_corpus.POEM_TAGS, (
        f"与 POEM_TAGS 不一致：{train.SPECIAL_TOKENS} vs {build_corpus.POEM_TAGS}")
    print("  SPECIAL_TOKENS == POEM_TAGS（逐项含顺序）✓")


def test_data_len_matches_meta():
    """data 长度 == meta 四数组长度 == len(text) − 3×标签数（T3/T4 两套计数口径一致）。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    meta = onp.load(META)
    L = len(text) - 3 * _tag_count(text)
    for name in ("weights", "tone_labels", "rhyme_labels", "zone_tags"):
        assert len(meta[name]) == len(data), f"{name} 长度 {len(meta[name])} != len(data) {len(data)}"
    assert len(data) == L, f"data 长度应为 L={L}，实为 {len(data)}"
    n_special = int(onp.isin(data, onp.asarray(special_ids)).sum())
    assert n_special == _tag_count(text), f"data 中特殊 id 个数应等于标签数，实为 {n_special}"
    assert int(data[0]) in special_ids, "data 首 token 应为诗体特殊 id（语料以标签开头）"
    print(f"  len(data)={len(data)} == meta 四数组 == L ✓")


# ── load_corpus 返回结构与 load_val 口径 ──

def test_load_corpus_returns_six_tuple():
    """load_corpus 返回 6 元组，且各元素类型/语义符合规格。"""
    pack = _corpus()
    assert isinstance(pack, tuple) and len(pack) == 6, f"应为 6 元组，实为 {len(pack)}"
    text, chars, stoi, itos, data, special_ids = pack
    assert isinstance(text, str) and len(text) > 0
    assert isinstance(chars, list)
    assert isinstance(stoi, dict) and isinstance(itos, dict)
    assert data.dtype == onp.int64
    assert len(special_ids) == 5
    print("  load_corpus 返回 (text, chars, stoi, itos, data, special_ids) ✓")


def test_load_val_encodes_specials_and_skips_oov():
    """load_val 用训练词表编码：特殊 token 逐个成 id、OOV 跳过；文件缺失返回 None。"""
    text, chars, stoi, itos, data, special_ids = _corpus()
    assert train.load_val(os.path.join(BASE, "data", "__no_such_val__.txt"), stoi) is None
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("<五绝>床前明月光\U0001F600\n")     # 1 特殊 token + 5 汉字 + 1 OOV + 1 换行
        arr = train.load_val(path, stoi)
        assert arr.dtype == onp.int64, f"应为 int64，实为 {arr.dtype}"
        assert int(arr[0]) == special_ids[0], "首 token 应为 <五绝> 特殊 id"
        assert len(arr) == 7, f"1 特殊 + 5 汉字 + 换行，OOV 跳过 → 应为 7，实为 {len(arr)}"
        assert list(arr[1:6]) == [stoi[c] for c in "床前明月光"], "汉字应逐字编码"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("  load_val 特殊 token 逐个成 id、OOV 跳过 ✓")


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
    test_encode_special_token_first,
    test_encode_multiple_specials,
    test_encode_skips_oov,
    test_encode_boundaries,
    test_vocab_size_and_pure_text,
    test_special_ids_appended,
    test_special_tokens_cross_module_locked,
    test_data_len_matches_meta,
    test_load_corpus_returns_six_tuple,
    test_load_val_encodes_specials_and_skips_oov,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部词表与编码器单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
