# -*- coding: utf-8 -*-
"""评审器打分 score_candidate 单测（M3 段：句长 + 押韵 + 平仄 + 模型 logprob 加权）

覆盖对象：prosody.score_candidate（签名
    score_candidate(text, punct_ids=DEFAULT_PUNCT, pattern_len=None, logprob=None) -> float）。

打分口径（计划 §7.1 / §7.2，死规则）：
    score = W_LINE_LEN·句长合规率 + W_RHYME·押韵主指标 + W_TONE·平仄合规率 + W_LOGP·模型平均 logprob
    权重 W_LINE_LEN=1.0、W_RHYME=1.5、W_TONE=1.0、W_LOGP=0.5；logprob=None 时该分项记 0。
    句长分项取 check_line_len(text, pattern_len)；押韵分项取 check_rhyme(text).main；
    平仄分项取 check_tone(text, pattern_len).main。

样本用字均经 `prosody.tone_of / rhyme_of / check_*` 在终端实测确认（本文件内另以 check_* 前提自校验，
不使用臆断字音）；期望分值由各分项手工代入公式推算，非事后抄录程序输出。

纯标准库实现（不 import pytest），可直接 `python test/test_critic.py` 运行；
逐项打印 ✓/✗ 与汇总，任一失败时进程以非零码退出。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prosody

PUNCT = "，。！？；：、"

# 合律样本（五言，四半句各命中 A/B/C/D 之一，偶半句末字 霜/光 同下平七阳）：
#   白日依山尽 [1,1,0,0,1] = A；天高白玉霜 [0,0,1,1,0] = B；
#   春江明月雪 [0,0,0,1,1] = C；夜雪白霜光 [1,1,1,0,0] = D
# 分项：句长 1.0、押韵 1.0、平仄 1.0 → 1.0*1.0 + 1.5*1.0 + 1.0*1.0 = 3.5
GOOD = "白日依山尽，天高白玉霜。春江明月雪，夜雪白霜光。"
# 仅句长错：首半句「白日依山尽」→「白日依山」（4 字）
#   分项：句长 check_line_len=3/4=0.75、押韵 1.0、平仄 1.0 → 1.0*0.75 + 1.5*1.0 + 1.0*1.0 = 3.25
BAD_LEN = "白日依山，天高白玉霜。春江明月雪，夜雪白霜光。"
# 仅押韵错：末半句末字「光」(下平七阳) → 「天」(下平一先)，平仄不变、句长不变
#   分项：句长 1.0、押韵 0.0、平仄 1.0 → 1.0*1.0 + 1.5*0.0 + 1.0*1.0 = 2.0
BAD_RHYME = "白日依山尽，天高白玉霜。春江明月雪，夜雪白霜天。"
# 仅平仄错：首半句必论位（0 基下标 1/3/4）由 日(仄)山(平)尽(仄) 改 春(平)山(平)尽(仄)
#   = (0,0,1)，不匹配任何 A/B/C/D 的必论位组合 → 该半句失律，平仄 main=3/4=0.75
#   分项：句长 1.0、押韵 1.0、平仄 0.75 → 1.0*1.0 + 1.5*1.0 + 1.0*0.75 = 3.25
BAD_TONE = "白春依山尽，天高白玉霜。春江明月雪，夜雪白霜光。"


def test_critic_good_beats_bad_same_meter():
    """合律样本得分严格大于同诗体（同 pattern_len）的失律样本。"""
    s_good = prosody.score_candidate(GOOD, pattern_len=5)
    s_bad_len = prosody.score_candidate(BAD_LEN, pattern_len=5)
    s_bad_tone = prosody.score_candidate(BAD_TONE, pattern_len=5)
    # 手工推算：GOOD=3.5、BAD_LEN=3.25、BAD_TONE=3.25（分项构成见文件头）
    assert abs(s_good - 3.5) < 1e-9, f"合律样本应 3.5，实为 {s_good}"
    assert s_good > s_bad_len, f"合律 {s_good} 应 > 失律(句长) {s_bad_len}"
    assert s_good > s_bad_tone, f"合律 {s_good} 应 > 失律(平仄) {s_bad_tone}"
    print(f"  合律 {s_good} > 失律(句长) {s_bad_len}、失律(平仄) {s_bad_tone} ✓")


def test_critic_line_len_variant_lowers_line_len_component():
    """仅句长错：总分下降，且降幅精确等于句长分项降幅（对照 check_line_len 定位）。"""
    ll_good = prosody.check_line_len(GOOD, 5)
    ll_bad = prosody.check_line_len(BAD_LEN, 5)
    assert abs(ll_good - 1.0) < 1e-9 and abs(ll_bad - 0.75) < 1e-9, \
        f"前提：句长合规率应 1.0→0.75，实为 {ll_good}→{ll_bad}"
    # 对照：押韵分项与平仄分项在本变体中不变，证明总分下降只来自句长分项
    assert prosody.check_rhyme(BAD_LEN).main == prosody.check_rhyme(GOOD).main
    assert prosody.check_tone(BAD_LEN, 5).main == prosody.check_tone(GOOD, 5).main
    s_good = prosody.score_candidate(GOOD, pattern_len=5)
    s_bad = prosody.score_candidate(BAD_LEN, pattern_len=5)
    assert s_bad < s_good, f"句长变体应拉低总分，实为 {s_bad} vs {s_good}"
    assert abs((s_good - s_bad) - prosody.W_LINE_LEN * (ll_good - ll_bad)) < 1e-9, \
        f"总分降幅应等于 W_LINE_LEN*(Δ句长)={prosody.W_LINE_LEN * (ll_good - ll_bad)}，实为 {s_good - s_bad}"
    print(f"  句长分项 {ll_good}→{ll_bad} 拉低总分 {s_good}→{s_bad} ✓")


def test_critic_rhyme_variant_lowers_rhyme_component():
    """仅押韵错：总分下降，且 check_rhyme().main 下降（对照定位到押韵分项）。"""
    r_good = prosody.check_rhyme(GOOD).main
    r_bad = prosody.check_rhyme(BAD_RHYME).main
    assert abs(r_good - 1.0) < 1e-9 and abs(r_bad - 0.0) < 1e-9, \
        f"前提：押韵主指标应 1.0→0.0，实为 {r_good}→{r_bad}"
    assert r_bad < r_good, f"check_rhyme().main 应下降，实为 {r_good}→{r_bad}"
    # 对照：句长与平仄分项不变
    assert prosody.check_line_len(BAD_RHYME, 5) == prosody.check_line_len(GOOD, 5)
    assert prosody.check_tone(BAD_RHYME, 5).main == prosody.check_tone(GOOD, 5).main
    s_good = prosody.score_candidate(GOOD, pattern_len=5)
    s_bad = prosody.score_candidate(BAD_RHYME, pattern_len=5)
    assert s_bad < s_good, f"押韵变体应拉低总分，实为 {s_bad} vs {s_good}"
    assert abs((s_good - s_bad) - prosody.W_RHYME * (r_good - r_bad)) < 1e-9, \
        f"总分降幅应等于 W_RHYME*(Δ押韵)={prosody.W_RHYME * (r_good - r_bad)}，实为 {s_good - s_bad}"
    print(f"  押韵分项 {r_good}→{r_bad} 拉低总分 {s_good}→{s_bad} ✓")


def test_critic_tone_variant_lowers_tone_component():
    """仅平仄错（必论位改反、不改句长与押韵）：总分下降，且 check_tone().main 下降。"""
    t_good = prosody.check_tone(GOOD, 5).main
    t_bad = prosody.check_tone(BAD_TONE, 5).main
    assert abs(t_good - 1.0) < 1e-9 and abs(t_bad - 0.75) < 1e-9, \
        f"前提：平仄合规率应 1.0→0.75，实为 {t_good}→{t_bad}"
    assert t_bad < t_good, f"check_tone().main 应下降，实为 {t_good}→{t_bad}"
    # 对照：句长与押韵分项不变（证明平仄错未夹带其它分项变化）
    assert prosody.check_line_len(BAD_TONE, 5) == prosody.check_line_len(GOOD, 5)
    assert prosody.check_rhyme(BAD_TONE).main == prosody.check_rhyme(GOOD).main
    s_good = prosody.score_candidate(GOOD, pattern_len=5)
    s_bad = prosody.score_candidate(BAD_TONE, pattern_len=5)
    assert s_bad < s_good, f"平仄变体应拉低总分，实为 {s_bad} vs {s_good}"
    assert abs((s_good - s_bad) - prosody.W_TONE * (t_good - t_bad)) < 1e-9, \
        f"总分降幅应等于 W_TONE*(Δ平仄)={prosody.W_TONE * (t_good - t_bad)}，实为 {s_good - s_bad}"
    print(f"  平仄分项 {t_good}→{t_bad} 拉低总分 {s_good}→{s_bad} ✓")


def test_critic_logprob_three_forms():
    """logprob 三形态：None 记 0；浮点按 W_LOGP 线性计入；可调用对象等价于其返回值。"""
    base = prosody.score_candidate(GOOD, pattern_len=5, logprob=None)
    assert abs(base - 3.5) < 1e-9, f"logprob=None 时基准应 3.5，实为 {base}"
    # 浮点：-1.0 nats → 3.5 + 0.5*(-1.0) = 3.0
    s_float = prosody.score_candidate(GOOD, pattern_len=5, logprob=-1.0)
    assert abs(s_float - 3.0) < 1e-9, f"logprob=-1.0 应得 3.0，实为 {s_float}"
    assert abs(s_float - (base + prosody.W_LOGP * (-1.0))) < 1e-9
    # 可调用对象：f(text) 返回值须与直接传该浮点等价，且入参文本与打分文本一致
    seen = []

    def f(text):
        seen.append(text)
        return -2.0

    s_call = prosody.score_candidate(GOOD, pattern_len=5, logprob=f)
    assert seen == [GOOD], f"logprob 应传入打分文本本身，实收 {seen}"
    assert abs(s_call - prosody.score_candidate(GOOD, pattern_len=5, logprob=-2.0)) < 1e-9, \
        "可调用对象应等价于直接传入其返回值"
    assert abs(s_call - (base + prosody.W_LOGP * (-2.0))) < 1e-9, \
        f"logprob=-2.0 应得 {base - 1.0}，实为 {s_call}"
    print(f"  logprob None→{base}、float(-1)→{s_float}、callable(-2)→{s_call} ✓")


def test_critic_backward_compatible_default_pattern_len():
    """向后兼容：score_candidate(text) 不传新参数即可用，且等价于按首个半句推断句长的显式调用。"""
    s_default = prosody.score_candidate(GOOD)                      # 仅旧签名
    s_explicit = prosody.score_candidate(GOOD, pattern_len=5)      # 首个半句 5 字 → 推断 5
    assert abs(s_default - s_explicit) < 1e-9, \
        f"默认调用应等价于 pattern_len=5，实为 {s_default} vs {s_explicit}"
    assert prosody.line_len_of(GOOD, PUNCT)[0] == 5, "前提：首个半句应为 5 字"
    print(f"  旧签名 {s_default} == 显式 pattern_len=5 {s_explicit} ✓")


def test_critic_empty_and_invalid_pattern_len():
    """边界：空文本不抛异常且给出有限分值；pattern_len ∉ {5,7} 时平仄分项按既有无口径为 0。"""
    s_empty = prosody.score_candidate("")
    assert isinstance(s_empty, float) and math.isfinite(s_empty), f"空文本应得有限分值，实为 {s_empty}"
    # pattern_len=6：check_tone 既有口径直接返回 main=0.0，不抛异常；其余分项照算
    t6 = prosody.check_tone(GOOD, 6)
    assert t6.main == 0.0, f"pattern_len=6 时平仄分项应为 0.0，实为 {t6.main}"
    s6 = prosody.score_candidate(GOOD, pattern_len=6)
    expect6 = prosody.W_LINE_LEN * prosody.check_line_len(GOOD, 6) \
        + prosody.W_RHYME * prosody.check_rhyme(GOOD).main \
        + prosody.W_TONE * 0.0
    assert abs(s6 - expect6) < 1e-9, f"pattern_len=6 应得 {expect6}，实为 {s6}"
    print(f"  空文本 {s_empty}（有限）；pattern_len=6 → {s6}（平仄分项 0）✓")


def _run(fn):
    """执行单项测试并把断言失败转为 ✗ 记录，返回是否通过。"""
    try:
        fn()
        print(f"✓ {fn.__name__}")
        return True
    except AssertionError as e:
        print(f"✗ {fn.__name__}: {e}")
        return False


TESTS = (
    test_critic_good_beats_bad_same_meter,
    test_critic_line_len_variant_lowers_line_len_component,
    test_critic_rhyme_variant_lowers_rhyme_component,
    test_critic_tone_variant_lowers_tone_component,
    test_critic_logprob_three_forms,
    test_critic_backward_compatible_default_pattern_len,
    test_critic_empty_and_invalid_pattern_len,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部评审器打分（M3）单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
