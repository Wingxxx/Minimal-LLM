"""句长约束（格律处理器）单测：句长推断、三态掩码、关闭/覆盖/降级、端到端句长恒等

覆盖对象：meter.infer_line_len / meter.make_meter_processor / GPT.generate(logits_processor)
主逻辑用 CPU 小模型（d=16/2 头/1 层/小词表，ctx 视用例取 32 或 64）快速验证，不碰全唐诗大语料。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import meter
from model.gpt import GPT

NEG = float("-inf")
ID_COMMA, ID_PERIOD = 8, 9          # 小词表里「，」「。」的 id
ID_NL = 10     # 换行符 id（0-7 汉字，8=「，」9=「。」10=换行）

ID_NL0 = 0                                    # 换行符 id 取 0（真实语料即为 0，须防真值判断陷阱）
ID_C0, ID_P0 = 9, 10                          # 该用例小词表：0=换行、1-8 汉字、9=「，」、10=「。」
ID_Q = 11                                     # 「？」（低频标点，既非「，。」亦非汉字）


def _allowed(logits):
    """返回未被 -inf 屏蔽的下标列表。"""
    return [i for i, v in enumerate(np.asarray(logits)) if v != NEG]


def test_newline_id_zero_boundary():
    """换行符 id 为 0（假值）时仍须正确放行与屏蔽。"""
    proc = meter.make_meter_processor([ID_C0, ID_P0], ID_C0, ID_P0, line_len=5,
                                      non_hanzi_ids=[ID_NL0, ID_C0, ID_P0],
                                      newline_id=ID_NL0)
    logits = np.zeros(11)
    # 「。」后：放行汉字 + 换行（id=0）
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_P0]]))
    assert _allowed(out) == [ID_NL0] + list(range(1, 9)), f"id=0 的换行应被放行，实为 {_allowed(out)}"
    # 换行之后：换行须被重新屏蔽
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_P0, ID_NL0]]))
    assert _allowed(out) == list(range(1, 9)), f"换行应被重新屏蔽，实为 {_allowed(out)}"
    print("newline_id_zero_boundary PASS")


def test_other_non_hanzi_blocked():
    """「？」等低频非汉字既不可断句、也不得偷占句长：半句内一律屏蔽。"""
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD,
                                      line_len=5,
                                      non_hanzi_ids=[ID_COMMA, ID_PERIOD, ID_NL, ID_Q],
                                      newline_id=ID_NL)
    logits = np.zeros(12)
    out = proc(logits, np.array([[1, 2, 3]]))
    assert _allowed(out) == list(range(8)), f"「？」应被屏蔽，实为 {_allowed(out)}"
    # 强制断句位也只放行目标标点，「？」不得顶替
    out = proc(logits, np.array([[1, 2, 3, 4, 5]]))
    assert _allowed(out) == [ID_COMMA], f"强制断句位应只留「，」，实为 {_allowed(out)}"
    print("other_non_hanzi_blocked PASS")


def test_infer_line_len():
    """句长推断三情形：末尾未断 / 末尾已标点回退 / 皆无→默认 5。"""
    punct = [ID_COMMA, ID_PERIOD]
    assert meter.infer_line_len([1, 2, 3, 4, 5], punct) == 5, "末尾未断五言应推断为 5"
    assert meter.infer_line_len([1, 2, 3, 4, 5, 6, 7], punct) == 7, "末尾未断七言应推断为 7"
    # 末尾是「。」→ 回退最后一个完整半句（七言）
    ids = [1, 2, 3, 4, 5, ID_COMMA, 1, 2, 3, 4, 5, 6, 7, ID_PERIOD]
    assert meter.infer_line_len(ids, punct) == 7, "末尾已标点应回退完整半句 7"
    # 末尾是「，」→ 回退前一个半句（五言）
    assert meter.infer_line_len([1, 2, 3, 4, 5, ID_COMMA], punct) == 5
    # 空 / 全标点 → 默认 5
    assert meter.infer_line_len([], punct) == 5
    assert meter.infer_line_len([ID_PERIOD], punct) == 5
    print("infer_line_len PASS")


def test_processor_states():
    """三态掩码：cur<L 屏蔽标点、cur==L 只留目标标点、cur>L 不约束。"""
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD, line_len=5)
    logits = np.zeros(10)                       # 10 个候选：0-7 常用字，8=「，」9=「。」
    # cur=2 < 5：禁止提前断句 → 标点 8/9 全屏蔽
    out = proc(logits, np.array([[1, 2]]))
    assert _allowed(out) == list(range(8)), f"cur<L 应屏蔽标点，实为 {_allowed(out)}"
    # cur=5 == 5、已出现标点 0（奇数半句）→ 强制「，」
    out = proc(logits, np.array([[1, 2, 3, 4, 5]]))
    assert _allowed(out) == [ID_COMMA], f"奇数半句应强制「，」，实为 {_allowed(out)}"
    assert out[ID_COMMA] == logits[ID_COMMA], "目标标点须保留原始得分"
    # cur=5 == 5、已出现标点 1（偶数半句）→ 强制「。」
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_COMMA, 1, 2, 3, 4, 5]]))
    assert _allowed(out) == [ID_PERIOD], f"偶数半句应强制「。」，实为 {_allowed(out)}"
    # cur=7 > 5：提示词已越界 → 不约束
    out = proc(logits, np.array([[1, 2, 3, 4, 5, 6, 7]]))
    assert _allowed(out) == list(range(10)), "cur>L 应放行全部"
    print("processor_states PASS")


def test_processor_off_and_override():
    """line_len=0 / punct 缺失 → 不生成处理器；显式 line_len=7 覆盖自动推断。"""
    assert meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD, line_len=0) is None
    assert meter.make_meter_processor(None, ID_COMMA, ID_PERIOD) is None
    assert meter.make_meter_processor([], ID_COMMA, ID_PERIOD) is None
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD, line_len=7)
    logits = np.zeros(10)
    assert _allowed(proc(logits, np.array([[1, 2, 3, 4, 5]]))) == list(range(8)), "5<7 不应断句"
    assert _allowed(proc(logits, np.array([[1, 2, 3, 4, 5, 6, 7]]))) == [ID_COMMA], "7 应强制断句"
    print("processor_off_and_override PASS")


def test_auto_infer_freeze():
    """line_len=None：首次调用从提示词推断并冻结，后续调用不得被新半句带偏。"""
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD)
    logits = np.zeros(10)
    # 首调：提示词为七言（7 字无标点）→ L 冻结为 7 → 到 7 强制断
    assert _allowed(proc(logits, np.array([[1, 2, 3, 4, 5, 6, 7]]))) == [ID_COMMA]
    # 二调：若未冻结会重推成 5（此时末尾半句 5 字）并强制断；冻结后仍是 7 → 只屏蔽标点
    assert _allowed(proc(logits, np.array([[1, 2, 3, 4, 5]]))) == list(range(8)), "句长未冻结"
    print("auto_infer_freeze PASS")


def test_end_to_end_line_len():
    """端到端：小模型挂处理器，按「，。」切分后每个完整半句的可见汉字数恒等于提示词句长。"""
    chars = list("床前明月光疑是地上霜") + ["，", "。", "\n"]   # 13 字：0-9 汉字，10=「，」11=「。」12=换行
    stoi = {c: i for i, c in enumerate(chars)}
    g = GPT(vocab_size=len(chars), d_model=16, n_head=2, n_layer=1, ctx_len=64)   # ctx ≥ 提示词 5 + 新生成 30
    prompt = "床前明月光"                                       # 五言提示 → L=5
    pid = np.array([[stoi[c] for c in prompt]], dtype=np.int64)
    proc = meter.make_meter_processor(
        [stoi["，"], stoi["。"]], stoi["，"], stoi["。"],
        non_hanzi_ids=[stoi["，"], stoi["。"], stoi["\n"]], newline_id=stoi["\n"])
    out = g.generate(pid, 30, temperature=1.0, top_k=None, logits_processor=proc)
    punct = {stoi["，"], stoi["。"]}
    segs, cur = [], []
    for i in out[0]:
        i = int(i)
        if i in punct:
            segs.append(("".join(cur), True))
            cur = []
        else:
            cur.append(chars[i])
    if cur:
        segs.append(("".join(cur), False))
    assert segs[0][0] == prompt, "提示词所在半句不应被改动"
    bad = [s for s, ended in segs
           if ended and sum(1 for c in s if "\u4e00" <= c <= "\u9fff") != 5]
    assert not bad, f"存在可见汉字数≠5 的完整半句：{bad}"
    print("end_to_end_line_len PASS")


def test_newline_not_counted():
    """换行符不计入句长：末尾为「。」+换行时 cur 归零 → 只放行汉字。"""
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD,
                                      line_len=5, non_hanzi_ids=[ID_COMMA, ID_PERIOD, ID_NL],
                                      newline_id=ID_NL)
    logits = np.zeros(11)
    # 「。」后已出换行：cur=0 < 5 → 只留汉字；换行因上一 token 是换行而不再放行
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_PERIOD, ID_NL]]))
    assert _allowed(out) == list(range(8)), f"换行后应只允许汉字，实为 {_allowed(out)}"
    print("newline_not_counted PASS")


def test_newline_only_after_period():
    """换行仅在「。」断句后的下一步放行；「，」之后与半句进行中一律禁绝。"""
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD,
                                      line_len=5, non_hanzi_ids=[ID_COMMA, ID_PERIOD, ID_NL],
                                      newline_id=ID_NL)
    logits = np.zeros(11)
    # 「。」刚好断句：cur=0 → 放行汉字 + 换行
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_PERIOD]]))
    assert _allowed(out) == list(range(8)) + [ID_NL], f"「。」后应放行换行，实为 {_allowed(out)}"
    # 「，」之后不放行换行
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_COMMA]]))
    assert _allowed(out) == list(range(8)), f"「，」后不应放行换行，实为 {_allowed(out)}"
    print("newline_only_after_period PASS")


def test_newline_does_not_steal_quota():
    """核心回归：半句内已出换行 + 4 汉字时仍视为 4 字（未满），不得强制断句。"""
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD,
                                      line_len=5, non_hanzi_ids=[ID_COMMA, ID_PERIOD, ID_NL],
                                      newline_id=ID_NL)
    logits = np.zeros(11)
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_PERIOD, ID_NL, 1, 2, 3, 4]]))
    assert _allowed(out) == list(range(8)), f"4 汉字不应触发断句，实为 {_allowed(out)}"
    print("newline_does_not_steal_quota PASS")


def test_newline_disabled_without_newline_id():
    """未给 newline_id 时换行彻底禁绝：即便 non_hanzi_ids 含换行，「。」后也不放行。"""
    proc = meter.make_meter_processor([ID_COMMA, ID_PERIOD], ID_COMMA, ID_PERIOD,
                                      line_len=5, non_hanzi_ids=[ID_COMMA, ID_PERIOD, ID_NL])
    logits = np.zeros(11)
    out = proc(logits, np.array([[1, 2, 3, 4, 5, ID_PERIOD]]))
    assert _allowed(out) == list(range(8)), f"未给 newline_id 时不应放行换行，实为 {_allowed(out)}"
    print("newline_disabled_without_newline_id PASS")


if __name__ == "__main__":
    test_infer_line_len()
    test_processor_states()
    test_processor_off_and_override()
    test_auto_infer_freeze()
    test_newline_not_counted()
    test_newline_only_after_period()
    test_newline_does_not_steal_quota()
    test_newline_disabled_without_newline_id()
    test_newline_id_zero_boundary()
    test_other_non_hanzi_blocked()
    test_end_to_end_line_len()
    print("\n全部句长约束单测通过")
