# -*- coding: utf-8 -*-
"""语料构建（build_corpus）单测：诗体标签 + 逐位置元信息数组

覆盖对象：build_corpus.POEM_TAGS / tag_of / tag_poem / build_meta / _half_line_spans
口径（对齐计划 §4.2 / §4.4 / §6.1）：
  每行 = 一联 = 两个半句 → 绝句 2 行、律诗 4 行；诗体标签顺序 <五绝> <七绝> <五律> <七律> <杂言>。
  meta 四数组与 token 流等长：L = len(text) − 3 × 特殊 token 个数（每个标签在文本中占 4 字符、在 token 流中计 1 个，故每标签少记 3）。
  韵脚 = 偶半句（第 2/4/6/8 半句）末字；对仗区 = 仅 <五律>/<七律> 的第 2 行（颔联）+ 第 3 行（颈联）汉字。
全部用例基于合成小样本，不触发全量 57k 首管线；不依赖任何测试框架，可直接
`python test/test_corpus.py` 运行（自带 ✓/✗ 汇总器），依赖仅为项目运行时依赖 numpy 与标准库
（导入 build_corpus 会连带导入 numpy），任一失败时进程以非零码退出。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))

import build_corpus
import prosody

# ── 合成诗样本（每行 = 一联 = 两半句）──
WUJUE = ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡。"]            # 2 行 × 2×5
QIJUE = ["两个黄鹂鸣翠柳，一行白鹭上青天。", "窗含西岭千秋雪，门泊东吴万里船。"]  # 2 行 × 2×7
WULV = ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡。",
        "白日依山尽，黄河入海流。", "欲穷千里目，更上一层楼。"]              # 4 行 × 2×5
QILV = ["岧嶤太华俯咸京，天外三峰削不成。", "武帝祠前云欲散，仙人掌上雨初晴。",
        "河山北枕秦关险，驿树西连汉畤平。", "借问路傍名利客，无如此处学长生。"]  # 4 行 × 2×7
ZAYAN = ["虎可搏，河难凭，公果溺死流海湄。"]                                 # 1 行、3 半句


def test_tag_of_forms():
    """诗体判定：2 行 5/7 字、4 行 5/7 字分别对应五绝/七绝/五律/七律，其余为杂言。"""
    assert build_corpus.tag_of(WUJUE) == "<五绝>", "2 行每半句 5 字应为 <五绝>"
    assert build_corpus.tag_of(QIJUE) == "<七绝>", "2 行每半句 7 字应为 <七绝>"
    assert build_corpus.tag_of(WULV) == "<五律>", "4 行每半句 5 字应为 <五律>"
    assert build_corpus.tag_of(QILV) == "<七律>", "4 行每半句 7 字应为 <七律>"
    assert build_corpus.tag_of(ZAYAN) == "<杂言>", "非绝句/律诗应为 <杂言>"
    print("  五绝/七绝/五律/七律/杂言 判定 ✓")


def test_tag_poem_format():
    """输出行格式：首行前恰一个诗体前缀，其后为原首行；行数不变、末尾换行不变。"""
    text = build_corpus.tag_poem(WULV)
    lines = text.splitlines()
    assert len(lines) == len(WULV), f"行数应保持 {len(WULV)}，实为 {len(lines)}"
    assert lines[0] == "<五律>" + WULV[0], "首行应为「前缀 + 原首行」"
    assert lines[1:] == WULV[1:], "其余行应逐字不变"
    assert text.count("<") == 1 and text.count(">") == 1, "全诗恰一个前缀 token"
    assert text.endswith("\n"), "末行后应保留换行"
    print(f"  前缀恰一个、行数 {len(lines)} 不变 ✓")


def test_poem_tags_order():
    """POEM_TAGS 顺序须与计划 §4.3 特殊 token 追加顺序逐项相同（防两处定义漂移）。"""
    assert build_corpus.POEM_TAGS == ("<五绝>", "<七绝>", "<五律>", "<七律>", "<杂言>"), \
        f"POEM_TAGS 顺序不符：{build_corpus.POEM_TAGS}"
    print("  POEM_TAGS 顺序锁定 ✓")


def test_half_line_spans_match_prosody():
    """半句切分口径与 prosody._half_lines 逐半句一致（换行/空白亦作断句符）。"""
    body = "\n".join(WULV) + "\n"
    spans = build_corpus._half_line_spans(body)
    got = [body[s:e] for s, e in spans]
    want = prosody._half_lines(body, prosody.DEFAULT_PUNCT)
    assert got == want, f"半句切分不一致：build={got} prosody={want}"
    print(f"  半句切分与 prosody 一致（{len(got)} 半句）✓")


def test_meta_lengths():
    """meta 四数组等长，且长度 == token 流长度 L（下标 = token 流位置 p）。

    口径：特殊 token 形如 <五绝>（4 字符）在 token 流中计 1 个 token，其余字符各计 1，故
    L = len(text) − 3×标签数（每个 4 字符标签净折 1 token，减 4 再加回 1），与计划 §4.4 一致。
    注：若误按「L = len(text) − 4×标签数」（纯正文字符数，比 token 流长度少「标签数」个）
    取数组长度，将比 T4 的 `data`（encode 后含特殊 token id）短 N 项，无法按 token 位置切片。
    本测试锁定 token 流长度。
    """
    poems = [WUJUE, QIJUE, WULV, QILV, ZAYAN]
    text, weights, tones, rhymes, zones = build_corpus.build_meta(poems)
    n_tag = len(poems)
    L = sum(1 + len("\n".join(p) + "\n") for p in poems)   # 1 标签 token + 正文字符逐字
    for name, arr in (("weights", weights), ("tone_labels", tones),
                      ("rhyme_labels", rhymes), ("zone_tags", zones)):
        assert len(arr) == L, f"{name} 长度 {len(arr)} 应等于 L={L}"
    assert L == len(text) - 3 * n_tag, "L 应等于 len(text) − 3×标签数（token 流长度）"
    print(f"  四数组等长 L={L}（len(text)={len(text)}, 标签数={n_tag}）✓")


def test_meta_semantics():
    """meta 语义（以 <五律> 样本）：韵脚/半句末标点/对仗区 zone 与权重，及优先级裁定。"""
    text, weights, tones, rhymes, zones = build_corpus.build_meta([WULV])
    body = "\n".join(WULV) + "\n"

    def tok(ch, start=0):
        """字符在 token 流中的下标（标签占 1 位，故正文偏移 + 1）。"""
        return 1 + body.index(ch, start)

    # 韵脚：第 2 个半句「疑是地上霜」末字 霜 → zone=1、权重 2.0
    i = tok("霜")
    assert zones[i] == 1, f"偶半句末字应为韵脚 zone=1，实为 {zones[i]}"
    assert weights[i] == 2.0, f"韵脚权重应 2.0，实为 {weights[i]}"
    assert tones[i] == prosody.tone_of("霜"), "平仄标签应与 prosody.tone_of 一致"
    assert rhymes[i] == prosody.rhyme_of("霜"), "韵部标签应与 prosody.rhyme_of 一致"

    # 半句末标点：首个逗号 → zone=2、权重 1.5
    j = tok("，")
    assert zones[j] == 2, f"半句末标点应 zone=2，实为 {zones[j]}"
    assert weights[j] == 1.5, f"半句末标点权重应 1.5，实为 {weights[j]}"
    assert tones[j] == 2 and rhymes[j] == 0, "标点平仄应 2、韵部应 0"

    # 对仗区：第 2 行（颔联）首字 举 → zone=4
    k = tok("举")
    assert zones[k] == 4, f"颔联汉字应 zone=4，实为 {zones[k]}"

    # 对仗区内的韵脚（乡，同时是偶半句末字）：zone 取 4（优先级 4>1），权重仍 2.0（独立取最大）
    m = tok("乡")
    assert zones[m] == 4, f"对仗区优先于韵脚：乡 应 zone=4，实为 {zones[m]}"
    assert weights[m] == 2.0, f"权重按身份最大值独立计算，应 2.0，实为 {weights[m]}"

    # 对仗区行内的标点不属对仗区，仍为半句末标点
    line2_start = len(WULV[0]) + 1
    p = tok("，", line2_start)
    assert zones[p] == 2, f"对仗区行内标点应 zone=2，实为 {zones[p]}"
    print("  韵脚/标点/对仗区语义与 3>4>1>2>0 优先级 ✓")


def test_meta_uncovered_char():
    """未覆盖汉字：tone_labels=2、rhyme_labels=0（不得静默当平声/韵部）。"""
    assert prosody._rhyme_set_of("疎") == frozenset(), "前提校验：疎 应为韵书未覆盖字"
    poems = [["疎烟淡柳，远岫孤云。"]]
    text, weights, tones, rhymes, zones = build_corpus.build_meta(poems)
    body = "疎烟淡柳，远岫孤云。\n"
    i = 1 + body.index("疎")
    assert tones[i] == 2, f"未覆盖字平仄标签应 2，实为 {tones[i]}"
    assert rhymes[i] == 0, f"未覆盖字韵部标签应 0，实为 {rhymes[i]}"
    print("  未覆盖字 → tone=2 / rhyme=0 ✓")


def test_train_text_same_source_as_val():
    """训练文本与验证文本须同源同格式：build_meta 产出的训练文本与 tag_poem 逐首拼接逐字节一致。

    训练文本在 build_meta 内以「标签 + 正文」两段拼接，验证文本在 main 内以 tag_poem 逐首拼接；
    两处口径必须一致（一联一行、诗间不插空行），否则训练/验证分布漂移。本测试以合成诗列表锁定二者等价。
    """
    poems = [WUJUE, QIJUE, WULV, QILV, ZAYAN]
    text = build_corpus.build_meta(poems)[0]
    want = "".join(build_corpus.tag_poem(p) for p in poems)
    assert text == want, "build_meta 训练文本与 tag_poem 拼接不一致"
    print(f"  训练文本与 tag_poem 拼接逐字节一致（{len(text)} 字符）✓")


def test_tone_label_multitone_char():
    """跨平/仄多读字：tone_labels=2（未覆盖，不得静默取平或仄单值）。"""
    assert prosody.tone_of("望") is None, "前提校验：望 应为跨平/仄多读字（tone_of 返 None）"
    assert prosody._tone_set_of("望") == frozenset({0, 1}), "前提校验：望 平仄集合应为 {平, 仄}"
    poems = [["举头望明月"]]          # 单行小样本，含目标字 望
    text, weights, tones, rhymes, zones = build_corpus.build_meta(poems)
    body = "举头望明月\n"
    i = 1 + body.index("望")          # 标签占 1 个 token，故正文偏移 + 1
    assert tones[i] == 2, f"跨平/仄多读字平仄标签应 2，实为 {tones[i]}"
    print("  跨平/仄多读字 望 → tone=2 ✓")


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
    test_tag_of_forms,
    test_tag_poem_format,
    test_poem_tags_order,
    test_half_line_spans_match_prosody,
    test_meta_lengths,
    test_meta_semantics,
    test_meta_uncovered_char,
    test_train_text_same_source_as_val,
    test_tone_label_multitone_char,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部语料构建单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
