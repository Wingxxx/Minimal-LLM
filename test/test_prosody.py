# -*- coding: utf-8 -*-
"""格律工具（prosody）单测：句长 + 押韵 + 平仄句式 + 对仗

覆盖对象：prosody.tone_of / rhyme_of / line_len_of / check_line_len / check_rhyme /
    check_tone / score_candidate / check_duizhang
数据契约（data/pingshui.json，由 tools/build_pingshui.py 生成）：
    { "<韵部名>": {"tone": "平|上|去|入", "chars": ["东", "同", ...]}, ... }，共 106 韵部。
多读字口径：任一读音匹配即合规；tone_of 对跨平/仄多读字返回 None，内部经私有 helper 保留读音集合。
纯标准库实现（不 import pytest），可直接 `python test/test_prosody.py` 运行；
逐项打印 ✓/✗ 与汇总，任一失败时进程以非零码退出。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prosody

POEM = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
PUNCT = "，。！？；：、"


def _tone_list(s):
    """返回字符串逐字的 tone_of 列表（供测试前提自校验，避免臆断字音）。"""
    return [prosody.tone_of(c) for c in s]


def test_tone_of_known():
    """查表：东/啼/炉→平(0)；月/扫→仄(1)（扫为上/去多读，皆归仄）。"""
    assert prosody.tone_of("东") == 0, "东（上平一东）应为平"
    assert prosody.tone_of("啼") == 0, "啼（上平八齐）应为平"
    assert prosody.tone_of("炉") == 0, "炉（上平七虞）应为平"
    assert prosody.tone_of("月") == 1, "月（入声六月）应为仄"
    assert prosody.tone_of("扫") == 1, "扫（上/去多读，皆仄）应为仄"
    print("  东/啼/炉→平(0)，月/扫→仄(1) ✓")


def test_tone_of_multi_reading():
    """跨平/仄多读→None（不静默取一）；全部读音同侧的多读字仍返回该侧。"""
    for ch in ("望", "看", "乡"):
        assert prosody.tone_of(ch) is None, f"{ch}（平+仄多读）应返回 None"
    assert prosody.tone_of("扫") == 1, "扫（上+去，同属仄）应返回 1 而非 None"
    print("  望/看/乡→None，扫→1 ✓")


def test_tone_of_unseen():
    """韵书未见字符（含非汉字符号）→None。"""
    assert prosody.tone_of("Ω") is None
    assert prosody.tone_of("A") is None
    print("  未见字→None ✓")


def test_rhyme_of_known():
    """韵部 id（1..106 空间）：东/同同上平一东(=1)；洽=入声十七洽(=106)；光/霜/乡同下平七阳。"""
    assert prosody.rhyme_of("东") == prosody.rhyme_of("同"), "东/同应同上平一东"
    assert prosody.rhyme_of("东") == 1, "上平一东按规范键序应为 id 1（0 保留给 unknown 标签）"
    assert prosody.rhyme_of("洽") == 106, "入声十七洽按规范键序应为 id 106"
    assert prosody.rhyme_of("光") == prosody.rhyme_of("霜") == prosody.rhyme_of("乡"), \
        "光/霜/乡应同下平七阳"
    assert prosody.rhyme_of("东") != prosody.rhyme_of("光"), "东与光不应同韵部"
    # id 空间锁定：全表已知字的韵部 id 必落在 1..106，不得出现 0（0 专供 unknown 标签）
    prosody._load()
    all_ids = [i for ids in prosody._RHYME_ID.values() for i in ids]
    assert all_ids and min(all_ids) >= 1 and max(all_ids) <= 106, \
        "韵部 id 应全部落在 1..106（0 保留给 unknown 标签）"
    print("  东/同 同韵(id=1)；洽 id=106；全表 id∈[1,106]；光/霜/乡 同韵 ✓")


def test_rhyme_of_unseen():
    """韵书未见字符→None。"""
    assert prosody.rhyme_of("Ω") is None
    assert prosody.rhyme_of("A") is None
    print("  未见字→None ✓")


def test_line_len_matches_meter():
    """对拍：line_len_of 各半句汉字数与 meter._count_trailing 口径一致（单半句 + 含标点多半句）。"""
    import meter
    samples = ["床前明月光", "疑是地上霜", "\n疑是地上霜", "白日依山尽",
               "举头望明月", "低头思故乡", "海内存知己"]
    for s in samples:
        stop = {ord(c) for c in s if not ("\u4e00" <= c <= "\u9fff")}
        want = meter._count_trailing([ord(c) for c in s], stop)
        got = prosody.line_len_of(s, PUNCT)[-1]
        assert got == want, f"对拍不一致 {s!r}: prosody={got} meter={want}"
    # 含标点多半句对拍：逐半句独立调用 meter._count_trailing，
    # 与 line_len_of(POEM, PUNCT) 的完整列表对齐，覆盖原用例仅对拍单半句 [-1] 的盲区。
    halves, cur = [], []
    for ch in POEM:
        if ch in PUNCT:
            if cur:
                halves.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        halves.append("".join(cur))
    want_list = []
    for h in halves:
        stop = {ord(c) for c in h if not ("\u4e00" <= c <= "\u9fff")}
        want_list.append(meter._count_trailing([ord(c) for c in h], stop))
    got_list = prosody.line_len_of(POEM, PUNCT)
    assert got_list == want_list, f"多半句对拍不一致：prosody={got_list} meter={want_list}"
    print(f"  {len(samples)} 例单半句 + {len(halves)} 例多半句对拍一致 ✓")


def test_line_len_split_and_newline():
    """半句切分：按标点分句；换行与空白不计入句长。"""
    assert prosody.line_len_of(POEM, PUNCT) == [5, 5, 5, 5], "五言绝句应切出四个 5 字半句"
    assert prosody.line_len_of("床前明月光，疑是地上霜。\n", PUNCT) == [5, 5], "换行不得产生半句"
    assert prosody.line_len_of("", PUNCT) == [], "空文本应返回空列表"
    print("  切分 [5,5,5,5] 与换行口径 ✓")


def test_check_line_len():
    """句长合规率：五言样本 1.0；掺入错字数样本 <1.0（此处为 3/4）。"""
    assert prosody.check_line_len(POEM, 5) == 1.0
    bad = "床前明月，疑是地上霜。举头望明月，低头思故乡。"     # 首半句 4 字
    r = prosody.check_line_len(bad, 5)
    assert 0.0 <= r < 1.0
    assert abs(r - 3 / 4) < 1e-9, f"应为 3/4，实为 {r}"
    print(f"  五言 {prosody.check_line_len(POEM, 5):.4f}，掺错 {r:.4f} ✓")


def test_check_rhyme_positive():
    """合律样本：偶半句（霜/乡，同下平七阳）同韵 → 主指标 1.0，且首句入韵（光）。"""
    res = prosody.check_rhyme(POEM)
    assert res.main == 1.0, f"偶半句同韵应 1.0，实为 {res.main}"
    assert res.first_rhymed is True, "光/霜 同下平七阳，首句应判为入韵"
    print(f"  合律 main={res.main} first_rhymed={res.first_rhymed} ✓")


def test_check_rhyme_negative():
    """末字改异韵（乡→人，上平十一真）→ 失韵：主指标 0.0。"""
    bad = "床前明月光，疑是地上霜。举头望明月，低头思故人。"
    res = prosody.check_rhyme(bad)
    assert res.main == 0.0, f"失韵应 0.0，实为 {res.main}"
    print(f"  失韵 main={res.main} ✓")


def test_check_rhyme_skip_uncovered_pair():
    """未覆盖偶半句整体跳过：末字未覆盖者不计入 matched/total，亦不得算作失韵。

    样本偶数半句末字依次为 光(基准)/疎/霜/乡；其中「疎」韵书未覆盖（证据：_rhyme_set_of 为空集），
    应整体跳过。故可比对两对（霜、乡）且均与基准（下平七阳）同韵 → main=1.0、total_pairs=2
    （而非把疎算作失韵得 2/3）。
    本用例前提「疎未覆盖」先做真实校验，不使用臆测字。
    """
    assert prosody._rhyme_set_of("疎") == frozenset(), "前提校验：疎 应为韵书未覆盖字"
    text = ("白日依山尽，黄河入海光。欲穷千里目，更上一层疎。"
            "春眠不觉晓，处处闻啼霜。举头望明月，低头思故乡。")
    segs = prosody._half_lines(text, PUNCT)
    assert segs[1::2][-1][-1] == "乡", "样本末次偶半句末字应为 乡"
    res = prosody.check_rhyme(text)
    assert res.main == 1.0, f"未覆盖偶半句应跳过，可比对两组均同韵应 1.0，实为 {res.main}"
    assert res.matched_pairs == 2, f"应命中 2 对，实为 {res.matched_pairs}"
    assert res.total_pairs == 2, f"未覆盖项应从分母剔除，应 2 对，实为 {res.total_pairs}"
    print(f"  跳过未覆盖对: main={res.main} total={res.total_pairs} ✓")


def test_check_rhyme_reference_first_covered():
    """首个偶半句末字未覆盖（实测语料 0.78% 的系统性偏差）：基准应取首个已覆盖者，main 不得误判 0.0。

    样本偶数半句末字依次为 疎(未覆盖)/霜/光；若误以 疎 为基准则 ref 为空 → 全诗 main=0.0（旧缺陷）。
    修正后基准取首个已覆盖的「霜」（下平七阳），其后「光」亦属七阳 → main=1.0、total_pairs=1。
    """
    assert prosody._rhyme_set_of("疎") == frozenset(), "前提校验：疎 应为韵书未覆盖字"
    assert prosody._rhyme_set_of("霜") == prosody._rhyme_set_of("光") != frozenset(), \
        "前提校验：霜/光 应同韵且已覆盖"
    text = "朝辞白帝，疎。千里江陵，霜。两岸猿声，光。"
    res = prosody.check_rhyme(text)
    assert res.main == 1.0, f"基准应取首个已覆盖偶半句（霜），main 不应为 0.0，实为 {res.main}"
    assert res.matched_pairs == 1, f"应命中 1 对，实为 {res.matched_pairs}"
    assert res.total_pairs == 1, f"应 1 对（疎 未覆盖不计），实为 {res.total_pairs}"
    print(f"  首个偶半句未覆盖: main={res.main} total={res.total_pairs} ✓")


def test_check_rhyme_first_rhymed_two_halves():
    """首句入韵不受偶半句数量限制：仅两个半句时 total_pairs=0、main=0.0，但 first_rhymed 应如实为 True。

    旧实现 len(even)<2 早返回恒置 first_rhymed=False，会漏报「光（首半句）与霜（基准）同下平七阳」。
    """
    res = prosody.check_rhyme("床前明月光，疑是地上霜。")
    assert res.first_rhymed is True, "光/霜 同下平七阳，首句应判入韵（不受偶半句数量限制）"
    assert res.total_pairs == 0, f"基准之后无偶半句，应 0 对，实为 {res.total_pairs}"
    assert res.main == 0.0, f"total_pairs==0 时 main 应为 0.0，实为 {res.main}"
    print(f"  两半句 first_rhymed={res.first_rhymed} total=0 main={res.main} ✓")


def test_score_candidate():
    """综合分（M3）= 1.0·句长 + 1.5·押韵 + 1.0·平仄（+ 0.5·logprob，此处未启用记 0）。

    合律样本 POEM 分项：句长 1.0、押韵 1.0、平仄 0.5（四半句中仅第 1、4 半句命中 A/B/C/D）
        → 1.0*1.0 + 1.5*1.0 + 1.0*0.5 = 3.0。
    失韵样本（乡→人，换上平十一真，异于基准霜之下平七阳）：句长 1.0、押韵 0.0、平仄 0.5
        （人仍为平，末半句必论位组合不变，平仄分项不变）→ 1.0*1.0 + 1.5*0.0 + 1.0*0.5 = 1.5。
    保留原意：合律样本得分严格高于失韵样本。
    """
    s_good = prosody.score_candidate(POEM)
    assert abs(s_good - 3.0) < 1e-9, f"合律样本应得 3.0，实为 {s_good}"
    bad = "床前明月光，疑是地上霜。举头望明月，低头思故人。"
    s_bad = prosody.score_candidate(bad)
    assert abs(s_bad - 1.5) < 1e-9, f"失韵样本应得 1.5，实为 {s_bad}"
    assert s_good > s_bad, f"合律 {s_good} 应严格大于失韵 {s_bad}"
    print("  score 合律 3.0 / 失韵 1.5（合律 > 失韵）✓")


# ---- M2：平仄句式 check_tone ----
# 样本用字均经 `prosody.tone_of` 实测确认（平=0/仄=1/未见或跨调=None），用例内另以 _tone_list 前提自校验。
# 平(0)：天 风 春 江 山 明 清 秋 人 年 东 河 前 声 高 来 归 黄 寒 依 床 光 霜
# 仄(1)：月 日 木 水 火 土 夜 雪 白 石 竹 玉 碧 落 尽 万 觉 海 入 上 下 里 目
# None ：疎（韵书未见）、望（跨平/仄多读）
P5_A = "白日依山尽"      # 仄仄平平仄（A 仄起仄收）
P5_B = "天高白玉寒"      # 平平仄仄平（B 平起平收）
P5_C = "春江明月雪"      # 平平平仄仄（C 平起仄收）
P5_D = "夜雪白霜天"      # 仄仄仄平平（D 仄起平收）
P7_A = "春江日落人归尽"  # 平平仄仄平平仄（A）
P7_B = "夜雪天高海月来"  # 仄仄平平仄仄平（B）
P7_C = "月落春江明水尽"  # 仄仄平平平仄仄（C）
P7_D = "春江月落白霜天"  # 平平仄仄仄平平（D）


def test_check_tone_five_patterns():
    """五言 A/B/C/D 四个基本句式各判为合规（main==1.0，matched==total==1）；pattern_len 默认按首半句推断。"""
    for name, line, pat in (("A", P5_A, [1, 1, 0, 0, 1]), ("B", P5_B, [0, 0, 1, 1, 0]),
                            ("C", P5_C, [0, 0, 0, 1, 1]), ("D", P5_D, [1, 1, 1, 0, 0])):
        assert _tone_list(line) == pat, f"前提：{line}（五言 {name}）平仄应为 {pat}"
        res = prosody.check_tone(line)
        assert res.main == 1.0, f"五言 {name} 合律应 main=1.0，实为 {res.main}"
        assert (res.matched, res.total) == (1, 1), f"五言 {name} 应 1/1，实为 {res.matched}/{res.total}"
    print("  五言 A/B/C/D 均 main=1.0 ✓")


def test_check_tone_seven_patterns():
    """七言 A/B/C/D 四个基本句式各判为合规（main==1.0）；pattern_len 默认按首半句推断为 7。"""
    for name, line, pat in (("A", P7_A, [0, 0, 1, 1, 0, 0, 1]), ("B", P7_B, [1, 1, 0, 0, 1, 1, 0]),
                            ("C", P7_C, [1, 1, 0, 0, 0, 1, 1]), ("D", P7_D, [0, 0, 1, 1, 1, 0, 0])):
        assert _tone_list(line) == pat, f"前提：{line}（七言 {name}）平仄应为 {pat}"
        res = prosody.check_tone(line)
        assert res.main == 1.0, f"七言 {name} 合律应 main=1.0，实为 {res.main}"
        assert (res.matched, res.total) == (1, 1), f"七言 {name} 应 1/1，实为 {res.matched}/{res.total}"
    print("  七言 A/B/C/D 均 main=1.0 ✓")


def test_check_tone_ignore_odd_positions():
    """一三五不论：仅 2/4/6 位 + 句末为必论位，1/3/5 位改成相反平仄后仍判合规。"""
    # 五言：A 句 白日依山尽 [1,1,0,0,1] → 改不论位（0 基 0/2）为反：天日木山尽 [0,1,1,0,1]
    mod5 = "天日木山尽"
    assert _tone_list(mod5) == [0, 1, 1, 0, 1], "前提：五言改位后平仄应为 [0,1,1,0,1]"
    res5 = prosody.check_tone(mod5)
    assert res5.main == 1.0, f"五言不论位改反后仍应合律，实为 {res5.main}"
    # 七言：A 句 春江日落人归尽 [0,0,1,1,0,0,1] → 改不论位（0 基 0/2/4）：月江明落海归尽 [1,0,0,1,1,0,1]
    mod7 = "月江明落海归尽"
    assert _tone_list(mod7) == [1, 0, 0, 1, 1, 0, 1], "前提：七言改位后平仄应为 [1,0,0,1,1,0,1]"
    res7 = prosody.check_tone(mod7)
    assert res7.main == 1.0, f"七言不论位改反后仍应合律，实为 {res7.main}"
    print("  不论位（1/3/5）改反仍合律 ✓")


def test_check_tone_required_position_violation():
    """必论位（2/4/6 位或句末）失律 → 该半句不计入 matched，main 相应下降。"""
    bad5 = "白高依山尽"      # 五言 2 位 日(1)→高(0)：必论位组合 (0,0,1) 不符任何句式
    assert _tone_list(bad5) == [1, 0, 0, 0, 1], "前提：五言失律样本平仄应为 [1,0,0,0,1]"
    res = prosody.check_tone("，".join([P5_A, bad5]))
    assert res.total == 2, f"两个五言半句应 total=2，实为 {res.total}"
    assert res.matched == 1, f"仅合律半句计入 matched，应 1，实为 {res.matched}"
    assert abs(res.main - 0.5) < 1e-9, f"main 应 0.5，实为 {res.main}"
    # 七言 4 位（0 基下标 3）改反亦失律：夜雪天高海月来 [1,1,0,0,1,1,0] → 夜雪天木海月来 [1,1,0,1,1,1,0]
    bad7 = "夜雪天木海月来"
    assert _tone_list(bad7) == [1, 1, 0, 1, 1, 1, 0], "前提：七言失律样本平仄应为 [1,1,0,1,1,1,0]"
    res7 = prosody.check_tone(bad7)
    assert (res7.matched, res7.total) == (0, 1), f"七言 4 位失律应 0/1，实为 {res7.matched}/{res7.total}"
    print("  必论位失律 main=0.5（五言 2 位）、0/1（七言 4 位）✓")


def test_check_tone_last_char_conjugate():
    """句末字属必论位，但 A/D（及 B/C）仅句末字互补，故单独翻转句末字只是切换为共轭句式、仍判合规。

    本用例如实锁定该交互（口径见计划 §7.1：「命中任一基本句式」+ 必论位含句末字）：末字翻转不单独
    致失律；须与其他必论位组合方可失律（见 test_check_tone_required_position_violation）。
    """
    flipped = "春江日落人归天"      # P7_A 末字 尽(1)→天(0)
    assert _tone_list(flipped) == [0, 0, 1, 1, 0, 0, 0], "前提：翻转末字后平仄应为 [0,0,1,1,0,0,0]"
    r = prosody.check_tone(flipped)
    assert (r.matched, r.total, r.main) == (1, 1, 1.0), \
        f"末字翻转命中共轭句式 D，应仍合规，实为 {r.matched}/{r.total}"
    print("  句末字翻转切换共轭句式（A↔D）仍合规 ✓")


def test_check_tone_skip_all_uncovered_halfline():
    """某半句全部必论位不可判定 → 整体不计入分子与分母（不得算作失律）。"""
    assert prosody.tone_of("疎") is None and prosody.tone_of("望") is None, \
        "前提：疎（未见）/望（跨调）均应不可判定"
    line_none = "疎望疎望疎"          # 必论位（1/3/4）分别为 望/望/疎，均 None
    assert [prosody.tone_of(line_none[i]) for i in (1, 3, 4)] == [None, None, None], \
        "前提：该半句必论位应全部不可判定"
    res = prosody.check_tone("，".join([P5_A, line_none]))
    assert res.total == 1, f"全不可判定半句应剔除，total 应为 1，实为 {res.total}"
    assert res.matched == 1, f"仅合律半句计入 matched，应为 1，实为 {res.matched}"
    assert res.main == 1.0, f"main 应为 1.0，实为 {res.main}"
    print("  全不可判定半句整体跳过 ✓")


def test_check_tone_gu_ping():
    """孤平：必论位与 B 句式相符且不论位首字为仄 → gu_ping=1；合法 B 句 gu_ping=0。"""
    gu5 = "月高木玉寒"          # [1,0,1,1,0] 必论位 (0,1,0)=B，首字 月(仄)
    assert _tone_list(gu5) == [1, 0, 1, 1, 0], "前提：五言孤平样本平仄应为 [1,0,1,1,0]"
    r5 = prosody.check_tone(gu5)
    assert (r5.total, r5.gu_ping) == (1, 1), f"五言孤平应计 1，实为 {r5.gu_ping}"
    assert r5.main == 1.0, "孤平句仍命中 B 句式，main 应为 1.0（违例单列，不并入 main）"
    gu7 = "月落夜天海玉寒"      # [1,1,1,0,1,1,0] 必论位 (1,0,1,0)=B（七言），第 3 字 夜(仄)
    assert _tone_list(gu7) == [1, 1, 1, 0, 1, 1, 0], "前提：七言孤平样本平仄应为 [1,1,1,0,1,1,0]"
    r7 = prosody.check_tone(gu7)
    assert (r7.total, r7.gu_ping) == (1, 1), f"七言孤平应计 1，实为 {r7.gu_ping}"
    assert prosody.check_tone(P5_B).gu_ping == 0, "合法五言 B 句首字为平，不得计孤平"
    assert prosody.check_tone(P7_B).gu_ping == 0, "合法七言 B 句第 3 字为平，不得计孤平"
    print("  孤平 五言/七言 各计 1，合法 B 句为 0 ✓")


def test_check_tone_san_ping():
    """三平调：半句末三字 tone_of 皆为平 → san_ping=1；否则 0。"""
    sp5 = "月落天高春"          # [1,1,0,0,0] 末三字 天高春 皆平
    assert _tone_list(sp5) == [1, 1, 0, 0, 0], "前提：五言三平调样本平仄应为 [1,1,0,0,0]"
    r5 = prosody.check_tone(sp5)
    assert (r5.total, r5.san_ping) == (1, 1), f"五言三平调应计 1，实为 {r5.san_ping}"
    sp7 = "春江月落天高声"      # [0,0,1,1,0,0,0] 末三字 天高声 皆平
    assert _tone_list(sp7) == [0, 0, 1, 1, 0, 0, 0], "前提：七言三平调样本平仄应为 [0,0,1,1,0,0,0]"
    r7 = prosody.check_tone(sp7)
    assert (r7.total, r7.san_ping) == (1, 1), f"七言三平调应计 1，实为 {r7.san_ping}"
    assert prosody.check_tone(P5_D).san_ping == 0, "仄仄仄平平（D）末三字非全平，不得计三平调"
    print("  三平调 五言/七言 各计 1，D 句为 0 ✓")


def test_check_tone_pattern_len_filter():
    """仅统计汉字数 == pattern_len 的半句：五言/七言混排时按 pattern_len 过滤。"""
    text = "，".join([P5_A, P7_D])
    assert prosody.line_len_of(text, PUNCT) == [5, 7], "前提：混排文本半句长应为 [5, 7]"
    r5 = prosody.check_tone(text, pattern_len=5)
    assert (r5.matched, r5.total, r5.main) == (1, 1, 1.0), \
        f"pattern_len=5 只应统计五言半句，实为 {r5.matched}/{r5.total}"
    r7 = prosody.check_tone(text, pattern_len=7)
    assert (r7.matched, r7.total, r7.main) == (1, 1, 1.0), \
        f"pattern_len=7 只应统计七言半句，实为 {r7.matched}/{r7.total}"
    print("  句长不符半句不计入 ✓")


def test_check_tone_invalid_pattern_len():
    """pattern_len ∉ {5,7} → 直接返回全零 ToneResult（不抛异常、不静默当合律）。"""
    for pl in (0, 4, 6, 8):
        r = prosody.check_tone(P5_A, pattern_len=pl)
        assert (r.main, r.matched, r.total, r.gu_ping, r.san_ping) == (0.0, 0, 0, 0, 0), \
            f"pattern_len={pl} 应返回全零，实为 {tuple(r)}"
    # pattern_len=None 按首个半句推断；首半句 4 字 → 推断 4 → 非 {5,7} → 全零
    r = prosody.check_tone("床前明月，疑是地上霜。")
    assert (r.main, r.matched, r.total) == (0.0, 0, 0), f"首半句 4 字推断应全零，实为 {tuple(r)}"
    print("  pattern_len∉{5,7} 及推断为 4 → 全零 ✓")


def test_check_tone_empty():
    """空文本 → ToneResult(0.0,0,0,0,0)；ToneResult 字段次序符合约定。"""
    r = prosody.check_tone("")
    assert (r.main, r.matched, r.total, r.gu_ping, r.san_ping) == (0.0, 0, 0, 0, 0), \
        f"空文本应全零，实为 {tuple(r)}"
    assert prosody.ToneResult._fields == ("main", "matched", "total", "gu_ping", "san_ping"), \
        "ToneResult 字段应依次为 main/matched/total/gu_ping/san_ping"
    print("  空文本全零，ToneResult 字段正确 ✓")


# ---- T2 Step 6：对仗必要条件度量 check_duizhang ----
# 对仗仅统计成篇律诗（半句数 >= 8）的颔联（半句下标 2/3）与颈联（下标 4/5）：对句平仄相对率 + 联内不重字率。
# 样本用字均经 `prosody.tone_of` 实测（平=0/仄=1/未见或跨调=None），用例内另以 _tone_list 前提自校验。
# 实测字音：春江日落人归尽[0,0,1,1,0,0,1]、夜雪天高海月来[1,1,0,0,1,1,0]、
#   河山北枕秦关险[0,0,1,1,0,0,1]、驿树西连汉畤平[1,1,0,0,None,1,0]、
#   夜月春江明海天[1,1,0,0,0,1,0]、白山雪落归高尽[1,0,1,1,0,0,1]。
DUZ_POS = ("春江日落人归尽，夜雪天高海月来。"     # 首联（不计入统计）
           "河山北枕秦关险，驿树西连汉畤平。"     # 颔联：平仄全相对且不重字
           "夜月春江明海天，白山雪落归高尽。"     # 颈联：平仄全相对且不重字
           "月落春江明水尽，春江月落白霜天。")    # 尾联（不计入统计）
# 颔联出句 2 位（0 基下标 1）由 山(平) 改为 月(仄)，与该联对句 树(仄) 同侧 → 该联虽句长合规但并非「全部相反」
DUZ_NOT_OPP = DUZ_POS.replace("河山北枕秦关险", "河月北枕秦关险")
# 颔联对句首字由 驿 改为与出句同字 河 → 联内重字（同字异位亦计重字），平仄必论位不变故仍相对
DUZ_CHONGZI = DUZ_POS.replace("驿树西连汉畤平", "河树西连汉畤平")
# 颔联出句截为 6 字 → 该联句长不符，整体跳过、不计入 pairs（仅剩颈联）
DUZ_SHORT = DUZ_POS.replace("河山北枕秦关险", "河山北枕秦关")


def _first_tagged_poem(tag):
    """从 data/corpus.txt 提取首个以 <tag> 起头的诗块（保留标点，供真实语料实测）。

    语料以行首标签分块且无闭合标签，故自首个「<tag>」行起逐行收集，遇到下一个标签行即止。
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "data", "corpus.txt"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    body, started = [], False
    for ln in lines:
        if ln.startswith("<" + tag + ">"):
            started = True
            body.append(ln[len(tag) + 2:])
            continue
        if started:
            if ln.startswith("<"):
                break
            body.append(ln)
    return "".join(body)


def test_check_duizhang_positive():
    """合律律诗：颔联与颈联平仄相对且联内不重字 → pairs==2、pingze==1.0、chongzi==1.0。

    统计半句（下标 2..5）的字音先经 _tone_list 实测自校验（不臆断平仄）；必论位（七言 0 基下标 1/3/5/6）
    两侧逐位相反，且各联两半句汉字集合无交集。
    """
    segs = prosody._half_lines(DUZ_POS, PUNCT)
    assert len(segs) == 8, f"前提：样本应为 8 个半句，实为 {len(segs)}"
    assert [prosody._trailing_hanzi(s) for s in segs[2:6]] == [7, 7, 7, 7], "前提：统计联半句应均为 7 字"
    assert _tone_list(segs[2]) == [0, 0, 1, 1, 0, 0, 1], "前提：颔联出句平仄应为 [0,0,1,1,0,0,1]"
    assert _tone_list(segs[3]) == [1, 1, 0, 0, None, 1, 0], "前提：颔联对句平仄应为 [1,1,0,0,None,1,0]"
    assert _tone_list(segs[4]) == [1, 1, 0, 0, 0, 1, 0], "前提：颈联出句平仄应为 [1,1,0,0,0,1,0]"
    assert _tone_list(segs[5]) == [1, 0, 1, 1, 0, 0, 1], "前提：颈联对句平仄应为 [1,0,1,1,0,0,1]"
    res = prosody.check_duizhang(DUZ_POS)
    assert res.pairs == 2, f"颔联 + 颈联应统计 2 联，实为 {res.pairs}"
    assert res.pingze == 1.0, f"两联皆全相对应 1.0，实为 {res.pingze}"
    assert res.chongzi == 1.0, f"两联皆不重字应 1.0，实为 {res.chongzi}"
    print(f"  合律 pairs={res.pairs} pingze={res.pingze} chongzi={res.chongzi} ✓")


def test_check_duizhang_not_opposite():
    """反例：颔联必论位两侧并非全部相反 → pingze < 1.0（颈联仍相对，故为 1/2）。"""
    segs = prosody._half_lines(DUZ_NOT_OPP, PUNCT)
    assert _tone_list(segs[2]) == [0, 1, 1, 1, 0, 0, 1], "前提：颔联出句改字后平仄应为 [0,1,1,1,0,0,1]"
    assert prosody.tone_of(segs[2][1]) == prosody.tone_of(segs[3][1]) == 1, \
        "前提：颔联 2 位（0 基下标 1）两侧应同为仄，构成「不相反」"
    res = prosody.check_duizhang(DUZ_NOT_OPP)
    assert res.pairs == 2, f"句长合规仍应统计 2 联，实为 {res.pairs}"
    assert abs(res.pingze - 0.5) < 1e-9, f"仅颈联相对应 0.5，实为 {res.pingze}"
    assert res.pingze < 1.0, f"不相对应使 pingze < 1.0，实为 {res.pingze}"
    assert res.chongzi == 1.0, f"重字未改，chongzi 应 1.0，实为 {res.chongzi}"
    print(f"  不相对 pairs={res.pairs} pingze={res.pingze} chongzi={res.chongzi} ✓")


def test_check_duizhang_repeated_char():
    """反例：颔联对句首字与出句同字 → 联内重字 → chongzi < 1.0（平仄未改，仍相对）。"""
    segs = prosody._half_lines(DUZ_CHONGZI, PUNCT)
    assert segs[2][0] == segs[3][0] == "河", "前提：颔联两半句首字应同为 河（同字异位）"
    assert _tone_list(segs[3]) == [0, 1, 0, 0, None, 1, 0], "前提：颔联对句改字后平仄应为 [0,1,0,0,None,1,0]"
    res = prosody.check_duizhang(DUZ_CHONGZI)
    assert res.pairs == 2, f"句长合规仍应统计 2 联，实为 {res.pairs}"
    assert res.pingze == 1.0, f"平仄未改应仍全相对，实为 {res.pingze}"
    assert abs(res.chongzi - 0.5) < 1e-9, f"仅颈联不重字应 0.5，实为 {res.chongzi}"
    assert res.chongzi < 1.0, f"联内重字应使 chongzi < 1.0，实为 {res.chongzi}"
    print(f"  重字 pairs={res.pairs} pingze={res.pingze} chongzi={res.chongzi} ✓")


def test_check_duizhang_jueju_all_zero():
    """绝句仅 4 个半句 → 一律返回 DuizhangResult(0.0, 0.0, 0)。

    绝句的第 2 联是尾联而非颔联，按颔联误计会污染口径，故以「成篇律诗」为统计前提；五绝/七绝均须全 0。
    """
    siete = "春江日落人归尽，夜雪天高海月来。河山北枕秦关险，驿树西连汉畤平。"     # 四个七言半句
    assert prosody.line_len_of(POEM, PUNCT) == [5, 5, 5, 5], "前提：POEM 应为 4 个五言半句"
    assert prosody.line_len_of(siete, PUNCT) == [7, 7, 7, 7], "前提：siete 应为 4 个七言半句"
    for text in (POEM, siete):
        res = prosody.check_duizhang(text)
        assert tuple(res) == (0.0, 0.0, 0), f"绝句（4 半句）应返回全 0，实为 {tuple(res)}"
    print("  绝句（4 半句）→ 全 0 ✓")


def test_check_duizhang_too_few_half_lines():
    """半句数 < 8 一律全 0：即使 6 个半句的三对联本可判定合规，也不得统计。"""
    six = ("春江日落人归尽，夜雪天高海月来。河山北枕秦关险，驿树西连汉畤平。"
           "夜月春江明海天，白山雪落归高尽。")
    assert prosody.line_len_of(six, PUNCT) == [7] * 6, "前提：six 应为 6 个七言半句"
    for text in ("", POEM, "春江日落人归尽", six):
        res = prosody.check_duizhang(text)
        assert tuple(res) == (0.0, 0.0, 0), f"半句数 < 8 应返回全 0，实为 {tuple(res)}"
    print("  半句数 < 8 → 全 0 ✓")


def test_check_duizhang_invalid_pattern_len():
    """pattern_len ∉ {5, 7} → 全 0（不抛异常、不静默当合律）；None 推断为非 {5,7} 时同样全 0。"""
    for pl in (0, 4, 6, 8):
        res = prosody.check_duizhang(DUZ_POS, pl)
        assert tuple(res) == (0.0, 0.0, 0), f"pattern_len={pl} 应返回全 0，实为 {tuple(res)}"
    # 首半句 4 字 → 推断 pattern_len=4 → 非 {5,7}；半句数取 8 以隔离 pattern_len 分支
    text4 = "床前明月，疑是地上霜。举头望明月，低头思故乡。" * 2     # 首半句 4 字，共 8 个半句
    assert prosody.line_len_of(text4, PUNCT) == [4, 5, 5, 5] * 2, "前提：样本应为 8 个半句且首半句 4 字"
    res = prosody.check_duizhang(text4)
    assert tuple(res) == (0.0, 0.0, 0), f"推断 pattern_len=4 应返回全 0，实为 {tuple(res)}"
    print("  pattern_len∉{5,7} 及推断为 4 → 全 0 ✓")


def test_check_duizhang_skip_wrong_length_couplet():
    """句长不符的联整体跳过：颔联出句截为 6 字 → 该联不入 pairs，仅统计颈联。"""
    segs = prosody._half_lines(DUZ_SHORT, PUNCT)
    assert [prosody._trailing_hanzi(s) for s in segs[2:6]] == [6, 7, 7, 7], \
        "前提：颔联出句应为 6 字、其余统计半句 7 字"
    res = prosody.check_duizhang(DUZ_SHORT)
    assert res.pairs == 1, f"句长不符的颔联应整体跳过，仅剩颈联应 1 联，实为 {res.pairs}"
    assert res.pingze == 1.0, f"仅剩颈联且全相对应 1.0，实为 {res.pingze}"
    assert res.chongzi == 1.0, f"仅剩颈联且不重字应 1.0，实为 {res.chongzi}"
    print(f"  句长不符联跳过 pairs={res.pairs} pingze={res.pingze} chongzi={res.chongzi} ✓")


def test_check_duizhang_corpus_sample():
    """真实语料实测：data/corpus.txt 首篇 <七律> 的 check_duizhang(text, 7) 数值须如实打印（不得编造）。"""
    text = _first_tagged_poem("七律")
    assert text, "前提：corpus.txt 应含 <七律> 语料"
    segs = prosody._half_lines(text, PUNCT)
    assert len(segs) >= 8, f"前提：首篇七律应含 >= 8 个半句，实为 {len(segs)}"
    res = prosody.check_duizhang(text, 7)
    print(f"  语料七律实测 pairs={res.pairs} pingze={res.pingze} chongzi={res.chongzi}"
          f"（颔联={segs[2]}／{segs[3]}）")
    assert res.pairs >= 1, f"成篇七律应至少统计到 1 联，实为 {res.pairs}"
    assert 0.0 <= res.pingze <= 1.0 and 0.0 <= res.chongzi <= 1.0, \
        f"度量须落在 0..1，实为 pingze={res.pingze} chongzi={res.chongzi}"
    print("  真实语料可测 ✓")


def test_check_duizhang_namedtuple_and_empty():
    """返回类型契约：DuizhangResult 字段依次为 pingze/chongzi/pairs；空文本返回全 0。"""
    assert prosody.DuizhangResult._fields == ("pingze", "chongzi", "pairs"), \
        "DuizhangResult 字段应依次为 pingze/chongzi/pairs"
    assert tuple(prosody.check_duizhang("")) == (0.0, 0.0, 0), "空文本应返回全 0"
    print("  DuizhangResult 字段正确，空文本全 0 ✓")


def test_api_present():
    """里程碑接口边界：M2 的 check_tone 与对仗评估 check_duizhang 均已暴露。"""
    assert hasattr(prosody, "check_tone"), "check_tone 属 M2，本段应已实现"
    assert hasattr(prosody, "check_duizhang"), "对仗评估 check_duizhang 应已实现"
    print("  check_tone / check_duizhang 均已暴露 ✓")


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
    test_tone_of_known,
    test_tone_of_multi_reading,
    test_tone_of_unseen,
    test_rhyme_of_known,
    test_rhyme_of_unseen,
    test_line_len_matches_meter,
    test_line_len_split_and_newline,
    test_check_line_len,
    test_check_rhyme_positive,
    test_check_rhyme_negative,
    test_check_rhyme_skip_uncovered_pair,
    test_check_rhyme_reference_first_covered,
    test_check_rhyme_first_rhymed_two_halves,
    test_score_candidate,
    test_check_tone_five_patterns,
    test_check_tone_seven_patterns,
    test_check_tone_ignore_odd_positions,
    test_check_tone_required_position_violation,
    test_check_tone_last_char_conjugate,
    test_check_tone_skip_all_uncovered_halfline,
    test_check_tone_gu_ping,
    test_check_tone_san_ping,
    test_check_tone_pattern_len_filter,
    test_check_tone_invalid_pattern_len,
    test_check_tone_empty,
    test_check_duizhang_positive,
    test_check_duizhang_not_opposite,
    test_check_duizhang_repeated_char,
    test_check_duizhang_jueju_all_zero,
    test_check_duizhang_too_few_half_lines,
    test_check_duizhang_invalid_pattern_len,
    test_check_duizhang_skip_wrong_length_couplet,
    test_check_duizhang_corpus_sample,
    test_check_duizhang_namedtuple_and_empty,
    test_api_present,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部格律工具单测通过（含对仗评估）")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
