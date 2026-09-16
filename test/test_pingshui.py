# -*- coding: utf-8 -*-
"""平水韵数据落盘与覆盖率单测：韵部结构、声调分布、字形归一、语料覆盖率、已知字目检

覆盖对象：data/pingshui.json（由 tools/build_pingshui.py 生成）
数据契约（详见 tools/build_pingshui.py docstring）：
    { "<韵部名>": {"tone": "平|上|去|入", "chars": ["东", "同", ...]}, ... }
纯标准库实现，可直接 `python test/test_pingshui.py` 运行；逐项打印 ✓/✗ 与汇总，
任一项失败时进程以非零码退出。
"""
import json
import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINGSHUI = os.path.join(BASE, "data", "pingshui.json")
CORPUS = os.path.join(BASE, "data", "corpus.txt")
CONFLICT = os.path.join(BASE, "data", "pingshui_conflicts.txt")
AUDIT = os.path.join(BASE, "data", "pingshui_audit.txt")

CJK = re.compile(r"[\u4e00-\u9fff]")
TONES = ("平", "上", "去", "入")
# 声调 -> 期望韵部数：平声 30（上平 15 + 下平 15）、上声 29、去声 30、入声 17
TONE_COUNTS = {"平": 30, "上": 29, "去": 30, "入": 17}
EXPECT_TOTAL = 106
COVER_THRESHOLD = 0.95

_DATA = None
_CHAR_TONES = None


def load_data():
    """加载并缓存韵书 JSON；文件缺失或不可解析时断言失败（给出可执行指引）。"""
    global _DATA
    if _DATA is not None:
        return _DATA
    assert os.path.exists(PINGSHUI), \
        f"缺少数据文件 {PINGSHUI}，请先运行 `python tools/build_pingshui.py` 生成"
    with open(PINGSHUI, encoding="utf-8") as f:
        _DATA = json.load(f)
    return _DATA


def char_tone_map():
    """构建 简体字 -> {声调,...} 映射（同一字多韵时保留全部声调）。"""
    global _CHAR_TONES
    if _CHAR_TONES is not None:
        return _CHAR_TONES
    m = {}
    for info in load_data().values():
        for ch in info["chars"]:
            m.setdefault(ch, set()).add(info["tone"])
    _CHAR_TONES = m
    return m


def test_json_loadable():
    """data/pingshui.json 存在、可解析、为对象。"""
    data = load_data()
    assert isinstance(data, dict) and data, "韵书应为非空对象"
    print(f"  韵部总数 {len(data)} | 字条合计 {sum(len(v['chars']) for v in data.values())}")


def test_total_and_tone_counts():
    """韵部总数 == 106，且各声调韵部数 == 平30 / 上29 / 去30 / 入17。"""
    data = load_data()
    assert len(data) == EXPECT_TOTAL, f"韵部总数应为 {EXPECT_TOTAL}，实为 {len(data)}"
    cnt = Counter(v["tone"] for v in data.values())
    for tone, want in TONE_COUNTS.items():
        assert cnt.get(tone, 0) == want, \
            f"{tone}声韵部数应为 {want}，实为 {cnt.get(tone, 0)}"
    print("  声调分布 " + " ".join(f"{t}{cnt[t]}" for t in TONES))


def test_tone_domain():
    """tone 取值只允许 平/上/去/入（仄为上/去/入合称，不得合并成「仄」）。"""
    data = load_data()
    bad = sorted({v["tone"] for v in data.values()} - set(TONES))
    assert not bad, f"出现非法声调取值：{bad}"


def test_chars_nonempty_and_unique():
    """每个韵部 chars 非空、元素均为单个汉字且组内无重复。"""
    data = load_data()
    for name, info in data.items():
        chars = info["chars"]
        assert chars, f"韵部「{name}」chars 为空"
        assert len(chars) == len(set(chars)), f"韵部「{name}」chars 存在重复字"
        for ch in chars:
            assert CJK.fullmatch(ch), f"韵部「{name}」含非单汉字项：{ch!r}"


def test_multi_reading_flagged():
    """多读字（同字跨多个韵部）必须显式记录于 data/pingshui_conflicts.txt 或 JSON 标记。

    本项仅校验多读现象存在时的可追溯性：若存在多读字，则冲突记录文件必须存在。
    """
    m = char_tone_map()
    multi = sorted(ch for ch, tones in m.items() if len(tones) > 1)
    if multi:
        assert os.path.exists(CONFLICT), \
            f"存在 {len(multi)} 个多读字，但缺少冲突记录文件 {CONFLICT}"
        print(f"  多读字（跨多声调）{len(multi)} 个，样例：{''.join(multi[:10])}")
    else:
        print("  无跨声调多读字")


def test_coverage_token_level():
    """token 级覆盖率 ≥ 95%：分子/分母均按汉字出现次数计，并打印去重覆盖率与未覆盖 Top10。"""
    assert os.path.exists(CORPUS), f"缺少语料 {CORPUS}"
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    freq = Counter(c for c in text if CJK.match(c))
    known = char_tone_map()
    denom = sum(freq.values())
    hit = sum(n for ch, n in freq.items() if ch in known)
    cov = hit / denom
    dedup = len(freq)
    dedup_hit = sum(1 for ch in freq if ch in known)
    miss = [(ch, n) for ch, n in freq.most_common() if ch not in known]
    print(f"  token 级覆盖率 {cov:.4%}（{hit}/{denom}） | "
          f"去重字符覆盖率 {dedup_hit / dedup:.4%}（{dedup_hit}/{dedup}）")
    print("  未覆盖 Top10：" + " ".join(f"{ch}{n}" for ch, n in miss[:10]))
    assert cov >= COVER_THRESHOLD, \
        f"token 级覆盖率 {cov:.4%} 低于门槛 {COVER_THRESHOLD:.0%}"


def test_eyeball_known_chars():
    """目检已知字：东→平声（上平一东）；月→入声（入声六月）。"""
    data = load_data()
    assert "上平一东" in data, "缺少韵部「上平一东」"
    assert data["上平一东"]["tone"] == "平", "上平一东 应为平声"
    assert "东" in data["上平一东"]["chars"], "「东」应归入上平一东"
    assert "入声六月" in data, "缺少韵部「入声六月」"
    assert data["入声六月"]["tone"] == "入", "入声六月 应为入声"
    assert "月" in data["入声六月"]["chars"], "「月」应归入入声六月"
    print("  东→上平一东(平) ✓ | 月→入声六月(入) ✓")


def test_regression_common_chars():
    """回归防线：常用简体字 啼 / 扫 / 炉 / 沉 必须被覆盖。

    上一轮「某字在韵部被 ≥2 源同时列出方采用」的交叉规则，在三源体量失衡
    （dockerian 14336 / charlesix59 8079 / weihethu 4349）下误杀单源合法字：
    「啼」繁简同形（仅 dockerian 收录）、「扫/炉/沉」亦然，导致整体丢弃数千合法字。
    本断言锁死该缺陷，防止交叉规则再次收紧为「多源一致」。
    """
    known = char_tone_map()
    missing = [ch for ch in ("啼", "扫", "炉", "沉") if ch not in known]
    assert not missing, f"常用字未被覆盖：{''.join(missing)}（交叉规则过严，误杀单源合法字）"
    print("  啼 / 扫 / 炉 / 沉 均已覆盖 ✓")


def test_audit_file_sections():
    """审计文件 data/pingshui_audit.txt 必须存在，且含「源支持数分布」与「丢弃清单」两节。

    置信度留痕要求：每字「源支持数」分布与「被丢弃字清单」须落盘，杜绝静默丢弃。
    """
    assert os.path.exists(AUDIT), f"缺少审计文件 {AUDIT}，请先运行 `python tools/build_pingshui.py`"
    with open(AUDIT, encoding="utf-8") as f:
        text = f.read()
    for section in ("源支持数分布", "丢弃清单"):
        assert section in text, f"审计文件缺少「{section}」小节"
    print("  审计文件含「源支持数分布」「丢弃清单」两节 ✓")


def test_conflict_source_attribution():
    """跨源韵部不一致之字必须带各源原始归属，而非只罗列韵部。

    例：「徘」dockerian→上平九佳、charlesix59→上平十灰，两源韵部分歧，
    冲突记录须逐源标注归属，便于人工核对分歧来源。
    """
    assert os.path.exists(CONFLICT), f"缺少冲突文件 {CONFLICT}"
    with open(CONFLICT, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    hit = [ln for ln in lines if ln.startswith("徘\t")]
    assert hit, "冲突文件缺少「徘」的记录（跨源韵部不一致字未落盘）"
    line = hit[0]
    for token in ("dockerian", "charlesix59", "上平九佳", "上平十灰"):
        assert token in line, f"「徘」冲突记录缺少源归属信息「{token}」（实际：{line}）"
    print(f"  徘 跨源归属：{line}")


def test_audit_section3_discarded_tokens():
    """审计文件须含第三节「被丢弃字中命中语料的 token Top20」，且节内列出真实丢弃字条目。

    该节须以确切标题落盘，并可见具体字目而非空标题；据此断言节内至少含一个
    实测存在于语料高频丢弃字（如「疎」「劒」），以证明该节内容非空。
    """
    title = "# 【③ 被丢弃字中命中语料的 token Top20】"
    assert os.path.exists(AUDIT), f"缺少审计文件 {AUDIT}"
    with open(AUDIT, encoding="utf-8") as f:
        text = f.read()
    assert title in text, f"审计文件缺少第三节标题：{title}"
    seg = text.split(title, 1)[1]
    nxt = seg.find("# 【")
    if nxt != -1:
        seg = seg[:nxt]
    assert seg.strip(), "第三节内容为空（被丢弃字 token Top20 未落盘）"
    hits = [ch for ch in ("疎", "劒") if ch in seg]
    assert hits, "第三节未列出任何被丢弃字条目（应为非空的字目清单）"
    print(f"  第三节存在，含被丢弃字条目：{'、'.join(hits)}")


def test_cross_conflicts_fully_archived():
    """跨源韵部不一致字须全量入档，条目数与审计文件记载一致，且逐字带各源原始归属。

    审计文件 ① 节记载跨源冲突字数（本项从审计文件解析，不硬编码期望值）；
    冲突文件 data/pingshui_conflicts.txt 中冲突类型含「跨源韵部不一致」的条目数
    须恰为该数，且每条第 3 字段（各源原始归属）非空——以此排除「静默丢弃」
    或仅落盘样例的情形。
    """
    assert os.path.exists(AUDIT), f"缺少审计文件 {AUDIT}"
    assert os.path.exists(CONFLICT), f"缺少冲突文件 {CONFLICT}"
    with open(AUDIT, encoding="utf-8") as f:
        audit_text = f.read()
    m = re.search(r"跨源韵部不一致字：(\d+)\s*字", audit_text)
    assert m, "审计文件缺少「跨源韵部不一致字：N 字」记载"
    expect = int(m.group(1))

    with open(CONFLICT, encoding="utf-8") as f:
        rows = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]
    cross = []
    for ln in rows:
        fields = ln.split("\t")
        assert len(fields) >= 2, f"冲突记录字段不足：{ln!r}"
        if "跨源韵部不一致" in fields[1].split(","):
            cross.append(fields)
    assert len(cross) == expect, \
        f"跨源冲突字条目数 {len(cross)} 与审计记载 {expect} 不一致（疑静默丢弃或仅落盘样例）"
    bad = [f[0] for f in cross if len(f) < 3 or not f[2].strip()]
    assert not bad, f"以下跨源冲突字缺少各源原始归属：{''.join(bad[:20])}"
    print(f"  跨源冲突字 {len(cross)} 条全量入档，均带源归属（与审计记载一致）")


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
    test_json_loadable,
    test_total_and_tone_counts,
    test_tone_domain,
    test_chars_nonempty_and_unique,
    test_multi_reading_flagged,
    test_coverage_token_level,
    test_eyeball_known_chars,
    test_regression_common_chars,
    test_audit_file_sections,
    test_conflict_source_attribution,
    test_audit_section3_discarded_tokens,
    test_cross_conflicts_fully_archived,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部平水韵单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
