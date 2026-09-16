# -*- coding: utf-8 -*-
"""格律工具（M1/M2 段：句长 + 押韵 + 平仄）：基于平水韵韵书为候选诗句打分的纯逻辑模块。

职责：提供字→平仄 / 字→韵部的确定性查表，以及诗句的句长合规率、押韵合规率、平仄句式合规率与综合评分。
仅依赖标准库，不依赖模型、不依赖 numpy 或任何第三方运行时依赖。

数据契约（data/pingshui.json，由 tools/build_pingshui.py 生成）：
    { "<韵部名>": {"tone": "平|上|去|入", "chars": ["东", "同", ...]}, ... }
    共 106 韵部；tone 仅取 平/上/去/入（仄为 上/去/入 的合称，不在数据中单独出现）。

韵部 id 映射（确定性）：按 JSON 键序（即「上平一东 … 入声十七洽」规范序）编号 1..106
    （0 保留给 unknown 标签，对齐计划 §4.4），不使用 set 迭代序或 hash。
    韵书惰性加载并缓存于模块级，仅首次调用读盘。

多读字口径（任一读音匹配即合规）：同一字若对应多个韵部/声调，不静默取单值。
    - tone_of：跨平/仄多读（如「望」平+仄）返回 None；内部经 _tone_set_of 保留全部声调。
    - rhyme_of：公开签名返回单个韵部 id（取词典序首个 = 最小 id）；多值经 _rhyme_set_of 保留。
    - 押韵判定用韵部集合交集：命中任一读音即视为匹配，避免多读字被误判失韵。

声调编码：0=平；1=仄（上/去/入 皆归仄）；None=韵书未见（或跨调多读，详见 tone_of）。

里程碑边界：本模块已实现 M1 段（句长 + 押韵）、M2 段（平仄句式 check_tone，含孤平/三平调
    违例计数）、M3 段（score_candidate 综合打分：句长 + 押韵 + 平仄 + 调用方注入的模型 logprob）
    与对仗必要条件度量 check_duizhang（评估侧：返回对句平仄相对率 + 联内不重字率 + 可比联对数，
    仅统计成篇律诗的颔联与颈联）。
    对仗为独立评估指标，不入 score_candidate 综合打分。
"""
import json
import os
from collections import namedtuple

_BASE = os.path.dirname(os.path.abspath(__file__))
_PINGSHUI_PATH = os.path.join(_BASE, "data", "pingshui.json")

# 声调 -> 平/仄 二值编码：上/去/入 同为仄(1)
_TONE_CODE = {"平": 0, "上": 1, "去": 1, "入": 1}

# 默认断句标点：作半句分隔符（换行与空白另按 isspace 处理，不计入句长）
DEFAULT_PUNCT = "，。！？；：、"
DEFAULT_LINE_LEN = 5        # 无法从文本推断目标句长时的兜底（五言）

# score_candidate 权重（计划 §7.2 初值）：句长 + 押韵 + 平仄 + 模型平均 logprob
W_LINE_LEN = 1.0
W_RHYME = 1.5
W_TONE = 1.0       # M2：平仄句式合规率（check_tone.main）
W_LOGP = 0.5       # M3：模型平均 token logprob（nats，≤0；由调用方注入，本模块不依赖模型）
# W_DUIZHANG = 0.0   # 对仗属 T2 Step 6 / 评估侧，不入 score_candidate 打分

# check_rhyme 具名返回：main=偶半句同韵部率(0..1)；first_rhymed=首句是否入韵；
# matched_pairs/total_pairs=偶半句比对数（供诊断）。
RhymeResult = namedtuple("RhymeResult", "main first_rhymed matched_pairs total_pairs")

# check_tone 具名返回：main=半句句式合规率(0..1)；matched/total=计入比对的半句数；
# gu_ping/san_ping=孤平/三平调违例数（单列，不计入 main）。
ToneResult = namedtuple("ToneResult", "main matched total gu_ping san_ping")

# check_duizhang 具名返回：pingze=对句平仄相对率(0..1)；chongzi=联内不重字率(0..1)；
# pairs=可比联对数（颔联 + 颈联中句长与可判位均合格者）。
DuizhangResult = namedtuple("DuizhangResult", "pingze chongzi pairs")

# 对仗统计的半句下标：律诗 8 个半句分 4 联，仅取颔联(2,3)与颈联(4,5)；
# 首联(0,1)与尾联(6,7)不计入（尾联本无对仗要求）。
_DUIZHANG_COUPLETS = ((2, 3), (4, 5))

# 五/七言 4 基本句式（0=平 1=仄），行序固定为 A/B/C/D（见计划文档 §7.1）：
#   A 仄起仄收、B 平起平收、C 平起仄收、D 仄起平收
_TONE_PATTERNS = {
    5: ((1, 1, 0, 0, 1),    # A 仄仄平平仄
        (0, 0, 1, 1, 0),    # B 平平仄仄平
        (0, 0, 0, 1, 1),    # C 平平平仄仄
        (1, 1, 1, 0, 0)),   # D 仄仄仄平平
    7: ((0, 0, 1, 1, 0, 0, 1),   # A 平平仄仄平平仄
        (1, 1, 0, 0, 1, 1, 0),   # B 仄仄平平仄仄平
        (1, 1, 0, 0, 0, 1, 1),   # C 仄仄平平平仄仄
        (0, 0, 1, 1, 1, 0, 0)),  # D 平平仄仄仄平平
}
# 必论位（0 基下标）：仅 2/4/6 位 + 句末字；1/3/5 位一律不论（见计划文档 §7.1「一三五不论」）
_REQUIRED_POS = {5: (1, 3, 4), 7: (1, 3, 5, 6)}
_B_IDX = 1                            # B 句式在 _TONE_PATTERNS 中的行序（孤平判定基准）
_LEAD_POS = {5: 0, 7: 2}              # 不论位首字下标：五言第 1 字 / 七言第 3 字（孤平判定用）

# 韵书缓存（惰性加载，模块级）
_TABLE = None        # 原始 JSON 对象（同时作「已加载」哨兵）
_RHYME_ID = None     # {字: set(韵部 id)}
_TONE_SET = None     # {字: set(平仄编码)}


def _load():
    """惰性加载并缓存韵书，构建 字→韵部 id 集合 / 字→平仄集合 两张索引。

    韵部 id 由 JSON 键序决定（enumerate(data.values())，Python dict 保序），
    映射确定且与文件规范序一致；重复调用直接复用缓存，不重读文件。
    """
    global _TABLE, _RHYME_ID, _TONE_SET
    if _TABLE is not None:
        return
    if not os.path.exists(_PINGSHUI_PATH):
        raise FileNotFoundError(
            "缺少韵书数据 %s，请先运行 `python tools/build_pingshui.py` 生成" % _PINGSHUI_PATH)
    with open(_PINGSHUI_PATH, encoding="utf-8") as f:
        data = json.load(f)
    rhyme_id, tone_set = {}, {}
    for idx, info in enumerate(data.values()):        # 键序 +1 即 1 基 id（1..106），确定性映射
        tone = _TONE_CODE[info["tone"]]
        for ch in info["chars"]:
            rhyme_id.setdefault(ch, set()).add(idx + 1)   # 0 保留给 unknown 标签，对齐计划 §4.4
            tone_set.setdefault(ch, set()).add(tone)
    _TABLE, _RHYME_ID, _TONE_SET = data, rhyme_id, tone_set


def tone_of(ch):
    """返回字的平仄：0=平、1=仄（上/去/入 皆归仄）、None=韵书未见或跨平/仄多读。

    跨平/仄多读（读音集合同时含 0 与 1，如「望」平+仄）返回 None，不静默取单值；
    全部读音同属一侧（如「扫」上+去，皆仄）仍返回该侧。多读的完整集合见 _tone_set_of。
    """
    _load()
    ts = _TONE_SET.get(ch)
    if not ts or len(ts) != 1:
        return None
    return next(iter(ts))


def rhyme_of(ch):
    """返回字的韵部 id（1..106，取词典序首个 = 最小 id；0 保留给 unknown 标签，对齐计划 §4.4）；韵书未见返回 None。

    多读字（属多个韵部）此处只暴露首个 id，完整集合见 _rhyme_set_of；
    押韵判定请用集合交集，勿以本函数单值比较，以免多读字被误判失韵。
    """
    _load()
    rs = _RHYME_ID.get(ch)
    return min(rs) if rs else None


def _tone_set_of(ch):
    """字全部读音的平仄集合（⊆ {0, 1}）；韵书未见返回空集合。供 check_* 做「任一匹配」。"""
    _load()
    return frozenset(_TONE_SET.get(ch, ()))


def _rhyme_set_of(ch):
    """字全部读音的韵部 id 集合；韵书未见返回空集合。供押韵判定取交集（任一匹配即合规）。"""
    _load()
    return frozenset(_RHYME_ID.get(ch, ()))


def _is_hanzi(ch):
    """基本汉字区判定（U+4E00–U+9FFF），与项目其余模块口径一致。"""
    return "\u4e00" <= ch <= "\u9fff"


def _count_trailing(ids, stop_set):
    """末尾连续汉字数：从尾部往前数，遇任一非汉字 token（stop_set 中）即止。

    与 meter._count_trailing 口径逐字等价。此处按其口径等价实现而非 import meter——
    meter 顶层 `from model.backend import np` 会在导入期改写进程环境变量
    （CUPY_CACHE_DIR / CUDA_CACHE_PATH），并在装有 cupy 时触发其导入与 CUDA 告警，
    对纯逻辑模块属非预期副作用。等价性由 test/test_prosody.py 的对拍用例锁定。
    """
    n = 0
    for t in reversed(ids):
        if int(t) in stop_set:
            break
        n += 1
    return n


def _trailing_hanzi(seg):
    """半句末尾连续汉字数（遇非汉字即止）：把字符按 ord 编码后复用 _count_trailing 口径。"""
    stop = {ord(c) for c in seg if not _is_hanzi(c)}
    return _count_trailing([ord(c) for c in seg], stop)


def _half_lines(text, punct_ids):
    """按断句标点把文本切成半句列表（保留原字符，丢弃空半句）。

    断句符 = punct_ids ∪ 全部空白（isspace，含换行）；换行不计入句长，故作断句符处理。
    空半句（连续标点或换行夹缝，如「。」后的换行）丢弃，避免污染半句计数。
    """
    delims = set(punct_ids)
    segs, cur = [], []
    for ch in text:
        if ch in delims or ch.isspace():
            segs.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    segs.append("".join(cur))
    return [s for s in segs if s]


def line_len_of(text, punct_ids):
    """按标点切分半句，返回各半句的汉字数列表（list[int]，顺序即半句出现顺序）。

    参数 punct_ids：标点字符集合字符串（如 DEFAULT_PUNCT="，。！？；：、"），**非 token id**；
        与 meter.punct_ids（token id 集合）同名异义，严禁传整型 id（否则 `ch in delims` 恒不命中，
        静默退化为「整段一个半句」）。

    口径说明：换行与空白在本函数中**一律按断句符处理**（有意设计：换行不计入句长），故半句内不会
        含空白；这与 meter._count_trailing 仅在「半句内嵌空白」时结构不同（meter 把非汉字仅当作
        计数终止符、不切分半句），而语料 data/corpus.txt 无内嵌空白，故二者实际等价
        （等价性由 test/test_prosody.py 对拍用例锁定）。空半句不计。
    """
    return [_trailing_hanzi(seg) for seg in _half_lines(text, punct_ids)]


def check_line_len(text, expect, punct_ids=DEFAULT_PUNCT):
    """句长合规率 = 合规半句数 / 总半句数（expect ∈ {5, 7}）。

    合规半句 = 汉字数恰等于 expect 的半句；无半句时返回 0.0。
    参数 punct_ids：标点字符集合字符串（如 DEFAULT_PUNCT），**非 token id**；
        与 meter.punct_ids（token id 集合）同名异义，严禁传整型 id。
    """
    lens = line_len_of(text, punct_ids)
    if not lens:
        return 0.0
    return sum(1 for n in lens if n == expect) / len(lens)


def check_rhyme(text, punct_ids=DEFAULT_PUNCT):
    """押韵评估，返回 RhymeResult(main, first_rhymed, matched_pairs, total_pairs)。

    参数 punct_ids：标点字符集合字符串（如 DEFAULT_PUNCT="，。！？；：、"），**非 token id**；
        与 meter.punct_ids（token id 集合）同名异义，严禁传整型 id。

    主指标 main = 偶半句（第 2/4/6/8 半句）末字与「韵脚基准」同韵部率。口径（对齐计划 §7.1/§8.1）：
      - 韵脚基准取**首个「韵书已覆盖」的偶半句末字**（其韵部集合非空）；
        若全部偶半句末字均未覆盖，则 main=0.0（无可比对之对）。
      - 基准之后的偶半句逐个比对：末字**未覆盖者整体跳过**（不计入 matched_pairs/total_pairs，
        亦不得算作失韵）；末字已覆盖者计入 total_pairs，其韵部集合与基准取交集非空则计入 matched_pairs。
      - main = matched_pairs / total_pairs；total_pairs == 0 时 main = 0.0。
    同韵判定一律用韵部集合交集（任一读音匹配即合规），多读字不误判失韵。

    first_rhymed 记录首句（第 1 半句）末字是否与韵脚基准同韵，仅单独统计、不入韵不扣分（不影响 main）。
    基准取自偶半句（下标 1,3,5,...），恒不等于 segs[0]，故不存在「与自身比较」的情形；只要首半句与
    基准均可查即如实判定，**不受偶半句数量限制**（旧实现 len(even)<2 早返回会误报 False）。
    文本无半句、或无任何已覆盖偶半句末字（无基准）时 first_rhymed=False。
    """
    segs = _half_lines(text, punct_ids)
    if not segs:
        return RhymeResult(0.0, False, 0, 0)
    even = segs[1::2]                        # 第 2/4/6/8 半句（0 基下标 1,3,5,7）
    ref, ref_idx = frozenset(), -1
    for i, s in enumerate(even):             # 韵脚基准 = 首个已覆盖偶半句末字
        rs = _rhyme_set_of(s[-1])
        if rs:
            ref, ref_idx = rs, i
            break
    if ref_idx < 0:                          # 无任何已覆盖偶半句末字：无可比对之对
        return RhymeResult(0.0, False, 0, 0)
    matched = total = 0
    for s in even[ref_idx + 1:]:             # 仅比对基准之后的偶半句
        rs = _rhyme_set_of(s[-1])
        if not rs:                           # 末字未覆盖：整体跳过，不计入 total
            continue
        total += 1
        if ref & rs:
            matched += 1
    main = matched / total if total else 0.0
    first_rhymed = bool(ref & _rhyme_set_of(segs[0][-1]))
    return RhymeResult(main, first_rhymed, matched, total)


def check_tone(text, pattern_len=None, punct_ids=DEFAULT_PUNCT):
    """平仄评估，返回 ToneResult(main, matched, total, gu_ping, san_ping)。

    参数 pattern_len：目标句长，取值 5 或 7；None 时按**首个半句汉字数**推断
        （与 check_line_len/score_candidate 的推断风格一致；无半句时取 DEFAULT_LINE_LEN）。
        若 pattern_len ∉ {5, 7}，直接返回 ToneResult(0.0, 0, 0, 0, 0)——无可比对之半句，
        不抛异常、亦不静默当作合律。
    参数 punct_ids：标点字符集合字符串（如 DEFAULT_PUNCT="，。！？；：、"），**非 token id**；
        与 meter.punct_ids（token id 集合）同名异义，严禁传整型 id。

    口径（对齐计划 §7.1/§8.1）：
      - 仅统计汉字数恰等于 pattern_len 的半句（句长不符者不计入分子与分母）。
      - 一三五不论：仅 2/4/6 位 + 句末字为「必论位」（五言 0 基下标 {1,3,4}；七言 {1,3,5,6}），
        1/3/5 位一律不参与比对。
      - 半句合规：其必论位上凡 tone_of 可判定（∈{0,1}）者，须与 A/B/C/D 中至少一个句式逐位相等；
        tone_of 为 None（韵书未见或跨平/仄多读）的位跳过不判。若某半句全部必论位均不可判定，
        则该半句整体不计入分子与分母（口径同 check_rhyme 的未覆盖跳过，不得算作失律）。
      - main = matched / total；total == 0 时 main = 0.0。
      - 违例单列（不计入 main，仅计数，且仅在「必论位可比对」的半句上统计）：
        孤平 = 必论位与 B 句式逐位相符，且不论位首字（五言下标 0 / 七言下标 2）为仄；
        三平调 = 半句末三字 tone_of 皆为平(0)。
    """
    segs = _half_lines(text, punct_ids)
    if pattern_len is None:
        pattern_len = _trailing_hanzi(segs[0]) if segs else DEFAULT_LINE_LEN
    if pattern_len not in (5, 7):
        return ToneResult(0.0, 0, 0, 0, 0)
    patterns = _TONE_PATTERNS[pattern_len]
    required = _REQUIRED_POS[pattern_len]
    lead_pos = _LEAD_POS[pattern_len]
    matched = total = gu_ping = san_ping = 0
    for seg in segs:
        n = _trailing_hanzi(seg)
        if n != pattern_len:                     # 句长不符：不计入分子与分母
            continue
        chars = seg[-n:]
        decided = [(i, tone_of(chars[i])) for i in required]
        decided = [(i, t) for i, t in decided if t is not None]
        if not decided:                          # 必论位全不可判定：整体跳过
            continue
        total += 1
        if any(all(t == pat[i] for i, t in decided) for pat in patterns):
            matched += 1
        if all(t == patterns[_B_IDX][i] for i, t in decided) and tone_of(chars[lead_pos]) == 1:
            gu_ping += 1
        if all(tone_of(chars[j]) == 0 for j in range(n - 3, n)):    # 末三字皆为平
            san_ping += 1
    main = matched / total if total else 0.0
    return ToneResult(main, matched, total, gu_ping, san_ping)


def check_duizhang(text, pattern_len=None, punct_ids=DEFAULT_PUNCT):
    """对仗必要条件度量，返回 DuizhangResult(pingze, chongzi, pairs)。

    参数 pattern_len：目标句长，取值 5 或 7；None 时按**首个半句汉字数**推断
        （与 check_tone 的推断风格一致；无半句时取 DEFAULT_LINE_LEN）。
        若 pattern_len ∉ {5, 7}，直接返回 DuizhangResult(0.0, 0.0, 0)——无可比对之联，
        不抛异常、亦不静默当作合律。
    参数 punct_ids：标点字符集合字符串（如 DEFAULT_PUNCT="，。！？；：、"），**非 token id**；
        与 meter.punct_ids（token id 集合）同名异义，严禁传整型 id。

    口径：
      - 统计前提为「成篇律诗」：半句数 < 8 一律返回 DuizhangResult(0.0, 0.0, 0)。绝句的第 2 联是
        尾联而非颔联，误按其计会污染口径，故不以绝句为统计对象。半句切分复用 _half_lines。
      - 仅统计颔联（半句下标 2、3）与颈联（半句下标 4、5）；首联与尾联不计入。
      - 联可比的必要条件：该联两半句的汉字数皆等于 pattern_len（汉字数用 _trailing_hanzi 口径）；
        否则该联整体跳过，分子与分母皆不入。
      - 对句平仄相对：在该联两半句的必论位（复用 _REQUIRED_POS[pattern_len] 与「一三五不论」口径）
        上逐位比对，**仅取两侧 tone_of 皆可判定（∈{0,1}）的位**；若该联无可判位，则整体跳过、
        不计入 pairs（口径同 check_tone 的「必论位全不可判定即跳过」）。可判位须**全部相反**
        （t1 != t2）方计为「相对」。
      - 联内不重字：该联两半句的汉字集合交集为空即合规（同字异位亦计为重字）。
      - pingze = 相对联数 / pairs，chongzi = 不重字联数 / pairs；pairs == 0 时二者皆为 0.0。
    """
    segs = _half_lines(text, punct_ids)
    if pattern_len is None:
        pattern_len = _trailing_hanzi(segs[0]) if segs else DEFAULT_LINE_LEN
    if pattern_len not in (5, 7):
        return DuizhangResult(0.0, 0.0, 0)
    if len(segs) < 8:                            # 统计前提：成篇律诗（8 半句 = 4 联）
        return DuizhangResult(0.0, 0.0, 0)
    required = _REQUIRED_POS[pattern_len]
    pairs = pingze = chongzi = 0
    for i, j in _DUIZHANG_COUPLETS:
        a, b = segs[i], segs[j]
        na, nb = _trailing_hanzi(a), _trailing_hanzi(b)
        if na != pattern_len or nb != pattern_len:
            continue                             # 句长不符：该联整体跳过
        ca, cb = a[-na:], b[-nb:]
        decided = [(tone_of(ca[p]), tone_of(cb[p])) for p in required]
        decided = [(t1, t2) for t1, t2 in decided if t1 is not None and t2 is not None]
        if not decided:                          # 无可判位：整体跳过
            continue
        pairs += 1
        if all(t1 != t2 for t1, t2 in decided):
            pingze += 1
        if not (set(ca) & set(cb)):              # 联内两半句汉字集合无交集
            chongzi += 1
    if pairs == 0:
        return DuizhangResult(0.0, 0.0, 0)
    return DuizhangResult(pingze / pairs, chongzi / pairs, pairs)


def score_candidate(text, punct_ids=DEFAULT_PUNCT, pattern_len=None, logprob=None):
    """候选诗句综合分（M3）= W_LINE_LEN·句长合规率 + W_RHYME·押韵主指标 + W_TONE·平仄合规率 + W_LOGP·logprob。

    分项来源：check_line_len(text, pattern_len)、check_rhyme(text).main、check_tone(text, pattern_len).main。

    参数 pattern_len：目标句长，取值 5 或 7；None 时沿用「首个半句汉字数推断」
        （无半句取 DEFAULT_LINE_LEN=5），与 check_tone 的推断风格一致。
        注意：M1 旧假设在「首半句本身错长」时会把错长当基准，故调用方应显式传入已知诗体句长。
    参数 logprob：模型平均 token logprob（nats，≤0），**由调用方注入**——本模块不依赖模型。
        接受浮点，或可调用对象 f(text) -> float（传入的 text 即打分文本本身）；None 表示不启用，该项记 0。
        不做归一化（保持可解释性；权重为初值，可调）。
    参数 punct_ids：标点字符集合字符串（如 DEFAULT_PUNCT），**非 token id**；
        与 meter.punct_ids（token id 集合）同名异义，严禁传整型 id。
    向后兼容：score_candidate(text) 仍可用（pattern_len 与 logprob 默认 None）。
    对仗（W_DUIZHANG）属 T2 Step 6 / 评估侧，不参与本打分。
    """
    lens = line_len_of(text, punct_ids)
    expect = pattern_len if pattern_len is not None else (lens[0] if lens else DEFAULT_LINE_LEN)
    ll = check_line_len(text, expect, punct_ids)
    rh = check_rhyme(text, punct_ids).main
    tn = check_tone(text, pattern_len, punct_ids).main
    if callable(logprob):
        lp = float(logprob(text))
    elif logprob is None:
        lp = 0.0
    else:
        lp = float(logprob)
    return W_LINE_LEN * ll + W_RHYME * rh + W_TONE * tn + W_LOGP * lp
