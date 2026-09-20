# -*- coding: utf-8 -*-
"""格律内化评估与 A/B 对照工具（T11）：口径固化 → 指标计算 → 闸门判定 → 中文报告落盘。

职责：
  1) 口径固化：入口固定随机源（原生 numpy 流 + 计算后端流），prompt 池只从 data/val.txt 抽取
     该诗任一合格半句（半句 = 该诗按 prosody.DEFAULT_PUNCT 切分后的任一段，须纯汉字且 5/7 字）、
     按诗体分层、同诗体内去重；候选半句凡其字符串出现在 data/corpus.txt 者一律剔除、不入池
     （碰撞排除：源数据含 corpus/val 逐字重复诗，此类文本模型已训练过，不满足评估 prompt
     「模型未训练过」的前提）。抽取结果落盘 data/eval-prompts.txt 复用（已存在即读，不重抽）；
     续写长度 96；A/B 各组必须同设备，否则响亮失败。
  2) 指标（单一来源）：句长合规率、押韵合规率（主指标 = 偶半句末字同韵部率，首句入韵单独统计）、
     平仄合规率（必论位 = 2/4/6 位 + 句末字，一三五不论）与其单列的孤平/三平调违例数，
     以及对仗必要条件率（对句平仄相对率 + 联内不重字率，只计律诗 <五律>/<七律>、绝句不入该指标，
     按联级合并：率 = Σ分子联数 / Σ可比联数），一律复用 prosody 的既有函数
     （check_line_len / check_rhyme / check_tone / check_duizhang），本工具不另立字音判据。
  3) 对照与闸门：旧档 model.npz（自带纯文本词表）、M1/M2 档（扩词表后的新词表）均以同一 prompt 池、
     同一采样流、同一设备评估；输出各组指标与 95% 置信区间半宽（1.96·√(p(1−p)/n)；对仗两子指标
     以可比联数为 n），并给出 M1（句长 ≥ 95% 且 押韵 ≥ 90%）与 M2（平仄 ≥ 85% 且较 M1 提升）的闸门判定。
  4) 报告：写盘中文报告（样本量 / 置信区间 / 旧模型同口径基线 / 三组 A/B 对照结构 / 对仗口径说明 /
     对仗区分度自检结论）；M3 档未就绪时第 2、3 组如实标注「待 M3」并留空，区分度自检标「未自检」，
     不编造数字。
  5) 押韵口径相位披露（死规则 §8.2，2026-09-15 追加）：给出两组实算并作为报告固定章节落盘——
     「真诗同口径上限」（real_poem_ceiling_stats：把 val 四诗体真诗按半句位截断续写，实算对齐截断
     与错位截断两类的押韵/句长合规率，并按 prompt 池实际相位分布加权；只用于判定指标本身是否有效、
     不得为模型低分开脱）与「模型实测按相位拆分」（split_rhyme_by_parity：对每组给出「对齐子集 /
     错位子集」两条押韵合规率与样本量，数据一律取自本轮已生成文本、不额外生成）；再由
     judge_parity_attribution 按各组两子集率差与该组置信区间半宽之和实算，输出归因结论——差异
     均不显著即判「低押韵合规率不可归因于相位错位、闸门判定成立」，某组对齐子集显著高于错位子集
     即点名该组并写「须按 §九 停手报主子复核口径」；终端打印与报告第 2、3 条共用该函数文案，
     不得只作定性声明，也不得拿取样口径当借口否认模型缺陷。

模型载入口径：
  - 旧档 model.npz 为本项目扩词表迁移之前的存档，其词表是「剥去诗体标签后的纯文本字符表」（8196 项），
    不含 5 个诗体特殊 token。故按该档自带的 _chars 构造 字↔id 映射并按名载入参数；载入时断言
    _chars 与当前语料纯文本字符表逐项一致（防词表漂移后用错词表），不一致即报错，不静默降级。
    纯汉字 prompt 在旧词表（8196）与新词表（8201，特殊 token 追加于末尾）下编码得到的 id 相同，
    故三组 prompt 的输入 token 流完全一致，差异只能归因于模型。
  - M1/M2 档使用当前语料词表（8201），按名载入共有参数并断言无缺失参数（辅助头配置由档内
    rhyme_head/tone_head 的有无与类别数自动识别）。

用法（命令行）：
    # 正式全量 A/B（每诗体目标 40 条；旧模型 + M1 + M2 三组均约束关；M3 档就绪时再并跑其三组）
    python tools/prosody_eval.py

    # PowerShell 下启用 GPU（A/B 全部组必须同设备，本工具会校验一致性，不一致即报错）
    $env:MINIMAL_GPU=1; python tools/prosody_eval.py

    # 冒烟（每诗体仅取前 2 条；样本不足时报告只给参考值并标注「不作闸门依据」）
    python tools/prosody_eval.py --limit 2 --report _probe/ab_smoke.txt

    # 指定模型子集与报告路径
    python tools/prosody_eval.py --models old,m2 --report test/output-gelv-ab.txt

产物：
  data/eval-prompts.txt  —— prompt 池（每行「诗体\\t半句」），固定种子抽取后落盘复用。
  test/output-gelv-ab.txt —— A/B 评估报告（中文），含样本量、置信区间、旧模型基线与三组对照结构。
"""
import argparse
import math
import os
import random
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（本脚本在 tools/ 下）
if BASE not in sys.path:            # 使项目根下的 prosody/train/model 可导入
    sys.path.insert(0, BASE)
_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:          # 使同目录的模块可导入
    sys.path.insert(0, _TOOLS)

try:                                # Windows 控制台常为 GBK：正文含 GBK 外字符时转义显示而非崩溃
    sys.stdout.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass

import prosody                     # noqa: E402
import train                       # noqa: E402
import meter                       # noqa: E402
from model.backend import np, onp, as_numpy   # noqa: E402
from model.gpt import GPT          # noqa: E402

# ─────────────────────────────── 路径与定档常量 ───────────────────────────────
DATA_DIR = os.path.join(BASE, "data")
VAL_PATH = os.path.join(DATA_DIR, "val.txt")                     # 评估 prompt 唯一来源
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")               # 训练语料（碰撞排除对照，只读一次）
POOL_PATH = os.path.join(DATA_DIR, "eval-prompts.txt")           # prompt 池（落盘复用）
REPORT_PATH = os.path.join(BASE, "test", "output-gelv-ab.txt")   # A/B 报告
OLD_MODEL = os.path.join(BASE, "model.npz")                      # 旧档（纯文本词表 8196）
MODEL_M1 = os.path.join(BASE, "model-gelv-m1.npz")               # M1 档（词表 8201，仅韵部头）
MODEL_M2 = os.path.join(BASE, "model-gelv-m2.npz")               # M2 档（词表 8201，韵部头 + 平仄头）
MODEL_M3 = os.path.join(BASE, "model-gelv-m3.npz")               # M3 档（未就绪时为占位）

SEED = 0                       # 入口固定随机种子（各组生成前复位，共享同一采样流）
N_PER_TAG = 40                 # 每诗体目标 prompt 条数（合计目标 160）
NEW_TOKENS = 96                # 每首续写 token 数
TEMPERATURE = 1.0              # 采样温度
TOP_K = 20                     # 采样 top_k
MIN_SAMPLES_PER_TAG = 30       # 比例指标样本硬下限：少于该样本量的合规率不作闸门依据
MIN_SAMPLES_TOTAL = 120        # 合计样本硬下限
Z_95 = 1.96                    # 95% 置信区间的正态分位点

# 闸门阈值（与里程碑量产目标一致）
M1_MIN_LINE_LEN = 0.95         # M1：句长合规率下限
M1_MIN_RHYME = 0.90            # M1：押韵合规率下限
M2_MIN_TONE = 0.85             # M2：平仄合规率下限

# 进入 prompt 池的诗体（<杂言> 句长不定、无法定 pattern_len，不入池）
POOL_TAGS = ("<五绝>", "<七绝>", "<五律>", "<七律>")
# 诗体控制 token 全集：用于按诗边界切分 data/val.txt（含 <杂言>，避免其诗行被并入上一首）
ALL_TAGS = ("<五绝>", "<七绝>", "<五律>", "<七律>", "<杂言>")
# 诗体 → 目标句长（pattern_len）
TAG_LEN = {"<五绝>": 5, "<七绝>": 7, "<五律>": 5, "<七律>": 7}

# 对仗必要条件率的适用诗体：只计律诗（<五律>/<七律>）；绝句无对仗要求，不入该指标
DUIZHANG_TAGS = ("<五律>", "<七律>")
DUIZHANG_MIN_DIFF_PP = 10.0          # 区分度自检下限：参与判定的子指标差值（百分点绝对值）任一项 ≥ 该值判有效
DUIZHANG_SATURATED_CHONGZI = 0.90    # 旧模型联内不重字率 ≥ 该值即近饱和：该子指标作废且不参与有效性判定

# 模型架构（旧档与 M1/M2 档一致，否则无法按名继承权重）
ARCH = dict(d_model=256, n_head=8, n_layer=6, ctx_len=128)

# 分组定义：kind -> (模型档路径, 是否开启句长约束, 报告用中文标签)
GROUP_SPECS = {
    "old": (OLD_MODEL, False, "旧模型 model.npz（约束关）"),
    "m1": (MODEL_M1, False, "M1 模型 model-gelv-m1.npz（约束关）"),
    "m2": (MODEL_M2, False, "M2 模型 model-gelv-m2.npz（约束关）"),
    "m3": (MODEL_M3, False, "M3 模型 model-gelv-m3.npz（约束关）"),
    "m3_cons": (MODEL_M3, True, "M3 模型 model-gelv-m3.npz（约束开）"),
}

# 第五节 A/B 对照的组序与编号（组号, kind, 报告用名, 约束开关）：报告第二、五节共用，避免编号漂移
AB_GROUPS = (("1", "old", "旧模型", "关"),
             ("2", "m3", "新 M3 模型", "关"),
             ("3", "m3_cons", "新 M3 模型", "开"))
_AB_NO = {kind: no for no, kind, _, _ in AB_GROUPS}

# tag 臂不可执行时的报告措辞（§15.8 死规则）：被过滤的组不生成、不评分、不出数，如实留空并注明
TAG_ARM_UNAVAILABLE_NOTE = (
    "tag 臂不可执行（档内词表不含诗体 token：旧档为剥去诗体标签的纯文本字符表，"
    "train.encode 会把「<诗体>」前缀按词表外字符逐个跳过），如实留空、不编造")

_VOCAB = None      # 当前语料词表缓存：(chars, stoi, itos)
_CORPUS_CACHE = {}  # corpus_path -> 全文缓存（同一路径整轮只读盘一次，供碰撞判定复用）
_POOL_STATS = None  # 最近一次构建/复用 prompt 池的碰撞排除统计（两条路径皆写回，供报告取用）


# ─────────────────────────── Step 0：口径固化与 prompt 池 ───────────────────────────

def set_seed(seed=SEED):
    """固定随机源：原生 numpy 流（onp）与计算后端流（np）一并复位到同一种子。

    numpy 与 cupy 是两套独立 RNG 流，故二者都须复位；A/B 各组生成前调用本函数，
    使各组从同一采样流出发，差异只能归因于模型。
    """
    onp.random.seed(seed)
    np.random.seed(seed)


def corpus_text(path=CORPUS_PATH):
    """读取并缓存 corpus 全文（同一路径整轮只读盘一次，供候选半句的碰撞判定复用）。

    碰撞排除须把每条候选半句与 corpus 全文做子串判定，逐条读盘代价过高，故按绝对路径缓存。
    """
    key = os.path.abspath(path)
    if key not in _CORPUS_CACHE:
        with open(path, encoding="utf-8") as f:
            _CORPUS_CACHE[key] = f.read()
    return _CORPUS_CACHE[key]


def pool_exclusion_stats():
    """最近一次 prompt 池（无论新建还是复用落盘池）的碰撞排除统计，返回 dict（未建池时为 None）。

    返回 {"collision": {诗体: 排除条数}, "collision_total": int,
          "supply_before": {诗体: 排除前候选量}, "supply_after": {诗体: 排除后供给量},
          "taken": {诗体: 实际抽取条数}}；
    「复用落盘池」路径亦按固定种子重算后写回同一份统计（§8.2 死规则：报告须写明实际排除条数
    与排除后的逐诗体样本量，复用池时不得静默吞掉），故两条路径取到的统计完全一致。
    """
    return _POOL_STATS


def corpus_vocab():
    """当前语料词表（惰性缓存）——返回 (chars, stoi, itos)。"""
    global _VOCAB
    if _VOCAB is None:
        _, chars, stoi, itos, _, _ = train.load_corpus()
        _VOCAB = (chars, stoi, itos)
    return _VOCAB


def _is_pure_hanzi(s):
    """是否纯汉字且汉字数为 5 或 7（可定 pattern_len 的合格半句）。"""
    return len(s) in (5, 7) and all("\u4e00" <= c <= "\u9fff" for c in s)


def _poem_tag_of(line):
    """若该行以任一诗体控制 token 起首则返回该 token，否则返回 None（用于判定诗边界）。"""
    for tag in ALL_TAGS:
        if line.startswith(tag):
            return tag
    return None


def _collect_poem(tag, lines, buckets, seen, corpus, collision):
    """把一首诗正文（list[str]，已剥去诗体 token）按半句切分，合格者按诗体去重入桶。

    半句切分复用 prosody._half_lines（断句标点与空白皆作分隔符、丢弃空段），与
    prosody.check_line_len / check_rhyme 的口径单一来源一致。合格半句 = 纯汉字且 5/7 字。
    去重后做碰撞排除：候选半句若为 corpus 全文的子串（源数据含 corpus/val 逐字重复诗，
    此类文本模型已训练过）则剔除、不入桶，并计入 collision[tag] 如实上报。
    """
    if tag not in POOL_TAGS:
        return
    for seg in prosody._half_lines("\n".join(lines), prosody.DEFAULT_PUNCT):
        if not _is_pure_hanzi(seg) or seg in seen[tag]:
            continue
        seen[tag].add(seg)
        if seg in corpus:
            collision[tag] += 1
            continue
        buckets[tag].append(seg)


def collect_pool_candidates(val_path=VAL_PATH, corpus_path=CORPUS_PATH):
    """扫 data/val.txt，按诗体收集该诗全部合格半句，返回 (候选桶, 碰撞排除条数)。

    半句 = 该诗正文按 prosody.DEFAULT_PUNCT 切分后的任一段，故一首诗可贡献多条候选。
    以任一诗体 token 起首的行判定诗边界（含 <杂言>，避免其诗行被并入上一首）。
    候选半句凡其字符串出现在 corpus_path（data/corpus.txt，只读一次并缓存复用）中者剔除、
    不入桶（碰撞排除：这些文本模型已训练过，不满足评估 prompt「模型未训练过」的前提）。
    返回 ({诗体: [半句, ...]}, {诗体: 排除条数})；<杂言> 不入池。
    """
    buckets = {tag: [] for tag in POOL_TAGS}
    seen = {tag: set() for tag in POOL_TAGS}
    collision = {tag: 0 for tag in POOL_TAGS}
    corpus = corpus_text(corpus_path)
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    cur_tag, buf = None, []
    for line in text.split("\n"):
        tag = _poem_tag_of(line)
        if tag is not None:
            _collect_poem(cur_tag, buf, buckets, seen, corpus, collision)
            cur_tag, buf = tag, [line[len(tag):]]
        elif cur_tag is not None:
            buf.append(line)
    _collect_poem(cur_tag, buf, buckets, seen, corpus, collision)
    return buckets, collision


def _split_poem_blocks(text):
    """按诗体控制 token 起首行把文本切成诗块，返回 list[(诗体, [正文行, ...])]。

    切块口径与 collect_pool_candidates 单一来源：以 _poem_tag_of 判定诗边界（含 <杂言>，
    避免其诗行并入上一首），诗块正文 = 该诗首行剔除诗体 token 后的余文 + 其后各行；
    空白行一律剔除（行尾换行等产物），避免因空行差异误判。
    """
    blocks, cur_tag, buf = [], None, []
    for line in text.split("\n"):
        tag = _poem_tag_of(line)
        if tag is not None:
            if cur_tag is not None:
                blocks.append((cur_tag, [x for x in buf if x.strip()]))
            cur_tag, buf = tag, [line[len(tag):]]
        elif cur_tag is not None:
            buf.append(line)
    if cur_tag is not None:
        blocks.append((cur_tag, [x for x in buf if x.strip()]))
    return blocks


def val_corpus_duplicate_stats(val_path=VAL_PATH, corpus_path=CORPUS_PATH):
    """统计 data/val.txt 中「整诗逐字相同地出现在 data/corpus.txt 里」的诗首数与占比。

    判定口径：先按诗体 token 起首行把两文件各切成诗块（_split_poem_blocks，与
    collect_pool_candidates 的切块口径单一来源）；某 val 诗块的全部正文行按换行拼接后，
    若与 corpus 某诗块的正文拼接串逐字相同，即判该首重复。合计 400 首、重复 9 首、
    占比 2.25%——系 tools/build_corpus.py 诗级划分未去重的 T1 遗留缺陷（见计划
    §7.2「隔离强化」与 §十 风险表），M1/M2/M3 的 val 指标由此含该比例乐观偏差。

    纯读文件、无副作用（只读 val_path 与 corpus_path）；文件缺失或不可读即抛异常
    （响亮失败），不静默返回空统计、不编造数字。
    返回 {"dup_poems": int, "total_poems": int, "ratio": float}。
    """
    with open(val_path, encoding="utf-8") as f:
        val_text = f.read()
    with open(corpus_path, encoding="utf-8") as f:
        corpus_text = f.read()
    corpus_bodies = {"\n".join(body) for _, body in _split_poem_blocks(corpus_text)}
    val_blocks = _split_poem_blocks(val_text)
    dup = sum(1 for _, body in val_blocks if "\n".join(body) in corpus_bodies)
    total = len(val_blocks)
    return {"dup_poems": dup, "total_poems": total,
            "ratio": (dup / total) if total else 0.0}


def corpus_tag_lead_stats(corpus_path=CORPUS_PATH):
    """统计 corpus 中「以诗体控制 token 起首」的诗首数与占比（供报告披露裸半句 prompt 的口径代价）。

    切块复用 _split_poem_blocks（与其余口径单一来源），返回 {"total_poems", "tag_leading", "ratio"}；
    并断言二者相等（切块口径即以诗体 token 起首行起诗块），不等即口径异常、响亮失败。
    纯读文件、无副作用。
    """
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    blocks = _split_poem_blocks(text)
    total = len(blocks)
    leading = sum(1 for tag, _ in blocks if tag in ALL_TAGS)
    assert leading == total, \
        f"{os.path.basename(corpus_path)} 有 {total - leading} 个诗块未以诗体 token 起首，口径异常"
    return {"total_poems": total, "tag_leading": leading,
            "ratio": (leading / total) if total else 0.0}


def poem_token_len_stats(corpus_path=CORPUS_PATH):
    """按诗体统计 corpus 中「一首完整诗所需的 token 数」（供报告披露 new_tokens 相对单首的余量）。

    字数口径 = 诗体控制 token 1 个 + 正文各行的字符数之和 + 行间换行数（行数 − 1；行末换行不计入）；
    取该诗体全部诗中**出现次数最多**者（众数；同频时 Counter 保插入序，故结果确定）。返回 {诗体: int}。
    纯读文件、无副作用。
    """
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    counts = {tag: Counter() for tag in POOL_TAGS}
    for tag, body in _split_poem_blocks(text):
        if tag in counts:
            counts[tag][1 + sum(len(x) for x in body) + (len(body) - 1)] += 1
    return {tag: counts[tag].most_common(1)[0][0] for tag in POOL_TAGS if counts[tag]}


def val_half_index(val_path=VAL_PATH):
    """把 val 各诗切半句，返回 {(诗体, 半句): 0 基半句下标}（供报告实算押韵对齐代价）。

    按诗体 token 切诗块（_split_poem_blocks）→ 按 prosody.DEFAULT_PUNCT 切半句
    （prosody._half_lines，与 prosody.check_* 的口径单一来源）→ 记合格半句（纯汉字且 5/7 字）
    在该诗内的下标；同一半句字符串在同诗体多首诗中重复时取**首现**下标（与候选收集的遍历顺序一致）。
    纯文本、不载模型、无副作用。
    """
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    idx = {}
    for tag, body in _split_poem_blocks(text):
        if tag not in POOL_TAGS:
            continue
        for i, seg in enumerate(prosody._half_lines("\n".join(body), prosody.DEFAULT_PUNCT)):
            if _is_pure_hanzi(seg):
                idx.setdefault((tag, seg), i)
    return idx


def half_parity_stats(pool, val_path=VAL_PATH):
    """实算 prompt 池各条取自原诗半句位的奇偶分布（条数与占比），供报告披露押韵对齐代价。

    偶数位 = 0 基半句下标为偶（即第 1/3/5/7 半句）；奇数位 = 0 基半句下标为奇
    （即第 2/4/6/8 半句，正是押韵主指标所计的「偶半句」韵位——取自该位者续写后半句奇偶与原诗韵位错位）。
    池中每一条都须能在 val 中定位（否则抛断言、不静默漏计）。返回
    {"even", "odd", "total", "even_ratio", "odd_ratio", "unlocated"}。
    """
    idx = val_half_index(val_path)
    even = odd = 0
    unlocated = []
    for tag, prompt in pool:
        i = idx.get((tag, prompt))
        if i is None:
            unlocated.append((tag, prompt))
        elif i % 2 == 0:
            even += 1
        else:
            odd += 1
    assert not unlocated, \
        f"池中有 {len(unlocated)} 条无法在 {os.path.basename(val_path)} 定位原诗半句下标：{unlocated[:5]}"
    total = even + odd
    return {"even": even, "odd": odd, "total": total, "unlocated": 0,
            "even_ratio": (even / total) if total else 0.0,
            "odd_ratio": (odd / total) if total else 0.0}


def real_poem_ceiling_stats(pool, val_path=VAL_PATH):
    """真诗同口径上限（完美续写基准）：把 val 四诗体真诗按半句位截断续写后实算押韵/句长合规率。

    口径（§8.2 死规则 2026-09-15）：真诗的「完美模型」输出即原诗余下部分，故在 0 基半句下标 i 处
    截断时，受评文本 = halves[i:]（prompt 半句 + 余下部分，与评估侧「prompt + 续写」同构）。
    按 0 基下标偶（对齐截断，即第 1/3/5/7 半句）与下标奇（错位截断，即第 2/4/6/8 半句，正是韵位）
    分为两类；每类率 = 该类各截断点的 prosody.check_rhyme(文本).main / prosody.check_line_len(文本,
    TAG_LEN[诗体]) 的算术平均（与 score_group 的逐样本均值口径一致）。再按 prompt 池实际相位分布
    （half_parity_stats 的 even/odd 条数）加权，给出池加权上限（押韵与句长各一个）。

    句长一律以 TAG_LEN 按诗体取值：<五绝>/<五律> 为 5 字、<七绝>/<七律> 为 7 字，**禁止**用首半句
    字数推断（否则五律被误当 7 字）。切诗块复用 _split_poem_blocks、切半句复用 prosody._half_lines
    与 prosody.DEFAULT_PUNCT（口径单一来源）；押韵/句长一律复用 prosody 既有函数。

    纯文本、确定性（POOL_TAGS 键序与诗内半句顺序皆固定）、无副作用（不读模型、不写盘）；
    返回 {"poems": int, "aligned": {"n", "rhyme", "line_len"}, "misaligned": {...},
          "pool_even", "pool_odd", "pool_total", "pool_weighted_rhyme", "pool_weighted_line_len"}。
    """
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    groups = {0: [], 1: []}                 # 奇偶相位 -> [(押韵率, 句长率), ...]
    poems = 0
    for tag, body in _split_poem_blocks(text):
        if tag not in POOL_TAGS:
            continue
        poems += 1
        plen = TAG_LEN[tag]
        halves = prosody._half_lines("\n".join(body), prosody.DEFAULT_PUNCT)
        for i, seg in enumerate(halves):
            if not _is_pure_hanzi(seg):
                continue
            suffix = "，".join(halves[i:])   # 完美续写：prompt 半句 + 原诗余下部分
            groups[i % 2].append((prosody.check_rhyme(suffix).main,
                                  prosody.check_line_len(suffix, plen)))

    def _mean(recs, k):
        return sum(r[k] for r in recs) / len(recs) if recs else 0.0

    out = {"poems": poems}
    for name, parity in (("aligned", 0), ("misaligned", 1)):
        out[name] = {"n": len(groups[parity]),
                     "rhyme": _mean(groups[parity], 0),
                     "line_len": _mean(groups[parity], 1)}
    hp = half_parity_stats(pool, val_path)
    total = hp["total"]
    out.update({
        "pool_even": hp["even"], "pool_odd": hp["odd"], "pool_total": total,
        "pool_weighted_rhyme": ((hp["even"] * out["aligned"]["rhyme"]
                                 + hp["odd"] * out["misaligned"]["rhyme"]) / total) if total else 0.0,
        "pool_weighted_line_len": ((hp["even"] * out["aligned"]["line_len"]
                                    + hp["odd"] * out["misaligned"]["line_len"]) / total) if total else 0.0,
    })
    return out


def split_rhyme_by_parity(texts, pool, val_path=VAL_PATH):
    """把某组已生成文本按 prompt 在原诗中的半句位奇偶拆成「对齐子集 / 错位子集」，各算押韵合规率。

    口径（§8.2 死规则 2026-09-15）：相位定位复用 val_half_index（0 基半句下标，与 half_parity_stats
    同一来源），与 prompt 池一一对应；押韵一律 prosody.check_rhyme(text).main。数据一律取自本轮
    **已生成文本**（不额外生成、不改抽样口径）。返回
    {"aligned": {"n", "rhyme"}, "misaligned": {"n", "rhyme"}, "total"}；两子集样本量之和恒等于池条数。
    池中任一条若无法在 val 定位原诗半句下标，即抛断言、不静默漏计（口径异常须响亮失败）。
    """
    assert len(texts) == len(pool), f"生成文本数 {len(texts)} 与 prompt 数 {len(pool)} 不一致"
    idx = val_half_index(val_path)
    buckets = {0: [], 1: []}
    for (tag, prompt), text in zip(pool, texts):
        i = idx.get((tag, prompt))
        assert i is not None, \
            f"prompt {tag} {prompt!r} 无法在 {os.path.basename(val_path)} 定位原诗半句下标"
        buckets[i % 2].append(prosody.check_rhyme(text).main)
    mean = lambda v: sum(v) / len(v) if v else 0.0        # noqa: E731
    return {"aligned": {"n": len(buckets[0]), "rhyme": mean(buckets[0])},
            "misaligned": {"n": len(buckets[1]), "rhyme": mean(buckets[1])},
            "total": len(texts)}


def read_pool_file(path):
    """读回已落盘的 prompt 池，返回 list[(诗体, 半句)]。"""
    pool = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                tag, _, prompt = ln.partition("\t")
                pool.append((tag, prompt))
    return pool


def compute_eval_pool(val_path=VAL_PATH, n_per_tag=N_PER_TAG, seed=SEED, corpus_path=CORPUS_PATH):
    """纯函数（确定性、无副作用、不写盘）：按固定种子产出候选池与碰撞排除统计。

    只从 val_path 抽取（严禁取训练语料）。逐诗收集该诗全部合格半句入候选桶（同诗体内去重），
    候选半句命中 corpus_path（data/corpus.txt）者按碰撞排除死规则剔除、不入桶。再按诗体分层
    固定种子抽样：候选多于 n_per_tag 者用 random.Random(seed) 截取 n_per_tag 条，候选不多于
    n_per_tag 者全取（不报错、不退出的降级取样）。
    返回 (pool, stats)：pool = list[(诗体, 半句)]（POOL_TAGS 键序、诗体内按抽样顺序）；
    stats = {"collision": {诗体: 排除条数}, "collision_total": int,
             "supply_before": {诗体: 排除前候选量}, "supply_after": {诗体: 排除后供给量},
             "taken": {诗体: 实际抽取条数}}。
    「重建池」与「复用落盘池时按固定种子重算校验」两条路径共用本函数（口径单一来源），
    以杜绝「仅重建路径才统计、复用路径静默吞掉排除条数」的缺陷（§8.2 死规则）。
    """
    cands, collision = collect_pool_candidates(val_path, corpus_path)
    rng = random.Random(seed)
    pool, taken = [], {}
    for tag in POOL_TAGS:                   # 键序固定：<五绝>/<七绝>/<五律>/<七律>
        supply = cands[tag]
        chosen = rng.sample(supply, n_per_tag) if len(supply) > n_per_tag else list(supply)
        taken[tag] = len(chosen)
        pool.extend((tag, prompt) for prompt in chosen)
    stats = {
        "collision": dict(collision),
        "collision_total": sum(collision.values()),
        "supply_before": {tag: len(cands[tag]) + collision[tag] for tag in POOL_TAGS},
        "supply_after": {tag: len(cands[tag]) for tag in POOL_TAGS},
        "taken": dict(taken),
    }
    return pool, stats


def _announce_pool(stats, val_path=VAL_PATH, corpus_path=CORPUS_PATH):
    """把池的碰撞排除统计与实际抽取分布如实打印（不静默掩盖）。

    逐诗体实际条数低于比例指标硬下限（MIN_SAMPLES_PER_TAG）时打印告警：该诗体合规率不作闸门依据。
    """
    for tag in POOL_TAGS:
        if stats["taken"][tag] < MIN_SAMPLES_PER_TAG:
            print(f"[告警] 诗体 {tag} 在 {os.path.basename(val_path)} 中合格半句仅 "
                  f"{stats['supply_after'][tag]} 条，实际抽取 {stats['taken'][tag]} 条，"
                  f"低于比例指标硬下限 {MIN_SAMPLES_PER_TAG}；该诗体合规率不作闸门依据")
    print(f"[池] 碰撞排除（候选半句命中 {os.path.basename(corpus_path)} 者剔除）：共 "
          f"{stats['collision_total']} 条（"
          + "、".join(f"{tag} {stats['collision'][tag]}" for tag in POOL_TAGS) + "）")
    print("[池] 排除后各诗体候选供给量→实际抽取条数：" + "、".join(
        f"{tag} {stats['supply_after'][tag]}→{stats['taken'][tag]}" for tag in POOL_TAGS))


def build_eval_pool(val_path=VAL_PATH, n_per_tag=N_PER_TAG, seed=SEED, pool_path=POOL_PATH,
                    corpus_path=CORPUS_PATH):
    """构建 prompt 池，返回 list[(诗体, 半句)]；pool_path 已存在即读取复用（不重抽、不覆盖）。

    两条路径的碰撞排除统计**完全一致**（均为 compute_eval_pool 的确定性结果，写回
    pool_exclusion_stats() 供报告取用），故不存在「仅重建路径才统计」的静默吞掉（§8.2 死规则）：
      - 复用路径：读落盘池 → **同时**按固定种子重算 → 断言两者逐项相等（不等即池已陈旧/被篡改，
        抛清晰异常，绝不静默使用）→ 返回落盘池原文（池内容仍取落盘那份，不悄悄重抽或覆盖）；
      - 新建路径：重算结果落盘（每行「诗体\\t半句」）并返回。
    pool_path 为 None 时只算不落盘。实际条数低于比例指标硬下限时打印告警（不作闸门依据）。
    """
    global _POOL_STATS
    pool, stats = compute_eval_pool(val_path, n_per_tag, seed, corpus_path)
    _POOL_STATS = stats
    _announce_pool(stats, val_path, corpus_path)
    if pool_path is not None and os.path.exists(pool_path):
        on_disk = read_pool_file(pool_path)
        assert on_disk == pool, (
            f"落盘 prompt 池 {pool_path} 与按固定种子（seed={seed}、每诗体 {n_per_tag} 条）"
            f"重算的池不一致（落盘 {len(on_disk)} 条 vs 重算 {len(pool)} 条），该池已陈旧或被篡改；"
            "拒绝静默使用（死规则要求「已存在即读、不重抽」且内容须与确定性重建逐项一致）："
            "请删除该落盘文件后重抽")
        return on_disk
    if pool_path is not None:
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        with open(pool_path, "w", encoding="utf-8") as f:
            for tag, prompt in pool:
                f.write(f"{tag}\t{prompt}\n")
    return pool


def take_per_tag(pool, limit):
    """按诗体各取前 limit 条（冒烟用；limit 为 None 时原样返回）。"""
    if limit is None:
        return pool
    out, seen = [], {tag: 0 for tag in POOL_TAGS}
    for tag, prompt in pool:
        if seen[tag] < limit:
            seen[tag] += 1
            out.append((tag, prompt))
    return out


# ─────────────────────────────── 模型载入 ───────────────────────────────

def load_old_model(path=OLD_MODEL):
    """载入旧档（扩词表迁移之前）——按该档自带词表构造 字↔id 映射，返回 (model, stoi, itos)。

    旧档词表为「剥去诗体标签后的纯文本字符表」（8196 项），不含诗体特殊 token。载入前断言
    档内 _chars 与当前语料纯文本字符表逐项一致：不一致说明词表已漂移，用其评估必然错位，
    故直接报错而非静默降级。参数按名逐个写入（含形状校验），参数名不符即报错。
    """
    d = dict(onp.load(path))
    assert "_chars" in d, f"{path} 缺少词表快照 _chars，无法确定旧档词表"
    old_chars = [str(c) for c in d["_chars"]]
    chars, _, _ = corpus_vocab()
    assert old_chars == list(chars), (
        f"旧档 {path} 自带词表（{len(old_chars)} 项）与当前语料纯文本字符表"
        f"（{len(chars)} 项）逐项不一致，词表已漂移，拒绝用错词表评估")
    stoi = {c: i for i, c in enumerate(old_chars)}
    itos = {i: c for c, i in stoi.items()}
    model = GPT(vocab_size=len(old_chars), **ARCH)
    params = {k: onp.asarray(v) for k, v in d.items() if not k.startswith("_")}
    expected = [name for name, _ in model._named_params()]
    assert sorted(params) == sorted(expected), (
        f"旧档参数名与模型不匹配：多余 {sorted(set(params) - set(expected))} "
        f"缺失 {sorted(set(expected) - set(params))}")
    for name, p in model._named_params():
        arr = params[name]
        assert tuple(arr.shape) == tuple(p.shape), (
            f"旧档参数 {name} 形状不一致：档内 {tuple(arr.shape)} vs 模型 {tuple(p.shape)}")
        p[...] = np.asarray(arr)
    return model, stoi, itos


def load_new_model(path, n_rhyme=None, use_tone=None):
    """载入扩词表后的新档（M1/M2/M3）——按当前语料词表构造 字↔id 映射，返回 (model, stoi, itos)。

    辅助头配置由档内参数自动识别（无 rhyme_head 即 n_rhyme=0，其类别数取 rhyme_head.W 列数；
    有无 tone_head 即 use_tone），显式传入时以传入值为准。按名载入后断言无缺失参数——
    配置与档不符会产生缺失参数，此时直接报错而非静默保留随机初始化。
    """
    d = onp.load(path)
    names = set(d.files)
    if n_rhyme is None:
        n_rhyme = int(d["rhyme_head.W"].shape[1]) if "rhyme_head.W" in names else 0
    if use_tone is None:
        use_tone = "tone_head.W" in names
    chars, stoi, itos = corpus_vocab()
    model = GPT(vocab_size=len(stoi), n_rhyme=n_rhyme, use_tone=use_tone, **ARCH)
    missing = train.load_params_partial(path, model)
    assert missing == 0, (
        f"{path} 有 {missing} 个参数未载入（模型辅助头配置与该档不符），拒绝评估")
    return model, stoi, itos


def tag_arm_available(path):
    """该档能否执行 tag 臂：只读档内词表（_chars 项数与 tok_emb 行数），不载权重。

    档内词表若只覆盖纯文本字符表（旧档 8196 项），encode 会静默跳过「<诗体>」前缀，两臂将
    退化为同一份输出——故判为不可执行、由报告如实留空，严禁给出伪对照。新档词表在纯文本字符表
    之后追加了 5 个诗体 token（8196 + 5 = 8201），故可执行。
    """
    d = onp.load(path)
    return int(d["tok_emb"].shape[0]) == len(d["_chars"]) + len(train.SPECIAL_TOKENS)


def tag_arm_kinds(kinds):
    """把待跑组按「tag 臂可否执行」二分，返回 (runnable, skipped)，各自保持入参顺序。"""
    runnable, skipped = [], []
    for kind in kinds:
        path = GROUP_SPECS[kind][0]
        if os.path.exists(path) and tag_arm_available(path):
            runnable.append(kind)
        else:
            skipped.append(kind)
    return runnable, skipped


# ─────────────────────────────── 生成与打分 ───────────────────────────────

def generate_texts(model, stoi, itos, pool, new_tokens=NEW_TOKENS, constraint_on=False,
                   tag_prefix=False):
    """对 prompt 池逐条续写，返回生成文本列表。

    生成前复位随机源，使各组从同一采样流出发。constraint_on=True 时挂载句长约束处理器
    （每 prompt 独立处理器，句长在首次调用时按 prompt 推断并冻结）。

    tag_prefix=True 时把该 prompt 所属诗体 token 拼在 prompt **之前**（如「<五绝>新晴花枝下」），
    用于诊断臂——模型由此得知诗体；返回文本含该前缀（与模型所见逐字一致），因四项指标全部
    尾锚，前缀不影响评分口径。裸臂（False）分支与加前缀前逐字一致。
    """
    set_seed(SEED)
    punct_ids = non_hanzi_ids = newline_id = None
    if constraint_on:                   # 仅开约束分组才需要句读/非汉字/换行 id，关约束时不作断言
        punct_ids = [stoi[c] for c in ("，", "。") if c in stoi]
        assert len(punct_ids) == 2, "词表缺少中文句读「，」「。」，句长约束无法启用"
        non_hanzi_ids = [i for c, i in stoi.items() if not ("\u4e00" <= c <= "\u9fff")]
        newline_id = stoi.get("\n")
    texts = []
    for tag, prompt in pool:
        head = tag + prompt if tag_prefix else prompt
        ids = onp.asarray(train.encode(head, stoi))
        if tag_prefix:
            assert len(ids) == len(prompt) + 1, (
                f"tag 臂编码长度应为 prompt 字数 + 1（诗体 token 计 1 个 id）：{head!r} → {len(ids)}")
        else:
            assert len(ids) == len(prompt), f"prompt 含词表外字符或特殊 token：{prompt!r}"
        processor = None
        if constraint_on:
            processor = meter.make_meter_processor(
                punct_ids, stoi["，"], stoi["。"], non_hanzi_ids=non_hanzi_ids,
                newline_id=newline_id)
        out = model.generate(np.asarray(ids[None, :]), new_tokens, temperature=TEMPERATURE,
                             top_k=TOP_K, logits_processor=processor)
        gen = as_numpy(out[0])[len(ids):]
        texts.append(head + "".join(itos[int(i)] for i in gen))
    return texts


def ci_halfwidth(p, n):
    """比例指标 95% 置信区间半宽 = 1.96·√(p(1−p)/n)；n ≤ 0 时返回 0.0。"""
    if n <= 0:
        return 0.0
    return Z_95 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _aggregate(recs):
    """把逐样本指标记录聚合成 {n, 各均值, 置信区间, 违例计数, 对仗分子/分母与合并两子指标}。

    对仗两子指标按联级合并：dz_pingze = Σ相对联数 / Σ可比联数（绝句样本分子分母皆为 0，
    不影响合并值）；置信区间一律以可比联数为 n。pairs == 0 时两子指标皆 0.0。
    """
    n = len(recs)
    keys = ("line_len", "rhyme", "first_rhyme", "tone")
    if n == 0:
        out = {k: 0.0 for k in keys}
        out.update(n=0, gu_ping=0, san_ping=0, ci={k: 0.0 for k in keys},
                   dz_ok_pingze=0, dz_ok_chongzi=0, dz_pairs=0, dz_pingze=0.0, dz_chongzi=0.0)
        out["ci"]["dz_pingze"] = 0.0
        out["ci"]["dz_chongzi"] = 0.0
        return out
    out = {k: float(onp.mean([r[k] for r in recs])) for k in keys}
    out["n"] = n
    out["gu_ping"] = int(sum(r["gu_ping"] for r in recs))
    out["san_ping"] = int(sum(r["san_ping"] for r in recs))
    out["ci"] = {k: ci_halfwidth(out[k], n) for k in keys}
    out["dz_ok_pingze"] = int(sum(r["dz_ok_pingze"] for r in recs))
    out["dz_ok_chongzi"] = int(sum(r["dz_ok_chongzi"] for r in recs))
    out["dz_pairs"] = int(sum(r["dz_pairs"] for r in recs))
    out["dz_pingze"] = out["dz_ok_pingze"] / out["dz_pairs"] if out["dz_pairs"] else 0.0
    out["dz_chongzi"] = out["dz_ok_chongzi"] / out["dz_pairs"] if out["dz_pairs"] else 0.0
    out["ci"]["dz_pingze"] = ci_halfwidth(out["dz_pingze"], out["dz_pairs"])
    out["ci"]["dz_chongzi"] = ci_halfwidth(out["dz_chongzi"], out["dz_pairs"])
    return out


def score_group(texts, pool):
    """对一组生成文本打分并按诗体分桶，返回 {n, overall, per_tag}。

    四项指标一律取自 prosody 的既有函数：句长合规率 check_line_len、押韵主指标与首句入韵
    check_rhyme、平仄合规率与孤平/三平调 check_tone。对仗三项（相对联数 dz_ok_pingze、
    不重字联数 dz_ok_chongzi、可比联数 dz_pairs）取自 check_duizhang，且只对律诗
    （DUIZHANG_TAGS = <五律>/<七律>）计——绝句无对仗要求，三项恒为 0、不入该指标。
    本函数不实现任何字音判据。诗体决定 pattern_len（五言 5 / 七言 7）。
    """
    assert len(texts) == len(pool), f"生成文本数 {len(texts)} 与 prompt 数 {len(pool)} 不一致"
    per_tag = {tag: [] for tag in POOL_TAGS}
    for (tag, _), text in zip(pool, texts):
        plen = TAG_LEN[tag]
        rh = prosody.check_rhyme(text)
        tn = prosody.check_tone(text, plen)
        dz = prosody.check_duizhang(text, plen) if tag in DUIZHANG_TAGS else None
        per_tag[tag].append({
            "line_len": prosody.check_line_len(text, plen),
            "rhyme": rh.main,
            "first_rhyme": float(rh.first_rhymed),
            "tone": tn.main,
            "gu_ping": tn.gu_ping,
            "san_ping": tn.san_ping,
            "dz_ok_pingze": int(round(dz.pingze * dz.pairs)) if dz else 0,
            "dz_ok_chongzi": int(round(dz.chongzi * dz.pairs)) if dz else 0,
            "dz_pairs": int(dz.pairs) if dz else 0,
        })
    return {"n": len(texts),
            "overall": _aggregate([r for v in per_tag.values() for r in v]),
            "per_tag": {tag: _aggregate(v) for tag, v in per_tag.items()}}


def merged_lvshi_duizhang(group_result):
    """把某组结果里 <五律> + <七律> 的分子分母相加，返回律诗合并后的对仗指标（报告与自检共用）。

    率 = Σ分子联数 / Σ可比联数（联级合并，非逐样本率平均）；置信区间以可比联数为 n
    （同一首诗的样本非独立，区间略偏乐观）。pairs == 0 时两子指标皆 0.0。
    """
    per_tag = group_result["per_tag"]
    ok_pingze = int(sum(per_tag[t]["dz_ok_pingze"] for t in DUIZHANG_TAGS))
    ok_chongzi = int(sum(per_tag[t]["dz_ok_chongzi"] for t in DUIZHANG_TAGS))
    pairs = int(sum(per_tag[t]["dz_pairs"] for t in DUIZHANG_TAGS))
    dz_pingze = ok_pingze / pairs if pairs else 0.0
    dz_chongzi = ok_chongzi / pairs if pairs else 0.0
    return {"dz_ok_pingze": ok_pingze, "dz_ok_chongzi": ok_chongzi, "dz_pairs": pairs,
            "dz_pingze": dz_pingze, "dz_chongzi": dz_chongzi,
            "ci": {"dz_pingze": ci_halfwidth(dz_pingze, pairs),
                   "dz_chongzi": ci_halfwidth(dz_chongzi, pairs)}}


def judge_duizhang_discrimination(results):
    """对仗指标区分度自检：新旧模型律诗合并两子指标的差值（百分点绝对值）是否足以支撑指标有效性。

    须同时具备旧模型基线组（old）与新 M3 组（m3）方做自检；缺任一组即如实报「未自检」，不编造结论。
    新旧一律取律诗合并两子指标（merged_lvshi_duizhang）。旧模型联内不重字率 ≥ 90% 即该子指标
    近饱和、作废（改以对句平仄相对率单独作主指标）；已作废的子指标不参与有效性判定，故有效性
    只在参与判定的子指标（未作废者，见返回字段 judged_submetrics）中判定——任一差值 ≥ 10 个百分点
    判「有效」，否则判「指标失效」（不得靠已判无区分度的子指标救活整个指标）。返回 dict：未自检为
    {"checked": False, "reason": ...}；已自检含 checked/void_chongzi/judged_submetrics/valid、
    新旧各两子指标与两差值（百分点绝对值）。
    """
    missing = [k for k in ("old", "m3") if k not in results]
    if missing:
        if missing == ["m3"]:
            reason = "缺 M3 档，自检未做"
        elif missing == ["old"]:
            reason = "缺旧模型基线，指标有效性未自检"
        else:
            reason = "缺旧模型基线与 M3 档，自检未做"
        return {"checked": False, "reason": reason}
    old = merged_lvshi_duizhang(results["old"])
    m3 = merged_lvshi_duizhang(results["m3"])
    diff_pingze = abs(m3["dz_pingze"] - old["dz_pingze"]) * 100.0
    diff_chongzi = abs(m3["dz_chongzi"] - old["dz_chongzi"]) * 100.0
    void_chongzi = bool(old["dz_chongzi"] >= DUIZHANG_SATURATED_CHONGZI)
    # 已作废的子指标不参与有效性判定（否则会出现靠一个已判无区分度的子指标救活整个指标）
    judged_submetrics = ("pingze",) if void_chongzi else ("pingze", "chongzi")
    diffs = {"pingze": diff_pingze, "chongzi": diff_chongzi}
    return {"checked": True,
            "void_chongzi": void_chongzi,
            "judged_submetrics": judged_submetrics,
            "valid": bool(any(diffs[k] >= DUIZHANG_MIN_DIFF_PP for k in judged_submetrics)),
            "old_dz_pingze": old["dz_pingze"], "old_dz_chongzi": old["dz_chongzi"],
            "old_dz_pairs": old["dz_pairs"],
            "m3_dz_pingze": m3["dz_pingze"], "m3_dz_chongzi": m3["dz_chongzi"],
            "m3_dz_pairs": m3["dz_pairs"],
            "diff_pingze_pp": diff_pingze, "diff_chongzi_pp": diff_chongzi}


# ─────────────────────────────── 分组执行与闸门 ───────────────────────────────

def assert_same_device(devices):
    """断言全部分组的计算设备一致（numpy 与 cupy 是两套 RNG 流，混用则差异不可归因于模型）。"""
    uniq = sorted(set(devices))
    assert len(uniq) <= 1, f"A/B 各组计算设备不一致：{devices}（须在同一设备上跑）"


def run_group(kind, pool, constraint_on=None, new_tokens=NEW_TOKENS, tag_prefix=False):
    """跑一组：载入模型 → 生成 → 打分 + 按相位拆分押韵，返回结果字典（含设备、标签、约束开关、
    是否加诗体前缀）。

    相位拆分（§8.2 死规则 2026-09-15）一律取自本轮**已生成文本**（generate_texts 的返回值，
    不额外生成、不改抽样口径），随结果落进 rhyme_by_parity 供报告取用。
    tag_prefix=True 即 tag 臂：prompt 前拼诗体 token（由 generate_texts 负责），结果字典内如实
    记录该开关（tag_prefix 字段），供报告第九节区分两臂。
    """
    assert kind in GROUP_SPECS, f"未知分组 {kind}"
    path, spec_constraint, label = GROUP_SPECS[kind]
    if constraint_on is None:
        constraint_on = spec_constraint
    assert os.path.exists(path), f"模型档不存在：{path}"
    if kind == "old":
        model, stoi, itos = load_old_model(path)
    else:
        model, stoi, itos = load_new_model(path)
    texts = generate_texts(model, stoi, itos, pool, new_tokens, constraint_on, tag_prefix)
    res = score_group(texts, pool)
    res["rhyme_by_parity"] = split_rhyme_by_parity(texts, pool)
    res.update(kind=kind, model_path=path, constraint=constraint_on,
               device=np.__name__, label=label, tag_prefix=tag_prefix)
    return res


def default_kinds():
    """默认分组：旧模型 + M1 + M2（均约束关）；M3 档存在时追加其三组（约束关 + 约束开）。"""
    kinds = ["old", "m1", "m2"]
    if os.path.exists(MODEL_M3):
        kinds += ["m3", "m3_cons"]
    return kinds


def judge_gates(results):
    """闸门判定：M1（句长 ≥ 95% 且 押韵 ≥ 90%）、M2（平仄 ≥ 85% 且较 M1 提升）。

    M1 判定附旧模型同口径基线；M2 判定附 M1 对照值。返回 {组名: {pass, reason, ...}}。
    """
    out = {}
    m1, old, m2 = results.get("m1"), results.get("old"), results.get("m2")
    if m1 is not None:
        o = m1["overall"]
        ok = (o["line_len"] >= M1_MIN_LINE_LEN) and (o["rhyme"] >= M1_MIN_RHYME)
        out["m1"] = {
            "pass": bool(ok),
            "reason": (f"句长合规 {_pct(o['line_len'])}（阈值 {_pct(M1_MIN_LINE_LEN)}）、"
                       f"押韵合规 {_pct(o['rhyme'])}（阈值 {_pct(M1_MIN_RHYME)}）"),
            "line_len": o["line_len"], "rhyme": o["rhyme"],
            "baseline": (old["overall"] if old is not None else None),
        }
    if m2 is not None and m1 is not None:
        t, tm1 = m2["overall"]["tone"], m1["overall"]["tone"]
        ok = (t >= M2_MIN_TONE) and (t > tm1)
        out["m2"] = {
            "pass": bool(ok),
            "reason": (f"平仄合规 {_pct(t)}（阈值 {_pct(M2_MIN_TONE)}），"
                       f"M1 平仄 {_pct(tm1)}，差值 {_pct(t - tm1)}"),
            "tone": t, "tone_m1": tm1,
        }
    return out


def judge_parity_attribution(results):
    """相位归因判据（死规则 §8.2）：按各组已算出的 rhyme_by_parity 实算两子集率差与显著性，定结论措辞。

    判据：组内 |对齐子集率 − 错位子集率| 与「该组两子集 95% 置信区间半宽之和」（ci_halfwidth =
    1.96·√(p(1−p)/n)，两子集各自算半宽再相加）比较；差值超出该和者判该组差异显著。结论只有两支：
      - 存在某组「对齐子集显著高于错位子集」→ verdict = "needs_review"：点名该组实算事实，
        并写「须按 §九 停手报主子复核口径」；
      - 否则 → verdict = "not_attributable"：写「低押韵合规率不可归因于相位错位，闸门判定成立」。
    措辞一律只陈述实算事实与规格判据：既不得把闸门不达标误判为口径伪影，也不得拿取样口径当借口
    否认模型缺陷（两端同禁）。无任一组含 rhyme_by_parity 时判未检（checked = False），如实留空。

    返回 {"checked", "reason"（未检时）, "entries"（逐组实算事实）, "significant_aligned_higher"
    （显著偏高组的报告用名）, "directions_uniform", "verdict", "facts"（逐组两子集实测文案）,
    "lines"（判据文案）}；facts / lines 由终端打印与报告落盘共用，避免两处漂移。
    """
    entries, facts = [], []
    for kind in ("old", "m1", "m2", "m3", "m3_cons"):
        if kind not in results:
            continue
        r = results[kind]
        rbp = r.get("rhyme_by_parity")
        if rbp is None:
            facts.append(f"{r['label']}：本轮结果未含相位拆分（非正式入口），如实留空、不编造。")
            continue
        a, m = rbp["aligned"], rbp["misaligned"]
        hw = ci_halfwidth(a["rhyme"], a["n"]) + ci_halfwidth(m["rhyme"], m["n"])
        diff = a["rhyme"] - m["rhyme"]
        entries.append({"kind": kind, "label": r["label"],
                        "aligned_n": a["n"], "aligned_rhyme": a["rhyme"],
                        "misaligned_n": m["n"], "misaligned_rhyme": m["rhyme"],
                        "diff": diff, "hw_sum": hw,
                        "significant": bool(abs(diff) > hw),
                        "aligned_higher": bool(diff > 0.0)})
        facts.append(f"{r['label']}：对齐子集 n={a['n']} 押韵 {_pct(a['rhyme'])}；"
                     f"错位子集 n={m['n']} 押韵 {_pct(m['rhyme'])}")
    if not entries:
        return {"checked": False, "reason": "本轮各组结果均未含相位拆分",
                "entries": [], "significant_aligned_higher": [], "directions_uniform": True,
                "verdict": "unchecked", "facts": facts, "lines": []}
    sig_high = [e for e in entries if e["significant"] and e["aligned_higher"]]
    n_up = sum(1 for e in entries if e["diff"] > 0.0)
    n_down = sum(1 for e in entries if e["diff"] < 0.0)
    uniform = not (n_up and n_down)
    lines = ["判据（由本次实算推导，不写死）：组内两子集押韵合规率之差与「两子集 95% 置信区间半宽"
             "之和」（1.96·√(p(1−p)/n)，两子集各自算半宽后相加）逐组比较，差值超出该和者判该组差异显著。"]
    for e in entries:
        lines.append(f"{e['label']}：对齐 {_pct(e['aligned_rhyme'])}（n={e['aligned_n']}） − "
                     f"错位 {_pct(e['misaligned_rhyme'])}（n={e['misaligned_n']}） = "
                     f"{e['diff'] * 100.0:+.2f} 个百分点，该组置信区间半宽之和 "
                     f"{e['hw_sum'] * 100.0:.2f} 个百分点，判"
                     f"{'显著' if e['significant'] else '不显著'}")
    if sig_high:
        e = sig_high[0]
        lines.append(f"结论：{e['label']} 的对齐子集押韵 {_pct(e['aligned_rhyme'])}"
                     f"（n={e['aligned_n']}）显著高于错位子集押韵 {_pct(e['misaligned_rhyme'])}"
                     f"（n={e['misaligned_n']}），差值 {e['diff'] * 100.0:+.2f} 个百分点超出该组"
                     f"置信区间半宽之和 {e['hw_sum'] * 100.0:.2f} 个百分点；相位错位对该组押韵合规率"
                     "存在显著口径影响，须按 §九 停手报主子复核口径。")
        verdict = "needs_review"
    else:
        if uniform:
            extra = ("（各组差值方向一致，但均落在置信区间内，不构成显著差异）"
                     if len(entries) > 1 else "（该组差值落在置信区间内）")
        else:
            extra = f"（方向亦不一致：{n_up} 组对齐偏高、{n_down} 组错位偏高）"
        lines.append("结论：各组「对齐子集」与「错位子集」的押韵合规率差异均不显著"
                     f"{extra}，低押韵合规率不可归因于相位错位，闸门判定成立。")
        verdict = "not_attributable"
    return {"checked": True, "entries": entries,
            "significant_aligned_higher": [e["label"] for e in sig_high],
            "directions_uniform": uniform, "verdict": verdict,
            "facts": facts, "lines": lines}


def run_arm(kinds, pool, new_tokens, tag_prefix):
    """跑一臂的全部组：逐组载入 → 生成 → 打分，返回 ({组名: 结果}, [设备...])。

    每组生成前由 generate_texts 复位随机源，故「先裸臂后 tag 臂」与「只跑裸臂」的裸臂结果一致。
    """
    results, devices = {}, []
    for kind in kinds:
        res = run_group(kind, pool, new_tokens=new_tokens, tag_prefix=tag_prefix)
        results[kind] = res
        devices.append(res["device"])
        o = res["overall"]
        print(f"[评估{'·tag' if tag_prefix else ''}] {res['label']}：样本 {res['n']} | "
              f"句长 {_pct(o['line_len'])} 押韵 {_pct(o['rhyme'])} "
              f"首句入韵 {_pct(o['first_rhyme'])} 平仄 {_pct(o['tone'])} | "
              f"孤平 {o['gu_ping']} 三平调 {o['san_ping']}")
    return results, devices


def run_eval(pool, kinds=None, new_tokens=NEW_TOKENS, report_path=REPORT_PATH, pool_stats=None,
             arms=("bare",)):
    """执行全部分组评估、断言同设备、写盘中文报告，返回 {组名: 结果}。

    pool_stats 为 prompt 池的碰撞排除统计（prosody_eval.pool_exclusion_stats()；复用落盘池时
    亦由 build_eval_pool 重算写回，非 None），随 meta 传入写盘函数，供报告如实写明实际排除条数
    与排除后逐诗体样本量。meta 另附 half_parity = half_parity_stats(pool)（池内 prompt 取自
    原诗半句位的奇偶分布，实算），供报告第七节披露押韵对齐代价。

    arms 为臂名序列（取值 bare / tag）：裸臂必跑（报告第一至八节的唯一数据来源），tag 臂仅在
    arms 含 "tag" 时追加——待跑组先由 tag_arm_kinds 按档内词表过滤（不可执行者不生成、不评分），
    返回值仍是**裸臂**结果字典（向后兼容）；同设备断言覆盖两臂全部组。
    """
    kinds = kinds if kinds is not None else default_kinds()
    results, devices = run_arm(kinds, pool, new_tokens, tag_prefix=False)
    tag_results, tag_skipped = None, ()
    if "tag" in arms:
        runnable, tag_skipped = tag_arm_kinds(kinds)
        tag_results, tag_devices = run_arm(runnable, pool, new_tokens, tag_prefix=True)
        devices += tag_devices
    assert_same_device(devices)
    # 终端同屏打印押韵口径相位披露的两组实算（报告为准）：真诗同口径上限 + 各组按相位拆分。
    ceil = real_poem_ceiling_stats(pool)
    print(f"[相位] 真诗同口径上限：对齐截断 n={ceil['aligned']['n']} "
          f"押韵 {_pct(ceil['aligned']['rhyme'])} 句长 {_pct(ceil['aligned']['line_len'])}；"
          f"错位截断 n={ceil['misaligned']['n']} 押韵 {_pct(ceil['misaligned']['rhyme'])} "
          f"句长 {_pct(ceil['misaligned']['line_len'])}；池加权上限 押韵 "
          f"{_pct(ceil['pool_weighted_rhyme'])} 句长 {_pct(ceil['pool_weighted_line_len'])}"
          f"（池 even {ceil['pool_even']} / odd {ceil['pool_odd']}）")
    # 逐组两子集实测文案与归因结论文案一律取自判据函数（终端与报告第八节第 2、3 条同源，避免两处漂移）。
    par = judge_parity_attribution(results)
    for ln in par["facts"] + par["lines"]:
        print(f"[相位] {ln}")
    per_tag_n = {tag: 0 for tag in POOL_TAGS}
    for tag, _ in pool:
        per_tag_n[tag] += 1
    meta = {"pool_path": POOL_PATH, "val_path": VAL_PATH, "corpus_path": CORPUS_PATH, "seed": SEED,
            "new_tokens": new_tokens, "temperature": TEMPERATURE, "top_k": TOP_K,
            "n_per_tag": N_PER_TAG, "per_tag_n": per_tag_n, "device": devices[0],
            "pool_stats": pool_stats, "half_parity": half_parity_stats(pool)}
    write_report(results, report_path, meta, tag_results=tag_results, tag_skipped=tuple(tag_skipped))
    print(f"[报告] 已写入 {report_path}")
    return results


# ─────────────────────────────── 报告落盘 ───────────────────────────────

def _pct(x):
    """把 0..1 的比例格式化为百分数文本。"""
    return f"{x * 100.0:.2f}%"


def _pm(rate, halfwidth):
    """把比例与其 95% 置信区间半宽格式化（如 96.25%±2.98%）。"""
    return f"{_pct(rate)}±{_pct(halfwidth)}"


ARMS = ("bare", "tag")                     # 评估臂：裸臂（基线）/ tag 臂（诊断）
TAG_ARM_METRICS = (("line_len", "句长"), ("rhyme", "押韵"), ("tone", "平仄"))
TAG_ARM_RHYME_BAND = (0.06, 0.08)          # 第二问判据带：§十四 M1 裸臂押韵实测 6.74% 所在量级
TAG_ARM_REDLINE = (
    "本节的 tag 臂仅作诊断对照，绝不替换、绝不修改闸门口径，绝不用以重判 M1；§十四 的闸门结论与"
    "全部阈值继续有效，本步不作任何改动。押韵在「对齐子集」上实测 6.93%、真诗同口径基准 65.78%"
    "（差 -58.85 个百分点，见第八节），已实证为真实缺陷，不得以取样口径为由改判。")


def _pp(x):
    """把两个比例之差格式化为带符号的百分点文本（如 +44.00）。"""
    return f"{x * 100.0:+.2f}"


def _threshold_pct(x):
    """把阈值格式化为整数百分点文本（如 95%），供直答问句使用。"""
    return f"{x * 100.0:.0f}%"


def _tag_arm_answers(bare, tag):
    """直答两问的正文行：① tag 臂 M1 句长是否 ≥ M1_MIN_LINE_LEN；② 押韵是否仍落在
    TAG_ARM_RHYME_BAND。缺任一侧数据时如实留空、不编造。"""
    if "m1" not in bare or "m1" not in tag:
        return ["     M1 组在某一臂缺数据，两问如实留空、不编造。"]
    b, t = bare["m1"]["overall"], tag["m1"]["overall"]
    lo, hi = TAG_ARM_RHYME_BAND
    q1 = "是" if t["line_len"] >= M1_MIN_LINE_LEN else "否"
    q2 = "是" if lo <= t["rhyme"] <= hi else "否"
    return [
        f"     ① tag 臂 M1 句长 {_pct(t['line_len'])}（裸臂 {_pct(b['line_len'])}，"
        f"差值 {_pp(t['line_len'] - b['line_len'])} 个百分点）"
        f"是否 ≥ {_threshold_pct(M1_MIN_LINE_LEN)}：{q1}",
        f"     ② tag 臂 M1 押韵 {_pct(t['rhyme'])}（裸臂 {_pct(b['rhyme'])}，"
        f"差值 {_pp(t['rhyme'] - b['rhyme'])} 个百分点）是否仍落在 "
        f"{_threshold_pct(lo)}–{_threshold_pct(hi)} 区间（即无实质提升）：{q2}",
    ]


def tag_arm_section(bare, tag, skipped=()):
    """生成报告第九节的正文行（纯函数，供单测直接复算；不含节标题）。

    逐组给 句长/押韵/平仄 的「裸臂值 / tag 臂值 / 差值（百分点）」，差值一律 = tag 臂 − 裸臂；
    skipped 中的组如实注明不可执行、不输出任何对照数字。所有数字一律取自入参结果字典，不写死。
    """
    lo, hi = TAG_ARM_RHYME_BAND
    L = ["  1. 两臂定义与同源约束：",
         "     裸臂 = prompt 原文（与第一至八节口径逐字一致）；tag 臂 = 该 prompt 所属诗体 token"
         "前缀 + prompt（如「<五绝>新晴花枝下」）。",
         "     同源：同一 prompt 池 / 同一 SEED / 同一 new_tokens / 同一 temperature·top_k / 同一设备；"
         "池、碰撞排除、分层抽样、闸门阈值一律未动。",
         "     评分口径：两臂共用 prosody.py 既有评分函数，一行未改；四项指标全部尾锚"
         "（句长 _trailing_hanzi、平仄 seg[-n:]、押韵 s[-1]、首句入韵 segs[0][-1]），"
         "诗体 token 位于半句首部（「<」「>」非汉字），故对四项指标无影响。",
         f"  2. 两臂各组总表（差值 = tag 臂 − 裸臂，单位：个百分点；"
         f"比例指标附 95% 置信区间半宽；押韵参考带 {_threshold_pct(lo)}–{_threshold_pct(hi)}）："]
    for kind, res in bare.items():
        label = res["label"]
        if kind in skipped:
            L.append(f"     {label}：{TAG_ARM_UNAVAILABLE_NOTE}")
            continue
        if kind not in tag:
            L.append(f"     {label}：本轮 tag 臂无该组数据，如实留空、不编造。")
            continue
        for key, name in TAG_ARM_METRICS:
            b, tv = res["overall"][key], tag[kind]["overall"][key]
            bci, tci = res["overall"]["ci"][key], tag[kind]["overall"]["ci"][key]
            L.append(f"     {label} {name}：裸臂 {_pct(b)}±{_pct(bci)} / "
                     f"tag 臂 {_pct(tv)}±{_pct(tci)} / 差值 {_pp(tv - b)} 个百分点")
    L.append("  3. 逐诗体明细（n 为该诗体样本量；n < 30 者不作为判据；"
             "差值 = tag 臂 − 裸臂，单位：百分点）：")
    for kind, res in bare.items():
        if kind in skipped or kind not in tag:
            continue
        for tg in POOL_TAGS:
            b, t = res["per_tag"][tg], tag[kind]["per_tag"][tg]
            note = "（n < 30，不作判据）" if b["n"] < MIN_SAMPLES_PER_TAG else ""
            cells = " ".join(
                f"{name} 裸臂 {_pct(b[key])} / tag 臂 {_pct(t[key])} / 差值 {_pp(t[key] - b[key])}"
                for key, name in TAG_ARM_METRICS)
            L.append(f"     {res['label']} {tg} n={b['n']} {cells}{note}")
    L.append("  4. 直答两问（取值一律出自本节第 2、3 条实算，不外推）：")
    L.extend(_tag_arm_answers(bare, tag))
    L.append(f"  5. 红线声明：{TAG_ARM_REDLINE}")
    return L


def _samples_ok(meta):
    """样本量是否满足硬下限（每诗体 ≥ 30 且 合计 ≥ 120）。"""
    n = meta["per_tag_n"]
    return all(n[tag] >= MIN_SAMPLES_PER_TAG for tag in POOL_TAGS) \
        and sum(n.values()) >= MIN_SAMPLES_TOTAL


def write_report(results, path, meta, tag_results=None, tag_skipped=()):
    """把评估结果写盘为中文报告，返回路径。

    报告固定包含：生成口径（含碰撞排除的实际排除条数与排除后逐诗体供给量）、各组指标
    （附置信区间，另附只计律诗的对仗平仄相对率/联内不重字率）与分诗体明细（绝句标注对仗不适用）、
    M1/M2 闸门判定（附旧模型同口径基线）、三组 A/B 对照结构、对仗区分度自检结论与口径声明
    （含 val 语料重复致乐观偏差的实测披露，以及裸半句 prompt、续写超长、押韵对齐三项口径局限
    的实算数字——数字一律由 corpus_tag_lead_stats / poem_token_len_stats / half_parity_stats 实算，
    不写死字面量）。
    M3 档未就绪时第 2、3 组标注「待 M3」并留空，区分度自检标注「未自检」，一律不编造数字。
    tag_results 非 None 时（tag 臂已跑）在第八节之后**纯追加**第九节「诗体前缀口径对照」
    （由 tag_arm_section 渲染，tag_skipped 中的组如实注明 tag 臂不可执行）；为 None 时第一至八节
    逐字节不变。
    """
    L = []
    A = L.append
    A("=" * 78)
    A("格律内化重训练 A/B 评估报告（T11：评估与 A/B 对照）")
    A("=" * 78)
    A("")
    A("一、生成口径（各组一致）")
    A(f"  1. prompt 池：{meta['pool_path']}；只从 {meta['val_path']} 抽取该诗任一合格半句"
      "（半句 = 该诗按 prosody.DEFAULT_PUNCT 切分后的任一段，须纯汉字且 5/7 字，同诗体内去重），"
      "<杂言> 不入池")
    A("  2. 分层抽样：每诗体目标 %d 条；实际 %s；合计 %d 条"
      % (meta["n_per_tag"],
         "、".join(f"{t} {meta['per_tag_n'][t]} 条" for t in POOL_TAGS),
         sum(meta["per_tag_n"].values())))
    A(f"  3. 续写长度 new_tokens = {meta['new_tokens']}；采样 temperature = "
      f"{meta['temperature']}，top_k = {meta['top_k']}；随机种子 = {meta['seed']}"
      "（各组生成前复位，共享同一采样流）")
    A(f"  4. 计算设备 = {meta['device']}（各组必须一致，不一致即报错）")
    A("  5. 指标口径：句长/押韵/平仄一律取自 prosody 既有函数（字音判据单一来源）；"
      "押韵主指标 = 偶半句末字同韵部率")
    A("     （未覆盖韵脚字从分子与分母一并剔除），首句入韵单独统计；平仄必论位 = "
      "2/4/6 位 + 句末字（一三五不论），")
    A("     孤平与三平调违例数单列。")
    ps = meta.get("pool_stats")
    if not ps:
        # 仅当调用方未提供池统计（外部只写报告的调用或单测）时走到；正式入口 run_eval 恒传入由
        # 复用/重建路径统一重算出的完整统计，故正式报告不会出现本分支，更不会静默吞掉排除条数。
        A("  6. 碰撞排除：调用方未提供池统计（非正式入口），排除统计未记录。")
    else:
        A("  6. 碰撞排除（候选半句字符串出现在 %s 者剔除、不入池——此类文本模型已训练过，"
          "不满足「模型未训练过」的前提）："
          % os.path.basename(meta.get("corpus_path", CORPUS_PATH)))
        A("     排除前碰撞 %d 条（%s）；排除后逐诗体候选供给量 %s 条；实际入池每诗体见上条 2。"
          % (ps["collision_total"],
             "、".join(f"{t} {ps['collision'][t]}" for t in POOL_TAGS),
             "、".join(f"{t} {ps['supply_after'][t]}" for t in POOL_TAGS)))
    A("")
    A("二、各组指标（比例指标附 95% 置信区间半宽 1.96·√(p(1−p)/n)）")
    A("  组号沿用第五节 A/B 对照编号（1=旧模型、2=新 M3 模型、3=新 M3 模型（约束开））；"
      "M1/M2 为训练中间档，不占 A/B 组号")
    for kind in ("old", "m1", "m2", "m3", "m3_cons"):
        if kind not in results:
            if kind in _AB_NO:                   # M3 未就绪：如实留空占位，不编造数字
                A(f"  组 {_AB_NO[kind]}  {GROUP_SPECS[kind][2]}：待 M3"
                  "（model-gelv-m3.npz 未就绪，如实留空）")
            continue
        r = results[kind]
        o = r["overall"]
        m = merged_lvshi_duizhang(r)
        head = f"  组 {_AB_NO[kind]}" if kind in _AB_NO else "  训练中间档"
        A(f"{head}  {r['label']}  样本量 n={r['n']}")
        A(f"        句长 {_pm(o['line_len'], o['ci']['line_len'])}  "
          f"押韵 {_pm(o['rhyme'], o['ci']['rhyme'])}  "
          f"首句入韵 {_pm(o['first_rhyme'], o['ci']['first_rhyme'])}  "
          f"平仄 {_pm(o['tone'], o['ci']['tone'])}  孤平 {o['gu_ping']}  三平调 {o['san_ping']}  "
          f"对仗平仄相对率 {_pm(m['dz_pingze'], m['ci']['dz_pingze'])} / "
          f"联内不重字率 {_pm(m['dz_chongzi'], m['ci']['dz_chongzi'])}"
          f"（n=可比联数 {m['dz_pairs']}，仅计律诗）")
    A("")
    A("  分诗体明细（n 为该诗体样本量；n < %d 者不作为闸门依据）：" % MIN_SAMPLES_PER_TAG)
    for kind in ("old", "m1", "m2", "m3", "m3_cons"):
        if kind not in results:
            continue
        A(f"    {results[kind]['label']}")
        for tag in POOL_TAGS:
            t = results[kind]["per_tag"][tag]
            flag = "  样本不足" if t["n"] < MIN_SAMPLES_PER_TAG else ""
            if tag in DUIZHANG_TAGS:
                dz_txt = (f"对仗平仄相对率 {_pm(t['dz_pingze'], t['ci']['dz_pingze'])} / "
                          f"联内不重字率 {_pm(t['dz_chongzi'], t['ci']['dz_chongzi'])}"
                          f"（n=可比联数 {t['dz_pairs']}）")
            else:
                dz_txt = "对仗不适用（绝句无对仗要求）"
            A(f"      {tag}  n={t['n']}  句长 {_pm(t['line_len'], t['ci']['line_len'])}  "
              f"押韵 {_pm(t['rhyme'], t['ci']['rhyme'])}  "
              f"首句入韵 {_pm(t['first_rhyme'], t['ci']['first_rhyme'])}  "
              f"平仄 {_pm(t['tone'], t['ci']['tone'])}  {dz_txt}{flag}")
    A("")
    gates = judge_gates(results)
    A("三、M1 闸门（句长合规 ≥ %s 且 押韵合规 ≥ %s；附旧模型同口径基线）"
      % (_pct(M1_MIN_LINE_LEN), _pct(M1_MIN_RHYME)))
    if "m1" in gates:
        g = gates["m1"]
        if g["baseline"] is not None:
            b = g["baseline"]
            A(f"  旧模型基线（约束关，n={b['n']}）：句长 {_pct(b['line_len'])}"
              f"±{_pct(b['ci']['line_len'])}，押韵 {_pct(b['rhyme'])}"
              f"±{_pct(b['ci']['rhyme'])}")
        A(f"  M1 模型（约束关）：句长 {_pct(g['line_len'])}，押韵 {_pct(g['rhyme'])}")
        A(f"  判定：{'达标' if g['pass'] else '未达标'}（{g['reason']}）")
    else:
        A("  M1 模型未纳入本次评估，暂不出判定。")
    A("")
    A("四、M2 闸门（平仄合规 ≥ %s 且较 M1 显著提升；孤平/三平调违例单列）"
      % _pct(M2_MIN_TONE))
    if "m2" in gates:
        g = gates["m2"]
        A(f"  判定：{'达标' if g['pass'] else '未达标'}（{g['reason']}）")
    else:
        A("  M2 模型未纳入本次评估，暂不出判定。")
    A("")
    A("五、三组 A/B 对照表（旧模型约束关 / 新 M3 模型约束关 / 新 M3 模型约束开）")
    for idx, kind, name, cons in AB_GROUPS:
        if kind in results:
            r = results[kind]
            o = r["overall"]
            A(f"  组 {idx}  {name} + 约束{cons}：n={r['n']}，句长 {_pct(o['line_len'])}，"
              f"押韵 {_pct(o['rhyme'])}，首句入韵 {_pct(o['first_rhyme'])}，"
              f"平仄 {_pct(o['tone'])}，孤平 {o['gu_ping']}，三平调 {o['san_ping']}")
        else:
            A(f"  组 {idx}  {name} + 约束{cons}：待 M3（model-gelv-m3.npz 未就绪，如实留空）")
    A("")
    A("六、对仗必要条件率区分度自检（新指标有效性前置校验）")
    A("  口径：对仗指标只计律诗（<五律>/<七律>），绝句无对仗要求、不入该指标；按联级合并，"
      "率 = Σ分子联数 / Σ可比联数（严禁逐样本率再求平均）；")
    A("        置信区间以可比联数为 n（同一首诗的样本非独立，区间略偏乐观）。")
    dz = judge_duizhang_discrimination(results)
    if not dz["checked"]:
        A(f"  未自检：{dz['reason']}；本轮不得以对仗指标充当 M3 闸门依据。")
    else:
        A(f"  旧模型（律诗合并，n=可比联数 {dz['old_dz_pairs']}）："
          f"对句平仄相对率 {_pct(dz['old_dz_pingze'])}，联内不重字率 {_pct(dz['old_dz_chongzi'])}")
        A(f"  新 M3 模型（律诗合并，n=可比联数 {dz['m3_dz_pairs']}）："
          f"对句平仄相对率 {_pct(dz['m3_dz_pingze'])}，联内不重字率 {_pct(dz['m3_dz_chongzi'])}")
        A(f"  新旧差值（百分点绝对值）：对句平仄相对率 {dz['diff_pingze_pp']:.2f} 个百分点，"
          f"联内不重字率 {dz['diff_chongzi_pp']:.2f} 个百分点")
        if dz["void_chongzi"]:
            A(f"  【子指标作废】旧模型联内不重字率 {_pct(dz['old_dz_chongzi'])} ≥ "
              f"{_pct(DUIZHANG_SATURATED_CHONGZI)}（近饱和），联内不重字率已作废，"
              "不参与有效性判定；有效性仅由对句平仄相对率判定。")
        A("  参与有效性判定的子指标：" + "、".join(
            {"pingze": "对句平仄相对率", "chongzi": "联内不重字率"}[k]
            for k in dz["judged_submetrics"]))
        if dz["valid"]:
            A(f"  判定：指标有效（参与判定的子指标差值至少一项 ≥ "
              f"{DUIZHANG_MIN_DIFF_PP:.0f} 个百分点），可作为 M3 闸门依据。")
        else:
            A(f"  判定：【指标失效】参与判定的子指标差值均 < {DUIZHANG_MIN_DIFF_PP:.0f} 个百分点，"
              "对仗指标区分度不足，不得充当 M3 闸门依据。")
    A("")
    A("七、口径声明与遗留")
    if _samples_ok(meta):
        A("  1. 样本量满足硬下限（每诗体 ≥ %d 且 合计 ≥ %d），闸门判定可用。"
          % (MIN_SAMPLES_PER_TAG, MIN_SAMPLES_TOTAL))
    else:
        short = [t for t in POOL_TAGS if meta["per_tag_n"][t] < MIN_SAMPLES_PER_TAG]
        A("  1. 【样本不足】以下诗体未达硬下限 %d 条：%s；合计 %d 条（下限 %d）。"
          % (MIN_SAMPLES_PER_TAG, "、".join(short) if short else "无",
             sum(meta["per_tag_n"].values()), MIN_SAMPLES_TOTAL))
        A("     上述合规率仅作参考，不得作为放行依据（置信区间过宽）。")
    A("  2. 对仗必要条件率（对句平仄相对率 + 联内不重字率）已实现，真实口径为：只计律诗"
      "（<五律>/<七律>；绝句无对仗要求，标「不适用」而非记 0）；")
    A("     按联级合并（率 = Σ分子联数 / Σ可比联数，非逐样本率平均）；置信区间以可比联数为 n"
      "（同诗样本非独立，区间略偏乐观）；须先过区分度自检，")
    A("     新旧差值过小即判指标失效，失效期间不得充当 M3 闸门依据；两子指标取自 "
      "prosody.check_duizhang，本工具不另立字音判据。")
    A("  3. 押韵主指标按「主韵部一致」判定，不做邻韵豁免；未覆盖韵脚字从分子与分母一并剔除。")
    A("  4. 取样口径代价：同一首诗可贡献多条 prompt，样本之间并非完全独立，比例指标置信区间"
      "略偏乐观，不得当作完全独立样本解读。")
    vb = val_corpus_duplicate_stats()
    A(f"  5. val 语料重复致乐观偏差（T1 遗留，已实证）：{os.path.basename(VAL_PATH)} 与 "
      f"{os.path.basename(CORPUS_PATH)} 有 {vb['dup_poems']} 首整诗逐字相同"
      f"（{vb['dup_poems']}/{vb['total_poems']} = {_pct(vb['ratio'])}），系诗级划分未去重；"
      "M1/M2/M3 的 val 指标含该比例乐观偏差，不得当作完全独立、未污染的评估集解读。")
    A("     prompt 侧泄漏已由碰撞排除消除（见第一节第 6 条）；根治（诗级去重 + 重建语料 + "
      "重训 M1/M2）代价过高，本轮不做。")
    corpus_name = os.path.basename(meta.get("corpus_path", CORPUS_PATH))
    cb = corpus_tag_lead_stats(meta.get("corpus_path", CORPUS_PATH))
    A(f"  6. 口径局限（裸半句）：评估 prompt 为裸半句（不含诗体 token），而 {corpus_name} 中 "
      f"{cb['tag_leading']}/{cb['total_poems']} 首（{_pct(cb['ratio'])}）均以诗体 token 起首；"
      "模型无从得知「该诗体应有字数」，句长合规率含该口径代价。")
    tl = poem_token_len_stats(meta.get("corpus_path", CORPUS_PATH))
    A(f"  7. 口径局限（续写超长）：new_tokens = {meta['new_tokens']} 超出一首所需（"
      + "、".join(f"{t} {tl[t]}" for t in POOL_TAGS if t in tl)
      + " token）；尾部会续出第二首（无诗体 token），其半句一并计入句长/押韵分母。")
    hp = meta.get("half_parity")
    if hp is None:
        # 外部只写报告的调用（正式入口 run_eval 恒传入实算结果）：按落盘池如实重算，不编造数字。
        pp = meta.get("pool_path")
        hp = half_parity_stats(read_pool_file(pp if pp and os.path.exists(pp) else POOL_PATH))
    A(f"  8. 口径局限（押韵对齐）：押韵主指标为「偶半句（第 2/4/6/8 半句）末字同韵部率」；"
      "若 prompt 取自原诗奇数位半句（0 基半句下标为奇，即第 2/4/6/8 半句），"
      "续写文本的半句奇偶与原诗韵位错位。实算：池 "
      f"{hp['total']} 条中取自偶数位半句（0 基下标 0/2/4/6，即第 1/3/5/7 半句）"
      f"{hp['even']} 条（{_pct(hp['even_ratio'])}）、取自奇数位半句（0 基下标 1/3/5/7，"
      f"即第 2/4/6/8 半句）{hp['odd']} 条（{_pct(hp['odd_ratio'])}）。")
    A("")
    A("八、押韵口径相位披露（死规则 §8.2，2026-09-15 追加：两组实算，不得只作定性声明）")
    # 真诗同口径上限：数字一律由 real_poem_ceiling_stats 实算（不写死字面量）；池相位分布复用
    # half_parity_stats（与第七节第 8 条同一来源）。句长按诗体 TAG_LEN 取值，禁止用首半句字数推断。
    pp_pool = meta.get("pool_path")
    pool_for_ceiling = read_pool_file(pp_pool if pp_pool and os.path.exists(pp_pool) else POOL_PATH)
    ceil = real_poem_ceiling_stats(pool_for_ceiling)
    # 归因判据（逐组两子集率差、置信区间半宽之和与结论）一律由 judge_parity_attribution 实算，
    # 终端打印与本节第 2、3 条同源；结论措辞禁止写死，也禁止为模型的低合规率开脱。
    par = judge_parity_attribution(results)
    A("  1. 真诗同口径上限（完美续写基准）：把 %s 四诗体真诗按半句位截断续写"
      "（「完美模型」输出即原诗余下部分，受评文本 = 截断半句起至诗末的半句串，"
      "与评估侧「prompt + 续写」同构）；句长一律按该诗体句长取值"
      "（<五绝>/<五律> 5 字、<七绝>/<七律> 7 字，不得用首半句字数推断——否则五律被误当 7 字）；"
      "押韵/句长一律复用 prosody.check_rhyme / check_line_len。实算（共 %d 首）："
      % (os.path.basename(meta.get("val_path", VAL_PATH)), ceil["poems"]))
    A(f"     对齐截断：n={ceil['aligned']['n']}  押韵 {ceil['aligned']['rhyme'] * 100:.2f}%  "
      f"句长 {ceil['aligned']['line_len'] * 100:.2f}%"
      "（0 基半句下标为偶，即第 1/3/5/7 半句）")
    A(f"     错位截断：n={ceil['misaligned']['n']}  押韵 {ceil['misaligned']['rhyme'] * 100:.2f}%  "
      f"句长 {ceil['misaligned']['line_len'] * 100:.2f}%"
      "（0 基半句下标为奇，即第 2/4/6/8 半句，正是押韵主指标所计的韵位）")
    A(f"     池加权上限（按本题 prompt 池实际相位分布加权：偶数位 {ceil['pool_even']} 条 / "
      f"奇数位 {ceil['pool_odd']} 条，合计 {ceil['pool_total']} 条）："
      f"押韵池加权上限 {ceil['pool_weighted_rhyme'] * 100:.2f}%，"
      f"句长池加权上限 {ceil['pool_weighted_line_len'] * 100:.2f}%")
    A("     用途（判定押韵主指标本身是否有效）：真诗在对齐截断下的率值即「完美续写」下该指标"
      "所能达到的基准值，与模型「对齐子集」实算值同源可比；")
    A("     模型低分若同时出现在对齐子集上，即说明低分与相位错位无关。本基准只用于判定指标"
      "有效性，不得以之为模型的低押韵合规率开脱。")
    A("     模型对齐子集实算值 vs 真诗对齐基准（差值 = 模型值 − 基准）：")
    if par["entries"]:
        for e in par["entries"]:
            gap = (e["aligned_rhyme"] - ceil["aligned"]["rhyme"]) * 100.0
            A(f"       {e['label']}：对齐子集 {_pct(e['aligned_rhyme'])}（n={e['aligned_n']}） vs "
              f"真诗对齐基准 {_pct(ceil['aligned']['rhyme'])}（n={ceil['aligned']['n']}），"
              f"差值 {gap:+.2f} 个百分点")
    else:
        A("       本轮无含相位拆分的组，如实留空、不编造。")
    A("  2. 模型实测按相位拆分（相位定位复用 val_half_index，与池一一对应；数据一律取自本轮"
      "已生成文本，不额外生成、不改抽样口径）：")
    for ln in par["facts"]:
        A(f"     {ln}")
    if par["lines"]:
        A(f"  3. {par['lines'][0]}")
        for ln in par["lines"][1:]:
            A(f"     {ln}")
    else:
        A(f"  3. 判据未检：{par['reason']}；本轮不得据此判定相位归因，如实留空、不编造。")
    if tag_results is not None:
        A("九、诗体前缀口径对照（诊断；2026-09-16 主子准行甲案：本步不改闸门口径、不重判 M1）")
        for ln in tag_arm_section(results, tag_results, tag_skipped):
            A(ln)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


def parse_arms(text):
    """解析 --arms：逗号分隔的臂名序列，去空白、去重保序；空序列、非法臂名、缺裸臂一律响亮失败。

    裸臂是报告第一至八节的唯一数据来源，故不可省略（省去即无法做「零漂移」自证）。
    """
    arms = tuple(dict.fromkeys(a.strip() for a in text.split(",") if a.strip()))
    assert arms, "--arms 不得为空（取值 bare / tag）"
    bad = [a for a in arms if a not in ARMS]
    assert not bad, f"未知臂名 {bad}（取值 {' / '.join(ARMS)}）"
    assert "bare" in arms, "裸臂是报告第一至八节的唯一数据来源，不可省略"
    return arms


def main(argv=None):
    """命令行入口：构建/读取 prompt 池 → 运行指定分组 → 写盘报告。"""
    ap = argparse.ArgumentParser(description="格律内化评估与 A/B 对照（旧模型 / M1 / M2 / M3）")
    ap.add_argument("--models", default=None,
                    help="分组列表（逗号分隔，取值 old,m1,m2,m3,m3_cons）；默认根据档存在性自动选择")
    ap.add_argument("--limit", type=int, default=None,
                    help="每诗体仅取前 N 条 prompt（冒烟用）；不给则用池内全部")
    ap.add_argument("--n-per-tag", type=int, default=N_PER_TAG, help="每诗体 prompt 条数")
    ap.add_argument("--new-tokens", type=int, default=NEW_TOKENS, help="每首续写 token 数")
    ap.add_argument("--report", default=REPORT_PATH, help="报告落盘路径")
    ap.add_argument("--arms", default="bare,tag",
                    help="评估臂（逗号分隔，取值 bare / tag）；默认两臂全跑作口径对照")
    args = ap.parse_args(argv)

    pool = build_eval_pool(n_per_tag=args.n_per_tag)
    pool = take_per_tag(pool, args.limit)
    kinds = args.models.split(",") if args.models else default_kinds()
    arms = parse_arms(args.arms)
    print(f"[臂] {' + '.join(arms)}")
    print(f"[池] {len(pool)} 条（" + "、".join(
        f"{t} {sum(1 for x, _ in pool if x == t)}" for t in POOL_TAGS) + "）")
    run_eval(pool, kinds, new_tokens=args.new_tokens, report_path=args.report,
             pool_stats=pool_exclusion_stats(), arms=arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
