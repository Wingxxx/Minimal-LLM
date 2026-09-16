# -*- coding: utf-8 -*-
"""全唐诗语料构建管线：JSON 分卷 → 简体纯诗句流 corpus.txt + 独立 val.txt + 逐位置元信息 meta.npz

数据源：chinese-poetry 仓库《全唐诗》（繁体存储，poet.tang.{0..57000}.json，每卷 1000 首，
约 5.7 万首）。本脚本做五件事：
  1) 字段级过滤：只取每条记录的 paragraphs（正文联句），丢弃 title/author/id/tags——
     组诗序号（如"帝京篇十首 一"）等标题噪声随 title 一并去除
  2) 繁→简 + 校注剥离 + 字符白名单：opencc t2s 统一字形；段落内圆括号（…）及其嵌套
     内容整体剥离（诗句正文不使用括号，括号内是敦煌卷子/拾遗诗的校记、异文、出处注）；
     剥净后仍 ≥30 字或含引号/冒号/书名号的行判为叙述性校记注，整段丢弃。
     依赖 opencc-python-reimplemented（一次性转换依赖，仅数据管线用，不污染运行时）：
         pip install opencc-python-reimplemented
     该依赖惰性导入（见 _converter），使本模块可在未装 opencc 的环境下被单测导入。
  3) 诗体标签：每首诗首行前插入一个诗体控制 token（POEM_TAGS 之一），
     口径为「每行 = 一联 = 两个半句」→ 绝句 2 行、律诗 4 行（详见 tag_of）
  4) 诗级划分：全量诗随机打乱（固定 seed 可复现），末尾 400 首留作独立验证集 val.txt，
     其余为训练集 corpus.txt——训练/验证同构（一联一行，诗间不插空行），
     验证集不参与训练，val loss 用于分辨"学规律"还是"背语料"
  5) 逐位置元信息 meta.npz：与训练集语料同一次遍历产出 weights / tone_labels /
     rhyme_labels / zone_tags 四个等长数组（对应 data/corpus.txt，val.txt 不产 meta），
     供训练期纯切片取用，避免逐批解析格律的解释开销（详见 build_meta）
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（脚本在 tools/ 下）
if BASE not in sys.path:        # 使项目根下的 prosody 可导入（平仄/韵部口径的单一来源）
    sys.path.insert(0, BASE)

import numpy as np
import prosody

# Windows 控制台常为 GBK：正文里可能出现 GBK 外字符（CJK 扩展区等），
# 打印时以 \\uXXXX 转义显示而不是抛 UnicodeEncodeError 崩溃
try:
    sys.stdout.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass

RAW_GLOB = os.path.join(BASE, "_probe", "raw", "poet.tang.*.json")
TRAIN_OUT = os.path.join(BASE, "data", "corpus.txt")
VAL_OUT = os.path.join(BASE, "data", "val.txt")
META_OUT = os.path.join(BASE, "data", "meta.npz")
VAL_N = 400          # 留出验证诗数
SEED = 0             # 打乱种子（可复现划分）
# 正文允许字符白名单：汉字 + 中文标点 + 半角标点 + 空白。
# 括号（全/半角）不在白名单——诗句正文不使用括号，其角色只是校注的界符，
# 先经剥括号步骤整体移除，残余孤立括号在此被滤除。
ALLOWED = set("，。！？；：、…「」『』《》〈〉——…·,.;:!? '\"")
# 诗体控制 token（单一来源，防漂移）：顺序须与词表特殊 token 追加顺序逐项相同
# （train.SPECIAL_TOKENS 须与之逐项相等，由 test/test_corpus.py 断言锁定）。
POEM_TAGS = ("<五绝>", "<七绝>", "<五律>", "<七律>", "<杂言>")
# zone_tags 取值含义（下标即取值）：0=常态 1=韵脚 2=半句末标点 3=特殊 token 4=对仗区
ZONE_NAMES = ("常态", "韵脚", "半句末标点", "特殊token", "对仗区")

_CC = None           # opencc t2s 转换器缓存（惰性构造；未装 opencc 不影响模块导入）


def _converter():
    """惰性构造并缓存 opencc 繁→简转换器（仅数据构建流程使用）。"""
    global _CC
    if _CC is None:
        from opencc import OpenCC
        _CC = OpenCC("t2s")
    return _CC


def clean_para(p, weird):
    """清洗单个段落：繁→简 → 剥圆括号校注 → 白名单过滤 → 诗词判据。

    返回清洗后字符串；若整段判为叙述性校记注则返回 None（调用方丢弃该段）。
    流程：
      1) opencc t2s 整段一次转换（繁体归简体）
      2) 剥除（…）及嵌套：诗句正文不使用括号，括号内是拾遗/敦煌卷子诗的
         校记、异文、出处注，整体剥离以还原纯诗句（循环剥最内层到无可剥）
      3) 逐字符过白名单：只留汉字与中文标点；被滤字符计入 weird
      4) 判据：剥净后单行仍 ≥30 字（长叙述），或仍含「」『』：冒号（引文/冒号
         只出现在校记叙述、诗句不用）→ 判为注记整段丢弃
    """
    s = _converter().convert(p)
    # 剥配对圆括号（支持嵌套，剥到不再变化为止）
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"[（(][^（）()]*[）)]", "", s)
    keep = []
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff" or ch in ALLOWED:
            keep.append(ch)
        else:
            weird[ch] += 1
    s = "".join(keep).strip()
    # 剥后无汉字 → 纯标点/符号残留（如注文被剥光后孤立的句读），非诗句
    if not re.search(r"[\u4e00-\u9fff]", s):
        return None
    if len(s) >= 30 or any(m in s for m in ("「", "」", "『", "』", "：", "《", "》", "〈", "〉")):
        return None
    return s


def _is_hanzi(ch):
    """基本汉字区判定（U+4E00–U+9FFF），与语料白名单口径一致。"""
    return "\u4e00" <= ch <= "\u9fff"


def _half_line_spans(text):
    """按断句符切分文本，返回各半句的 (start, end) 字符下标区间（end 为开区间）。

    断句符 = prosody.DEFAULT_PUNCT ∪ 全部空白（含换行），与 prosody._half_lines 口径逐半句一致：
    换行/空白作断句符处理、空半句丢弃。区间即该半句在 text 中的连续非断句符字符跨度。
    本函数在 prosody 口径（字符串结果）之外额外给出位置，供元信息逐 token 标注定位。
    """
    delims = set(prosody.DEFAULT_PUNCT)
    spans, start = [], None
    for i, ch in enumerate(text):
        if ch in delims or ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


def tag_of(lines):
    """判定诗体，返回 POEM_TAGS 中对应标签。

    口径（计划 §4.2，实测）：每行 = 一联 = 两个半句 → 绝句 2 行、律诗 4 行。
    判定条件：行数 ∈ {2, 4} 且每行恰两个半句，且全部半句汉字数同为 5（五言）或 7（七言）；
    其余（排律/乐府/长短句/残句）一律 <杂言>。半句汉字数由 prosody.line_len_of 给出。
    """
    n = len(lines)
    per_line = [prosody.line_len_of(ln, prosody.DEFAULT_PUNCT) for ln in lines]
    if n in (2, 4) and all(len(ls) == 2 for ls in per_line):
        flat = [x for ls in per_line for x in ls]
        if flat and all(x == 5 for x in flat):
            return POEM_TAGS[0] if n == 2 else POEM_TAGS[2]
        if flat and all(x == 7 for x in flat):
            return POEM_TAGS[1] if n == 2 else POEM_TAGS[3]
    return POEM_TAGS[4]


def tag_poem(lines):
    """在诗首行前插入诗体控制 token，返回整首诗的语料文本。

    返回 = 「标签 + 首行」+ "\\n" + 其余行 + "\\n"（行间以 \\n 连接、末行后保留 \\n），
    与既有语料落盘格式一致（总行数不变，仅首行多一个前缀 token）。空诗返回空串。
    """
    if not lines:
        return ""
    out = list(lines)
    out[0] = tag_of(lines) + out[0]
    return "\n".join(out) + "\n"


def build_meta(poems):
    """由诗列表一次遍历产出整份语料文本与逐 token 元信息。

    返回 (text, weights, tone_labels, rhyme_labels, zone_tags)，四个列表长度均 = token 数 L：
    每个诗体标签（4 字符）计 1 个 token，正文逐字符各计 1（含换行），故
    L = len(text) − 3 × 标签个数（4 字符标签净折 1 token；即"纯正文字符数 + 标签数"）。
    注：计划 §4.4 写作「L = len(text) − 4 × 标签个数」，该式得到的是纯正文字符数，
    比 token 流长度少「标签个数」项（每个标签少记 1），按此取值将无法与 T4 编码后的
    `data`（含特殊 token id）按 token 位置对齐；此处按「每特殊 token 计 1」的定义取 token 流长度。
    逐位置语义（下标 = token 流位置，见计划 §4.4）：
      weights      —— 该位置作为 target 的损失权重，按身份取最大 {韵脚 2.0, 半句末标点 1.5,
                      特殊 token 1.0, 其余 1.0}；对仗区不烘焙（训练期另由 duizhang_weight 抬升）。
      tone_labels  —— 0=平 1=仄 2=未覆盖/非汉字（prosody.tone_of 为 None 记 2，不静默当平）。
      rhyme_labels —— 0=unknown，1..106=韵部（prosody.rhyme_of 已为 1 基，None 记 0，勿再加 1）。
      zone_tags    —— 0=常态 1=韵脚 2=半句末标点 3=特殊 token 4=对仗区；
                      优先级 3 > 4 > 1 > 2 > 0（重叠时取高者）。
    韵脚 = 偶半句（第 2/4/6/8 半句）末字；对仗区 = 仅 <五律>/<七律> 第 2、3 行的全部汉字。
    """
    texts = []
    weights, tone_labels, rhyme_labels, zone_tags = [], [], [], []
    for lines in poems:
        if not lines:
            continue
        tag = tag_of(lines)
        body = "\n".join(lines) + "\n"
        texts.append(tag)
        texts.append(body)
        # 特殊 token 本身：权重 1.0、zone 3、无平仄/韵部标签
        weights.append(1.0)
        tone_labels.append(2)
        rhyme_labels.append(0)
        zone_tags.append(3)

        n = len(body)
        # 各行在 body 中的字符区间（行末换行不计入行内容）
        line_bounds, off = [], 0
        for ln in lines:
            line_bounds.append((off, off + len(ln)))
            off += len(ln) + 1
        # 韵脚：偶半句（下标 1,3,5,...）内最后一个汉字
        is_rhyme = [False] * n
        for j, (s, e) in enumerate(_half_line_spans(body)):
            if j % 2 == 1:
                for k in range(e - 1, s - 1, -1):
                    if _is_hanzi(body[k]):
                        is_rhyme[k] = True
                        break
        is_punct = [ch in prosody.DEFAULT_PUNCT for ch in body]
        # 对仗区：仅五律/七律的第 2、3 行（0 基下标 1、2）全部汉字（不含标点）
        is_dui = [False] * n
        if tag in (POEM_TAGS[2], POEM_TAGS[3]):
            for li in (1, 2):
                if li < len(line_bounds):
                    s, e = line_bounds[li]
                    for k in range(s, e):
                        if _is_hanzi(body[k]):
                            is_dui[k] = True
        # 逐字符填四数组
        for i, ch in enumerate(body):
            weights.append(2.0 if is_rhyme[i] else 1.5 if is_punct[i] else 1.0)
            if _is_hanzi(ch):
                t = prosody.tone_of(ch)
                r = prosody.rhyme_of(ch)
                tone_labels.append(2 if t is None else t)
                rhyme_labels.append(0 if r is None else r)
            else:
                tone_labels.append(2)
                rhyme_labels.append(0)
            if is_dui[i]:
                zone_tags.append(4)
            elif is_rhyme[i]:
                zone_tags.append(1)
            elif is_punct[i]:
                zone_tags.append(2)
            else:
                zone_tags.append(0)
    return "".join(texts), weights, tone_labels, rhyme_labels, zone_tags


def main():
    poems = []          # 每首 = 段落行列表
    weird = Counter()   # 全量扫描被白名单过滤的非常见字符（潜在噪声源）
    n_raw, n_drop = 0, 0      # 有效诗数 / 空正文丢弃数
    n_note_seg, n_note_poem = 0, 0  # 注记判据剔除：段数 / 整诗被剔光的条数
    files = sorted(glob.glob(RAW_GLOB))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for it in data:
            paras = it.get("paragraphs") or []
            if not paras:
                n_drop += 1
                continue
            lines = []
            for p in paras:
                c = clean_para(p, weird)
                if c is None:        # 整段判为叙述性校记注
                    n_note_seg += 1
                elif c:
                    lines.append(c)
            if lines:
                poems.append(lines)
                n_raw += 1
            else:
                n_note_poem += 1     # 原有正文但清洗后全为注记 → 整条作废

    print("解析诗数:", n_raw, "| 空正文丢弃:", n_drop,
          "| 注记段剔除:", n_note_seg, "| 全注条作废:", n_note_poem)
    print("被过滤的非常见字符 Top20:", dict(weird.most_common(20)))

    # 诗级随机打乱，末尾 VAL_N 首作验证集
    random.seed(SEED)
    random.shuffle(poems)
    val_poems, train_poems = poems[:VAL_N], poems[VAL_N:]

    # 训练集：同一次遍历同时产出语料文本与逐位置元信息（不额外读盘）
    text, weights, tone_labels, rhyme_labels, zone_tags = build_meta(train_poems)
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        f.write(text)

    # 验证集：仅写带诗体前缀的语料（不参与加权训练，不产 meta）
    val_text = "".join(tag_poem(lines) for lines in val_poems)
    with open(VAL_OUT, "w", encoding="utf-8") as f:
        f.write(val_text)

    # 语料与 meta 落盘（四数组与训练集 token 流等长）
    w = np.asarray(weights, dtype=np.float32)
    t = np.asarray(tone_labels, dtype=np.int8)
    r = np.asarray(rhyme_labels, dtype=np.int16)
    z = np.asarray(zone_tags, dtype=np.int8)
    # L = token 流长度 = 纯正文字符数 + 标签数（每 4 字符标签净折 1 token）
    L = len(text) - 3 * len(train_poems)
    # 不变量校验（落盘前）：四数组长度须等于 L，且 L 须与语料文本自洽。
    # build_meta 对空诗列表 continue 跳过，而此处标签数按诗数计——
    # 若传入含空项的诗列表，二者会错位，故显式校验以阻断数组与 token 位置的错配。
    assert (len(w) == len(t) == len(r) == len(z) == L), (
        f"meta 四数组长度与 token 流长度 L 不一致：weights={len(w)} tone_labels={len(t)} "
        f"rhyme_labels={len(r)} zone_tags={len(z)} L={L}；"
        f"build_meta 会跳过空诗项而 L 按诗数计入，不一致说明语料与逐位置元信息已错位")
    assert L == len(text) - 3 * len(train_poems), (
        f"token 流长度不自洽：L={L} != len(text)-3×标签数="
        f"{len(text)}-3×{len(train_poems)}")
    np.savez(META_OUT, weights=w, tone_labels=t, rhyme_labels=r, zone_tags=z)
    print(f"[meta] L={L} | len(text)={len(text)} | 标签数={len(train_poems)} | "
          f"数组长度 {len(w)}/{len(t)}/{len(r)}/{len(z)} | 落盘 {os.path.basename(META_OUT)}")

    # 诗体分布（训练集）
    form = Counter(tag_of(ps) for ps in train_poems)
    n_train = len(train_poems)
    print("[诗体分布(训练集)] " + " ".join(
        f"{tg}={form[tg]}({form[tg] / n_train:.1%})" for tg in POEM_TAGS))

    # zone_tags 分布与关键占比
    zone_cnt = np.bincount(z, minlength=len(ZONE_NAMES))
    print("[zone_tags 分布] " + " ".join(
        f"{ZONE_NAMES[i]}={int(zone_cnt[i])}({zone_cnt[i] / L:.2%})" for i in range(len(ZONE_NAMES))))
    n_rhyme = int((w == 2.0).sum())     # 韵脚（按身份最大独立计数，含对仗区内的韵脚）
    n_punct = int((w == 1.5).sum())     # 半句末标点
    print(f"[权重身份] 韵脚(2.0)={n_rhyme} | 半句末标点(1.5)={n_punct} | "
          f"对仗区占比={int(zone_cnt[4]) / L:.2%}")

    # 逐文件统计：字数/行数/词表/低频字占比
    for tag, ps, txt in (("训练", train_poems, text), ("验证", val_poems, val_text)):
        chars = sorted(set(txt))
        freq = Counter(txt)
        rare = sum(1 for c in chars if freq[c] < 5)
        print(f"[{tag}] {len(ps)} 首 | {len(txt)} 字符 | {len(txt.splitlines())} 行 | "
              f"含前缀词表 {len(chars)} | 频次<5 字符 {rare} ({rare / max(len(chars), 1):.1%})")
    # 纯文本字符集（剥去 < > 前缀字符）应与旧词表 8196 一致
    pure = sorted(set(text) - {"<", ">"})
    print(f"[纯文本字符集] {len(pure)} 字符（旧词表 8196，不变）")

    # 抽样目检：训练/验证各 3 首完整诗（含诗体前缀）
    random.seed(SEED)
    for tag, ps in (("训练抽样", train_poems), ("验证抽样", val_poems)):
        print(f"──── {tag} ────")
        for lines in random.sample(ps, 3):
            print(tag_poem(lines), end="")
            print()
    # 目检：首个五律样本，核对颔联/颈联汉字 zone_tags==4
    for lines in train_poems:
        if tag_of(lines) == POEM_TAGS[2]:
            _, _, _, _, z1 = build_meta([lines])
            print("──── 五律目检（数字为 zone_tags：1=韵脚 2=标点 3=标签 4=对仗区 0=常态）────")
            print(tag_poem(lines), end="")
            off = 1                       # 跳过特殊 token
            for ln in lines:
                marks = []
                for ch in ln:
                    marks.append(f"{ch}{z1[off]}")
                    off += 1
                print(" ".join(marks))
                off += 1                  # 换行
            break


if __name__ == "__main__":
    main()
