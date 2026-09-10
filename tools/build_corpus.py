# -*- coding: utf-8 -*-
"""全唐诗语料构建管线：JSON 分卷 → 简体纯诗句流 corpus.txt + 独立 val.txt

数据源：chinese-poetry 仓库《全唐诗》（繁体存储，poet.tang.{0..57000}.json，每卷 1000 首，
约 5.7 万首）。本脚本做四件事：
  1) 字段级过滤：只取每条记录的 paragraphs（正文联句），丢弃 title/author/id/tags——
     组诗序号（如"帝京篇十首 一"）等标题噪声随 title 一并去除
  2) 繁→简 + 校注剥离 + 字符白名单：opencc t2s 统一字形；段落内圆括号（…）及其嵌套
     内容整体剥离（诗句正文不使用括号，括号内是敦煌卷子/拾遗诗的校记、异文、出处注）；
     剥净后仍 ≥30 字或含引号/冒号/书名号的行判为叙述性校记注，整段丢弃。
     依赖 opencc-python-reimplemented（一次性转换依赖，仅数据管线用，不污染运行时）：
         pip install opencc-python-reimplemented
  3) 诗级划分：全量诗随机打乱（固定 seed 可复现），末尾 400 首留作独立验证集 val.txt，
     其余为训练集 corpus.txt——训练/验证同构（一联一行，诗间不插空行），
     验证集不参与训练，val loss 用于分辨"学规律"还是"背语料"
  4) 统计上报：字数/行数/词表/低频字占比/非白名单字符抽样，供人工目检
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter

from opencc import OpenCC

# Windows 控制台常为 GBK：正文里可能出现 GBK 外字符（CJK 扩展区等），
# 打印时以 \\uXXXX 转义显示而不是抛 UnicodeEncodeError 崩溃
try:
    sys.stdout.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（脚本在 tools/ 下）
RAW_GLOB = os.path.join(BASE, "_probe", "raw", "poet.tang.*.json")
TRAIN_OUT = os.path.join(BASE, "data", "corpus.txt")
VAL_OUT = os.path.join(BASE, "data", "val.txt")
VAL_N = 400          # 留出验证诗数
SEED = 0             # 打乱种子（可复现划分）
# 正文允许字符白名单：汉字 + 中文标点 + 半角标点 + 空白。
# 括号（全/半角）不在白名单——诗句正文不使用括号，其角色只是校注的界符，
# 先经剥括号步骤整体移除，残余孤立括号在此被滤除。
ALLOWED = set("，。！？；：、…「」『』《》〈〉——…·,.;:!? '\"")

cc = OpenCC("t2s")


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
    s = cc.convert(p)
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

    def dump(poems, path):
        with open(path, "w", encoding="utf-8") as f:
            for lines in poems:
                f.write("\n".join(lines) + "\n")   # 联与联换行；诗尾接下一诗首联（无空行）

    dump(train_poems, TRAIN_OUT)
    dump(val_poems, VAL_OUT)

    for tag, path, ps in (("训练", TRAIN_OUT, train_poems), ("验证", VAL_OUT, val_poems)):
        text = open(path, encoding="utf-8").read()
        chars = sorted(set(text))
        freq = Counter(text)
        rare = sum(1 for c in chars if freq[c] < 5)
        print(f"[{tag}] {len(ps)} 首 | {len(text)} 字符 | {len(text.splitlines())} 行 | "
              f"词表 {len(chars)} | 频次<5 字符 {rare} ({rare / max(len(chars), 1):.1%})")

    # 抽样目检：训练/验证各 3 首完整诗
    random.seed(SEED)
    for tag, ps in (("训练抽样", train_poems), ("验证抽样", val_poems)):
        print(f"──── {tag} ────")
        for lines in random.sample(ps, 3):
            print("\n".join(lines))
            print()


if __name__ == "__main__":
    main()
