# -*- coding: utf-8 -*-
"""T11 评估与 A/B 工具（tools/prosody_eval.py）单测：口径固化、prompt 池、指标、置信区间、报告。

覆盖对象与判据：
  - set_seed：入口随机源可复现（原生 numpy 流与计算后端流各自复位到同一种子）。
  - build_eval_pool：prompt 只从 data/val.txt 抽取该诗任一合格半句（半句 = 该诗按
    prosody.DEFAULT_PUNCT 切分后的任一段，纯汉字且 5/7 字）、按诗体分层、同诗体内去重、落盘复用；
    断言每诗体取满 40 条（合计 160）且不低于 30 条硬下限，池中每一条都命中 val 合格半句集合
    （用与实现不同源的正则独立解析），并与 data/corpus.txt 隔离（碰撞排除：候选半句见于 corpus
    者剔除，另以独立核算的碰撞条数作证据断言，证明排除动作真实生效）；另以合成样例断言
    「某诗体候选 ≤40 时全取且不抛异常」。
  - score_group：句长/押韵/首句入韵/平仄四项指标一律由 prosody 既有函数给出
    （本工具不得另立字音判据）；首句入韵与押韵主指标分别独立统计；对仗三项（相对联数、
    不重字联数、可比联数）取自 prosody.check_duizhang，且只对律诗（<五律>/<七律>）计，
    绝句无对仗要求、三项恒为 0。
  - 对仗口径：律诗两子指标按联级合并（率 = Σ分子联数 / Σ可比联数，非逐样本率平均），
    置信区间以可比联数为 n；区分度自检须同时具备旧模型与 M3 组，缺任一组即如实报「未自检」，
    旧模型联内不重字率 ≥ 90% 判该子指标作废且不再参与有效性判定；参与判定的子指标差值
    均 < 10 个百分点判指标失效，至少一项 ≥ 10 个百分点判指标有效。
  - ci_halfwidth：比例指标 95% 置信区间半宽 1.96·√(p(1−p)/n)。
  - judge_gates / write_report：M1、M2 闸门判定；中文报告含样本量、置信区间、旧模型同口径基线、
    三组 A/B 对照结构、对仗口径说明与区分度自检段落、样本不足标注与「待 M3」占位
    （不得编造数字）。
  - val_corpus_duplicate_stats / write_report：val 与 corpus 整诗逐字重复的统计口径（与独立算法
    比对、锁定实证锚点 9/400 约 2.25%），以及报告第七节对该乐观偏差的如实披露（实测数字取自
    统计函数、根因与「本轮不做」的根治声明齐备）。
  - load_old_model / load_new_model：旧档自带词表（纯文本 8196 字符）与新档词表（8196+5 特殊 token）
    的正确载入；纯汉字 prompt 在两套词表下逐 id 编码相同。
  - 相位归因判据（judge_parity_attribution）：按各组已算出的对齐/错位两子集押韵率与样本量实算
    率差与该组两子集置信区间半宽之和，据此定结论——均不显著（方向亦不一致）即判「低押韵合规率
    不可归因于相位错位、闸门判定成立」，某组对齐子集显著高于错位子集即点名该组并写「须按 §九
    停手报主子复核口径」；两端措辞皆不得为模型的低合规率开脱（不得以取样口径否定闸门判定）。
    报告第八节第 3 条与终端打印共用该函数文案，不得两处漂移。
  - 极短生成：旧档模型对 1 条 prompt 生成 4 个 token，验证解码链路可用（仅 1 次生成）。

产物只落 _probe/prosody_eval_test/ 并在 main 结束前清理；不写 data/ 与 test/ 下的文件。
无第三方测试框架依赖，直接 `python test/test_prosody_eval.py` 运行；逐项打印 ✓/✗ 与中文汇总，
任一失败时进程以非零码退出。
"""
import math
import os
import re
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))

import numpy as onp

import prosody
import train
import prosody_eval
from model.backend import np, as_numpy
from model.gpt import GPT

PROBE = os.path.join(BASE, "_probe", "prosody_eval_test")
POOL_TAGS = prosody_eval.POOL_TAGS
LVSHI_TAGS = ("<五律>", "<七律>")      # 对仗指标适用诗体（绝句无对仗要求）；独立书写以交叉校验实现常量

# 合成样本（字音口径经 prosody 实测，用例内另做前提自校验，不臆断平仄/韵部）
GOOD5 = "白日依山尽，天高白玉霜。春江明月雪，夜雪白霜光。"
GOOD7 = "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。"
WULV = ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡。",
        "白日依山尽，黄河入海流。", "欲穷千里目，更上一层楼。"]
QILV = ["岧嶤太华俯咸京，天外三峰削不成。", "武帝祠前云欲散，仙人掌上雨初晴。",
        "河山北枕秦关险，驿树西连汉畤平。", "借问路傍名利客，无如此处学长生。"]
# 押韵主指标为 0 但首句入韵为真的两半句样本（用于锁定首句入韵单列口径）
TWO_HALVES = "床前明月光，疑是地上霜。"
# 对仗样例（八半句成篇律诗；口径经 prosody 实测，用例内另做前提自校验）：
#   DZ_REL   —— 颔联与颈联皆可比且皆相对（pairs=2，pingze=1.0）
#   DZ_IRREL —— 仅颔联可比且不相对（颈联两半句为七言，按五言口径跳过；pairs=1，pingze=0.0）
DZ_REL = "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。"
DZ_IRREL = "白日依山尽，黄河入海流。床前明月光，疑是地上霜。两个黄鹂鸣翠柳，一行白鹭上青天。举头望明月，低头思故乡。"


def _mkprobe():
    """建立本次用例的临时产物目录并返回其路径。"""
    os.makedirs(PROBE, exist_ok=True)
    return PROBE


def _val_halflines(val_path):
    """独立解析 data/val.txt，返回全部 (诗体, 半句) 集合。

    刻意采用正则切分（与实现侧 prosody._half_lines 的字符级切分手法不同），避免用例与实现
    同源而失去校验力；半句口径 = 该诗正文按断句标点/空白切分后的任一段，须纯汉字且 5/7 字。
    用于证明 prompt 池的每一条都确实是 val 的合格半句（只从 val 抽取）。
    """
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    out, cur_tag, buf = set(), None, []

    def flush():
        if cur_tag in POOL_TAGS:
            for seg in re.split(r"[，。！？；：、\s]+", "\n".join(buf)):
                if _is_pure_hanzi(seg):
                    out.add((cur_tag, seg))

    for line in text.split("\n"):
        m = re.match(r"<[^>]+>", line)
        if m:
            flush()
            cur_tag, buf = m.group(0), [line[m.end():]]
        elif cur_tag is not None:
            buf.append(line)
    flush()
    return out


def _is_pure_hanzi(s):
    """是否纯汉字且汉字数为 5 或 7。"""
    return len(s) in (5, 7) and all("\u4e00" <= c <= "\u9fff" for c in s)


# ─────────────────────────── 口径固化（Step 0）───────────────────────────

def test_set_seed_reproducible():
    """set_seed 同时复位原生 numpy 流与计算后端流：同种子两次取值逐元素相同。"""
    prosody_eval.set_seed(1234)
    a1, a2 = onp.random.rand(4), onp.random.rand(4)
    prosody_eval.set_seed(1234)
    b1, b2 = onp.random.rand(4), onp.random.rand(4)
    assert onp.array_equal(a1, b1) and onp.array_equal(a2, b2), \
        "set_seed 后原生 numpy 流不可复现"
    prosody_eval.set_seed(7)
    c1 = as_numpy(np.random.rand(4))
    prosody_eval.set_seed(7)
    c2 = as_numpy(np.random.rand(4))
    assert onp.array_equal(c1, c2), "set_seed 后计算后端随机流不可复现"
    print("  原生流与后端流均可复现 ✓")


def test_build_eval_pool_from_val_only():
    """prompt 池取 val 该诗任一合格半句、分层、同诗体内去重、落盘复用；定档每诗体 40 条；
    池与 data/corpus.txt 隔离（碰撞排除），并以独立核算的证据断言证明排除动作真实生效。"""
    d = _mkprobe()
    pool_path = os.path.join(d, "eval-prompts.txt")
    if os.path.exists(pool_path):
        os.remove(pool_path)
    pool = prosody_eval.build_eval_pool(val_path=prosody_eval.VAL_PATH, n_per_tag=40,
                                        seed=0, pool_path=pool_path)
    assert pool, "prompt 池不得为空"
    assert all(tag in POOL_TAGS for tag, _ in pool), "prompt 池不得含 <杂言> 等非抽样诗体"

    # 独立解析 val 的全部合格半句（正则切分，与实现侧 prosody._half_lines 不同源）
    val_hl = _val_halflines(prosody_eval.VAL_PATH)
    taken = {tag: [p for t, p in pool if t == tag] for tag in POOL_TAGS}

    # 定档：每诗体 40 条、合计 160 条
    assert len(pool) == 160, f"池总量应为 160，实为 {len(pool)}"
    for tag in POOL_TAGS:
        assert len(taken[tag]) == 40, f"{tag} 应取满 40 条，实为 {len(taken[tag])}"

    # 硬下限：每诗体 ≥30 条
    for tag in POOL_TAGS:
        assert len(taken[tag]) >= prosody_eval.MIN_SAMPLES_PER_TAG, \
            f"{tag} 样本量 {len(taken[tag])} 低于硬下限 {prosody_eval.MIN_SAMPLES_PER_TAG}"

    # 同诗体内无重复
    for tag in POOL_TAGS:
        assert len(set(taken[tag])) == len(taken[tag]), f"{tag} 池内出现重复半句"

    # 只从 val 抽取：池中每一条都须命中 val 合格半句集合（口径越界即失败）
    miss = [(t, p) for t, p in pool if (t, p) not in val_hl]
    assert not miss, f"以下 prompt 不是 val 的合格半句（口径越界）：{miss[:5]}"

    # 与训练语料隔离：池中每一条都不得出现在 data/corpus.txt
    with open(os.path.join(BASE, "data", "corpus.txt"), encoding="utf-8") as f:
        corpus_text = f.read()
    leak = [(t, p) for t, p in pool if p in corpus_text]
    assert not leak, f"以下 prompt 出现在训练语料 data/corpus.txt：{leak[:5]}"

    # 证据断言：碰撞排除须真实生效（而非恰好无碰撞）——用与实现不同源的正则解析独立核算
    # val 合格半句中命中 corpus 者，并与实现上报的排除统计逐诗体比对。
    hit = {tag: sorted(p for t, p in val_hl if t == tag and p in corpus_text)
           for tag in POOL_TAGS}
    assert sum(len(v) for v in hit.values()) >= 1, \
        "前提：val 合格半句须至少 1 条命中 data/corpus.txt，否则排除断言无证据力"
    stats = prosody_eval.pool_exclusion_stats()
    assert stats is not None, "新建池后应能取到碰撞排除统计（pool_exclusion_stats）"
    want = {tag: len(hit[tag]) for tag in POOL_TAGS}
    assert stats["collision"] == want, \
        f"实际排除条数与独立核算不符：{stats['collision']} vs {want}"
    assert stats["collision_total"] == sum(want.values()) >= 1, \
        "实际排除合计须与独立核算一致且 ≥1（证明排除动作真实生效）"
    assert all(stats["supply_after"][tag] < stats["supply_before"][tag]
               for tag in POOL_TAGS if want[tag]), \
        "命中 corpus 的诗体，其排除后供给量须严格小于排除前"
    assert sum(stats["supply_after"].values()) < sum(stats["supply_before"].values()), \
        "碰撞排除须减少候选供给量"

    # 落盘格式：每行「诗体\t半句」，行数等于池条数
    with open(pool_path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    assert len(lines) == len(pool), f"落盘行数 {len(lines)} 与池条数 {len(pool)} 不一致"
    for ln in lines:
        assert ln.count("\t") == 1, f"行格式应为「诗体\\t半句」：{ln!r}"
        tag, _, prompt = ln.partition("\t")
        assert tag in POOL_TAGS and prompt, f"行内容非法：{ln!r}"

    # 已存在即读不重抽：再次构建须返回同一内容
    pool2 = prosody_eval.build_eval_pool(val_path=prosody_eval.VAL_PATH, n_per_tag=40,
                                         seed=0, pool_path=pool_path)
    assert pool2 == pool, "池已落盘却重抽（应读取复用）"

    print("  池 " + "、".join(f"{t} {len(taken[t])}" for t in POOL_TAGS)
          + f"；合计 {len(pool)} 条，每诗体 ≥30、同诗体内无重、全部命中 val 合格半句"
          + "且与 corpus 隔离 ✓；碰撞排除实测剔除 "
          + f"{stats['collision_total']} 条（{'、'.join(f'{t} {want[t]}' for t in POOL_TAGS)}），"
          + f"候选供给 {sum(stats['supply_before'].values())}→{sum(stats['supply_after'].values())} 条")


def test_build_eval_pool_undersupply_takes_all():
    """合成样例：某诗体候选 ≤40 时全取（不抛异常、不断言失败），并如实记录实际条数。

    本用例只验「供给 ≤40 全取」分支，故传入与合成 val 无任何碰撞的 corpus，隔离碰撞排除
    （该死规则由 test_build_eval_pool_from_val_only 以独立核算覆盖）。
    """
    d = _mkprobe()
    val_path = os.path.join(d, "val_small.txt")
    pool_path = os.path.join(d, "pool_small.txt")
    corpus_path = os.path.join(d, "corpus_small.txt")
    # <五绝> 2 首（各 4 个合格半句，供给 8≤40）；<七律> 1 首（4 个半句，供给 4≤40）；
    # <七绝>/<五律> 无此诗体（供给 0≤40）——三者皆应全取而非报错。
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("<五绝>床前明月光，疑是地上霜。\n"
                "举头望明月，低头思故乡。\n"
                "<五绝>白日依山尽，黄河入海流。\n"
                "欲穷千里目，更上一层楼。\n"
                "<七律>两个黄鹂鸣翠柳，一行白鹭上青天。\n"
                "窗含西岭千秋雪，门泊东吴万里船。\n")
    # 合成 corpus：与上述合成 val 的候选半句无子串重叠，故碰撞排除不触发。
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("<五绝>春眠不觉晓，处处闻啼鸟。\n")
    if os.path.exists(pool_path):
        os.remove(pool_path)
    pool = prosody_eval.build_eval_pool(val_path=val_path, n_per_tag=40, seed=0,
                                        pool_path=pool_path, corpus_path=corpus_path)
    got = {tag: sorted(p for t, p in pool if t == tag) for tag in POOL_TAGS}
    assert got["<五绝>"] == sorted(["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡",
                                    "白日依山尽", "黄河入海流", "欲穷千里目", "更上一层楼"]), \
        f"<五绝> 候选 8≤40 应全取，实为 {got['<五绝>']}"
    assert got["<七律>"] == sorted(["两个黄鹂鸣翠柳", "一行白鹭上青天",
                                    "窗含西岭千秋雪", "门泊东吴万里船"]), \
        f"<七律> 候选 4≤40 应全取，实为 {got['<七律>']}"
    assert got["<七绝>"] == [] and got["<五律>"] == [], "无候选诗体应取空而非报错"
    assert len(pool) == 12, f"合计应 12 条，实为 {len(pool)}"
    print("  供给 ≤40 全取（<五绝>8、<七律>4、<七绝>/<五律>0），未抛异常 ✓")


# ─────────────────────────── 指标与统计（Step 1/2）───────────────────────────

def test_ci_halfwidth():
    """比例指标 95% 置信区间半宽 = 1.96·√(p(1−p)/n)；n=30 时 p=0.9 约 ±0.107、p=0.5 约 ±0.18。"""
    assert abs(prosody_eval.ci_halfwidth(0.9, 30) - 1.96 * math.sqrt(0.09 / 30)) < 1e-12
    assert abs(prosody_eval.ci_halfwidth(0.9, 30) - 0.107) < 0.001
    assert abs(prosody_eval.ci_halfwidth(0.5, 30) - 0.18) < 0.003
    assert prosody_eval.ci_halfwidth(0.5, 0) != prosody_eval.ci_halfwidth(0.5, 0) or \
        prosody_eval.ci_halfwidth(0.5, 0) == 0.0, "n=0 时半宽须为 NaN 或 0，不得抛异常"
    print("  p=0.9/n=30→0.107，p=0.5/n=30→0.179 ✓")


def test_score_group_matches_prosody():
    """score_group 四项指标与 prosody 既有函数逐一吻合（不另立字音判据），并按诗体分桶。"""
    pool = [("<五绝>", "白日依山尽"), ("<七绝>", "两个黄鹂鸣翠柳"),
            ("<五律>", "床前明月光"), ("<七律>", "岧嶤太华俯咸京")]
    texts = [GOOD5, GOOD7, "".join(WULV), "".join(QILV)]
    res = prosody_eval.score_group(texts, pool)
    assert res["n"] == 4 and set(res["per_tag"]) == set(POOL_TAGS), "分诗体结构不符"
    for tag in POOL_TAGS:
        assert res["per_tag"][tag]["n"] == 1, f"{tag} 应 1 条样本"

    exp = {"line_len": [], "rhyme": [], "first_rhyme": [], "tone": [],
           "gu_ping": 0, "san_ping": 0}
    for (tag, _), text in zip(pool, texts):
        plen = prosody_eval.TAG_LEN[tag]
        rh = prosody.check_rhyme(text)
        tn = prosody.check_tone(text, plen)
        exp["line_len"].append(prosody.check_line_len(text, plen))
        exp["rhyme"].append(rh.main)
        exp["first_rhyme"].append(float(rh.first_rhymed))
        exp["tone"].append(tn.main)
        exp["gu_ping"] += tn.gu_ping
        exp["san_ping"] += tn.san_ping
    ov = res["overall"]
    for key in ("line_len", "rhyme", "first_rhyme", "tone"):
        want = float(onp.mean(exp[key]))
        assert abs(ov[key] - want) < 1e-12, f"{key} 应 {want}，实为 {ov[key]}"
    assert ov["gu_ping"] == exp["gu_ping"] and ov["san_ping"] == exp["san_ping"], \
        "孤平/三平调违例数应为各组样本计数之和"
    assert set(ov["ci"]) >= {"line_len", "rhyme", "tone"}, "overall 须附各指标置信区间"
    print(f"  句长 {ov['line_len']:.4f} 押韵 {ov['rhyme']:.4f} 平仄 {ov['tone']:.4f} "
          f"（与 prosody 逐项一致）✓")


def test_first_rhymed_reported_separately():
    """首句入韵与押韵主指标分别统计：样本含「主指标 0 而入韵为真」者，两均值须各自独立。"""
    texts = [TWO_HALVES, GOOD5, GOOD7]
    pool = [("<五绝>", "床前明月光"), ("<五绝>", "白日依山尽"), ("<七绝>", "两个黄鹂鸣翠柳")]
    rh0 = prosody.check_rhyme(TWO_HALVES)
    assert rh0.main == 0.0 and rh0.first_rhymed is True, \
        "前提：两半句样本应「主指标 0 而首句入韵真」"
    exp_first = [float(prosody.check_rhyme(t).first_rhymed) for t in texts]
    exp_main = [prosody.check_rhyme(t).main for t in texts]
    assert set(exp_first) == {0.0, 1.0} and exp_main != exp_first, \
        "前提：两均值应可区分（否则本用例无检定力）"
    ov = prosody_eval.score_group(texts, pool)["overall"]
    assert abs(ov["rhyme"] - float(onp.mean(exp_main))) < 1e-12, "押韵主指标统计不符"
    assert abs(ov["first_rhyme"] - float(onp.mean(exp_first))) < 1e-12, "首句入韵统计不符"
    assert ov["rhyme"] != ov["first_rhyme"], "首句入韵须与押韵主指标分开统计"
    print(f"  押韵主指标 {ov['rhyme']:.4f} ≠ 首句入韵 {ov['first_rhyme']:.4f}（独立统计）✓")


# ─────────────────────────── 闸门判定与报告 ───────────────────────────

def _fake_group(kind, label, line_len, rhyme, tone, constraint=False, n=160, first=0.5,
                gu=0, san=0, device="numpy", dz_pairs=40, dz_ok_pingze=20, dz_ok_chongzi=20):
    """构造与 run_group 同构的结果字典（用于闸门与报告用例，不加载模型）。

    对仗三项按联数直接给定：<五律>/<七律> 每个诗体记 dz_pairs 个可比联、dz_ok_pingze 个相对联、
    dz_ok_chongzi 个不重字联（分子分母皆为整数，故合并率可精确断言）；<五绝>/<七绝> 恒为 0——
    绝句无对仗要求、不入该指标。派生两子指标按联级合并，置信区间以可比联数为 n。
    """
    ntag = n // len(POOL_TAGS)
    ci = {"line_len": 0.01, "rhyme": 0.01, "first_rhyme": 0.01, "tone": 0.01}
    per_tag = {}
    for tag in POOL_TAGS:
        lv = tag in LVSHI_TAGS
        pairs = dz_pairs if lv else 0
        ok_p = dz_ok_pingze if lv else 0
        ok_c = dz_ok_chongzi if lv else 0
        per_tag[tag] = {"n": ntag, "line_len": line_len, "rhyme": rhyme, "first_rhyme": first,
                        "tone": tone, "gu_ping": gu, "san_ping": san, "ci": dict(ci),
                        "dz_ok_pingze": ok_p, "dz_ok_chongzi": ok_c, "dz_pairs": pairs,
                        "dz_pingze": ok_p / pairs if pairs else 0.0,
                        "dz_chongzi": ok_c / pairs if pairs else 0.0}
        per_tag[tag]["ci"]["dz_pingze"] = prosody_eval.ci_halfwidth(per_tag[tag]["dz_pingze"], pairs)
        per_tag[tag]["ci"]["dz_chongzi"] = prosody_eval.ci_halfwidth(per_tag[tag]["dz_chongzi"], pairs)
    tot_pairs = sum(per_tag[t]["dz_pairs"] for t in POOL_TAGS)
    tot_ok_p = sum(per_tag[t]["dz_ok_pingze"] for t in POOL_TAGS)
    tot_ok_c = sum(per_tag[t]["dz_ok_chongzi"] for t in POOL_TAGS)
    overall = {"n": n, "line_len": line_len, "rhyme": rhyme, "first_rhyme": first,
               "tone": tone, "gu_ping": gu, "san_ping": san, "ci": dict(ci),
               "dz_ok_pingze": tot_ok_p, "dz_ok_chongzi": tot_ok_c, "dz_pairs": tot_pairs,
               "dz_pingze": tot_ok_p / tot_pairs if tot_pairs else 0.0,
               "dz_chongzi": tot_ok_c / tot_pairs if tot_pairs else 0.0}
    overall["ci"]["dz_pingze"] = prosody_eval.ci_halfwidth(overall["dz_pingze"], tot_pairs)
    overall["ci"]["dz_chongzi"] = prosody_eval.ci_halfwidth(overall["dz_chongzi"], tot_pairs)
    return {"kind": kind, "label": label, "model_path": f"{kind}.npz", "constraint": constraint,
            "device": device, "n": n, "overall": overall, "per_tag": per_tag}


def test_judge_gates():
    """M1（句长≥0.95 且 押韵≥0.90）与 M2（平仄≥0.85 且较 M1 提升）闸门判定正反例。"""
    ok = {"old": _fake_group("old", "旧模型", 0.70, 0.55, 0.50),
          "m1": _fake_group("m1", "M1 模型", 0.96, 0.91, 0.60),
          "m2": _fake_group("m2", "M2 模型", 0.97, 0.92, 0.87)}
    g = prosody_eval.judge_gates(ok)
    assert g["m1"]["pass"] is True, "M1 达标样本应判通过"
    assert g["m2"]["pass"] is True, "M2 达标样本应判通过"
    assert g["m1"]["baseline"]["n"] == 160, "M1 判定须附旧模型同口径基线"

    bad = {"old": _fake_group("old", "旧模型", 0.70, 0.55, 0.50),
           "m1": _fake_group("m1", "M1 模型", 0.94, 0.91, 0.60),
           "m2": _fake_group("m2", "M2 模型", 0.97, 0.92, 0.80)}
    g2 = prosody_eval.judge_gates(bad)
    assert g2["m1"]["pass"] is False, "句长 0.94 < 0.95 应判不通过"
    assert g2["m2"]["pass"] is False, "平仄 0.80 < 0.85 应判不通过"
    # 平仄达 0.85 但未较 M1 提升（相等）亦不通过
    flat = {"old": bad["old"], "m1": _fake_group("m1", "M1", 0.96, 0.91, 0.86),
            "m2": _fake_group("m2", "M2", 0.96, 0.91, 0.86)}
    assert prosody_eval.judge_gates(flat)["m2"]["pass"] is False, \
        "平仄未较 M1 提升不得判通过"
    print("  M1/M2 闸门判定正反例均正确 ✓")


def test_assert_same_device():
    """A/B 同设备断言：设备不一致须响亮失败。"""
    prosody_eval.assert_same_device(["numpy", "numpy"])
    try:
        prosody_eval.assert_same_device(["numpy", "cupy"])
        raise AssertionError("设备不一致应报错，却静默通过")
    except AssertionError as e:
        assert "设备" in str(e), f"错误信息应点名设备：{e}"
    print("  同设备断言（numpy×2 通过，numpy/cupy 报错）✓")


def test_write_report_structure():
    """报告结构：中文、含样本量/置信区间/旧模型基线/三组 A/B 结构/待 M3 占位、数字不编造。"""
    d = _mkprobe()
    path = os.path.join(d, "ab-report.txt")
    results = {"old": _fake_group("old", "旧模型 model.npz（约束关）", 0.70, 0.55, 0.50),
               "m1": _fake_group("m1", "M1 模型（约束关）", 0.96, 0.91, 0.62),
               "m2": _fake_group("m2", "M2 模型（约束关）", 0.97, 0.92, 0.87)}
    meta = {"pool_path": "data/eval-prompts.txt", "val_path": "data/val.txt",
            "seed": 0, "new_tokens": 96, "temperature": 1.0, "top_k": 20, "n_per_tag": 40,
            "per_tag_n": {tag: 40 for tag in POOL_TAGS}, "device": "numpy"}
    prosody_eval.write_report(results, path, meta)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert text.strip(), "报告不得为空"
    for key in ("样本量", "置信区间", "旧模型", "约束关", "约束开", "待 M3",
                "句长", "押韵", "首句入韵", "平仄", "孤平", "三平调"):
        assert key in text, f"报告缺少关键字段：{key}"
    assert "None" not in text and "nan" not in text, "报告不得出现空值/NaN 占位"
    assert text.count("待 M3") >= 2, "第 2、3 组在 M3 就绪前须各标「待 M3」"
    # M3 就绪时应填入真实数字（用同构结果替换占位）
    results3 = dict(results)
    results3["m3"] = _fake_group("m3", "M3 模型（约束关）", 0.98, 0.95, 0.90)
    results3["m3_cons"] = _fake_group("m3_cons", "M3 模型（约束开）", 0.99, 0.99, 0.99,
                                      constraint=True)
    path3 = os.path.join(d, "ab-report-m3.txt")
    prosody_eval.write_report(results3, path3, meta)
    with open(path3, encoding="utf-8") as f:
        text3 = f.read()
    assert text3.count("待 M3") == 0, "M3 结果齐备时不应再出现占位"
    print(f"  报告 {os.path.basename(path)} 结构完整、占位与实填均正确 ✓")


# ─────────────────────────── 对仗指标与区分度自检（T11 Step 3）───────────────────────────

def test_score_group_duizhang_matches_prosody():
    """score_group 的对仗三项与 prosody.check_duizhang 逐项一致（律诗）；绝句不入该指标（恒 0）。"""
    assert hasattr(prosody_eval, "DUIZHANG_TAGS"), "实现缺少对仗适用诗体常量 DUIZHANG_TAGS"
    assert set(prosody_eval.DUIZHANG_TAGS) == set(LVSHI_TAGS), \
        f"对仗指标只应对律诗计，实为 {prosody_eval.DUIZHANG_TAGS}"
    pool = [("<五绝>", "白日依山尽"), ("<七绝>", "两个黄鹂鸣翠柳"),
            ("<五律>", "床前明月光"), ("<七律>", "岧嶤太华俯咸京")]
    wulv, qilv = "".join(WULV), "".join(QILV)
    res = prosody_eval.score_group([GOOD5, GOOD7, wulv, qilv], pool)
    for tag, text in (("<五律>", wulv), ("<七律>", qilv)):
        plen = prosody_eval.TAG_LEN[tag]
        dz = prosody.check_duizhang(text, plen)
        assert dz.pairs >= 1, f"前提：{tag} 真实样例须含可比联，否则本用例无检定力"
        rec = res["per_tag"][tag]
        for key in ("dz_ok_pingze", "dz_ok_chongzi", "dz_pairs"):
            assert key in rec, f"score_group 逐样本记录缺少对仗字段 {key}"
        assert rec["dz_pairs"] == dz.pairs, f"{tag} dz_pairs 应为 {dz.pairs}，实为 {rec['dz_pairs']}"
        assert rec["dz_ok_pingze"] == round(dz.pingze * dz.pairs), \
            f"{tag} dz_ok_pingze 应为 {round(dz.pingze * dz.pairs)}，实为 {rec['dz_ok_pingze']}"
        assert rec["dz_ok_chongzi"] == round(dz.chongzi * dz.pairs), \
            f"{tag} dz_ok_chongzi 应为 {round(dz.chongzi * dz.pairs)}，实为 {rec['dz_ok_chongzi']}"
    for tag, text in (("<五绝>", GOOD5), ("<七绝>", GOOD7)):
        assert prosody.check_duizhang(text, prosody_eval.TAG_LEN[tag]).pairs == 0, \
            f"前提：{tag} 样例无可比联"
        rec = res["per_tag"][tag]
        assert (rec["dz_pairs"], rec["dz_ok_pingze"], rec["dz_ok_chongzi"]) == (0, 0, 0), \
            f"{tag} 为绝句、无对仗要求，三项须恒为 0（不得拿 0 冒充合规）：{rec}"
    ov = res["overall"]
    want_pairs = sum(prosody.check_duizhang(t, prosody_eval.TAG_LEN[g]).pairs
                     for g, t in (("<五律>", wulv), ("<七律>", qilv)))
    assert ov["dz_pairs"] == want_pairs == 4, f"聚合可比联数应为律诗之和，实为 {ov['dz_pairs']}"
    assert abs(ov["dz_pingze"] - ov["dz_ok_pingze"] / want_pairs) < 1e-12, \
        "聚合须按联级合并（Σ相对联数 / Σ可比联数）"
    print(f"  律诗对仗三项与 check_duizhang 逐项一致（<五律> {res['per_tag']['<五律>']['dz_pairs']} 联、"
          f"<七律> {res['per_tag']['<七律>']['dz_pairs']} 联）；绝句三项恒 0 ✓")


def test_duizhang_merge_by_couplet_not_sample_mean():
    """对仗按联级合并（率 = Σ分子联数 / Σ可比联数），不得按逐样本率再求平均。

    构造两个律诗样本：pairs=2 全相对（率 1.0）与 pairs=1 全不相对（率 0.0）；联级合并率为 2/3，
    而逐样本率平均为 0.5——二者必须可区分（本断言即口径锁定）。
    """
    assert hasattr(prosody_eval, "merged_lvshi_duizhang"), "实现缺少律诗合并对仗指标函数"
    d_rel = prosody.check_duizhang(DZ_REL, 5)
    d_irr = prosody.check_duizhang(DZ_IRREL, 5)
    assert (d_rel.pairs, d_rel.pingze) == (2, 1.0), f"前提：样本一应为 pairs=2 全相对，实为 {d_rel}"
    assert (d_irr.pairs, d_irr.pingze) == (1, 0.0), f"前提：样本二应为 pairs=1 全不相对，实为 {d_irr}"
    pool = [("<五律>", "国破山河在"), ("<五律>", "白日依山尽")]
    res = prosody_eval.score_group([DZ_REL, DZ_IRREL], pool)
    m = prosody_eval.merged_lvshi_duizhang(res)
    assert m["dz_pairs"] == 3, f"可比联数应为 2+1=3，实为 {m['dz_pairs']}"
    assert m["dz_ok_pingze"] == 2 and m["dz_ok_chongzi"] == 3, \
        f"分子须为两样本之和，实为 {m['dz_ok_pingze']}/{m['dz_ok_chongzi']}"
    assert abs(m["dz_pingze"] - 2.0 / 3.0) < 1e-12, f"联级合并率应为 2/3，实为 {m['dz_pingze']}"
    assert abs((d_rel.pingze + d_irr.pingze) / 2 - 0.5) < 1e-12, "前提：逐样本率平均应为 0.5"
    assert abs(m["dz_pingze"] - 0.5) > 1e-6, "联级合并率不得等于逐样本率平均（口径锁定失败）"
    assert abs(m["ci"]["dz_pingze"] - prosody_eval.ci_halfwidth(m["dz_pingze"], 3)) < 1e-12, \
        "对仗置信区间半宽须以可比联数为 n"
    print(f"  联级合并 {m['dz_ok_pingze']}/{m['dz_pairs']}={m['dz_pingze']:.4f}"
          f" ≠ 逐样本率平均 0.5 ✓")


def test_judge_duizhang_discrimination():
    """区分度自检：缺组→未自检；旧不重字率饱和→作废且不再参与有效性判定；
    参与判定的子指标差值均 <10 个百分点→失效，至少一项 ≥10→有效。"""
    assert hasattr(prosody_eval, "judge_duizhang_discrimination"), "实现缺少区分度自检函数"
    judge = prosody_eval.judge_duizhang_discrimination
    # 缺 M3 档：未自检，reason 点明缺 M3
    r1 = judge({"old": _fake_group("old", "旧模型", 0.70, 0.55, 0.50)})
    assert r1["checked"] is False, f"缺 M3 档不得给出自检结论：{r1}"
    assert "M3" in r1["reason"], f"reason 须点明缺 M3 档：{r1['reason']}"
    # 缺旧模型基线：未自检（指标有效性无从校验）
    r2 = judge({"m3": _fake_group("m3", "M3 模型", 0.98, 0.95, 0.90)})
    assert r2["checked"] is False and "旧" in r2["reason"], f"缺旧模型基线须未自检：{r2}"
    # 旧模型联内不重字率饱和（≥90%）→ 该子指标作废；平仄差值 20 个百分点 → 有效
    r3 = judge({"old": _fake_group("old", "旧模型", 0.70, 0.55, 0.50,
                                   dz_ok_pingze=20, dz_ok_chongzi=38),
                "m3": _fake_group("m3", "M3 模型", 0.98, 0.95, 0.90,
                                  dz_ok_pingze=28, dz_ok_chongzi=39)})
    assert r3["checked"] is True, "新旧齐备须做自检"
    assert r3["void_chongzi"] is True, f"旧模型不重字率 ≥90% 须判该子指标作废：{r3['old_dz_chongzi']}"
    assert abs(r3["diff_pingze_pp"] - 20.0) < 1e-9, f"平仄差应为 20 个百分点：{r3['diff_pingze_pp']}"
    assert r3["valid"] is True, "参与判定的子指标差值 ≥10 个百分点须判指标有效"
    # 两差值均 <10 个百分点 → 指标失效
    r4 = judge({"old": _fake_group("old", "旧模型", 0.70, 0.55, 0.50,
                                   dz_ok_pingze=20, dz_ok_chongzi=20),
                "m3": _fake_group("m3", "M3 模型", 0.98, 0.95, 0.90,
                                  dz_ok_pingze=22, dz_ok_chongzi=21)})
    assert r4["checked"] is True and r4["valid"] is False, f"差值均 <10 个百分点须判失效：{r4}"
    assert abs(r4["diff_pingze_pp"] - 5.0) < 1e-9 and abs(r4["diff_chongzi_pp"] - 2.5) < 1e-9, \
        f"差值应为 5.0 / 2.5 个百分点，实为 {r4['diff_pingze_pp']} / {r4['diff_chongzi_pp']}"
    assert r4["void_chongzi"] is False and abs(r4["old_dz_chongzi"] - 0.5) < 1e-12, \
        "旧模型不重字率 0.5 未饱和，不得判作废"
    for key in ("old_dz_pingze", "old_dz_chongzi", "m3_dz_pingze", "m3_dz_chongzi"):
        assert isinstance(r4[key], float), f"自检返回须含新旧两子指标数值 {key}"
    # 作废子指标不得救活指标：旧模型不重字率饱和（触发作废），且仅不重字率差值 ≥10 个百分点
    # （20.0）、平仄相对率差值 <10 个百分点（5.0）→ 参与判定者仅平仄、其差值不足，须判失效。
    r5 = judge({"old": _fake_group("old", "旧模型", 0.70, 0.55, 0.50,
                                   dz_ok_pingze=30, dz_ok_chongzi=38),
                "m3": _fake_group("m3", "M3 模型", 0.98, 0.95, 0.90,
                                  dz_ok_pingze=32, dz_ok_chongzi=30)})
    assert r5["checked"] is True and r5["void_chongzi"] is True, \
        f"旧模型不重字率 0.95 ≥90% 须判该子指标作废：{r5['old_dz_chongzi']}"
    assert abs(r5["diff_pingze_pp"] - 5.0) < 1e-9 and abs(r5["diff_chongzi_pp"] - 20.0) < 1e-9, \
        f"前提：平仄差 5.0、不重字率差 20.0 个百分点，实为 {r5['diff_pingze_pp']} / {r5['diff_chongzi_pp']}"
    assert r5["valid"] is False, \
        "已作废的不重字率差值 ≥10 个百分点不得救活指标（参与判定者仅平仄，其差值 <10）"
    # 未作废（旧模型两子指标均 <90%）时按原口径：平仄相对率差值 ≥10 个百分点 → 有效
    r6 = judge({"old": _fake_group("old", "旧模型", 0.70, 0.55, 0.50,
                                   dz_ok_pingze=20, dz_ok_chongzi=20),
                "m3": _fake_group("m3", "M3 模型", 0.98, 0.95, 0.90,
                                  dz_ok_pingze=28, dz_ok_chongzi=21)})
    assert r6["checked"] is True and r6["void_chongzi"] is False, \
        "旧模型两子指标均 <90%，不得判作废"
    assert abs(r6["diff_pingze_pp"] - 20.0) < 1e-9, f"平仄差应为 20 个百分点：{r6['diff_pingze_pp']}"
    assert r6["valid"] is True, "未作废时任一子指标差值 ≥10 个百分点须判有效"
    # 参与判定的子指标名元组须在返回字段中自证：未作废含两子指标，作废后仅剩平仄
    for res, want in ((r4, ("pingze", "chongzi")), (r5, ("pingze",)),
                      (r6, ("pingze", "chongzi"))):
        assert res["judged_submetrics"] == want, \
            f"参与判定的子指标应为 {want}（须新增 judged_submetrics 字段自证）：{res['judged_submetrics']}"
    print("  缺组未自检 / 饱和作废（作废子指标不救活指标） / 参与判定差值<10 个百分点失效 / "
          "≥10 个百分点有效，判定均正确 ✓")


def test_write_report_duizhang_and_selfcheck():
    """报告含对仗口径说明与区分度自检段落：M3 未就绪标「未自检」；齐备列四项数值与判定。"""
    d = _mkprobe()
    meta = {"pool_path": "data/eval-prompts.txt", "val_path": "data/val.txt",
            "corpus_path": "data/corpus.txt", "seed": 0, "new_tokens": 96, "temperature": 1.0,
            "top_k": 20, "n_per_tag": 40, "per_tag_n": {tag: 40 for tag in POOL_TAGS},
            "device": "numpy", "pool_stats": None}
    base = {"old": _fake_group("old", "旧模型 model.npz（约束关）", 0.70, 0.55, 0.50)}
    # 情形一：M3 未就绪 → 自检段落如实标「未自检」，报告仍以「待 M3」留空
    p1 = os.path.join(d, "ab-dz-nom3.txt")
    prosody_eval.write_report(base, p1, meta)
    with open(p1, encoding="utf-8") as f:
        t1 = f.read()
    assert "自检" in t1 and "未自检" in t1, "M3 未就绪时须有自检段落并标注「未自检」"
    assert "待 M3" in t1, "M3 未就绪时须保留「待 M3」占位"
    assert "只计律诗" in t1 and "联级合并" in t1, "须写明对仗真实口径（只计律诗、联级合并）"
    assert "可比联数" in t1, "对仗置信区间须注明以可比联数为 n"
    assert "不适用" in t1, "绝句须标注对仗不适用（不得拿 0 冒充）"
    assert "None" not in t1 and "nan" not in t1, "报告不得出现 None/nan"
    # 情形二：M3 齐备（约束关 + 约束开）且一项差值 ≥10 个百分点 → 判定有效，自检段落列四项数值
    ok = dict(base)
    ok["m3"] = _fake_group("m3", "M3 模型 model-gelv-m3.npz（约束关）", 0.98, 0.95, 0.90,
                           dz_ok_pingze=28, dz_ok_chongzi=21)
    ok["m3_cons"] = _fake_group("m3_cons", "M3 模型 model-gelv-m3.npz（约束开）", 0.99, 0.99, 0.99,
                                constraint=True, dz_ok_pingze=30, dz_ok_chongzi=22)
    p2 = os.path.join(d, "ab-dz-valid.txt")
    prosody_eval.write_report(ok, p2, meta)
    with open(p2, encoding="utf-8") as f:
        t2 = f.read()
    assert t2.count("待 M3") == 0, "M3 齐备时不应再出现占位"
    assert "【指标失效】" not in t2 and "指标有效" in t2, "任一差值 ≥10 个百分点须判指标有效"
    for key in ("旧模型（律诗合并", "新 M3 模型（律诗合并", "个百分点"):
        assert key in t2, f"自检段落缺少：{key}"
    assert "None" not in t2 and "nan" not in t2, "报告不得出现 None/nan"
    # 情形三：两差值均 <10 个百分点 → 显著措辞判「指标失效」并声明不得充当 M3 闸门依据
    bad = dict(base)
    bad["m3"] = _fake_group("m3", "M3 模型 model-gelv-m3.npz（约束关）", 0.98, 0.95, 0.90,
                            dz_ok_pingze=22, dz_ok_chongzi=21)
    bad["m3_cons"] = _fake_group("m3_cons", "M3 模型 model-gelv-m3.npz（约束开）", 0.99, 0.99, 0.99,
                                 constraint=True, dz_ok_pingze=22, dz_ok_chongzi=21)
    p3 = os.path.join(d, "ab-dz-invalid.txt")
    prosody_eval.write_report(bad, p3, meta)
    with open(p3, encoding="utf-8") as f:
        t3 = f.read()
    assert "【指标失效】" in t3, "两差值均 <10 个百分点须以显著措辞写明指标失效"
    assert "不得充当 M3 闸门依据" in t3, "须声明失效指标不得充当 M3 闸门依据"
    # 情形四：旧模型联内不重字率饱和（≥90%）触发作废，且仅不重字率差值 ≥10 个百分点、
    # 平仄相对率差值 <10 个百分点 → 报告须写明作废说明（不参与有效性判定、仅由平仄判定）并判失效。
    void = dict(base)
    void["old"] = _fake_group("old", "旧模型 model.npz（约束关）", 0.70, 0.55, 0.50,
                              dz_ok_pingze=30, dz_ok_chongzi=38)
    void["m3"] = _fake_group("m3", "M3 模型 model-gelv-m3.npz（约束关）", 0.98, 0.95, 0.90,
                             dz_ok_pingze=32, dz_ok_chongzi=30)
    p4 = os.path.join(d, "ab-dz-void.txt")
    prosody_eval.write_report(void, p4, meta)
    with open(p4, encoding="utf-8") as f:
        t4 = f.read()
    assert "【子指标作废】" in t4, "旧模型不重字率 ≥90% 须在报告自检段落判该子指标作废"
    assert "不参与有效性判定" in t4, "作废说明须写明该子指标不参与有效性判定"
    assert "有效性仅由对句平仄相对率判定" in t4, "作废说明须写明有效性仅由对句平仄相对率判定"
    assert "【指标失效】" in t4, \
        "不重字率作废后参与判定者仅平仄，其差值 <10 个百分点须判指标失效（不得被作废子指标救活）"
    print(f"  报告 {os.path.basename(p1)}/{os.path.basename(p2)}/{os.path.basename(p3)}"
          f"/{os.path.basename(p4)}"
          "：自检段落（未自检/有效/失效/作废）与对仗口径说明均正确 ✓")


# ─────────────────────────── val 语料重复与偏差披露 ───────────────────────────

def _val_corpus_dup_independent(val_path, corpus_path):
    """独立核算 val 与 corpus 整诗逐字重复的诗首数（与实现手法不同源）。

    手法：正则 `(?m)^(?=<[^>]+>)` 按「以诗体 token 起首的行」切行块，逐块取正文（剔除空白行）
    按换行拼接，再与 corpus 的行块正文集合比对——不复用实现侧的切块函数与判定逻辑。
    返回 (重复诗首数, val 诗总数)。
    """
    def blocks(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        out = []
        for chunk in re.split(r"(?m)^(?=<[^>]+>)", text):
            lines = chunk.split("\n")
            if not lines or not lines[0]:
                continue
            body_lines = [re.sub(r"^<[^>]+>", "", lines[0])] + lines[1:]
            out.append("\n".join(x for x in body_lines if x.strip()))
        return out

    corpus_blocks = set(blocks(corpus_path))
    val_blocks = blocks(val_path)
    return sum(1 for b in val_blocks if b in corpus_blocks), len(val_blocks)


def test_val_corpus_duplicate_stats():
    """val 语料重复统计：与独立算法（正则行块分割 + 逐块比对）一致，并锁定实证锚点 9/400（约 2.25%）。"""
    assert hasattr(prosody_eval, "val_corpus_duplicate_stats"), \
        "实现缺少 val 语料重复统计函数 val_corpus_duplicate_stats"
    stats = prosody_eval.val_corpus_duplicate_stats()
    want_dup, want_total = _val_corpus_dup_independent(prosody_eval.VAL_PATH,
                                                       prosody_eval.CORPUS_PATH)
    assert (stats["dup_poems"], stats["total_poems"]) == (want_dup, want_total), \
        f"统计与独立算法不符：实现 {stats} vs 独立 dup={want_dup} total={want_total}"
    assert stats["dup_poems"] == 9, f"实证重复诗首数应为 9，实为 {stats['dup_poems']}"
    assert stats["total_poems"] == 400, f"val 诗总数应为 400，实为 {stats['total_poems']}"
    assert abs(stats["ratio"] - 9.0 / 400.0) < 1e-12, \
        f"占比应为 9/400，实为 {stats['ratio']}"
    assert abs(stats["ratio"] - 0.0225) < 1e-9, f"占比应约为 2.25%，实为 {stats['ratio']}"
    print(f"  独立算法一致：整诗重复 {stats['dup_poems']}/{stats['total_poems']}"
          f"={stats['ratio'] * 100:.2f}% ✓")


def test_write_report_annotates_val_bias():
    """报告须如实披露 val 语料重复的乐观偏差：含「乐观偏差」、取自统计函数的实测 N/总数与占比、
    根因（诗级划分未去重）、prompt 侧泄漏已由碰撞排除消除、根治本轮不做。"""
    assert hasattr(prosody_eval, "val_corpus_duplicate_stats"), \
        "实现缺少 val 语料重复统计函数 val_corpus_duplicate_stats"
    d = _mkprobe()
    stats = prosody_eval.val_corpus_duplicate_stats()
    path = os.path.join(d, "ab-val-bias.txt")
    meta = {"pool_path": "data/eval-prompts.txt", "val_path": "data/val.txt",
            "corpus_path": "data/corpus.txt", "seed": 0, "new_tokens": 96, "temperature": 1.0,
            "top_k": 20, "n_per_tag": 40, "per_tag_n": {tag: 40 for tag in POOL_TAGS},
            "device": "numpy", "pool_stats": None}
    results = {"old": _fake_group("old", "旧模型 model.npz（约束关）", 0.70, 0.55, 0.50)}
    prosody_eval.write_report(results, path, meta)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "乐观偏差" in text, "报告须写明语料重复致 val 指标的乐观偏差"
    frag = f"{stats['dup_poems']}/{stats['total_poems']}"
    assert frag in text, f"报告须含取自统计函数的实测重复首数占比 {frag}"
    ratio_txt = f"{stats['ratio'] * 100:.2f}%"
    assert ratio_txt in text, f"报告须含实测占比 {ratio_txt}（数字不得写死）"
    assert "未去重" in text, "报告须写明根因（诗级划分未去重）"
    assert "碰撞排除" in text, "报告须写明 prompt 侧泄漏已由碰撞排除消除"
    assert "本轮不做" in text, "报告须写明根治（诗级去重 + 重建语料 + 重训 M1/M2）本轮不做"
    print(f"  报告已披露 val 语料重复乐观偏差 {frag}={ratio_txt}，根治声明齐备 ✓")


# ─────────────────── 复用落盘池的排除统计 / 陈旧池防呆 / 口径局限 ───────────────────

def test_build_eval_pool_reuse_reports_full_stats():
    """池文件已存在（复用路径）时仍须产出完整碰撞排除统计，且与重建路径逐项一致（§8.2 死规则）。"""
    d = _mkprobe()
    pool_path = os.path.join(d, "reuse-pool.txt")
    if os.path.exists(pool_path):
        os.remove(pool_path)
    pool_build = prosody_eval.build_eval_pool(pool_path=pool_path)
    stats_build = prosody_eval.pool_exclusion_stats()
    assert stats_build is not None, "新建池后应能取到碰撞排除统计"
    prosody_eval._POOL_STATS = None            # 清空统计，验证复用路径自身会按固定种子重算并写回
    pool_reuse = prosody_eval.build_eval_pool(pool_path=pool_path)
    stats_reuse = prosody_eval.pool_exclusion_stats()
    assert pool_reuse == pool_build, "复用路径须读取落盘池原文（不得重抽、不得覆盖）"
    assert stats_reuse is not None, "复用落盘池时 pool_stats 不得为 None（§8.2：不得静默吞掉）"
    for key in ("collision", "collision_total", "supply_before", "supply_after", "taken"):
        assert stats_reuse[key] == stats_build[key], \
            f"复用路径与重建路径的 {key} 不一致：{stats_reuse[key]} vs {stats_build[key]}"
    assert stats_reuse["collision_total"] >= 1, "实证前提：val 候选须至少 1 条命中 corpus"
    print(f"  复用落盘池：pool_stats 完整且与重建逐项一致（实际排除 "
          f"{stats_reuse['collision_total']} 条）✓")


def test_write_report_reuse_pool_records_exclusions():
    """复用落盘池时报告仍须写出实际排除条数与排除后逐诗体供给量，且不得出现「未记录」字样。"""
    d = _mkprobe()
    pool_path = os.path.join(d, "reuse-report-pool.txt")
    if os.path.exists(pool_path):
        os.remove(pool_path)
    pool = prosody_eval.build_eval_pool(pool_path=pool_path)
    ps = prosody_eval.pool_exclusion_stats()
    assert ps is not None, "复用路径亦须有碰撞排除统计"
    assert hasattr(prosody_eval, "half_parity_stats"), \
        "实现缺少半句奇偶统计函数 half_parity_stats"
    path = os.path.join(d, "ab-reuse.txt")
    meta = {"pool_path": pool_path, "val_path": prosody_eval.VAL_PATH,
            "corpus_path": prosody_eval.CORPUS_PATH, "seed": 0, "new_tokens": 96,
            "temperature": 1.0, "top_k": 20, "n_per_tag": 40,
            "per_tag_n": {tag: 40 for tag in POOL_TAGS}, "device": "numpy",
            "pool_stats": ps, "half_parity": prosody_eval.half_parity_stats(pool)}
    results = {"old": _fake_group("old", "旧模型 model.npz（约束关）", 0.70, 0.55, 0.50)}
    prosody_eval.write_report(results, path, meta)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "未记录" not in text, "复用落盘池时不得再出现「未记录」（§8.2 死规则）"
    col = "、".join(f"{t} {ps['collision'][t]}" for t in POOL_TAGS)
    sup = "、".join(f"{t} {ps['supply_after'][t]}" for t in POOL_TAGS)
    assert f"排除前碰撞 {ps['collision_total']} 条（{col}）" in text, \
        "报告须写出实际排除条数（合计与逐诗体）"
    assert f"排除后逐诗体候选供给量 {sup} 条" in text, \
        "报告须写出排除后逐诗体供给量"
    print(f"  复用池报告写出实际排除 {ps['collision_total']} 条与逐诗体供给量，无「未记录」✓")


def test_build_eval_pool_stale_file_fails_loudly():
    """落盘池与按固定种子重算逐项不一致（陈旧/被篡改）时必须响亮失败，且不得覆盖落盘池。"""
    d = _mkprobe()
    pool_path = os.path.join(d, "stale-pool.txt")
    if os.path.exists(pool_path):
        os.remove(pool_path)
    prosody_eval.build_eval_pool(pool_path=pool_path)
    with open(pool_path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().split("\n") if ln]
    lines[0], lines[1] = lines[1], lines[0]     # 篡改：交换前两条，制造与重算不一致的陈旧池
    with open(pool_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        prosody_eval.build_eval_pool(pool_path=pool_path)
    except AssertionError as e:
        assert "不一致" in str(e), f"错误信息应点明池已陈旧/来源不同：{e}"
    else:
        raise AssertionError("STALE POOL WAS SILENTLY ACCEPTED")
    with open(pool_path, encoding="utf-8") as f:
        assert [ln for ln in f.read().split("\n") if ln] == lines, \
            "校验失败时不得覆盖落盘池（不得悄悄重抽）"
    print("  陈旧/篡改池触发响亮失败（断言异常），且未覆盖落盘池 ✓")


def _val_halflines_indexed(val_path):
    """独立实现（与工具函数不同源）：正则按行块切诗块 + 正则切半句，重建 (诗体, 半句)→首现 0 基半句下标。"""
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    idx = {}
    for chunk in re.split(r"(?m)^(?=<[^>]+>)", text):
        lines = chunk.split("\n")
        if not lines or not lines[0]:
            continue
        m = re.match(r"<[^>]+>", lines[0])
        tag = m.group(0)
        if tag not in POOL_TAGS:
            continue
        body = [lines[0][m.end():]] + lines[1:]
        segs = [s for s in re.split(r"[，。！？；：、\s]+", "\n".join(body)) if s]
        for i, seg in enumerate(segs):
            if _is_pure_hanzi(seg):
                idx.setdefault((tag, seg), i)
    return idx


def test_half_parity_stats_matches_independent():
    """半句奇偶分布：工具函数结果须与独立实现（正则切块 + 独立重建半句下标）逐项一致，并锁定实证锚点 70/90。"""
    assert hasattr(prosody_eval, "half_parity_stats"), \
        "实现缺少半句奇偶统计函数 half_parity_stats"
    pool = prosody_eval.read_pool_file(prosody_eval.POOL_PATH)
    assert len(pool) == 160, f"前提：落盘池应为 160 条，实为 {len(pool)}"
    hp = prosody_eval.half_parity_stats(pool)
    idx = _val_halflines_indexed(prosody_eval.VAL_PATH)
    even = sum(1 for t, p in pool if idx.get((t, p)) is not None and idx[(t, p)] % 2 == 0)
    odd = sum(1 for t, p in pool if idx.get((t, p)) is not None and idx[(t, p)] % 2 == 1)
    assert (hp["even"], hp["odd"]) == (even, odd), \
        f"与独立实现不一致：工具 {hp['even']}/{hp['odd']} vs 独立 {even}/{odd}"
    assert hp["total"] == len(pool) == 160 and hp["unlocated"] == 0, \
        f"池内每一条都应能定位原诗半句下标：{hp}"
    assert (even, odd) == (70, 90), f"实证锚点应为 偶数位 70 / 奇数位 90，实为 {even}/{odd}"
    assert abs(hp["even_ratio"] - 70 / 160) < 1e-12 and abs(hp["odd_ratio"] - 90 / 160) < 1e-12, \
        "占比须由条数实算（70/160 与 90/160）"
    print(f"  池 {hp['total']} 条：偶数位 {hp['even']}（{hp['even_ratio'] * 100:.2f}%）、"
          f"奇数位 {hp['odd']}（{hp['odd_ratio'] * 100:.2f}%），与独立实现一致 ✓")


def test_old_model_vocab_and_prompt_id_parity():
    """旧档按自带词表（纯文本 8196）载入；纯汉字 prompt 在旧/新词表下逐 id 相同。"""
    model, stoi, itos = prosody_eval.load_old_model()
    assert model.vocab_size == 8196, f"旧档词表应为 8196，实为 {model.vocab_size}"
    chars = train.load_corpus()[1]
    assert stoi == {c: i for i, c in enumerate(chars)}, "旧档词表应与语料纯文本字符表逐项一致"
    assert len(itos) == len(stoi) == 8196, "旧档 id→字 映射应与词表等长"
    new_stoi = train.load_corpus()[2]
    for prompt in ("床前明月光", "两个黄鹂鸣翠柳"):
        a = [int(x) for x in train.encode(prompt, stoi)]
        b = [int(x) for x in train.encode(prompt, new_stoi)]
        assert a == b, f"prompt {prompt} 在旧/新词表下编码不一致：{a} vs {b}"
    print("  旧档词表 8196 载入成功，纯汉字 prompt 旧/新词表编码一致 ✓")


def test_new_model_head_config():
    """新档辅助头配置：M1 仅韵部头、M2 韵部头+平仄头；缺失参数为 0（配置不符即响亮失败）。"""
    m1, stoi1, _ = prosody_eval.load_new_model(prosody_eval.MODEL_M1)
    assert m1.rhyme_head is not None and m1.tone_head is None, \
        "M1 应仅挂韵部头（无平仄头）"
    assert len(stoi1) == 8201, f"新档词表应为 8201，实为 {len(stoi1)}"
    m2, stoi2, _ = prosody_eval.load_new_model(prosody_eval.MODEL_M2)
    assert m2.rhyme_head is not None and m2.tone_head is not None, "M2 应同时挂韵部头与平仄头"
    assert stoi1 == stoi2, "M1/M2 应共用同一词表"
    print("  M1 仅韵部头、M2 韵部头+平仄头，缺失参数 0 ✓")


def test_generate_path_smoke():
    """极短生成：旧档模型对 1 条 prompt 生成 4 个 token，验证解码链路与 id 范围。"""
    model, stoi, itos = prosody_eval.load_old_model()
    prosody_eval.set_seed(0)
    ids = onp.asarray(train.encode("床前明月光", stoi))
    assert len(ids) == 5, "前提：prompt 应编码为 5 个 token"
    out = model.generate(np.asarray(ids[None, :]), 4, temperature=1.0, top_k=20,
                         logits_processor=None)
    arr = as_numpy(out)
    assert arr.shape == (1, 9), f"生成序列形状应为 (1,9)，实为 {arr.shape}"
    gen = arr[0][5:]
    assert all(0 <= int(i) < 8196 for i in gen), f"旧档生成 id 应落在旧词表内：{gen}"
    text = "床前明月光" + "".join(itos[int(i)] for i in gen)
    assert len(text) == 9, "拼回的文本长度应为 prompt + 4"
    print(f"  生成 4 token：{text!r} ✓")


# ─────────── 押韵口径相位披露（真诗同口径上限 + 模型按相位拆分，死规则 §8.2）───────────

# 合成夹具：四诗体各一首（<五绝>/<七绝> 各 4 半句，<五律>/<七律> 各 8 半句）。
# 半句显式列字（不经实现侧解析），使期望值可在用例内独立复算；<五律> 半句恒为 5 字，
# 专门锁定「句长按诗体取 5 字」而非误用 7 字。
CEIL_WUJUE = ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡"]
CEIL_QIJUE = ["两个黄鹂鸣翠柳", "一行白鹭上青天", "窗含西岭千秋雪", "门泊东吴万里船"]
CEIL_WULV = ["国破山河在", "城春草木深", "感时花溅泪", "恨别鸟惊心",
             "烽火连三月", "家书抵万金", "白头搔更短", "浑欲不胜簪"]
CEIL_QILV = ["岧嶤太华俯咸京", "天外三峰削不成", "武帝祠前云欲散", "仙人掌上雨初晴",
             "河山北枕秦关险", "驿树西连汉畤平", "借问路傍名利客", "无如此处学长生"]
CEIL_POEMS = (("<五绝>", CEIL_WUJUE), ("<七绝>", CEIL_QIJUE),
              ("<五律>", CEIL_WULV), ("<七律>", CEIL_QILV))


def _write_ceiling_val(path, poems=CEIL_POEMS):
    """把合成诗块落盘为 val 文件（每首一行「诗体 + 半句以「，」相连」）。"""
    with open(path, "w", encoding="utf-8") as f:
        for tag, halves in poems:
            f.write(tag + "，".join(halves) + "。\n")
    return path


def _expected_parity_rates(poems):
    """用例侧独立实算：逐诗显式列出的半句 → 逐截断点取余下部分 → 按 0 基下标奇偶分类求均值。

    只复用 prosody 的字音判据（check_rhyme / check_line_len），聚合口径在用例内独立书写，
    故可校验实现侧的分类与率值；返回 {0: [(押韵, 句长), ...], 1: [...]}。
    """
    groups = {0: [], 1: []}
    for tag, halves in poems:
        plen = prosody_eval.TAG_LEN[tag]
        for i in range(len(halves)):
            suffix = "，".join(halves[i:])
            groups[i % 2].append((prosody.check_rhyme(suffix).main,
                                  prosody.check_line_len(suffix, plen)))
    return groups


def _tree_snapshot(dirs):
    """对给定目录取「条目名集合」快照（用于证明被测函数只读不写）。"""
    out = set()
    for dd in dirs:
        if os.path.isdir(dd):
            out.update(os.path.join(dd, name) for name in os.listdir(dd))
    return out


def test_real_poem_ceiling_stats_synthetic():
    """真诗同口径上限（合成夹具）：对齐/错位两类的划分与率值、池相位加权值算术正确；
    句长须按诗体 TAG_LEN 取值（五律 5 字，锁定不得误用 7 字）。"""
    assert hasattr(prosody_eval, "real_poem_ceiling_stats"), \
        "实现缺少真诗同口径上限函数 real_poem_ceiling_stats"
    d = _mkprobe()
    val_path = _write_ceiling_val(os.path.join(d, "ceiling_val.txt"))
    pool = [("<五绝>", "床前明月光"), ("<五绝>", "举头望明月"),
            ("<五绝>", "疑是地上霜"), ("<七绝>", "两个黄鹂鸣翠柳")]
    st = prosody_eval.real_poem_ceiling_stats(pool, val_path=val_path)
    exp = _expected_parity_rates(CEIL_POEMS)
    # 划分：四诗体半句数 4/4/8/8 → 0 基偶位 12 个、奇位 12 个
    assert st["aligned"]["n"] == 12 and st["misaligned"]["n"] == 12, \
        f"两类截断点划分应为各 12 个，实为 {st['aligned']['n']}/{st['misaligned']['n']}"
    for key, p in (("aligned", 0), ("misaligned", 1)):
        want_rhyme = float(onp.mean([x[0] for x in exp[p]]))
        want_len = float(onp.mean([x[1] for x in exp[p]]))
        assert abs(st[key]["rhyme"] - want_rhyme) < 1e-12, \
            f"{key} 押韵率应 {want_rhyme}，实为 {st[key]['rhyme']}"
        assert abs(st[key]["line_len"] - want_len) < 1e-12, \
            f"{key} 句长率应 {want_len}，实为 {st[key]['line_len']}"
    # 句长须按诗体取值：五律半句 5 字合律；若误按 7 字口径则全不合律（锁定「5 字」）
    wulv_suffix = "，".join(CEIL_WULV)
    assert prosody.check_line_len(wulv_suffix, 5) == 1.0, "前提：五律夹具须为 5 字半句"
    assert prosody.check_line_len(wulv_suffix, 7) == 0.0, \
        "前提：该夹具按 7 字口径应全不合律（本用例据此锁定不得误用 7 字）"
    # 池相位加权：偶数位 3 条 / 奇数位 1 条（half_parity_stats 实算）
    assert (st["pool_even"], st["pool_odd"], st["pool_total"]) == (3, 1, 4), \
        f"池相位计数应为 3/1，实为 {st['pool_even']}/{st['pool_odd']}"
    want_pw = (3 * st["aligned"]["rhyme"] + 1 * st["misaligned"]["rhyme"]) / 4
    assert abs(st["pool_weighted_rhyme"] - want_pw) < 1e-12, \
        f"押韵池加权上限应 {want_pw}，实为 {st['pool_weighted_rhyme']}"
    want_pwl = (3 * st["aligned"]["line_len"] + 1 * st["misaligned"]["line_len"]) / 4
    assert abs(st["pool_weighted_line_len"] - want_pwl) < 1e-12, \
        f"句长池加权上限应 {want_pwl}，实为 {st['pool_weighted_line_len']}"
    assert st["aligned"]["rhyme"] > st["misaligned"]["rhyme"] + 0.3, \
        "前提：合成夹具对齐截断应显著高于错位截断（否则本用例无检定力）"
    print(f"  对齐 n={st['aligned']['n']} 押韵 {st['aligned']['rhyme']:.4f}；"
          f"错位 n={st['misaligned']['n']} 押韵 {st['misaligned']['rhyme']:.4f}；"
          f"池加权 押韵 {st['pool_weighted_rhyme']:.4f} 句长 {st['pool_weighted_line_len']:.4f} ✓")


def test_real_poem_ceiling_stats_read_only_and_deterministic():
    """真诗同口径上限：纯文本、确定性（重复调用结果逐项一致）、无副作用（不产生任何落盘文件）。"""
    assert hasattr(prosody_eval, "real_poem_ceiling_stats"), \
        "实现缺少真诗同口径上限函数 real_poem_ceiling_stats"
    d = _mkprobe()
    _write_ceiling_val(os.path.join(d, "ceiling_val2.txt"))
    pool = prosody_eval.read_pool_file(prosody_eval.POOL_PATH)
    watch = (BASE, os.path.join(BASE, "data"), d)
    before = _tree_snapshot(watch)
    a = prosody_eval.real_poem_ceiling_stats(pool, val_path=prosody_eval.VAL_PATH)
    b = prosody_eval.real_poem_ceiling_stats(pool, val_path=prosody_eval.VAL_PATH)
    assert a == b, "重复调用结果须逐项一致（确定性）"
    assert _tree_snapshot(watch) == before, "该函数只读不写，不得产生任何落盘文件"
    for key in ("aligned", "misaligned", "pool_even", "pool_odd", "pool_total",
                "pool_weighted_rhyme", "pool_weighted_line_len"):
        assert key in a, f"返回值缺少字段 {key}"
    assert a["aligned"]["n"] == a["misaligned"]["n"] > 0, "val 四诗体两类截断点数应相等且非空"
    assert a["aligned"]["rhyme"] > 0.5 and a["misaligned"]["rhyme"] < 0.1, \
        f"真诗对齐截断应高、错位截断应趋近 0：{a['aligned']['rhyme']}/{a['misaligned']['rhyme']}"
    assert a["aligned"]["line_len"] > 0.99 and a["misaligned"]["line_len"] > 0.99, \
        "真诗在两种截断下句长合规率均应接近 1"
    print(f"  确定性一致、无落盘；真诗对齐押韵 {a['aligned']['rhyme']:.4f} / "
          f"错位 {a['misaligned']['rhyme']:.4f}，句长均 >0.99 ✓")


def test_split_rhyme_by_parity_synthetic():
    """模型实测按相位拆分：两子集押韵率与样本量正确，且两子集样本量之和 == 池条数。"""
    assert hasattr(prosody_eval, "split_rhyme_by_parity"), \
        "实现缺少按相位拆分函数 split_rhyme_by_parity"
    d = _mkprobe()
    val_path = _write_ceiling_val(os.path.join(d, "parity_val.txt"))
    pool = [("<五绝>", "床前明月光"), ("<五绝>", "疑是地上霜"),
            ("<五绝>", "举头望明月"), ("<五绝>", "低头思故乡")]
    texts = [GOOD5, TWO_HALVES, GOOD7, "白日依山尽，黄河入海流。"]
    res = prosody_eval.split_rhyme_by_parity(texts, pool, val_path=val_path)
    exp_aligned = [prosody.check_rhyme(texts[0]).main, prosody.check_rhyme(texts[2]).main]
    exp_misalg = [prosody.check_rhyme(texts[1]).main, prosody.check_rhyme(texts[3]).main]
    assert res["aligned"]["n"] == 2 and res["misaligned"]["n"] == 2, \
        f"两子集样本量应为 2/2，实为 {res['aligned']['n']}/{res['misaligned']['n']}"
    assert res["aligned"]["n"] + res["misaligned"]["n"] == len(pool) == res["total"], \
        "两子集样本量之和须等于池条数"
    assert abs(res["aligned"]["rhyme"] - float(onp.mean(exp_aligned))) < 1e-12, "对齐子集率不符"
    assert abs(res["misaligned"]["rhyme"] - float(onp.mean(exp_misalg))) < 1e-12, "错位子集率不符"
    assert float(onp.mean(exp_aligned)) != float(onp.mean(exp_misalg)), \
        "前提：两子集率须可区分（否则本用例无检定力）"
    print(f"  对齐子集 n=2 押韵 {res['aligned']['rhyme']:.4f}；"
          f"错位子集 n=2 押韵 {res['misaligned']['rhyme']:.4f}；样本量之和 == 池条数 4 ✓")


def test_write_report_parity_disclosure():
    """报告须设「真诗同口径上限」与「对齐子集 / 错位子集」固定章节，数字取自实算函数、成句可断言。"""
    assert hasattr(prosody_eval, "real_poem_ceiling_stats"), \
        "实现缺少真诗同口径上限函数 real_poem_ceiling_stats"
    d = _mkprobe()
    meta = {"pool_path": prosody_eval.POOL_PATH, "val_path": prosody_eval.VAL_PATH,
            "corpus_path": prosody_eval.CORPUS_PATH, "seed": 0, "new_tokens": 96,
            "temperature": 1.0, "top_k": 20, "n_per_tag": 40,
            "per_tag_n": {tag: 40 for tag in POOL_TAGS}, "device": "numpy",
            "pool_stats": None}
    pool = prosody_eval.read_pool_file(prosody_eval.POOL_PATH)
    ceil = prosody_eval.real_poem_ceiling_stats(pool)
    res = _fake_group("old", "旧模型 model.npz（约束关）", 0.70, 0.55, 0.50)
    res["rhyme_by_parity"] = {"aligned": {"n": 70, "rhyme": 0.11},
                              "misaligned": {"n": 90, "rhyme": 0.05}, "total": 160}
    path = os.path.join(d, "ab-parity.txt")
    prosody_eval.write_report({"old": res}, path, meta)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for key in ("真诗同口径上限", "对齐截断", "错位截断", "对齐子集", "错位子集",
                "池加权上限", "不可归因于相位错位"):
        assert key in text, f"报告缺少相位披露关键字段：{key}"
    assert f"对齐截断：n={ceil['aligned']['n']}" in text, "报告须写出对齐截断的实算样本量"
    assert f"押韵 {ceil['aligned']['rhyme'] * 100:.2f}%" in text, "报告须写出对齐截断的实算押韵率"
    assert f"押韵池加权上限 {ceil['pool_weighted_rhyme'] * 100:.2f}%" in text, \
        "报告须写出押韵池加权上限（数字取自实算）"
    assert "对齐子集 n=70 押韵 11.00%" in text and "错位子集 n=90 押韵 5.00%" in text, \
        "报告须按组写出两子集的样本量与押韵率"
    assert "None" not in text and "nan" not in text, "报告不得出现 None/nan"
    print(f"  报告含真诗同口径上限（对齐 {ceil['aligned']['rhyme'] * 100:.2f}% / "
          f"错位 {ceil['misaligned']['rhyme'] * 100:.2f}%，池加权 "
          f"{ceil['pool_weighted_rhyme'] * 100:.2f}%）与按组相位拆分 ✓")


# ───────── 相位归因判据（实算两子集差异 → 结论措辞，死规则 §8.2 禁两端）─────────

def _parity_meta():
    """相位判据用例共用的写盘 meta（与正式入口 run_eval 同构）；池统计留空由实现自行重算。"""
    return {"pool_path": prosody_eval.POOL_PATH, "val_path": prosody_eval.VAL_PATH,
            "corpus_path": prosody_eval.CORPUS_PATH, "seed": 0, "new_tokens": 96,
            "temperature": 1.0, "top_k": 20, "n_per_tag": 40,
            "per_tag_n": {tag: 40 for tag in POOL_TAGS}, "device": "numpy",
            "pool_stats": None}


def _fake_parity(kind, label, aligned_rhyme, misaligned_rhyme, n_a=70, n_m=90):
    """在 _fake_group 之上挂 rhyme_by_parity（两子集率与样本量），供相位归因判据用例使用。"""
    res = _fake_group(kind, label, 0.70, 0.55, 0.50)
    res["rhyme_by_parity"] = {"aligned": {"n": n_a, "rhyme": aligned_rhyme},
                              "misaligned": {"n": n_m, "rhyme": misaligned_rhyme},
                              "total": n_a + n_m}
    return res


def test_judge_parity_attribution_not_significant():
    """相位归因判据不显著支：各组两子集率差均落在各自置信区间半宽之和内、且方向不一致时，
    判据须输出「不可归因于相位错位、闸门判定成立」，并逐组列出实算率差与半宽之和；
    一律不得出现为模型低分开脱的表述（口径不得当借口否认模型缺陷）。"""
    assert hasattr(prosody_eval, "judge_parity_attribution"), \
        "实现缺少相位归因判据函数 judge_parity_attribution"
    results = {
        "old": _fake_parity("old", "旧模型 model.npz（约束关）", 0.0804, 0.0759),
        "m2": _fake_parity("m2", "M2 模型 model-gelv-m2.npz（约束关）", 0.0695, 0.0923),
        "m3_cons": _fake_parity("m3_cons", "M3 模型 model-gelv-m3.npz（约束开）", 0.2767, 0.3158),
    }
    j = prosody_eval.judge_parity_attribution(results)
    assert j["checked"] is True, "三组皆有相位拆分，判据须已检"
    assert j["verdict"] == "not_attributable", f"不显著支应判不可归因，实为 {j['verdict']}"
    assert not j["significant_aligned_higher"], "本例无「对齐显著高于错位」组，不得误报"
    assert j["directions_uniform"] is False, "本例差值方向不一致（1 正 2 负），须如实反映"
    text = "\n".join(j["lines"])
    assert "不可归因于相位错位" in text and "闸门判定成立" in text, \
        f"不显著支须判「不可归因于相位错位、闸门判定成立」，实为：\n{text}"
    assert "方向亦不一致" in text, f"不显著支须写出方向不一致这一实算事实，实为：\n{text}"
    for bad in ("不得据此判定模型未内化", "主要来自 prompt 相位错位", "取样口径代价", "为模型开脱"):
        assert bad not in text, f"判据不得出现为模型开脱的表述：{bad}"
    for kind in results:                       # 判据须由数据推导：逐组列出率差与半宽之和
        rbp = results[kind]["rhyme_by_parity"]
        hw = (prosody_eval.ci_halfwidth(rbp["aligned"]["rhyme"], rbp["aligned"]["n"])
              + prosody_eval.ci_halfwidth(rbp["misaligned"]["rhyme"], rbp["misaligned"]["n"]))
        assert f"{hw * 100:.2f} 个百分点" in text, \
            f"判据须列出 {kind} 的置信区间半宽之和 {hw * 100:.2f}（不得写死），实为：\n{text}"
        assert prosody_eval._pct(rbp["aligned"]["rhyme"]) in text \
            and prosody_eval._pct(rbp["misaligned"]["rhyme"]) in text, \
            f"判据须列出 {kind} 两子集实算率值，实为：\n{text}"
    d = _mkprobe()
    path = os.path.join(d, "ab-parity-notsiginificant.txt")
    prosody_eval.write_report(results, path, _parity_meta())
    with open(path, encoding="utf-8") as f:
        rep = f.read()
    assert "不可归因于相位错位" in rep and "闸门判定成立" in rep, "报告判据条须判不可归因"
    for bad in ("不得据此判定模型未内化", "主要来自 prompt 相位错位"):
        assert bad not in rep, f"报告不得出现为模型开脱的表述：{bad}"
    assert "真诗对齐基准" in rep, "第八节第 1 条须写明真诗对齐截断值作为「完美续写」基准"
    print("  相位归因判据不显著支：逐组实算率差/半宽之和 → 不可归因于相位错位、闸门判定成立 ✓")


def test_judge_parity_attribution_significant_aligned_higher():
    """相位归因判据显著支：某组「对齐子集显著高于错位子集」（差值超出该组两子集置信区间半宽
    之和）时，判据须点名该组实算事实并写「须按 §九 停手报主子复核口径」，不得判不可归因。"""
    assert hasattr(prosody_eval, "judge_parity_attribution"), \
        "实现缺少相位归因判据函数 judge_parity_attribution"
    label = "M3 模型 model-gelv-m3.npz（约束关）"
    results = {
        "old": _fake_parity("old", "旧模型 model.npz（约束关）", 0.0804, 0.0759),
        "m3": _fake_parity("m3", label, 0.6000, 0.0500),
    }
    j = prosody_eval.judge_parity_attribution(results)
    assert j["checked"] is True, "两组皆有相位拆分，判据须已检"
    assert j["verdict"] == "needs_review", f"显著支应判复核口径，实为 {j['verdict']}"
    assert j["significant_aligned_higher"] == [label], \
        f"须点名显著偏高的那一组，实为 {j['significant_aligned_higher']}"
    text = "\n".join(j["lines"])
    assert "报主子复核口径" in text and "§九" in text, \
        f"显著支须写「须按 §九 停手报主子复核口径」，实为：\n{text}"
    assert "不可归因于相位错位" not in text, "存在显著偏高组时不得判不可归因"
    d = _mkprobe()
    path = os.path.join(d, "ab-parity-significant.txt")
    prosody_eval.write_report(results, path, _parity_meta())
    with open(path, encoding="utf-8") as f:
        rep = f.read()
    assert "报主子复核口径" in rep, "报告判据条须与判据函数同源（同一文案落盘）"
    assert "不可归因于相位错位" not in rep, "报告在存在显著偏高组时不得判不可归因"
    print("  相位归因判据显著支：点名该组 → 须按 §九 停手报主子复核口径 ✓")


# ─────────── 诗体前缀口径对照（诊断；§十五 / §15.8）───────────

def test_encode_tag_prefix_is_single_token():
    """编码前提：诗体 token 前缀在 prompt 之前只占 1 个 id（tag 臂断言 len(ids)=len(prompt)+1 的依据）。"""
    stoi = train.load_corpus()[2]
    for tag, prompt, n in (("<五绝>", "新晴花枝下", 5), ("<七绝>", "两个黄鹂鸣翠柳", 7)):
        ids = train.encode(tag + prompt, stoi)
        assert len(ids) == n + 1, f"{tag}{prompt} 应编码为 {n + 1} 个 id，实为 {len(ids)}"
        assert len(train.encode(prompt, stoi)) == n, f"{prompt} 应编码为 {n} 个 id"
        assert int(ids[0]) >= 8196, "诗体 token 的 id 应落在纯文本字符表（8196）之后"
    print("  诗体前缀 <五绝>/<七绝> 各计 1 个 id，其 id ≥ 8196 ✓")


def test_generate_texts_tag_prefix_smoke():
    """tag 臂生成通道：返回文本以「诗体 token + prompt」起头，续写长度不变。

    续写长度按 token 计（诗体 token 计 1 个 id、占 4 字符，故字符数 ≠ token 数）：裸臂 =
    prompt 5 token + 4 token 续写；tag 臂 = 诗体 token 1 + prompt 5 token + 4 token 续写。
    """
    model, stoi, itos = prosody_eval.load_new_model(prosody_eval.MODEL_M1)
    pool = [("<五绝>", "新晴花枝下")]
    bare = prosody_eval.generate_texts(model, stoi, itos, pool, new_tokens=4)
    tag = prosody_eval.generate_texts(model, stoi, itos, pool, new_tokens=4, tag_prefix=True)
    n_prompt = len(train.encode("新晴花枝下", stoi))
    assert bare[0].startswith("新晴花枝下"), f"裸臂文本异常：{bare[0]!r}"
    assert tag[0].startswith("<五绝>新晴花枝下"), f"tag 臂文本异常：{tag[0]!r}"
    assert len(train.encode(bare[0], stoi)) == n_prompt + 4, \
        f"裸臂续写应为 4 个 token：{bare[0]!r} → {len(train.encode(bare[0], stoi))} token"
    assert len(train.encode(tag[0], stoi)) == 1 + n_prompt + 4, \
        f"tag 臂续写应为 4 个 token 且前缀占 1 个 id：{tag[0]!r} → " \
        f"{len(train.encode(tag[0], stoi))} token"
    print(f"  裸臂 {bare[0]!r} / tag 臂 {tag[0]!r} ✓")


def test_tag_arm_available_by_vocab():
    """tag 臂可执行性按档内词表判定：旧档（纯文本词表 8196）不可执行，M1/M2/M3 可执行。"""
    assert prosody_eval.tag_arm_available(prosody_eval.OLD_MODEL) is False, \
        "旧档词表不含诗体 token，tag 臂应判不可执行"
    for p in (prosody_eval.MODEL_M1, prosody_eval.MODEL_M2, prosody_eval.MODEL_M3):
        assert prosody_eval.tag_arm_available(p) is True, f"{p} 词表含诗体 token，应可执行"
    print("  旧档 tag 臂不可执行、新档可执行 ✓")


def test_tag_arm_kinds_splits_by_vocab():
    """待跑组按可执行性二分：旧档入 skipped，其余入 runnable；各自保持入参顺序。"""
    runnable, skipped = prosody_eval.tag_arm_kinds(["old", "m1", "m2", "m3", "m3_cons"])
    assert runnable == ["m1", "m2", "m3", "m3_cons"], f"可执行组异常：{runnable}"
    assert skipped == ["old"], f"应仅旧档不可执行：{skipped}"
    print("  分组过滤：old → skipped，其余 → runnable ✓")


def test_run_group_records_tag_prefix():
    """run_group 透传 tag_prefix 并在结果字典内如实记录（供第九节取用）；两臂均可跑通。"""
    pool = prosody_eval.read_pool_file(prosody_eval.POOL_PATH)[:2]
    bare = prosody_eval.run_group("old", pool, new_tokens=4)
    assert bare["tag_prefix"] is False and bare["n"] == 2, \
        f"裸臂结果异常：tag_prefix={bare.get('tag_prefix')} n={bare['n']}"
    tag = prosody_eval.run_group("m1", pool, new_tokens=4, tag_prefix=True)
    assert tag["tag_prefix"] is True and tag["n"] == 2, \
        f"tag 臂结果异常：tag_prefix={tag.get('tag_prefix')} n={tag['n']}"
    print("  run_group 记录 tag_prefix：裸臂 False / tag 臂 True ✓")


def test_tag_arm_section_totals_and_diff_recomputable():
    """第九节两臂总表：差值 = tag 臂 − 裸臂（百分点），可由行内两臂值复算（误差 ≤ 0.01pp）。"""
    bare = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.52, 0.0674, 0.9152)}
    tag = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.96, 0.068, 0.93)}
    text = "\n".join(prosody_eval.tag_arm_section(bare, tag))
    for want in ("+44.00", "+0.06", "+1.48"):
        assert want in text, f"两臂差值应含 {want}：{text}"
    assert "裸臂 52.00%" in text and "tag 臂 96.00%" in text, f"两臂绝对值缺失：{text}"
    print("  第九节两臂总表差值可复算 ✓")


def test_tag_arm_section_per_tag_threshold_note():
    """逐诗体明细：n < 30 者标注「不作判据」，n ≥ 30 者不标。"""
    bare = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.52, 0.07, 0.91, n=80)}
    tag = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.96, 0.07, 0.93, n=80)}
    text = "\n".join(prosody_eval.tag_arm_section(bare, tag))
    assert "n=20" in text and "（n < 30，不作判据）" in text, f"小样本须标注：{text}"
    bare2 = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.52, 0.07, 0.91)}
    tag2 = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.96, 0.07, 0.93)}
    text2 = "\n".join(prosody_eval.tag_arm_section(bare2, tag2))
    assert "n=40" in text2 and "不作判据" not in text2, f"大样本不应标注：{text2}"
    print("  逐诗体明细小样本标注正确 ✓")


def test_tag_arm_section_skipped_is_annotated():
    """tag 臂不可执行的组如实留空并注明，且不得输出任何对照数字（防伪对照）。"""
    bare = {"old": _fake_group("old", "旧模型 model.npz（约束关）", 0.70, 0.55, 0.50)}
    text = "\n".join(prosody_eval.tag_arm_section(bare, {}, skipped=("old",)))
    assert "不可执行" in text and "留空" in text, f"不可执行组须如实注明：{text}"
    assert "70.00%" not in text, f"不可执行的组不得输出对照数字（防伪对照）：{text}"
    print("  不可执行组如实留空、无伪对照 ✓")


def test_tag_arm_section_answers_two_questions():
    """直答两问取值出自实算：①句长是否 ≥ 95%；②押韵是否仍落在 6–8% 区间。"""
    bare = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.5209, 0.0674, 0.9152)}
    low = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.9000, 0.0680, 0.9300)}
    high = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.9700, 0.1500, 0.9300)}
    t1 = "\n".join(prosody_eval.tag_arm_section(bare, low))
    assert "是否 ≥ 95%：否" in t1 and "是否仍落在 6%–8% 区间（即无实质提升）：是" in t1, t1
    t2 = "\n".join(prosody_eval.tag_arm_section(bare, high))
    assert "是否 ≥ 95%：是" in t2 and "是否仍落在 6%–8% 区间（即无实质提升）：否" in t2, t2
    print("  直答两问取值出自实算 ✓")


def test_tag_arm_section_declares_redline():
    """第九节须含红线声明：不替换/不改闸门口径/不重判 M1 + §十四 实测缺口引用。"""
    text = "\n".join(prosody_eval.tag_arm_section({}, {}))
    for key in ("绝不替换", "绝不修改闸门口径", "绝不用以重判 M1", "6.93%", "65.78%", "58.85"):
        assert key in text, f"第九节缺少红线声明要素：{key}"
    print("  第九节红线声明齐备 ✓")


def test_write_report_tag_arm_appends_only():
    """接入第九节后，报告第一至八节与仅裸臂时逐字节一致（零漂移）；第九节为纯追加。"""
    d = _mkprobe()
    meta = {"pool_path": "data/eval-prompts.txt", "val_path": "data/val.txt", "seed": 0,
            "new_tokens": 96, "temperature": 1.0, "top_k": 20, "n_per_tag": 40,
            "per_tag_n": {tag: 40 for tag in POOL_TAGS}, "device": "numpy"}
    bare = {"old": _fake_group("old", "旧模型 model.npz（约束关）", 0.484, 0.0779, 0.8635),
            "m1": _fake_group("m1", "M1 模型（约束关）", 0.5209, 0.0674, 0.9152)}
    tag = {"m1": _fake_group("m1", "M1 模型（约束关）", 0.9100, 0.0690, 0.9300)}
    p1 = os.path.join(d, "ab-bare-only.txt")
    p2 = os.path.join(d, "ab-both-arms.txt")
    prosody_eval.write_report(bare, p1, meta)
    prosody_eval.write_report(bare, p2, meta, tag_results=tag, tag_skipped=("old",))
    with open(p1, encoding="utf-8") as f:
        a = f.read()
    with open(p2, encoding="utf-8") as f:
        b = f.read()
    assert "九、" not in a, "仅裸臂时不得出现第九节"
    assert b.startswith(a), "第九节必须为纯追加，第一至八节不得有任何改动（零漂移）"
    assert "九、诗体前缀口径对照" in b[len(a):], "新增内容应为第九节"
    print("  第九节纯追加、前八节零漂移 ✓")


def test_parse_arms():
    """--arms 解析：裸臂必备、去重保序、非法值响亮失败。"""
    assert prosody_eval.parse_arms("bare,tag") == ("bare", "tag")
    assert prosody_eval.parse_arms(" tag , bare , tag ") == ("tag", "bare")
    for bad in ("", "   ", "tag", "bare,tag,xxx"):
        try:
            prosody_eval.parse_arms(bad)
        except AssertionError:
            continue
        raise AssertionError(f"--arms={bad!r} 应报错（非法或省略裸臂）")
    print("  --arms 解析与非法值防呆 ✓")


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
    test_set_seed_reproducible,
    test_build_eval_pool_from_val_only,
    test_build_eval_pool_undersupply_takes_all,
    test_ci_halfwidth,
    test_score_group_matches_prosody,
    test_first_rhymed_reported_separately,
    test_judge_gates,
    test_assert_same_device,
    test_write_report_structure,
    test_score_group_duizhang_matches_prosody,
    test_duizhang_merge_by_couplet_not_sample_mean,
    test_judge_duizhang_discrimination,
    test_write_report_duizhang_and_selfcheck,
    test_val_corpus_duplicate_stats,
    test_write_report_annotates_val_bias,
    test_build_eval_pool_reuse_reports_full_stats,
    test_write_report_reuse_pool_records_exclusions,
    test_build_eval_pool_stale_file_fails_loudly,
    test_half_parity_stats_matches_independent,
    test_old_model_vocab_and_prompt_id_parity,
    test_new_model_head_config,
    test_generate_path_smoke,
    test_real_poem_ceiling_stats_synthetic,
    test_real_poem_ceiling_stats_read_only_and_deterministic,
    test_split_rhyme_by_parity_synthetic,
    test_write_report_parity_disclosure,
    test_judge_parity_attribution_not_significant,
    test_judge_parity_attribution_significant_aligned_higher,
    test_encode_tag_prefix_is_single_token,
    test_generate_texts_tag_prefix_smoke,
    test_tag_arm_available_by_vocab,
    test_tag_arm_kinds_splits_by_vocab,
    test_run_group_records_tag_prefix,
    test_tag_arm_section_totals_and_diff_recomputable,
    test_tag_arm_section_per_tag_threshold_note,
    test_tag_arm_section_skipped_is_annotated,
    test_tag_arm_section_answers_two_questions,
    test_tag_arm_section_declares_redline,
    test_write_report_tag_arm_appends_only,
    test_parse_arms,
)


def main():
    try:
        passed = sum(_run(t) for t in TESTS)
        total = len(TESTS)
    finally:
        shutil.rmtree(PROBE, ignore_errors=True)     # 环境洁癖：临时产物不残留
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部评估与 A/B 工具（T11）单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
