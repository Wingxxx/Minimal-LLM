# -*- coding: utf-8 -*-
"""T10 自蒸馏（tools/selftrain.py）单测：prompt 池隔离、候选筛选、混合配比与端到端小样本。

覆盖对象（对齐 T10 死规则「自蒸馏」与「T10 实施方案」）：
  - build_prompt_pool：分层抽样只取 data/corpus.txt 的首半句（纯汉字、5/7 字），按诗体各抽
    n_per_tag 条；并对 data/val.txt 的半句全集做只读排除，使池中任一条 prompt 均不出现在
    val.txt（评估集隔离死规则），且排除动作真实生效（val 与 corpus 含逐字重复诗）。
  - select_and_format：先按诗体行数把候选续写规范化成「语料格式正文」（截取前 n_lines×2 个
    半句，不足即弃），复判诗体一致后按 prosody.score_candidate 降序取 top-1，句长/平仄/押韵
    三项门槛同时满足才入选，否则返回 None。
  - build_mix：R = max(1, round(ρ/(1-ρ)·L_corpus/L_self))、四数组 numpy.tile 拼接、
    长度一致断言、词表漂移断言。
  - 端到端小样本（真跑 model-gelv-m2.npz）：prompt 取 16 条中前 12 条、K=4 / new_tokens=96，
    收集全部产出（允许为 0 的 prompt 跳过；诗体复判不一致率较高属硬规则，非门槛可放松），
    校验 selftrain.txt 与主语料格式一致（诗体 token 起首、行末换行、行数符合诗体、encode 无
    OOV）、mix-meta.npz 四数组长度与 mix token 流一致、词表仍为 8201 项。

纯函数层不加载模型、不写数据/ 目录，直接命中实现；端到端层为真跑，产物落 _probe/selftrain_smoke/
临时目录并在测试内清理（不污染 data/）。全部用例基于合成小样本或既有语料，不触发全量生成。

运行 `python test/test_selftrain.py`（端到端项需加载模型并生成，GPU 下约数十秒）；无第三方测试
框架依赖，逐项打印 ✓/✗ 与中文汇总，任一失败时进程以非零码退出。
"""
import inspect
import os
import re
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))

import numpy as onp

import build_corpus
import prosody
import train
import selftrain

SMOKE_DIR = os.path.join(BASE, "_probe", "selftrain_smoke")
POOL_TAGS = ("<五绝>", "<七绝>", "<五律>", "<七律>")
META_KEYS = ("weights", "tone_labels", "rhyme_labels", "zone_tags")

# ── 打分/门槛用合成诗（字音口径经 prosody 实测，参见 test/test_critic.py）──
# 四半句各命中 A/B/C/D，句长/押韵/平仄均 1.0
GOOD = "白日依山尽，天高白玉霜。春江明月雪，夜雪白霜光。"
# 首半句 4 字：句长合规率 3/4=0.75，押韵与平仄保持 1.0
BAD_LEN = "白日依山，天高白玉霜。春江明月雪，夜雪白霜光。"
# 末半句末字改韵：押韵主指标 0.0，句长与平仄保持 1.0
BAD_RHYME = "白日依山尽，天高白玉霜。春江明月雪，夜雪白霜天。"
# 首半句必论位改反：平仄合规率 3/4=0.75，句长与押韵保持 1.0
BAD_TONE = "白春依山尽，天高白玉霜。春江明月雪，夜雪白霜光。"
# 七言诗：用于「诗体不一致」用例（按五绝 pattern_len=5 校验时句长合规率为 0）
QIJUE = "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。"

# build_mix 合成样本（每行 = 一联 = 两半句；此处只作数据载体，不含字音断言）
WUJUE = ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡。"]
QIJUE_POEM = ["两个黄鹂鸣翠柳，一行白鹭上青天。", "窗含西岭千秋雪，门泊东吴万里船。"]
WUJUE2 = ["春眠不觉晓，处处闻啼鸟。", "夜来风雨声，花落知多少。"]
WULV = ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡。",
        "白日依山尽，黄河入海流。", "欲穷千里目，更上一层楼。"]
QILV = ["岧嶤太华俯咸京，天外三峰削不成。", "武帝祠前云欲散，仙人掌上雨初晴。",
        "河山北枕秦关险，驿树西连汉畤平。", "借问路傍名利客，无如此处学长生。"]


def _cand(text, logprob=None):
    """构造候选字典（与 generate_candidates 的产出一致：text 为完整生成文本）。"""
    return {"text": text, "logprob": logprob}


def _save_meta(path, meta_tuple):
    """按 data/meta.npz 的 dtype（float32/int8/int16/int8）落盘四数组，供 build_mix 消费。

    meta_tuple 取 build_corpus.build_meta 的返回值 (text, weights, tone_labels,
    rhyme_labels, zone_tags)；首项语料文本不落盘，仅取后四项数组。
    """
    _, w, tone, rhyme, zone = meta_tuple
    onp.savez(path,
              weights=onp.asarray(w, dtype=onp.float32),
              tone_labels=onp.asarray(tone, dtype=onp.int8),
              rhyme_labels=onp.asarray(rhyme, dtype=onp.int16),
              zone_tags=onp.asarray(zone, dtype=onp.int8))


def _corpus_firsts(corpus_path):
    """独立解析语料，返回全部 (诗体, 首半句) 集合。

    与实现侧刻意采用不同切分手法（正则 split），避免用例与实现同源而失去校验力。
    """
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    firsts = set()
    for line in text.split("\n"):
        for tag in build_corpus.POEM_TAGS:
            if line.startswith(tag):
                body = line[len(tag):]
                seg = re.split(r"[，。！？；：、\s]", body)[0]
                firsts.add((tag, seg))
                break
    return firsts


def _val_halves(val_path):
    """独立解析 data/val.txt，返回其全部半句集合。

    按断句标点（prosody.DEFAULT_PUNCT）与空白正则切分（与实现侧字符级切分手法不同），
    丢弃空段；用于独立核算「corpus 首半句与 val 半句」的碰撞数。
    """
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    return {seg for seg in re.split(r"[，。！？；：、\s]+", text) if seg}


# ─────────────────────────── 纯函数层（无需模型）───────────────────────────

def test_prompt_pool_stratified_and_isolated():
    """prompt 池：分层各 125 条、均命中 corpus 首半句集合、任一条不出现在 val.txt。"""
    pool = selftrain.build_prompt_pool(corpus_path=selftrain.CORPUS_PATH,
                                       n_per_tag=125, seed=selftrain.T10_SEED,
                                       pool_path=None)
    assert len(pool) == 500, f"应共 500 条，实为 {len(pool)}"
    for tag in POOL_TAGS:
        n = sum(1 for t, _ in pool if t == tag)
        assert n == 125, f"{tag} 应 125 条，实为 {n}"
    assert all(tag in POOL_TAGS for tag, _ in pool), "prompt 池不得含 <杂言> 等非抽样诗体"

    firsts = _corpus_firsts(selftrain.CORPUS_PATH)
    miss = [(t, p) for t, p in pool if (t, p) not in firsts]
    assert not miss, f"以下 prompt 不是 corpus 首半句（越界取样）：{miss[:5]}"

    with open(selftrain.VAL_PATH, encoding="utf-8") as f:
        val_text = f.read()
    leak = [p for _, p in pool if p in val_text]
    assert not leak, f"prompt 池泄漏评估集：{leak}"

    # 证据断言：val 与 corpus 确有逐字重复句，证明排除动作真实生效（而非恰好未抽中）
    val_halves = _val_halves(selftrain.VAL_PATH)
    collision = [(t, p) for t, p in firsts if any(p in hl for hl in val_halves)]
    assert len(collision) >= 1, "前提：corpus 首半句应与 val 半句存在碰撞，否则排除断言无证据力"
    print(f"  池 {len(pool)} 条（每体 125），全部为 corpus 首半句，val 无泄漏 ✓；"
          f"val 排除实测剔除候选 {len(collision)} 条（corpus×val 重复句）")


def test_select_and_format_top1_and_thresholds():
    """select_and_format：规范化成体 → 诗体复判 → 降序取 top-1 → 三项门槛；含负数与行数不足负例。"""
    tag = "<五绝>"
    plen = 5

    # ① 降序取 top-1：同为合法五绝者，低分在前、高分在后，结果须为高分者（GOOD > BAD_TONE）
    s_hi = prosody.score_candidate(GOOD, pattern_len=plen)
    s_lo = prosody.score_candidate(BAD_TONE, pattern_len=plen)
    assert s_hi > s_lo, f"前提：GOOD 应严格高于 BAD_TONE，实为 {s_hi} vs {s_lo}"
    out = selftrain.select_and_format([_cand(BAD_TONE), _cand(GOOD)], tag, plen)
    canon_good = "白日依山尽，天高白玉霜。\n春江明月雪，夜雪白霜光。"
    assert out == tag + canon_good + "\n", \
        f"应选取得分最高者 GOOD 的规范化成体，实为 {out!r}"

    # ② 句长门槛边界：规范化成体后句长恒合规（=1.0），故边界落在 1.0
    ll = prosody.check_line_len(GOOD, plen)
    assert abs(ll - 1.0) < 1e-9, f"前提：GOOD 句长应 1.0，实为 {ll}"
    assert selftrain.select_and_format([_cand(GOOD)], tag, plen, min_line_len=1.0,
                                       min_tone=0.0, min_rhyme=0.0) is not None, \
        "句长合规率等于门槛应入选"
    assert selftrain.select_and_format([_cand(GOOD)], tag, plen, min_line_len=1.01,
                                       min_tone=0.0, min_rhyme=0.0) is None, \
        "句长合规率低于门槛应弃"
    # 含错长半句者规范化后不构成目标诗体（复判为 <杂言>），在门槛前即弃
    assert selftrain.select_and_format([_cand(BAD_LEN)], tag, plen) is None, \
        "含 4 字半句者规范化后非目标诗体，应弃"

    # ③ 平仄门槛边界：BAD_TONE 平仄合规率 0.75
    tn = prosody.check_tone(BAD_TONE, plen).main
    assert abs(tn - 0.75) < 1e-9, f"前提：BAD_TONE 平仄应 0.75，实为 {tn}"
    assert selftrain.select_and_format([_cand(BAD_TONE)], tag, plen, min_line_len=0.0,
                                       min_tone=0.75, min_rhyme=0.0) is not None, \
        "平仄合规率等于门槛应入选"
    assert selftrain.select_and_format([_cand(BAD_TONE)], tag, plen, min_line_len=0.0,
                                       min_tone=0.76, min_rhyme=0.0) is None, \
        "平仄合规率低于门槛应弃"

    # ④ 押韵门槛边界：BAD_RHYME 押韵主指标 0.0
    rh = prosody.check_rhyme(BAD_RHYME).main
    assert abs(rh - 0.0) < 1e-9, f"前提：BAD_RHYME 押韵应 0.0，实为 {rh}"
    assert selftrain.select_and_format([_cand(BAD_RHYME)], tag, plen, min_line_len=0.0,
                                       min_tone=0.0, min_rhyme=0.0) is not None, \
        "押韵主指标等于门槛应入选"
    assert selftrain.select_and_format([_cand(BAD_RHYME)], tag, plen, min_line_len=0.0,
                                       min_tone=0.0, min_rhyme=0.01) is None, \
        "押韵主指标低于门槛应弃"

    # ⑤ 诗体复判不一致弃：七言规范化后复判为 <七绝>，与目标 <五绝> 不符而弃
    assert selftrain.select_and_format([_cand(QIJUE)], tag, plen) is None, \
        "规范化后诗体复判与目标不符者应弃"

    # ⑥ OOV 弃：候选整体含词表外字符时（传入 stoi）剔除——须在规范化前于完整正文上检出，
    # 否则位于句尾、将被「截取前 n_lines×2 个半句」截掉的 OOV 会漏检
    stoi = train.load_corpus()[2]
    oov = next(c for c in "龘䶮㐀鿿" if c not in stoi)
    assert selftrain.select_and_format([_cand(GOOD + oov)], tag, plen,
                                       min_line_len=0.0, min_tone=0.0, min_rhyme=0.0,
                                       stoi=stoi) is None, "含词表外字符的候选应弃"
    assert selftrain.select_and_format([_cand(GOOD)], tag, plen,
                                       min_line_len=0.0, min_tone=0.0, min_rhyme=0.0,
                                       stoi=stoi) is not None, "词表覆盖的候选应入选"

    # ⑦ 半句数不足 n_lines×2 即弃：3 个半句传 <五绝>（需 4 个）
    three_halves = "白日依山尽，天高白玉霜，春江明月雪。"
    assert selftrain.select_and_format([_cand(three_halves)], tag, plen) is None, \
        "半句数不足 n_lines×2 应弃"
    # ⑧ 4 个半句的五言文本传 <五律>（需 8 个）→ 弃
    assert selftrain.select_and_format([_cand(GOOD)], "<五律>", 5) is None, \
        "半句数不足 <五律> 所需的 8 个应弃"

    # ⑨ 正例：合法五律传 <五律>，门槛放松后返回以 <五律> 起首、4 行×2 半句、行末换行的串
    out_lv = selftrain.select_and_format([_cand("".join(WULV))], "<五律>", 5,
                                         min_line_len=0.0, min_tone=0.0, min_rhyme=0.0)
    assert out_lv is not None and out_lv.startswith("<五律>") and out_lv.endswith("\n"), \
        f"合法五律应入选为语料格式串，实为 {out_lv!r}"
    lv_lines = out_lv[len("<五律>"):-1].split("\n")
    assert len(lv_lines) == 4 and all(prosody.line_len_of(ln, prosody.DEFAULT_PUNCT) == [5, 5]
                                      for ln in lv_lines), \
        f"五律须为 4 行、每行 2 个五言半句，实为 {lv_lines}"
    print("  规范化成体、降序 top-1、诗体复判、三项门槛边界、行数不足弃、OOV 弃 均 ✓")


def test_build_mix_synthetic():
    """build_mix：R/share 计算、np.tile 拼接、长度一致断言、词表漂移断言。"""
    d = os.path.join(SMOKE_DIR, "mix_synth")
    os.makedirs(d, exist_ok=True)
    try:
        corpus_poems = [WUJUE, QIJUE_POEM, WULV, QILV, WUJUE2, WULV, QIJUE_POEM, WUJUE]
        self_poems = [WUJUE]
        corpus_text = "".join(build_corpus.tag_poem(p) for p in corpus_poems)
        self_text = "".join(build_corpus.tag_poem(p) for p in self_poems)
        cp, sp = os.path.join(d, "corpus.txt"), os.path.join(d, "self.txt")
        with open(cp, "w", encoding="utf-8") as f:
            f.write(corpus_text)
        with open(sp, "w", encoding="utf-8") as f:
            f.write(self_text)
        stoi = train.load_corpus(cp)[2]
        cm, sm = os.path.join(d, "meta.npz"), os.path.join(d, "self-meta.npz")
        _save_meta(cm, build_corpus.build_meta(corpus_poems))
        _save_meta(sm, build_corpus.build_meta(self_poems))

        mix_txt, mix_meta = os.path.join(d, "mix.txt"), os.path.join(d, "mix-meta.npz")
        res = selftrain.build_mix(cp, sp, cm, sm, mix_txt, mix_meta, stoi)

        L_corpus = int(len(train.encode(corpus_text, stoi)))
        L_self = int(len(train.encode(self_text, stoi)))
        R = max(1, round(selftrain.RHO / (1 - selftrain.RHO) * L_corpus / L_self))
        assert res["L_corpus"] == L_corpus and res["L_self"] == L_self, \
            f"L 值应为 {L_corpus}/{L_self}，实为 {res['L_corpus']}/{res['L_self']}"
        assert res["R"] == R, f"R 应为 {R}，实为 {res['R']}"
        assert abs(res["share"] - L_self * R / (L_corpus + L_self * R)) < 1e-12, \
            f"share 计算不符，实为 {res['share']}"
        assert R >= 2, f"前提：合成语料须足够大以令 R>1（锁定 np.tile 重复次数），实为 {R}"

        with open(mix_txt, encoding="utf-8") as f:
            assert f.read() == corpus_text + self_text * R, "mix.txt 应为 corpus 原文 + 自产原文×R"

        cmeta, smeta, mmeta = train.load_meta(cm), train.load_meta(sm), train.load_meta(mix_meta)
        mix_data = train.load_corpus(mix_txt)[4]
        for k in META_KEYS:
            exp = onp.concatenate([cmeta[k], onp.tile(smeta[k], R)])
            assert mmeta[k].dtype == cmeta[k].dtype, f"{k} dtype 应与既有 meta 对齐"
            assert onp.array_equal(mmeta[k], exp), f"{k} 应为 concat(corpus, tile(self, R))"
            assert len(mix_data) == len(mmeta[k]), \
                f"{k} 长度 {len(mmeta[k])} 与 mix token 流 {len(mix_data)} 不一致"

        # 长度一致断言（响亮失败）：自产 meta 末位截短后须抛 AssertionError
        bad = os.path.join(d, "self-meta-bad.npz")
        bad_meta = build_corpus.build_meta(self_poems)
        _save_meta(bad, (bad_meta[0],) + tuple(a[:-1] for a in bad_meta[1:]))
        try:
            selftrain.build_mix(cp, sp, cm, bad, os.path.join(d, "mix-bad.txt"),
                                os.path.join(d, "mix-bad.npz"), stoi)
            raise AssertionError("自产 meta 长度不符时应抛 AssertionError，实为静默通过")
        except AssertionError as e:
            assert "长度" in str(e), f"应为长度断言失败，实为：{e}"

        # 词表漂移断言（响亮失败）：自产集引入 corpus 词表外字符时须抛 AssertionError
        oov = next(c for c in "龘䶮㐀鿿" if c not in stoi)
        drift_poem = [oov * 5 + "，" + oov * 5 + "。", oov * 5 + "，" + oov * 5 + "。"]
        drift_text = "".join(build_corpus.tag_poem(p) for p in [WUJUE, drift_poem])
        dp, dm = os.path.join(d, "drift.txt"), os.path.join(d, "drift-meta.npz")
        with open(dp, "w", encoding="utf-8") as f:
            f.write(drift_text)
        _save_meta(dm, build_corpus.build_meta([WUJUE, drift_poem]))
        try:
            selftrain.build_mix(cp, dp, cm, dm, os.path.join(d, "mix-drift.txt"),
                                os.path.join(d, "mix-drift.npz"), stoi)
            raise AssertionError("混合后词表漂移时应抛 AssertionError，实为静默通过")
        except AssertionError as e:
            assert "词表" in str(e), f"应为词表漂移断言失败，实为：{e}"
        print(f"  R={res['R']} L_corpus={res['L_corpus']} L_self={res['L_self']} "
              f"share={res['share']:.4f}；tile/长度/漂移断言均 ✓")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ─────────────────────────── 端到端小样本（真跑）───────────────────────────

def test_end_to_end_small_sample():
    """真跑：prompt 16 条取前 12、K=4/new_tokens=96；收集全部产出并校验格式、OOV、mix 长度与词表。"""
    model_path = os.path.join(BASE, selftrain.DEFAULT_MODEL)
    assert os.path.exists(model_path), f"缺少基线模型 {model_path}"
    stoi, itos = train.load_corpus()[2], train.load_corpus()[3]
    model = selftrain.build_model(model_path)

    pool = selftrain.build_prompt_pool(n_per_tag=4, pool_path=None)[:12]
    assert len(pool) == 12, f"小样本应取 12 条 prompt，实为 {len(pool)}"

    d = os.path.join(SMOKE_DIR, "e2e")
    os.makedirs(d, exist_ok=True)
    try:
        formatted = []
        for tag, prompt in pool:
            cands = selftrain.generate_candidates(model, stoi, itos, tag, prompt,
                                                  k=4, new_tokens=96)
            assert len(cands) == 4, "每条 prompt 应产出 K=4 个候选"
            s = selftrain.select_and_format(cands, tag, selftrain.TAG_LEN[tag],
                                            min_line_len=0.0, min_tone=0.0, min_rhyme=0.0,
                                            stoi=stoi)
            if s is not None:                 # 诗体复判不一致者弃（硬规则，非门槛可放松）
                formatted.append((tag, s))
        assert len(formatted) >= 1, "12 条 prompt × K=4 应至少产出 1 首自产诗"
        print(f"  12 条 prompt × K=4 实际产出 {len(formatted)} 首（诗体复判不一致者已弃）")

        self_text = "".join(s for _, s in formatted)
        for tag, s in formatted:
            assert s.startswith(tag) and tag in build_corpus.POEM_TAGS, "每首须以诗体 token 起首"
            assert s.endswith("\n"), "每首须行末换行"
            n_lines = 2 if tag in ("<五绝>", "<七绝>") else 4
            assert len(selftrain.poem_lines(s)[1]) == n_lines, f"{tag} 应为 {n_lines} 行"

        # encode 无 OOV：除 4 字符诗体 token 折 1 个 id 外，逐字符 1:1，无跳过
        specials = sum(self_text.count(t) for t in train.SPECIAL_TOKENS)
        ids = train.encode(self_text, stoi)
        assert len(ids) == len(self_text) - 3 * specials, \
            "selftrain.txt 存在词表外字符（encode 跳字）"

        sp = os.path.join(d, "selftrain.txt")
        with open(sp, "w", encoding="utf-8") as f:
            f.write(self_text)
        sm = os.path.join(d, "selftrain-meta.npz")
        _save_meta(sm, build_corpus.build_meta([selftrain.poem_lines(s)[1] for _, s in formatted]))

        mix_txt, mix_meta = os.path.join(d, "mix.txt"), os.path.join(d, "mix-meta.npz")
        res = selftrain.build_mix(selftrain.CORPUS_PATH, sp, selftrain.META_PATH, sm,
                                  mix_txt, mix_meta, stoi)
        mmeta = train.load_meta(mix_meta)
        mix_data = train.load_corpus(mix_txt)[4]
        for k in META_KEYS:
            assert len(mix_data) == len(mmeta[k]), f"{k} 与 mix token 流长度不一致"
        assert len(train.load_corpus(mix_txt)[2]) == 8201, "混合后词表须仍为 8201 项"
        print(f"  {len(formatted)} 首自产诗（K=4/96 tok）→ selftrain.txt 格式/OOV/mix 长度/词表 均 ✓"
              f"（R={res['R']}, share={res['share']:.4f}）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ─────────────────────── train_m3 定档与骨干接线（猴子补丁）───────────────────────

def test_train_m3_backbone_matches_m2():
    """train_m3 必须显式传与 M2 档一致的骨干（骨干口径：d256/h8/L6），否则按名载参形状断言失败。

    事实来源为 model-gelv-m2.npz 本体，不得只硬编码常量：d_model 取档内 tok_emb.shape[1]，
    n_layer 取档内 blocks.{i}.* 键最大下标 +1；n_head 无档内证据，按项目骨干口径断言为 8。
    以猴子补丁把 train.train 临时替换为「只记录 kwargs 并立即返回」的桩函数（try/finally 还原），
    取回 train_m3 转发的入参；再用该入参（仅骨干与辅助头/词表相关项）**真正构建** GPT 并按名
    载入 M2 档，断言缺失参数为 0 且不抛形状断言——修复前 d_model/n_head/n_layer 未传入而退回
    train.train 默认 d64/h4/L2，此处必因 tok_emb 形状 (8201, 256) vs (8201, 64) 抛 AssertionError。
    """
    m2 = os.path.join(BASE, selftrain.DEFAULT_MODEL)
    assert os.path.exists(m2), f"缺少基线模型档 {m2}"
    arch = dict(onp.load(m2))
    d_model = int(arch["tok_emb"].shape[1])              # 档内证据：骨干宽度
    blk_ids = []
    for k in arch:                                       # 档内证据：blocks.{i}.* 的最大层下标
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            blk_ids.append(int(m.group(1)))
    assert blk_ids, "M2 档内应含 blocks.{i}.* 权重键，否则无法推出层数"
    n_layer = max(blk_ids) + 1

    captured = {}                                        # 捕获 train_m3 转发给 train.train 的入参
    orig_train = selftrain.train.train

    def _stub_train(**kwargs):
        captured.update(kwargs)
        return "STUB"                                    # 仅记录入参，不触发真实训练、不落任何档

    selftrain.train.train = _stub_train
    try:
        assert selftrain.train_m3() == "STUB", "train_m3 应把入参原样转发给 train.train 并返回其结果"
    finally:
        selftrain.train.train = orig_train               # 无论成败都还原生产函数
    kwargs = captured
    assert kwargs, "train_m3 未向 train.train 传入任何参数"

    # 真实复现（红的核心）：以捕获入参构建 GPT 并按名载入 M2 档。未捕获者按 train.train 的
    # 真实签名默认值补全（= train() 建模型时的实际取值），修复前即退回默认 d64/h4/L2 而形状断言失败
    stoi = train.load_corpus()[2]
    keys = ("d_model", "n_head", "n_layer", "ctx_len", "n_rhyme", "use_tone")
    defaults = inspect.signature(orig_train).parameters
    model_cfg = {k: kwargs.get(k, defaults[k].default) for k in keys}
    model = selftrain.GPT(vocab_size=len(stoi), **model_cfg)
    missing = train.load_params_partial(m2, model)
    assert missing == 0, f"M2 档应提供全部模型参数，实缺 {missing} 个"

    # 事实来源断言：骨干须与 M2 档一致
    assert kwargs.get("d_model") == d_model, (
        f"train_m3 未传或传错 d_model：应 {d_model}（M2 档 tok_emb 宽度），实为 {kwargs.get('d_model')}")
    assert kwargs.get("n_layer") == n_layer, (
        f"train_m3 未传或传错 n_layer：应 {n_layer}（M2 档 blocks 最大下标 +1），实为 {kwargs.get('n_layer')}")
    assert kwargs.get("n_head") == 8, (
        f"train_m3 未传或传错 n_head：骨干口径 d256/h8/L6，应 8，实为 {kwargs.get('n_head')}")

    # 既有定档不得被顺手改坏
    assert kwargs.get("max_steps") == 1000, f"max_steps 应为 1000，实为 {kwargs.get('max_steps')}"
    assert kwargs.get("anneal_total") == 1000, \
        f"anneal_total 应为 1000，实为 {kwargs.get('anneal_total')}"
    assert kwargs.get("ckpt_path") == "model-gelv-m3.npz", \
        f"ckpt_path 应为 model-gelv-m3.npz，实为 {kwargs.get('ckpt_path')}"
    assert kwargs.get("duizhang_weight") == 1.5, \
        f"duizhang_weight 应为 1.5，实为 {kwargs.get('duizhang_weight')}"
    assert kwargs.get("n_rhyme") == 107, f"n_rhyme 应为 107，实为 {kwargs.get('n_rhyme')}"
    assert kwargs.get("use_tone") is True, f"use_tone 应为 True，实为 {kwargs.get('use_tone')}"
    assert kwargs.get("save_every") == kwargs.get("print_every"), (   # 既有死规则：存档须对齐监控节拍
        f"save_every 须等于 print_every，实为 {kwargs.get('save_every')} vs {kwargs.get('print_every')}")
    print(f"  train_m3 骨干 d{kwargs['d_model']}/h{kwargs['n_head']}/L{kwargs['n_layer']} "
          f"与 M2 档一致（按名载参缺 {missing} 项）；既有定档 max_steps/anneal_total/ckpt_path/"
          f"duizhang_weight/n_rhyme/use_tone/save_every 均 ✓")


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
    test_prompt_pool_stratified_and_isolated,
    test_select_and_format_top1_and_thresholds,
    test_build_mix_synthetic,
    test_end_to_end_small_sample,
    test_train_m3_backbone_matches_m2,
)


def main():
    passed = sum(_run(t) for t in TESTS)
    total = len(TESTS)
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)   # 环境洁癖：临时产物目录不残留
    print(f"\n通过 {passed}/{total} 项")
    if passed == total:
        print("全部自蒸馏（T10）单测通过")
        return 0
    print("存在未通过项")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
