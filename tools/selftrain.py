# -*- coding: utf-8 -*-
"""T10 BoN 自蒸馏：prompt 池 → K 候选生成 → 格律打分筛选 → 混合配比 → 混合再训。

职责（对齐「自蒸馏」与「T10 实施方案」死规则）：
  1) prompt 池：只从 data/corpus.txt 取首半句（纯汉字、5/7 字），按诗体分层各抽 N//4 条，
     固定种子抽取并落盘复用。data/val.txt 仅以只读方式用于「排除」——把 val 的半句全集作
     排除集，剔除与之碰撞的 corpus 候选（源数据含逐字重复诗，只取 corpus 必然抽到与 val
     重复之句），绝不从 val 取样（prompt 隔离死规则：防评估集泄漏）。
  2) 生成：每 prompt 独立跑 K 次（temperature=1.0、top_k=20、关闭 meter 约束、new_tokens=96）；
     每完成一条 prompt 即向 data/_selftrain_raw.jsonl 追加一行（含候选列表），中断后按 idx 续跑。
  3) 打分：先按诗体行数把续写规范化成体（截取前 n_lines×2 个半句，不足即弃），复判诗体一致后
     以 prosody.score_candidate（句长 + 押韵 + 平仄 + 自模型平均 logprob）降序取 top-1；
     句长/平仄/押韵三项门槛同时满足才入选，否则弃。
  4) 混合：自产集循环重复 R 次使 token 占比 ρ，与原语料字节级拼接；meta 四数组同步 tile 拼接，
     并对长度一致与词表不漂移做响亮断言。

文件边界：仅本文件与 test/test_selftrain.py 新增；复用 train / prosody / build_corpus 既有接口，
不修改任何既有模块。跨里程碑载参一律走 train.load_params_partial，严禁 train.load_checkpoint
（后者逐名读 _opt.m.*，M1/M2 档缺辅助头键会直接 KeyError）。

用法：
    python tools/selftrain.py --stage all            # 生成 → 构建混合集 → 混合再训
    python tools/selftrain.py --stage gen --n 500    # 仅生成（约 2 小时长任务，可中断续跑）
    python tools/selftrain.py --stage build --n 500  # 仅构建自产集与混合集
    python tools/selftrain.py --stage train          # 仅混合再训
"""
import argparse
import json
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（脚本在 tools/ 下）
if BASE not in sys.path:            # 使项目根下的 train/model 可导入（直接运行本脚本时）
    sys.path.insert(0, BASE)
_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:          # 使同目录的 build_corpus 可导入
    sys.path.insert(0, _TOOLS)

try:                                # Windows 控制台常为 GBK：正文含 GBK 外字符时转义显示而非崩溃
    sys.stdout.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass

import build_corpus                # noqa: E402
import prosody                     # noqa: E402
import train                       # noqa: E402
from model.backend import np, onp, as_numpy   # noqa: E402
from model.gpt import GPT          # noqa: E402

# ─────────────────────────────── 路径与定档常量 ───────────────────────────────
DATA_DIR = os.path.join(BASE, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")             # 训练语料（prompt 唯一来源）
VAL_PATH = os.path.join(DATA_DIR, "val.txt")                   # 评估集（仅作 prompt 池排除，只读、不取样）
META_PATH = os.path.join(DATA_DIR, "meta.npz")                 # 既有语料 meta（混合时直接复用）
POOL_PATH = os.path.join(DATA_DIR, "selftrain-prompts.txt")    # prompt 池（每行 诗体\t首半句）
SELFTRAIN_PATH = os.path.join(DATA_DIR, "selftrain.txt")       # 过滤后的自产集
SELFTRAIN_META_PATH = os.path.join(DATA_DIR, "selftrain-meta.npz")
MIX_TEXT_PATH = os.path.join(DATA_DIR, "mix.txt")              # 原语料 + 自产集×R
MIX_META_PATH = os.path.join(DATA_DIR, "mix-meta.npz")
RAW_JSONL_PATH = os.path.join(DATA_DIR, "_selftrain_raw.jsonl")  # 生成阶段断点账本
LOG_PATH = os.path.join(BASE, "logs", "m3-selftrain.log")        # 配比与合规率记录（与 train 的 m3-train.log 分流）

T10_SEED = 0                       # prompt 池固定抽样种子（可复现）
N_PROMPTS = 500                    # prompt 定档总数（每诗体 125）
K = 8                              # 每 prompt 候选数
NEW_TOKENS = 96                    # 每候选续写长度
RHO = 0.2                          # 自产 token 在混合语料中的目标占比
# 诗体 → 目标句长（pattern_len）；<杂言> 句长不定，不入池
TAG_LEN = {"<五绝>": 5, "<七绝>": 7, "<五律>": 5, "<七律>": 7}
# 诗体 → 行数（绝句 2、律诗 4，每行 2 半句）；<杂言> 不定，不入此表
TAG_LINES = {"<五绝>": 2, "<七绝>": 2, "<五律>": 4, "<七律>": 4}
# 基线模型架构（与 S5/M1/M2 完全一致，否则权重无法按名继承）；n_rhyme/use_tone 挂载辅助头
MODEL_ARCH = dict(d_model=256, n_head=8, n_layer=6, ctx_len=128, n_rhyme=107, use_tone=True)
DEFAULT_MODEL = "model-gelv-m2.npz"
META_KEYS = ("weights", "tone_labels", "rhyme_labels", "zone_tags")


def _log(msg, log_path=LOG_PATH):
    """把一行记录同时打印并追加到训练日志（供 m3-selftrain.log 留痕）。"""
    print(msg)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ─────────────────────────────── 1) prompt 池 ───────────────────────────────

def _first_half_line(line):
    """取一行按断句标点/空白切分后的第 1 段（与 prosody 的半句切分口径一致）。"""
    delims = set(prosody.DEFAULT_PUNCT)
    for i, ch in enumerate(line):
        if ch in delims or ch.isspace():
            return line[:i]
    return line


def _is_pure_hanzi(s):
    """是否纯汉字且汉字数为 5 或 7（可定 pattern_len 的首半句）。"""
    return len(s) in (5, 7) and all("\u4e00" <= c <= "\u9fff" for c in s)


def _half_set(text):
    """把文本切成半句全集（半句保留其终止标点、丢弃空白与空段）。

    断句符 = prosody.DEFAULT_PUNCT ∪ 全部空白（含换行），与 prosody._half_lines 口径一致；
    用作 val 排除集。对纯汉字 5/7 字串而言「为某半句子串」等价于「为原文子串」。
    """
    delims = set(prosody.DEFAULT_PUNCT)
    halves, cur = set(), []
    for ch in text:
        if ch in delims:
            if cur:
                halves.add("".join(cur))
                cur = []
        elif ch.isspace():
            if cur:
                halves.add("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        halves.add("".join(cur))
    return halves


def _read_pool_file(path):
    """读回已落盘的 prompt 池，返回 list[(诗体, 首半句)]。"""
    pool = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                tag, _, prompt = ln.partition("\t")
                pool.append((tag, prompt))
    return pool


def build_prompt_pool(corpus_path=CORPUS_PATH, n_per_tag=125, seed=T10_SEED, pool_path=None,
                      val_path=VAL_PATH):
    """分层抽样 prompt 池，返回 list[(诗体, 首半句)]。

    只从 corpus_path 取样。逐行扫描：遇诗体 token 起首的行，取该行按断句标点切分的第 1 段，
    要求纯汉字且 5/7 字；同诗体内去重（保留首次出现序）。val_path（只读）的全部半句构成排除集，
    候选首半句为其任一子串者剔除——源数据含 corpus/val 逐字重复诗，仅取 corpus 必然抽到重复句。
    剔除在抽取之前完成（先过滤候选桶、再 rng.sample）。随后按诗体各用 random.Random(seed)
    抽取 n_per_tag 条；桶内候选不足 n_per_tag 时断言失败（响亮保护）。pool_path 已存在时直接
    读回复用，不重抽。
    """
    if pool_path is not None and os.path.exists(pool_path):
        return _read_pool_file(pool_path)
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    with open(val_path, encoding="utf-8") as f:
        val_halves = _half_set(f.read())
    buckets = {tag: [] for tag in TAG_LEN}
    seen = {tag: set() for tag in TAG_LEN}
    for line in text.split("\n"):
        for tag in build_corpus.POEM_TAGS:
            if line.startswith(tag):
                if tag in buckets:
                    prompt = _first_half_line(line[len(tag):])
                    if (_is_pure_hanzi(prompt) and prompt not in seen[tag]
                            and not any(prompt in hl for hl in val_halves)):
                        seen[tag].add(prompt)
                        buckets[tag].append(prompt)
                break
    rng = random.Random(seed)
    pool = []
    for tag in TAG_LEN:                       # 键序固定：<五绝>/<七绝>/<五律>/<七律>
        cands = buckets[tag]
        assert len(cands) >= n_per_tag, (
            f"诗体 {tag} 合格首半句仅 {len(cands)} 条，不足 n_per_tag={n_per_tag}")
        for prompt in rng.sample(cands, n_per_tag):
            pool.append((tag, prompt))
    if pool_path is not None:
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        with open(pool_path, "w", encoding="utf-8") as f:
            for tag, prompt in pool:
                f.write(f"{tag}\t{prompt}\n")
    return pool


# ──────────────────────────── 2) 候选生成与 logprob ────────────────────────────

def build_model(model_path=None):
    """按 T10 定档架构建 GPT 并按名载入基线参数，返回模型（跨里程碑走 load_params_partial）。"""
    path = model_path or DEFAULT_MODEL
    if not os.path.isabs(path):
        path = os.path.join(BASE, path)
    _, _, stoi, _, _, _ = train.load_corpus()
    model = GPT(vocab_size=len(stoi), **MODEL_ARCH)
    train.load_params_partial(path, model)
    return model


def _mean_logprob(model, ids, gen_start):
    """模型对生成段 token 的平均 log 概率（nats，≤0）。

    对 ids[:-1] 做一次仅前向（不建优化器、不反向），log-softmax 后 gather ids[1:]，
    仅对「生成段」位置求均值：目标 token 下标 j ≥ gen_start 对应前向位置 j-1，
    故起点取 max(gen_start-1, 0)（首个生成 token 由 prompt 末尾位置预测）。
    减最大值的 logsumexp 与 train 的数值稳定口径一致。
    """
    ids = onp.asarray(ids)
    assert ids.ndim == 1 and len(ids) >= 2, f"ids 须为一维且长度 ≥2，实为 {ids.shape}"
    logits = model.forward(np.asarray(ids[:-1][None, :]))          # [1, L-1, V]
    m = logits - logits.max(axis=-1, keepdims=True)
    logp = m - np.log(np.exp(m).sum(axis=-1, keepdims=True))
    tgt = np.asarray(ids[1:])
    picked = logp[0, np.arange(len(tgt)), tgt]                     # 各位置真实 token 的 log 概率
    seg = picked[max(int(gen_start) - 1, 0):]
    assert len(seg) > 0, "生成段为空，无法求平均 logprob"
    return float(as_numpy(seg).mean())


def generate_candidates(model, stoi, itos, tag, prompt, k=K, new_tokens=NEW_TOKENS, seed=None):
    """对单条 prompt 独立生成 K 个候选，返回 list[dict]（键：text、logprob）。

    temperature=1.0、top_k=20、logits_processor=None（关闭 meter 约束）。每候选的 text 为
    「prompt + 续写」完整生成文本；logprob 为模型对该完整序列生成段的平均 log 概率。
    seed 非 None 时先固定采样随机源，便于复现。
    """
    prompt_ids = train.encode(prompt, stoi)
    assert len(prompt_ids) == len(prompt), f"prompt 含词表外字符：{prompt!r}"
    if seed is not None:
        np.random.seed(seed)
    cands = []
    for _ in range(k):
        idx = np.asarray(prompt_ids[None, :])
        out = model.generate(idx, new_tokens, temperature=1.0, top_k=20,
                             logits_processor=None)
        gen_ids = as_numpy(out[0])[len(prompt_ids):]
        text = prompt + "".join(itos[int(i)] for i in gen_ids)
        full = onp.concatenate([onp.asarray(prompt_ids), gen_ids])
        cands.append({"text": text, "logprob": _mean_logprob(model, full, len(prompt_ids))})
    return cands


# ──────────────────────────── 3) 打分与筛选 ────────────────────────────

def _strip_special(text):
    """剥去文本中的诗体特殊 token。

    生成文本可能内嵌控制 token（模型把它们当作分隔符输出），而训练语料正文不含控制 token；
    不剥离会使 build_corpus.build_meta 的逐字符计数与 train.encode 的 token 折叠不一致，
    破坏 meta 与 token 流的等长断言。
    """
    for tok in train.SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text


def _has_oov(text, stoi):
    """文本是否含词表外字符（调用前已剥离特殊 token）。"""
    return any(c not in stoi for c in text)


def poem_lines(formatted):
    """把语料格式诗串拆回 (诗体, 行列表)，为 build_corpus.build_meta 的逆操作。

    formatted = 诗体 token + "\\n".join(lines) + "\\n"；无诗体前缀则抛 ValueError（响亮失败）。
    """
    for tag in build_corpus.POEM_TAGS:
        if formatted.startswith(tag):
            body = formatted[len(tag):]
            if body.endswith("\n"):
                body = body[:-1]
            return tag, body.split("\n")
    raise ValueError(f"非语料格式串（未以诗体 token 起首）：{formatted[:8]!r}")


def _canonical_body(text, n_lines):
    """把已剥去诗体特殊 token 的正文规范化成「语料格式正文」；半句不足 n_lines×2 时返回 None。

    口径：以 prosody.DEFAULT_PUNCT 任一字符为**保留终止符**（归入当前半句末尾）、空白为分隔符
    （不保留），切成半句序列并丢弃空段；取前 n_lines×2 个半句，按「每 2 个半句拼成 1 行、行间
    \\n」拼回（忽略模型自有换行，统一为一联一行）。new_tokens=96 的续写远超单首诗长，若不截断
    则整段判门槛恒为 0 入选，故本函数先截断成体。
    """
    delims = set(prosody.DEFAULT_PUNCT)
    halves, cur = [], []
    for ch in text:
        if ch in delims:
            cur.append(ch)
            halves.append("".join(cur))
            cur = []
        elif ch.isspace():
            if cur:
                halves.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        halves.append("".join(cur))
    halves = [h for h in halves if h]
    if len(halves) < n_lines * 2:
        return None
    chosen = halves[:n_lines * 2]
    return "\n".join("".join(chosen[2 * i:2 * i + 2]) for i in range(n_lines))


def select_and_format(cands, tag, pattern_len, min_line_len=1.0, min_tone=0.5,
                      min_rhyme=0.5, stoi=None):
    """K 候选先规范化成体、复判诗体，再按综合分降序取 top-1；三项门槛全过则返回语料格式诗串。

    流程（顺序不可调换）：
      1) 由 tag 定行数（绝句 2、律诗 4；其他 tag 直接返回 None）；
      2) 每个候选先剥诗体特殊 token，**在完整正文上**做 OOV 检查（stoi 非 None 时含词表外字符者
         整个剔除）——不可放在规范化之后，否则位于句尾、会被截断掉的 OOV 将漏检；
      3) _canonical_body 截前 n_lines×2 个半句成体，不足即弃；
      4) build_corpus.tag_of(成体各行) 须严格等于 tag，否则弃（保证 tag 与结构自洽）；
      5) 存活者按 prosody.score_candidate 降序取 top-1（同分保持候选原序）；
      6) top-1 须满足 check_line_len ≥ min_line_len、check_tone.main ≥ min_tone、
         check_rhyme.main ≥ min_rhyme，任一不过返回 None；
      7) 通过则返回 tag + 成体 + "\\n"。

    打分文本 = 规范化后的成诗正文（即最终入训练集的形态）。logprob 由调用方注入，其均值在
    **完整续写**上统计（生成阶段一次前向顺带算出），仅作候选间排序项；与打分文本（截断成体）
    口径不同，不得据其绝对值判断结构合规。stoi 仅用于第 2 步 OOV 检查。
    """
    n_lines = TAG_LINES.get(tag)
    if n_lines is None:
        return None
    scored = []
    for c in cands:
        body = _strip_special(c["text"])
        if stoi is not None and _has_oov(body, stoi):
            continue
        canon = _canonical_body(body, n_lines)
        if canon is None:
            continue
        if build_corpus.tag_of(canon.split("\n")) != tag:
            continue
        score = prosody.score_candidate(canon, pattern_len=pattern_len, logprob=c.get("logprob"))
        scored.append((score, canon))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)     # 稳定降序：同分保持候选出现序
    best = scored[0][1]
    if prosody.check_line_len(best, pattern_len) < min_line_len:
        return None
    if prosody.check_tone(best, pattern_len).main < min_tone:
        return None
    if prosody.check_rhyme(best).main < min_rhyme:
        return None
    return tag + best + "\n"


# ──────────────────────────── 4) 混合配比与 meta 对齐 ────────────────────────────

def build_mix(corpus_path, selftrain_path, corpus_meta_path, selftrain_meta_path,
              mix_text_path, mix_meta_path, stoi, rho=RHO):
    """构建混合训练集与其 meta，返回 {R, L_corpus, L_self, share}。

    配比：R = max(1, round(ρ/(1-ρ)·L_corpus/L_self))，L_* 以 len(train.encode(text, stoi)) 计。
    文本：mix.txt = corpus 原文 + selftrain 原文 × R（字节级追加，不重排、不插分隔符）。
    meta：四数组各为 concatenate([corpus_meta[k], tile(selftrain_meta[k], R)])；corpus 部分
    直接复用既有 meta（不重算，保证与 M1/M2 口径逐位一致），dtype 由拼接自然对齐。
    断言（响亮失败）：mix token 流长度与四数组等长；混合后词表逐项等于 corpus 词表。
    """
    with open(corpus_path, encoding="utf-8") as f:
        corpus_text = f.read()
    with open(selftrain_path, encoding="utf-8") as f:
        selftrain_text = f.read()
    L_corpus = int(len(train.encode(corpus_text, stoi)))
    L_self = int(len(train.encode(selftrain_text, stoi)))
    assert L_self > 0, "自产集为空，无法计算循环次数 R"
    R = max(1, round(rho / (1 - rho) * L_corpus / L_self))
    mix_text = corpus_text + selftrain_text * R
    os.makedirs(os.path.dirname(mix_text_path), exist_ok=True)
    with open(mix_text_path, "w", encoding="utf-8") as f:
        f.write(mix_text)

    corpus_meta = train.load_meta(corpus_meta_path)
    selftrain_meta = train.load_meta(selftrain_meta_path)
    mix_meta = {k: onp.concatenate([corpus_meta[k], onp.tile(selftrain_meta[k], R)])
                for k in META_KEYS}

    # 以 mix.txt 自身的词表编码：与 train.train(load_corpus) 的取数口径一致
    _, _, mix_stoi, _, mix_data, _ = train.load_corpus(mix_text_path)
    for k in META_KEYS:
        assert len(mix_data) == len(mix_meta[k]), (
            f"混合语料与 meta 长度不一致：mix_data={len(mix_data)} {k}={len(mix_meta[k])}")
    _, _, corpus_stoi, _, _, _ = train.load_corpus(corpus_path)
    assert sorted(set(mix_stoi)) == sorted(set(corpus_stoi)), (
        f"混合后词表漂移：mix={len(mix_stoi)} corpus={len(corpus_stoi)}")
    onp.savez(mix_meta_path, **mix_meta)
    share = (L_self * R) / (L_corpus + L_self * R)
    return {"R": R, "L_corpus": L_corpus, "L_self": L_self, "share": share}


# ──────────────────────────── 5) 混合再训 ────────────────────────────

def train_m3():
    """按 T10 实施方案定档超参调用 train.train 做混合再训。

    跨里程碑载参由 train.train 内部的 base_path（load_params_partial）承担，严禁 load_checkpoint。
    骨干须与基线档 model-gelv-m2.npz 一致（d256/h8/L6）：train.train 的骨干默认值为 d64/h4/L2，
    不显式传参会以 d64 建模型，按名载参时 tok_emb 形状 (8201, 256) vs (8201, 64) 不符即断言失败。
    """
    return train.train(
        data_path="data/mix.txt", meta_path="data/mix-meta.npz",
        base_path="model-gelv-m2.npz", max_steps=1000, anneal_total=1000,
        batch_size=16, ctx_len=128, warmup_steps=200, peak_lr=1.5e-4,
        d_model=256, n_head=8, n_layer=6,
        print_every=200, save_every=200, val_n_batch=16, seed=0,
        n_rhyme=107, use_tone=True, rhyme_classes=107, duizhang_weight=1.5,
        ckpt_path="model-gelv-m3.npz")


# ──────────────────────────── 断点账本与各阶段 ────────────────────────────

def _load_raw(raw_path=RAW_JSONL_PATH):
    """读生成阶段断点账本，返回 {idx: 记录}；文件不存在时空字典。"""
    recs = {}
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    rec = json.loads(ln)
                    recs[int(rec["idx"])] = rec
    return recs


def _append_raw(rec, raw_path=RAW_JSONL_PATH):
    """追加一行账本并 flush（每完成一条 prompt 立即落盘，中断不丢已完成结果）。"""
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def stage_gen(n=N_PROMPTS, k=K, new_tokens=NEW_TOKENS, model_path=None):
    """生成阶段：对 prompt 池逐条生成 K 候选并记账；按 idx 跳过已完成（断点续跑）。"""
    pool = build_prompt_pool(pool_path=POOL_PATH, n_per_tag=max(1, n // len(TAG_LEN)))[:n]
    done = _load_raw()
    todo = [(i, tag, p) for i, (tag, p) in enumerate(pool) if i not in done]
    _log(f"[gen] prompt {len(pool)} 条，已完成 {len(pool) - len(todo)}，待生成 {len(todo)}")
    if not todo:
        return
    model = build_model(model_path)
    _, _, stoi, itos, _, _ = train.load_corpus()
    for i, tag, prompt in todo:
        cands = generate_candidates(model, stoi, itos, tag, prompt, k=k, new_tokens=new_tokens)
        _append_raw({"idx": i, "tag": tag, "prompt": prompt, "candidates": cands})
        print(f"  [{i + 1}/{len(pool)}] {tag}{prompt} → {len(cands)} 候选")


def stage_build(n=N_PROMPTS):
    """构建阶段：读账本 → 逐条打分筛选 → 落盘自产集与其 meta → 构建混合集。"""
    pool = build_prompt_pool(pool_path=POOL_PATH, n_per_tag=max(1, n // len(TAG_LEN)))[:n]
    recs = _load_raw()
    _, _, stoi, _, _, _ = train.load_corpus()
    formatted = []
    n_cand = 0
    ll_sum = tn_sum = rh_sum = 0.0
    for i, (tag, prompt) in enumerate(pool):
        rec = recs.get(i)
        assert rec is not None, f"prompt #{i}（{tag}{prompt}）无生成记录，请先执行 --stage gen"
        cands = rec["candidates"]
        n_cand += len(cands)
        s = select_and_format(cands, tag, TAG_LEN[tag], stoi=stoi)
        if s is None:
            continue
        formatted.append((tag, s))
        ll_sum += prosody.check_line_len(s, TAG_LEN[tag])
        tn_sum += prosody.check_tone(s, TAG_LEN[tag]).main
        rh_sum += prosody.check_rhyme(s).main

    self_text = "".join(s for _, s in formatted)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SELFTRAIN_PATH, "w", encoding="utf-8") as f:
        f.write(self_text)
    _, w, t, r, z = build_corpus.build_meta([poem_lines(s)[1] for _, s in formatted])
    onp.savez(SELFTRAIN_META_PATH, weights=onp.asarray(w, dtype=onp.float32),
              tone_labels=onp.asarray(t, dtype=onp.int8),
              rhyme_labels=onp.asarray(r, dtype=onp.int16),
              zone_tags=onp.asarray(z, dtype=onp.int8))

    n_pick = len(formatted)
    rate = n_pick / n_cand if n_cand else 0.0
    _log(f"[build] 候选 {n_cand}，入选 {n_pick}（入选率 {rate:.2%}）；"
         f"入选集合规率均值 句长 {ll_sum / n_pick:.4f} 平仄 {tn_sum / n_pick:.4f} "
         f"押韵 {rh_sum / n_pick:.4f}" if n_pick else
         f"[build] 候选 {n_cand}，入选 0（入选率 0.00%）")
    res = build_mix(CORPUS_PATH, SELFTRAIN_PATH, META_PATH, SELFTRAIN_META_PATH,
                    MIX_TEXT_PATH, MIX_META_PATH, stoi)
    _log(f"[mix] R={res['R']} L_corpus={res['L_corpus']} L_self={res['L_self']} "
         f"实际占比 share={res['share']:.4f}（目标 ρ={RHO}）")


def main(argv=None):
    ap = argparse.ArgumentParser(description="T10 BoN 自蒸馏（prompt 池 → 生成 → 混合 → 混合再训）")
    ap.add_argument("--stage", choices=("gen", "build", "train", "all"), default="all",
                    help="执行阶段（all = gen → build → train）")
    ap.add_argument("--n", type=int, default=N_PROMPTS, help="prompt 总数（每诗体 n//4）")
    ap.add_argument("--k", type=int, default=K, help="每 prompt 候选数")
    ap.add_argument("--new-tokens", type=int, default=NEW_TOKENS, help="每候选续写长度")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="基线模型档（默认 model-gelv-m2.npz）")
    args = ap.parse_args(argv)
    if args.stage in ("gen", "all"):
        stage_gen(args.n, args.k, args.new_tokens, args.model)
    if args.stage in ("build", "all"):
        stage_build(args.n)
    if args.stage in ("train", "all"):
        train_m3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
