# -*- coding: utf-8 -*-
"""扩词表迁移：把旧 checkpoint(model.npz, V0=8196) 的权重按名灌入新词表模型，
产出 model-ext.npz（V = V0 + 5 = 8201），供带诗体特殊 token 的续训使用。

背景：旧档词表为 8196 个「剥标签纯文本字符」，新词表在其后追加 5 个诗体特殊
token（id 8196..8200，单一来源 train.SPECIAL_TOKENS）。为使旧行为无损继承，采用
「扩行/扩列 + 新行/新列严格全零」：零初始化下纯文本前向永不索引新 token，
lm_head 新列恒输出 0，故 logits 前 V0 列与旧档逐元素精确相等；切忌对新行用
randn 随机初始化。

不走 train.load_checkpoint：旧档 _chars(8196) 与新词表(8201) 不符，其 expect_chars
校验会拒载；本脚本直接读 npz、按名写入。目标函数将随词表改变，故不继承优化器
状态（_opt.* 一律忽略），新档只存模型参数（不含 _chars/优化器元数据）。

用法：`python tools/migrate_vocab.py`（默认 model.npz -> model-ext.npz）。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（脚本在 tools/ 下）
if BASE not in sys.path:        # 使项目根下的 train/model 可导入（直接运行本脚本时）
    sys.path.insert(0, BASE)

import numpy as onp

import train
from model.gpt import GPT

# 目标架构：d/n_layer/ctx 由旧档形状实测后断言，n_head 旧档不存故按已知训练配置常量
D_MODEL = 256
N_HEAD = 8
N_LAYER = 6
CTX_LEN = 128

DEFAULT_OLD = os.path.join(BASE, "model.npz")
DEFAULT_NEW = os.path.join(BASE, "model-ext.npz")

# 需扩行/扩列继承的参数名（词表维度相关），其余参数按名直接覆盖
_EXPAND_NAMES = ("tok_emb", "lm_head.W", "lm_head.b")


def _derive_arch(old):
    """从旧档参数形状实测推导 (V0, d_model, ctx_len, n_layer)。

    V0=tok_emb 行数、d=tok_emb 列数、ctx=pos_emb 行数、n_layer=blocks 层号最大值+1；
    层号须自 0 连续，否则判为档损坏。
    """
    V0, d = int(old["tok_emb"].shape[0]), int(old["tok_emb"].shape[1])
    ctx = int(old["pos_emb"].shape[0])
    layers = sorted({int(k.split(".")[1]) for k in old if k.startswith("blocks.")})
    if not layers:
        raise ValueError("旧档缺少 blocks.* 参数，无法推导层数")
    n_layer = layers[-1] + 1
    if layers != list(range(n_layer)):
        raise ValueError(f"旧档 blocks 层号不连续：{layers}")
    return V0, d, ctx, n_layer


def migrate(old_path=DEFAULT_OLD, new_path=DEFAULT_NEW, n_head=N_HEAD):
    """旧档 -> 新档扩词表迁移，返回摘要字典。

    步骤：实测旧档架构并断言与目标一致 -> 校验词表来源（旧档 _chars 须等于语料
    chars）-> 建新模型 -> 按名继承（共享参数直接覆盖；三处词表相关参数扩行/扩列、
    新行/新列全零）-> 落盘（仅参数）。返回 {V0,V,d_model,n_head,n_layer,ctx_len,
    三个新行/新列全零校验布尔}。
    """
    with onp.load(old_path) as f:           # 惰性取键：仅载入迁移所需数组，跳过 _opt.* 等元数据
        keep = [k for k in f.files if k == "_chars" or not k.startswith("_")]
        old = {k: f[k] for k in keep}
    V0, d, ctx, n_layer = _derive_arch(old)
    if (d, n_layer, ctx) != (D_MODEL, N_LAYER, CTX_LEN):
        raise ValueError(f"旧档架构 d{d}/L{n_layer}/ctx{ctx} 与目标 "
                         f"d{D_MODEL}/L{N_LAYER}/ctx{CTX_LEN} 不一致，拒绝迁移")

    # 词表来源校验：新词表纯文本部分 = 语料 chars（顺序不变）；旧档 _chars 须与之相等，
    # 否则说明语料/词表漂移，迁移会静默错位，直接报错
    _, chars, _, _, _, _ = train.load_corpus()
    old_chars = list(old.get("_chars", []))
    if list(chars) != old_chars:
        raise ValueError(f"词表漂移：语料 chars({len(chars)}) 与旧档 _chars({len(old_chars)}) "
                         "逐项不一致，拒绝迁移")
    if V0 != len(chars):
        raise ValueError(f"旧档 tok_emb 行数 {V0} 与语料 chars 数 {len(chars)} 不一致，拒绝迁移")
    specials = list(train.SPECIAL_TOKENS)
    V = V0 + len(specials)

    # 扩行/扩列继承：前 V0 部分逐元素复制，新行/新列严格全零（禁止随机初始化）
    old_tok, old_W, old_b = old["tok_emb"], old["lm_head.W"], old["lm_head.b"]
    tok_emb = onp.zeros((V, d), dtype=old_tok.dtype)
    tok_emb[:V0] = old_tok
    lm_W = onp.zeros((d, V), dtype=old_W.dtype)
    lm_W[:, :V0] = old_W[:, :V0]
    lm_b = onp.zeros((V,), dtype=old_b.dtype)
    lm_b[:V0] = old_b

    # 共享参数（pos_emb / blocks.* / ln_f.*）按名直接照搬旧数组
    params = {k: v for k, v in old.items()
              if not k.startswith("_") and k not in _EXPAND_NAMES}
    params["tok_emb"] = tok_emb
    params["lm_head.W"] = lm_W
    params["lm_head.b"] = lm_b

    new = GPT(vocab_size=V, d_model=d, n_head=n_head, n_layer=n_layer, ctx_len=ctx)
    expected = {name for name, _ in new._named_params()}
    if set(params) != expected:
        raise ValueError(f"参数名不匹配：多余 {sorted(set(params) - expected)} "
                         f"缺失 {sorted(expected - set(params))}")
    new.load_params(params)                 # 按名逐元素写入（CPU 零拷贝/GPU 显式拷入）
    new.save(new_path)                      # 仅参数落盘（无 _chars/优化器元数据）
    return {"V0": V0, "V": V, "d_model": d, "n_head": n_head,
            "n_layer": n_layer, "ctx_len": ctx,
            "new_tok_emb_zero": bool(onp.all(tok_emb[V0:] == 0)),
            "new_lm_head_W_zero": bool(onp.all(lm_W[:, V0:] == 0)),
            "new_lm_head_b_zero": bool(onp.all(lm_b[V0:] == 0))}


def main():
    """默认迁移 model.npz -> model-ext.npz，并回读落盘产物做真实性校验。"""
    info = migrate(DEFAULT_OLD, DEFAULT_NEW)
    V0 = info["V0"]
    with onp.load(DEFAULT_OLD) as f:        # 惰性取键：仅取前缀比对所需旧档数组，跳过 _opt.* 等元数据
        old_tok, old_W, old_b = f["tok_emb"], f["lm_head.W"], f["lm_head.b"]
    with onp.load(DEFAULT_NEW) as f:        # 回读产物：校验对象是磁盘上的实际交付物
        tok, W, b = f["tok_emb"], f["lm_head.W"], f["lm_head.b"]
    prefix_ok = (onp.array_equal(tok[:V0], old_tok)
                 and onp.array_equal(W[:, :V0], old_W)
                 and onp.array_equal(b[:V0], old_b))
    zero_ok = bool(onp.all(tok[V0:] == 0)
                   and onp.all(W[:, V0:] == 0)
                   and onp.all(b[V0:] == 0))
    print(f"迁移完成：{DEFAULT_OLD} -> {DEFAULT_NEW}")
    print(f"  词表 V0={info['V0']} -> V={info['V']}（新增 {info['V'] - info['V0']} 个诗体特殊 token）")
    print(f"  架构 d{info['d_model']}/h{info['n_head']}/L{info['n_layer']}/ctx{info['ctx_len']}")
    print(f"  回读产物校验：前缀与旧档逐元素相等={prefix_ok} "
          f"新行/新列严格全零={zero_ok} -> "
          f"{'通过 ✓' if (prefix_ok and zero_ok) else '失败 ✗'}")
    return 0 if (prefix_ok and zero_ok) else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
