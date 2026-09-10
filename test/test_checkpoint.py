"""续训 checkpoint 单测：save/load 往返、lr 回放、chars/调度签名拒载、原子写

覆盖对象：train.save_checkpoint / train.load_checkpoint / train._savez_atomic / train.train
均用 CPU 小模型（d=16/1 头/1 层/ctx=16/小词表）快速验证，不碰全唐诗大语料。
"""
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train
from model.gpt import GPT

V = 20                                     # 小词表
CHARS = list("床前明月光疑是地上霜举头望低思故乡。，")[:V]
SCHED = dict(anneal_total=200, warmup_steps=20, peak_lr=3e-4)
SEED = 0


def _make_gpt():
    """CPU 小 GPT：d=16 / 2 头 / 1 层 / ctx=16。"""
    return GPT(vocab_size=V, d_model=16, n_head=2, n_layer=1, ctx_len=16)


def _run_steps(gpt, opt, n=6):
    """同一批假数据推 n 步，制造非零的 m/v/t。"""
    x = np.random.randint(0, V, size=(4, 16))
    y = np.random.randint(0, V, size=(4, 16))
    for _ in range(n):
        loss = gpt.loss(gpt.forward(x), y)
        gpt.zero_grad()
        gpt.backward()
        opt.step()


def test_roundtrip():
    """save→load 往返：参数逐名一致、m/v 逐名一致、t 恢复。"""
    np.random.seed(SEED)
    g1, opt1 = _make_gpt(), None
    opt1 = train.AdamW(g1)
    _run_steps(g1, opt1, 6)
    assert opt1.t == 6, f"opt.t 应为 6，实为 {opt1.t}"
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "ckpt.npz")
        train.save_checkpoint(cp, g1, opt1, CHARS, **SCHED)
        g2, opt2 = _make_gpt(), train.AdamW(_make_gpt())
        t2 = train.load_checkpoint(cp, g2, opt2, expect_chars=CHARS, expect_sched=SCHED)
        assert t2 == 6 and opt2.t == 6, f"载入后 t 应为 6，实为 {t2}/{opt2.t}"
        d1, d2 = g1.dump_params(), g2.dump_params()
        assert set(d1) == set(d2), "参数名集合不一致"
        for k in d1:
            assert np.allclose(np.asarray(d1[k]), np.asarray(d2[k])), f"参数 {k} 不一致"
        for k in opt1.m:
            assert np.allclose(np.asarray(opt1.m[k]), np.asarray(opt2.m[k])), f"m[{k}] 不一致"
            assert np.allclose(np.asarray(opt1.v[k]), np.asarray(opt2.v[k])), f"v[{k}] 不一致"
    print("roundtrip PASS")


def _tiny_corpus_dir():
    """造 60 行小语料目录，返回 (dir, corpus_path, ckpt_path)。"""
    td = tempfile.mkdtemp()
    line = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    with open(os.path.join(td, "corpus.txt"), "w", encoding="utf-8") as f:
        for _ in range(60):
            f.write(line + "\n")
    return td, os.path.join(td, "corpus.txt"), os.path.join(td, "model.npz")


def _collect_lrs(**kw):
    """跑一次 train()，捕获每步打印的 lr 序列。kw 覆盖默认参数。"""
    buf = StringIO()
    with redirect_stdout(buf):
        train.train(print_every=1, demo_every=None, **kw)
    return [float(m) for m in re.findall(r"lr ([0-9.e+-]+)", buf.getvalue())]


def test_lr_replay():
    """lr 回放：续训段 lr 必须按全局步数走 get_lr 回放，而非冻结在 0.1×peak。

    第一段训 40 步、第二段续 20 步：拼接 lr 曲线应与从头训 60 步逐点相等。
    （旧冻结语义下第二段恒 3e-5，退火中段并非该值，故能区分新旧行为。）
    """
    td, corpus, cp = _tiny_corpus_dir()
    try:
        base = dict(data_path=corpus, ckpt_path=cp, max_steps=60, anneal_total=200,
                    warmup_steps=20, peak_lr=3e-4, d_model=16, n_head=2,
                    n_layer=1, ctx_len=16, batch_size=4, seed=SEED, save_every=None,
                    val_path=os.path.join(td, "no-such-val.txt"))  # 关闭 val，隔离外部文件
        expect = _collect_lrs(**base)                     # 从头训 60 步（参照曲线）
        # 第一段：训 40 步，正常存档 t=40
        lrs_a = _collect_lrs(**{**base, "max_steps": 40})
        assert lrs_a == expect[:40], "第一段 lr 曲线偏离从头训"
        # 第二段：resume 同一档，续 20 步（起点读档 t=40，自动衔接）
        lrs_b = _collect_lrs(**{**base, "max_steps": 20, "resume_path": cp})
        assert lrs_b == expect[40:60], f"续训段 lr 未按全局步数回放\nA={lrs_a}\nB={lrs_b}"
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)
    print("lr_replay PASS")


def test_reject_chars():
    """词表快照不一致 → 抛 ValueError（防换语料静默错位）。"""
    np.random.seed(SEED)
    g, opt = _make_gpt(), train.AdamW(_make_gpt())
    _run_steps(g, opt, 3)
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "ckpt.npz")
        train.save_checkpoint(cp, g, opt, CHARS, **SCHED)
        other = CHARS[:-1] + ["山"]                      # 末尾换一个字 → 词表不同
        try:
            train.load_checkpoint(cp, _make_gpt(), train.AdamW(_make_gpt()),
                                  expect_chars=other, expect_sched=SCHED)
            raise AssertionError("chars 不一致应拒载，却载入成功")
        except ValueError:
            pass
    print("reject_chars PASS")


def test_reject_sched():
    """调度签名不一致 → 抛 ValueError（防错传参静默错位）。"""
    np.random.seed(SEED)
    g, opt = _make_gpt(), train.AdamW(_make_gpt())
    _run_steps(g, opt, 3)
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "ckpt.npz")
        train.save_checkpoint(cp, g, opt, CHARS, **SCHED)
        wrong = dict(SCHED, warmup_steps=999)            # 热身步数改掉
        try:
            train.load_checkpoint(cp, _make_gpt(), train.AdamW(_make_gpt()),
                                  expect_chars=CHARS, expect_sched=wrong)
            raise AssertionError("调度签名不一致应拒载，却载入成功")
        except ValueError:
            pass
    print("reject_sched PASS")


def test_atomic_write():
    """原子写：os.replace 前中断，原档完好可读、内容不变。"""
    np.random.seed(SEED)
    g, opt = _make_gpt(), train.AdamW(_make_gpt())
    _run_steps(g, opt, 4)                                # t=4 的档 A
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "ckpt.npz")
        train.save_checkpoint(cp, g, opt, CHARS, **SCHED)
        gA, optA = _make_gpt(), train.AdamW(_make_gpt())
        assert train.load_checkpoint(cp, gA, optA, expect_chars=CHARS,
                                     expect_sched=SCHED) == 4
        # 模拟第二次写档在替换前崩溃：os.replace 抛异常
        _run_steps(g, opt, 2)                            # t=6 的档 B
        orig = train.os.replace
        def _boom(src, dst):
            raise OSError("simulated crash before rename")
        train.os.replace = _boom
        try:
            try:
                train.save_checkpoint(cp, g, opt, CHARS, **SCHED)
                raise AssertionError("模拟崩溃应抛错，却写档成功")
            except OSError:
                pass
        finally:
            train.os.replace = orig
        # 原档仍是 A（t=4），未被半截文件破坏
        g2, opt2 = _make_gpt(), train.AdamW(_make_gpt())
        assert train.load_checkpoint(cp, g2, opt2, expect_chars=CHARS,
                                     expect_sched=SCHED) == 4, "原档被损坏"
        # 临时文件应已清理（savez 失败路径不留 .tmp）
        leftovers = [f for f in os.listdir(td) if "tmp" in f]
        assert not leftovers, f"残留临时文件: {leftovers}"
    print("atomic_write PASS")


if __name__ == "__main__":
    test_roundtrip()
    test_lr_replay()
    test_reject_chars()
    test_reject_sched()
    test_atomic_write()
    print("\n全部 checkpoint 单测通过")
