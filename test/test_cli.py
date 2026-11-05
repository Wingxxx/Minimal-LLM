"""CLI 测试：提示词编码 / 续写截取 / 上下文越界

只测不落权重盘的纯逻辑（桩模型替身）；真档实跑由 `test/output-cli.txt` 留痕。
运行：`python test/test_cli.py`
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cli

_STOI = {"床": 0, "前": 1, "明": 2, "月": 3, "光": 4, "，": 5, "。": 6, "\n": 7,
         "<五绝>": 8, "<七绝>": 9, "<五律>": 10, "<七律>": 11, "<杂言>": 12}
_ITOS = {0: "床", 1: "前", 2: "明", 3: "月", 4: "光", 5: "，", 6: "。", 7: "\n",
         8: "<五绝>", 9: "<七绝>", 10: "<五律>", 11: "<七律>", 12: "<杂言>"}


class _StubModel:
    """GPT 替身：记录采样参数，按序回吐 gen_ids，返回「提示词 + 续写」拼好的序列。"""

    def __init__(self, gen_ids=(1, 2, 3), ctx_len=8):
        self.gen_ids = list(gen_ids)
        self.ctx_len = ctx_len
        self.kw = None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, **kw):
        self.kw = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        picked = self.gen_ids[:max_new_tokens]
        return np.concatenate([idx, np.asarray([picked], dtype=idx.dtype)], axis=1)


# ---------- 提示词编码 ----------

def test_encode_prompt_basic():
    """五个汉字 -> [1,5] 的 id 数组（列序与提示词一致）"""
    ids = cli.encode_prompt("床前明月光", _STOI)
    assert ids.shape == (1, 5), ids.shape
    assert ids[0].tolist() == [0, 1, 2, 3, 4], ids
    print("CLI encode prompt PASS", ids.tolist())

def test_encode_prompt_special_token():
    """诗体标签按 1 个 id 计（<五绝> 共 4 字符），不得拆成 4 个字符 id"""
    ids = cli.encode_prompt("<五绝>床前", _STOI)
    assert ids.shape == (1, 3), ids.shape
    assert ids[0].tolist() == [8, 0, 1], ids
    print("CLI encode special token PASS", ids.tolist())

def test_encode_prompt_all_oov_raises():
    """分词后为空必须响亮报错——严禁以换行符兜底掩盖输入错误"""
    try:
        cli.encode_prompt("abc", _STOI)
    except ValueError as exc:
        assert "为空" in str(exc), exc
        print("CLI encode empty raises PASS", exc)
        return
    raise AssertionError("全 OOV 提示词未被拒绝（疑似静默兜底）")


# ---------- 续写截取 ----------

def test_generate_text_excludes_prompt():
    """返回的文本只含续写部分，不得把提示词重复打印一遍"""
    m = _StubModel(gen_ids=(1, 2, 3))
    ids = cli.encode_prompt("床前明月光", _STOI)
    text = cli.generate_text(m, _ITOS, ids, 3, 0.8, 20, None)
    assert text == "前明月", repr(text)
    print("CLI generate strips prompt PASS", repr(text))

def test_generate_text_forwards_sampling_args():
    """采样参数须原样透传给 GPT.generate（CLI 只做起句与装配，不私改解码策略）"""
    m = _StubModel(gen_ids=(1,))
    ids = cli.encode_prompt("床", _STOI)
    cli.generate_text(m, _ITOS, ids, 1, 0.5, 7, 0.9)
    assert m.kw == dict(temperature=0.5, top_k=7, top_p=0.9), m.kw
    print("CLI sampling args PASS", m.kw)

def test_generate_text_ctx_overflow_raises():
    """prompt_tokens + max_tokens 超出 ctx_len 必须响亮报错（与 server 同口径）"""
    m = _StubModel(gen_ids=(1, 2, 3, 4), ctx_len=8)
    ids = cli.encode_prompt("床前明月光", _STOI)          # 5 个 token
    try:
        cli.generate_text(m, _ITOS, ids, 4, 0.8, 20, None)   # 5 + 4 > 8
    except ValueError as exc:
        assert "上下文" in str(exc) and "8" in str(exc), exc
        print("CLI ctx overflow raises PASS", exc)
        return
    raise AssertionError("超出 ctx_len 未被拒绝")


def main():
    test_encode_prompt_basic()
    test_encode_prompt_special_token()
    test_encode_prompt_all_oov_raises()
    test_generate_text_excludes_prompt()
    test_generate_text_forwards_sampling_args()
    test_generate_text_ctx_overflow_raises()
    print("ALL CLI TESTS PASS")


if __name__ == "__main__":
    main()
