"""CLI：加载权重档，本地直接生成（不经 HTTP、不起服务）

与 `server.py` 的分工：server 面向多客户端（串行排队 + SSE + 延迟统计）；
CLI 面向单机快速试模型——一条命令、一次生成、直接看文本。

启动：`python cli.py "床前明月光" --max-tokens 40`
诗体起句可直接把标签写进提示词：`python cli.py "<五绝>床前明月光"`
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import train
from model.backend import np as onp, as_numpy
from server import load_serving_model


def encode_prompt(prompt, stoi):
    """把提示词编码成 [1,T] 的 id 数组。

    复用 `train.encode`：先匹配诗体特殊 token（1 个 id），未匹配处逐字符查表。
    分词后为空（全为词表外字符）必须响亮报错——严禁以换行符兜底，
    否则客户端输错提示词也会被静默掩盖（口径与 server 的参数校验一致）。
    """
    ids = train.encode(prompt, stoi)
    if len(ids) == 0:
        raise ValueError("提示词 %r 分词后为空（全部字符不在词表内）" % prompt)
    return onp.asarray([ids], dtype=onp.int64)


def generate_text(model, itos, ids, max_tokens, temperature, top_k, top_p):
    """从 ids 续写 max_tokens 个 token，返回**续写部分**的文本（不含提示词）。

    越界必须响亮报错：`pos_emb` 只有 ctx_len 行，超长会让位置嵌入越界崩溃；
    与其让内核抛晦涩的形状错误，不如在入口按「提示词 + 续写 ≤ ctx_len」直接拒绝。
    """
    used = int(ids.shape[1])
    ctx_len = int(model.ctx_len)
    if used + int(max_tokens) > ctx_len:
        raise ValueError("上下文不足：提示词 %d + max_tokens %d > ctx_len %d"
                         % (used, int(max_tokens), ctx_len))
    out = model.generate(ids, int(max_tokens), temperature=temperature,
                         top_k=top_k, top_p=top_p)
    new = as_numpy(out)[0, used:]                  # 只取续写部分，避免重复打印提示词
    return "".join(itos.get(int(t), "") for t in new)


def main(argv=None):
    ap = argparse.ArgumentParser(description="极简 LLM 命令行生成")
    ap.add_argument("prompt", nargs="?", default="床前明月光", help="起句（可含 <五绝> 等诗体标签）")
    ap.add_argument("--model", default=os.path.join(BASE, "model-gelv-m3.npz"),
                    help="权重档路径（默认 model-gelv-m3.npz）")
    ap.add_argument("--max-tokens", type=int, default=40, help="续写 token 数（默认 40）")
    ap.add_argument("--temperature", type=float, default=0.8, help="采样温度（默认 0.8）")
    ap.add_argument("--top-k", type=int, default=20, help="Top-K 截断（默认 20；≤0 视为不启用）")
    ap.add_argument("--top-p", type=float, default=None, help="核采样阈值（默认不启用）")
    args = ap.parse_args(argv)

    # CLI 参数即用户输入边界：非法值当场拒绝，不把晦涩报错留给内核
    if args.max_tokens <= 0:
        ap.error("--max-tokens 必须 > 0")
    if args.temperature <= 0:
        ap.error("--temperature 必须 > 0")
    top_k = args.top_k if args.top_k > 0 else None

    model, stoi, itos = load_serving_model(args.model)
    try:
        ids = encode_prompt(args.prompt, stoi)
        text = generate_text(model, itos, ids, args.max_tokens,
                             args.temperature, top_k, args.top_p)
    except ValueError as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return 2
    print(args.prompt + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
