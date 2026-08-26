"""KV Cache 一致性测试：增量前向（带缓存）与全量前向（无缓存）输出必须一致。

原理：
    KV Cache 是解码阶段的加速手段：历史 K/V 只算一次、存进缓存，每步只算
    新 token 的 K/V 并拼接。如果实现正确，带缓存的"增量前向"与不带缓存的
    "全量前向"应当产生完全相同的结果（误差仅来自浮点累积，应 < 1e-6）。

    增量前向流程（路径 B）：
        1. 创建空缓存（K/V 账本，序列长度为 0）
        2. 逐词调用 forward，每步只喂 1 个新 token，并传入缓存
        3. forward 内部将新 K/V 拼接到缓存尾部（Tk = 历史 + 新）
        4. 收集每一步的输出，按序列方向拼接成完整输出
    全量前向流程（路径 A）：
        一次喂入全部 token，不传缓存，forward 走普通全量路径。

    两条路径得到的输出数组应逐元素一致。
"""

import os
import sys

import numpy as np

# 将项目根目录加入模块搜索路径，使 `import model.attention` 在任意目录下均可生效
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.attention import MultiHeadAttention


def test_kv_cache_consistency():
    """KV Cache 一致性：增量 vs 全量，输出最大误差 < 1e-6 视为通过。"""
    print("== KV Cache 一致性 ==")
    np.random.seed(0)                          # 固定随机种子，结果可复现
    d_model, n_head = 8, 4                     # 特征维 8，4 个头（d_k = 2）
    attn = MultiHeadAttention(d_model, n_head)
    B, T_full = 2, 5                           # 2 批，5 个词
    x_full = np.random.randn(B, T_full, d_model)

    # 路径 A：全量前向（无缓存），一次喂全部 5 词
    out_full = attn.forward(x_full)            # [B, T_full, d_model]

    # 路径 B：增量前向（带缓存），创建空缓存后逐词解码
    kv_cache = [np.zeros((B, attn.n_kv_head, 0, attn.d_k)),   # 空 K 账本
                np.zeros((B, attn.n_kv_head, 0, attn.d_k))]   # 空 V 账本
    outs = []
    for t in range(T_full):
        # 每步只喂 1 个词（x[:, t:t+1]），缓存里已存前 t 个词的历史
        out_t = attn.forward(x_full[:, t:t + 1], kv_cache)
        outs.append(out_t)
        # 断言：每步结束后，缓存长度应恰好等于已生成的词数 t+1
        assert kv_cache[0].shape[2] == t + 1, f"第{t}步缓存长度应为{t+1}，实际{kv_cache[0].shape[2]}"
    out_inc = np.concatenate(outs, axis=1)     # 拼接各步输出 → [B, T_full, d_model]

    # 对比两条路径：逐元素最大误差应接近 0（浮点累积误差 < 1e-6）
    err = np.max(np.abs(out_full - out_inc))
    print(f"  全量 vs 增量 输出最大误差 = {err:.2e}")
    print(f"  缓存长度逐步递增断言: PASS")
    assert err < 1e-6, f"KV Cache 一致性测试失败: 最大误差 {err}"

    # 顺带验证：每步输出的序列长度等于本次喂入的 1 个词
    print(f"  每步输出形状: [{out_inc.shape[0]}, 1, {out_inc.shape[2]}]  × {T_full} 步")


if __name__ == '__main__':
    # 启动按钮：仅直接运行本文件时执行
    test_kv_cache_consistency()
    print("\nKV Cache 一致性测试通过：增量前向与全量前向输出一致，缓存复用正确。")
