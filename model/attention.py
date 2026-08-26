import numpy as np
from .layers import Linear


class MultiHeadAttention:
    """多头注意力：QKV 投影 + 缩放点积 + causal mask + 多头切分/合并。

    核心公式（缩放点积注意力）：
        Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
    多头将特征维 d 切成 n_head 份（每份 d_k = d / n_head），每头独立
    执行上述公式，再拼回并经过输出投影 Wo，供后续层使用。
    输入输出形状均为 [B, T, d]（B 批大小 / T 序列长度 / d 特征维度）。
    """

    def __init__(self, d_model, n_head, n_kv_head=None):
        # d_k = d_model / n_head：每头分到的特征维度；n_kv_head：K/V 投影头数
        self.d_model, self.n_head = d_model, n_head
        # n_kv_head 默认等于 n_head（MHA）；设为 1 即 MQA，中间值即 GQA
        self.n_kv_head = n_kv_head or n_head
        self.d_k = d_model // n_head
        # scale = 1 / sqrt(d_k)：缩放系数。点积结果方差与 d_k 成正比，
        # 除以 sqrt(d_k) 把方差压回 ~1，防止 softmax 因数值过大而过早饱和
        self.scale = 1.0 / np.sqrt(self.d_k)
        # 4 个投影层：Wq/Wk/Wv 把输入投影成 Q/K/V 三种角色，
        # Wo 把多头注意力结果线性混合后映射回原空间。
        # 注意：Wk/Wv 输出维度按 n_kv_head 缩放（n_kv_head * d_k），
        # 实现 K/V 头共享（省参数量）；MHA 下 n_kv_head = n_head，即为 d
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, self.n_kv_head * self.d_k)
        self.Wv = Linear(d_model, self.n_kv_head * self.d_k)
        self.Wo = Linear(d_model, d_model)
        # 前向中间结果缓存（x/q/k/v/probs），backward 反向传播时使用
        self.cache = None

    def _to_heads(self, x):
        """把 Q/K/V 的 [B, T, d] 切成多头布局 [B, n_head, T, d_k]。"""
        B, T, D = x.shape
        # reshape：把特征维 d 按内存顺序切成 n_head 块，每块 d_k 维 →
        # 形状 [B, T, n_head, d_k]，此时"头"还在第 2 位
        # transpose(0, 2, 1, 3)：轴序变为 [B, n_head, T, d_k]，
        # 把头提到 batch 之后，使每头内 T 连续排列、后两维 [T, d_k]
        # 恰好是矩阵行结构，便于每头直接执行 Q @ K^T 的注意力运算
        return x.reshape(B, T, self.n_head, self.d_k).transpose(0, 2, 1, 3)

    def _from_heads(self, x):
        """把多头布局 [B, n_head, T, d_k] 并回 [B, T, d]，是 _to_heads 的逆操作。"""
        B, H, T, Dk = x.shape
        # transpose(0, 2, 1, 3)：轴序变回 [B, T, n_head, d_k]，头放回第 2 位
        # reshape：把 n_head 块 d_k 维拼接回完整特征维 n_head * d_k = d，
        # 得到与输入同形的 [B, T, d]，保证 MHA 输入输出形状一致
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.n_head * Dk)

    def _kv_heads(self, x):
        """K/V 专属切头：切成 [B, n_kv_head, T, d_k]（头数可少于 n_head，实现 GQA）。"""
        B, T, D = x.shape
        # 与 _to_heads 相同的切块+换轴，区别仅在头数用 n_kv_head：
        # reshape [B, T, n_kv_head, d_k] → transpose(0,2,1,3) → [B, n_kv_head, T, d_k]
        # Wk/Wv 的输出维度（n_kv_head * d_k）决定了这里的 D 能正好切成 n_kv_head 份
        return x.reshape(B, T, self.n_kv_head, self.d_k).transpose(0, 2, 1, 3)

    def _expand_kv(self, k, v):
        """把 n_kv_head 组 KV 复制展开为 n_head 组，供所有 Q 头参与注意力。"""
        # reps = n_head / n_kv_head：每组 KV 被共享的 Q 头数量
        reps = self.n_head // self.n_kv_head
        if reps == 1:
            return k, v
        # np.repeat(axis=1)：每份连续复制 reps 份，如 [KV0, KV1] → [KV0, KV0, KV1, KV1]
        # 效果：第 0/1 个 Q 头共享 KV0，第 2/3 个 Q 头共享 KV1（MQA 时 reps=n_head，全共享）
        return np.repeat(k, reps, axis=1), np.repeat(v, reps, axis=1)

    def _softmax(self, scores):
        """数值稳定的 softmax：softmax(s) = e^(s-m) / sum(e^(s-m))，m 为每行最大值。

        分子分母同乘 e^(-m) 不改变结果，但 e^(s-m) 最大为 e^0 = 1，
        避免 scores 较大时 e^s 溢出为 inf（softmax 平移不变性）。
        """
        # 第 1 步：取每行最大值 m（axis=-1 = 序列轴；keepdims=True 保留 [B,H,T,1] 便于广播）
        m = scores.max(axis=-1, keepdims=True)
        # 第 2 步：每个分数减去所在行最大值后再取指数，e^(s-m) ∈ (0, 1]，永不溢出
        e = np.exp(scores - m)
        # 第 3 步：除以该行所有指数之和，归一化成权重（每行权重和为 1）
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, x, kv_cache=None):
        """前向：QKV 投影 → 切头 → 缩放点积注意力（causal mask）→ 并头 → Wo。

        kv_cache：KV Cache 的容器列表 [k, v]，解码阶段传入以复用历史 K/V；
        为 None 时走普通全量前向（训练/预填充）。
        """
        B, T, D = x.shape
        # 投影并切头：Q 用 n_head 头，K/V 用 n_kv_head 头（GQA 省缓存）
        q = self._to_heads(self.Wq.forward(x))        # [B, n_head, T, d_k]
        k = self._kv_heads(self.Wk.forward(x))        # [B, n_kv_head, T, d_k]
        v = self._kv_heads(self.Wv.forward(x))
        # ── KV Cache：复用历史 K/V，避免解码阶段重复计算 ──
        # kv_cache 是长度为 2 的列表 [历史k, 历史v]，各自形状 [B, n_kv_head, 历史长度, d_k]
        if kv_cache is not None and kv_cache[0].shape[2] > 0:
            # np.concatenate(数组列表, axis=轴号)：把多个数组沿指定轴首尾相接成一个大数组。
            # axis=2 即"序列长度"轴（0:B / 1:头 / 2:序列 / 3:特征）。
            # 这里把 [历史K] 和 [本次新算的K] 接起来：历史2词 + 新1词 → 3词
            k = np.concatenate([kv_cache[0], k], axis=2)
            v = np.concatenate([kv_cache[1], v], axis=2)
        if kv_cache is not None:
            # 把拼接后的 K/V 写回缓存（只增不减），下次生成时直接复用
            kv_cache[0], kv_cache[1] = k, v
        # Tk：拼接后的总序列长度 = 历史长度 + 本次新词数
        Tk = k.shape[2]
        # 展开 K/V 到 n_head 组，参与逐头注意力
        k_exp, v_exp = self._expand_kv(k, v)          # [B, n_head, Tk, d_k]
        # 缩放点积注意力第 1 步：scores = Q @ K^T * scale，形状 [B, H, T, Tk]
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) * self.scale
        # causal mask：当前批次新来 T 个位置（索引 base ~ base+T-1），
        # 每个位置 i 只能看 j <= base + i 的历史位置，j > base + i 视为未来，置 -1e9
        base = Tk - T
        mask = np.arange(Tk)[None, None, None, :] > (base + np.arange(T)[None, None, :, None])
        scores = np.where(mask, -1e9, scores)
        # 第 2、3 步：softmax 归一化 → 加权求和取 V
        probs = self._softmax(scores)                 # [B, H, T, Tk]
        attn = probs @ v_exp                          # [B, H, T, d_k]
        # 并头 + 输出投影，恢复 [B, T, d]
        out = self.Wo.forward(self._from_heads(attn))
        # 缓存前向中间结果（反向传播 backward 使用）
        self.cache = (x, q, k_exp, v_exp, probs)
        return out

    def zero_grad(self):
        """清空全部投影层的梯度缓存（backward 用 += 累加，训练每步前必须归零）。"""
        self.Wq.zero_grad()
        self.Wk.zero_grad()
        self.Wv.zero_grad()
        self.Wo.zero_grad()

    def backward(self, dout):
        """反向传播：把上游梯度 dout 沿前向链路逐层回传，累加各投影层 dW/db。

        核心公式（缩放点积注意力的反向，P 为 softmax 权重矩阵）：
            dP      = dAttn @ V^T                           （权重矩阵梯度）
            dV      = P^T @ dAttn                           （V 的梯度）
            dScores = P ⊙ (dP - sum(dP ⊙ P, axis=-1))       （softmax 反向）
            dQ      = dScores @ K * scale                   （Q 的梯度）
            dK      = dScores^T @ Q * scale                 （K 的梯度）
        结构类操作的反向：切头/并头（reshape/transpose）原样转回；
        复制（_expand_kv）把各份梯度求和回 n_kv_head 份（同一数据多处使用，梯度累加）；
        拼接（concatenate）在训练路径不出现（kv_cache=None，Tk == T）。
        注意：backward 仅用于训练（无缓存路径），解码生成阶段不反向传播。
        """
        # 取出前向缓存：x 原始输入；q/k_exp/v_exp 多头布局；probs 注意力权重
        x, q, k_exp, v_exp, probs = self.cache
        B, H, T, Dk = q.shape
        Tk = k_exp.shape[2]
        reps = self.n_head // self.n_kv_head

        # ── 1. Wo 与并头反向 ──
        # Linear.backward(dout) 返回输入侧梯度 [B,T,d]，同时把 dWo/db 累加进 Wo 缓存
        d_merged = self.Wo.backward(dout)
        # _from_heads 的反向 = _to_heads（两者互逆）：[B,T,d] → [B,H,T,d_k]
        dAttn = self._to_heads(d_merged)

        # ── 2. 注意力核心反向（公式见方法 docstring）──
        # dP = dAttn @ V^T：把注意力输出误差分回权重矩阵（矩阵乘法反向，换位相乘）
        dP = dAttn @ v_exp.transpose(0, 1, 3, 2)              # [B,H,T,Tk]
        # dV = P^T @ dAttn：权重矩阵把误差传给 V
        dV_exp = probs.transpose(0, 1, 3, 2) @ dAttn          # [B,H,Tk,d_k]
        # softmax 反向：P ⊙ (dP - Σ(dP⊙P))。归一化是行内操作，故沿 axis=-1 求和；
        # P ⊙ 是 softmax 反向自带的权重因子
        dScores = probs * (dP - np.sum(dP * probs, axis=-1, keepdims=True))  # [B,H,T,Tk]
        # dQ = dScores @ K * scale（K 形状 [Tk,d_k]，无需转置；前向乘过 scale，反向再乘一次）
        dQ = (dScores @ k_exp) * self.scale  # [B,H,T,d_k]
        # dK = dScores^T @ Q * scale
        dK_exp = (dScores.transpose(0, 1, 3, 2) @ q) * self.scale  # [B,H,Tk,d_k]

        # ── 3. _expand_kv 反向：复制变求和 ──
        # 前向按头轴复制 reps 份；反向把每份梯度按组相加，还原 n_kv_head 份
        if reps > 1:
            dK = dK_exp.reshape(B, self.n_kv_head, reps, Tk, Dk).sum(axis=2)
            dV = dV_exp.reshape(B, self.n_kv_head, reps, Tk, Dk).sum(axis=2)
        else:
            dK, dV = dK_exp, dV_exp                            # 无复制，原样直通

        # ── 4. 切头反向：把多头布局转回 [B,T,d] ──
        # _to_heads 的反向 = _from_heads（互逆）：[B,H,T,d_k] → [B,T,d]
        dQ_flat = self._from_heads(dQ)
        # _kv_heads 反向：transpose + reshape 转回 [B,Tk,n_kv_head*d_k]
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B, Tk, self.n_kv_head * Dk)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B, Tk, self.n_kv_head * Dk)

        # ── 5. QKV 投影反向（Linear.backward 自动累加 dW/db 并返回输入侧梯度）──
        dx_q = self.Wq.backward(dQ_flat)                       # [B,T,d]
        dx_k = self.Wk.backward(dK_flat)                       # [B,Tk,d]
        dx_v = self.Wv.backward(dV_flat)                       # [B,Tk,d]
        # 训练时无缓存（Tk == T），三个 dx 同形，相加即为输入梯度 dx
        return dx_q + dx_k + dx_v
