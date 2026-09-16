from .backend import np, onp, as_numpy  # 计算后端 np：默认 numpy(CPU)，MINIMAL_GPU=1 切 cupy(GPU)；onp 恒为原生 numpy（落盘用），as_numpy 显式搬运
from .layers import Linear, LayerNorm, GELU
from .attention import MultiHeadAttention


class MLP:
    """前馈网络（FFN）：Linear -> GELU -> Linear。

    Block 中的"逐位置加工"部分：注意力负责 token 之间的信息交换，
    MLP 负责每个 token 自身的非线性变换（升维 4 倍思考 → 降维回 d）。
    输入输出形状均为 [B, T, d]。
    """

    def __init__(self, d_model):
        # fc1 升维到 4*d_model（经验倍数，给模型更大"思考空间"），fc2 降维回 d_model
        self.fc1 = Linear(d_model, 4 * d_model)
        # GELU 提供非线性：两个 Linear 之间必须夹激活，否则两层线性等价于一层
        self.gelu = GELU()
        self.fc2 = Linear(4 * d_model, d_model)

    def forward(self, x):
        # 前向链路：y = fc2(GELU(fc1(x)))，形状 [B,T,d] -> [B,T,4d] -> [B,T,4d] -> [B,T,d]
        return self.fc2.forward(self.gelu.forward(self.fc1.forward(x)))

    def backward(self, dy):
        # 反向：与前向严格逆序，每个 backward 返回输入侧梯度传给上一层
        dy = self.fc2.backward(dy)
        dy = self.gelu.backward(dy)
        return self.fc1.backward(dy)

    def zero_grad(self):
        # 清空两个 Linear 的梯度缓存（backward 用 += 累加，训练每步前必须清零）
        self.fc1.zero_grad()
        self.fc2.zero_grad()

    def get_params(self):
        # 参数名带前缀 fc1./fc2.，保证整个 GPT 内全局唯一，便于优化器按名寻址
        out = [("fc1." + a, b) for a, b in self.fc1.get_params()]
        out += [("fc2." + a, b) for a, b in self.fc2.get_params()]
        return out

    def get_grad(self, name):
        # 按"fc1.W"这类名字取梯度：先按前缀取子模块，再转发剩余名字
        m, pn = name.split(".", 1)
        return getattr(self, m).get_grad(pn)

    def set_param(self, name, val):
        # 按名设参（加载 checkpoint 权重时使用）
        m, pn = name.split(".", 1)
        getattr(self, m).set_param(pn, val)


class Block:
    """Transformer 层：LN -> MHA -> 残差 -> LN -> MLP -> 残差（Pre-LN 布局）。

    残差连接 y = x + f(x)：x 原样抄一份到输出，f 只做增量。
    好处是反向时梯度有一条"高速路"（dy/dx = 1 + df/dx 中的 1），
    不经过任何变换原样传回，深层也能稳定训练。
    输入输出形状均为 [B, T, d]，GPT 由多个 Block 首尾堆叠而成。
    """

    def __init__(self, d_model, n_head):
        # ln1/ln2：注意力前、MLP 前的归一化（Pre-LN：先归一化再进子层）
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        # 残差加号前的输入快照（备用：backward 实际依赖各子模块自身缓存）
        self.x_pre_attn = None
        self.x_pre_mlp = None

    def forward(self, x, kv_cache=None):
        # 残差①：x = x + attn(ln1(x))。kv_cache 透传给注意力（解码生成时复用历史 K/V）
        self.x_pre_attn = x
        h = self.ln1.forward(x)
        x = x + self.attn.forward(h, kv_cache)
        # 残差②：x = x + mlp(ln2(x))
        self.x_pre_mlp = x
        h2 = self.ln2.forward(x)
        x = x + self.mlp.forward(h2)
        return x

    def backward(self, dy):
        # ── MLP 分支反向：dy 穿回 mlp -> ln2，再加回直通梯度 ──
        # 前向 x = x_pre_mlp + mlp(ln2(x_pre_mlp))，反向两路相加
        dx_mlp = self.mlp.backward(dy)          # dy 传回 mlp 输入侧（= ln2 输出）
        dx_mlp = self.ln2.backward(dx_mlp)      # 再传回 ln2 输入侧（= 残差加号前的 x）
        dx_mlp = dx_mlp + dy                    # + 直通分支：x_pre_mlp 的梯度
        # ── 注意力分支反向：同一份梯度（= 注意力输出的梯度）穿回 attn -> ln1 ──
        dx_attn = self.attn.backward(dx_mlp)    # dx_mlp 同时也是 attn 输出的梯度
        dx_attn = self.ln1.backward(dx_attn)    # 传回 ln1 输入侧（= 残差加号前的 x）
        return dx_attn + dx_mlp                 # 再次两路相加：x_pre_attn 的梯度

    def zero_grad(self):
        # 清空全部子模块梯度缓存（backward 用 += 累加，训练每步前必须清零）
        self.ln1.zero_grad()
        self.attn.zero_grad()
        self.ln2.zero_grad()
        self.mlp.zero_grad()

    def get_params(self):
        # 参数名带前缀 ln1./attn./ln2./mlp.，GPT 内全局唯一
        out = [("ln1." + a, b) for a, b in self.ln1.get_params()]
        out += [("attn." + a, b) for a, b in self.attn.get_params()]
        out += [("ln2." + a, b) for a, b in self.ln2.get_params()]
        out += [("mlp." + a, b) for a, b in self.mlp.get_params()]
        return out

    def get_grad(self, name):
        # 按名取梯度：先按前缀取子模块，再转发剩余名字
        m, pn = name.split(".", 1)
        return getattr(self, m).get_grad(pn)

    def set_param(self, name, val):
        # 按名设参（加载 checkpoint 权重时使用）
        m, pn = name.split(".", 1)
        getattr(self, m).set_param(pn, val)


class GPT:
    """完整 GPT：词嵌入 + 位置嵌入 + N 个 Block + 最终 LN + 输出投影。

    数据流：idx(B,T) -> tok_emb 查表 + pos_emb 注入位置 -> blocks 堆叠
            -> ln_f -> lm_head -> logits(B,T,V)。V 为词表大小。
    """

    def __init__(self, vocab_size, d_model, n_head, n_layer, ctx_len,
                 n_rhyme=0, use_tone=False):
        # 词嵌入矩阵：第 i 行 = 词表第 i 个 token 的向量（可学习参数，初始化缩放同 Linear）
        self.tok_emb = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        # 位置嵌入矩阵：第 i 行 = 绝对位置 i 的向量（可学习参数，小初始化 0.02）
        self.pos_emb = np.random.randn(ctx_len, d_model) * 0.02
        # N 个 Block 首尾堆叠：每层 = LN + MHA + 残差 + LN + MLP + 残差
        self.blocks = [Block(d_model, n_head) for _ in range(n_layer)]
        # 输出端：最后一层归一化 + 投影到词表大小 V 的 logits
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        # 辅助头开关：平仄头（2 分类）与韵部头（n_rhyme 分类）。禁用时一律不构造
        # Linear——构造会消耗全局 RNG 状态，从而破坏「同 seed 下禁用路径零回归」。
        self.use_tone = use_tone
        self.n_rhyme = n_rhyme
        self.tone_head = Linear(d_model, 2) if use_tone else None
        self.rhyme_head = Linear(d_model, n_rhyme) if n_rhyme > 0 else None
        # 辅助头前向缓存（forward 写出；禁用时为 None）
        self.tone_logits = None
        self.rhyme_logits = None
        # 辅助头 logits 梯度缓存（aux_loss 写出，backward 读取；已乘 λ 与 mask 权重）
        self.tone_dlogits = None
        self.rhyme_dlogits = None
        # 嵌入梯度缓存（backward 按 token 索引 scatter add 回填时使用）
        self.d_tok_emb = np.zeros_like(self.tok_emb)
        self.d_pos_emb = np.zeros_like(self.pos_emb)
        # 记录配置（生成解码时初始化 KV Cache 需要）
        self.vocab_size, self.d_model = vocab_size, d_model
        self.n_head, self.n_layer, self.ctx_len = n_head, n_layer, ctx_len

    def forward(self, idx, kv_cache=None):
        """前向：idx(B,T) -> logits(B,T,V)。kv_cache 为 None 走全量，否则走增量解码。"""
        B, T = idx.shape
        # 关键坑修复：解码时新 token 的绝对位置 = 已有缓存长度，而非 0
        offset = 0
        if kv_cache is not None:
            # kv_cache[0][0]：第 0 层 Block 的 K 缓存，形状 [B, n_kv_head, 历史长度, d_k]
            offset = kv_cache[0][0].shape[2]
        # 词嵌入查表 tok_emb[idx] -> [B,T,d]，加位置嵌入（按 offset 偏移取；
        # 训练/预填充时 offset=0，等价于 pos_emb[:T]）
        x = self.tok_emb[idx] + self.pos_emb[offset:offset + T]
        # 缓存输入信息（backward 回填嵌入梯度时使用）
        self.x_cache = (idx, T, offset)
        # 逐层过 Block；解码时每层各取自己的缓存（kv_cache[i]）
        for i, blk in enumerate(self.blocks):
            blk_cache = kv_cache[i] if kv_cache is not None else None
            x = blk.forward(x, blk_cache)
        # 末端隐藏态 h：lm_head 与辅助头共用同一份 h，辅助头据此投影但不回灌 LM 分支
        h = self.ln_f.forward(x)
        # 输出投影：每个位置对词表 V 的得分，形状 [B,T,V]
        self.logits = self.lm_head.forward(h)
        # 辅助头前向：仅在启用时计算（禁用时置 None，不引入任何额外运算，保持旧路径逐位一致）
        self.tone_logits = self.tone_head.forward(h) if self.use_tone else None
        self.rhyme_logits = self.rhyme_head.forward(h) if self.n_rhyme > 0 else None
        return self.logits

    def loss(self, logits, targets, weights=None):
        """加权交叉熵损失：loss = -Σ(w·log P(真实下一token)) / Σw（logsumexp 数值稳定版）。

        logits(B,T,V) 是模型对每个位置的 V 个预测得分；targets(B,T) 是每个位置的真实
        下一 token。softmax 概率满足 log p = logits - logsumexp(logits)，其中
        logsumexp 先减每行最大值 m，保证 exp 参数 <= 0 永不上溢，数值稳定。

        weights(B,T) 为逐位置权重，与 targets 逐位置对齐（预测第 i 个目标 token 的
        重要度）；weight=0 表示忽略该位置（分子分母均不计）。weights=None 等价全 1 权重，
        此时 Σw = 位置数 N，退化为不加权均值（与旧实现逐位一致）。

        weights 会被搬运到当前计算后端，故可传原生 numpy 数组（GPU 下自动拷入显存），
        无需调用方自行保证与 logits 同设备。
        """
        B, T, V = logits.shape
        flat = logits.reshape(-1, V)               # [B,T,V] -> [B*T, V]，逐位置独立算
        tflat = targets.reshape(-1)                # [B,T] -> [B*T]，与 flat 逐行对齐
        m = flat.max(axis=-1, keepdims=True)       # 每行最大得分（防溢出减数）
        lse = np.log(np.exp(flat - m).sum(axis=-1, keepdims=True)) + m  # logsumexp
        log_probs = flat - lse                     # 每行 = 该位置各 token 的 log 概率
        # backward 需要 softmax 概率（probs - one-hot 联合梯度）、真实 token 索引与逐位置权重
        self.probs = np.exp(log_probs)             # 缓存 softmax 概率
        self.targets_flat = tflat                  # 缓存真实 token 索引
        # 权重展平为 [B*T]：None -> 全 1（Σw=N，等价旧均值口径）；显式权重展平后与 targets 对齐
        wflat = np.ones(len(tflat)) if weights is None else np.asarray(weights).reshape(-1)
        self.pos_weights_flat = wflat              # 缓存逐位置权重（backward 按同口径回传）
        # 取各位置"真实 token"的 log 概率，按权重加权求和取负，再除以权重之和（非位置数）
        picked = log_probs[np.arange(len(tflat)), tflat]   # 各位置真实 token 的 log 概率
        loss = -(wflat * picked).sum() / wflat.sum()       # 分母 = Σw
        return loss

    def _masked_ce(self, logits, target, mask, lam):
        """带位置掩码的交叉熵：L = -Σ(mask·log p[target])/Σmask，返回 (L, dlogits)。

        分母为 mask 之和（非位置数）；mask=0 位置分子分母均不计。log p 用 logsumexp
        数值稳定式计算（同 GPT.loss 手法）。dlogits = (probs - one-hot)·mask/Σmask·λ
        为已乘 λ 的 logits 梯度；mask 全 0 时分子与梯度均为 0，不产生 NaN。
        """
        C = logits.shape[-1]
        flat = logits.reshape(-1, C)                             # [B,T,C] -> [B*T, C]
        tflat = np.asarray(target).reshape(-1)                  # 标签搬运到当前计算后端
        mflat = np.asarray(mask, dtype=np.float64).reshape(-1)  # 掩码搬运到后端并转浮点
        m = flat.max(axis=-1, keepdims=True)
        lse = np.log(np.exp(flat - m).sum(axis=-1, keepdims=True)) + m
        log_probs = flat - lse
        picked = log_probs[np.arange(len(tflat)), tflat]        # 各位置目标类别的 log 概率
        denom = mflat.sum()
        safe = np.maximum(denom, 1e-12)                         # 全 0 掩码时防止 0/0 = NaN
        loss = -(mflat * picked).sum() / safe                   # 分母 = Σmask
        # logits 梯度：(probs - one-hot)·mask/Σmask·λ（mask=0 位置梯度为 0）
        dflat = np.exp(log_probs)
        dflat[np.arange(len(tflat)), tflat] -= 1
        dflat *= (mflat / safe)[:, None]
        dflat *= lam
        return loss, dflat.reshape(*logits.shape)

    def aux_loss(self, tone_target=None, tone_mask=None, rhyme_target=None, rhyme_pos_mask=None,
                 lam_tone=0.3, lam_rhyme=0.5):
        """辅助头损失：平仄头与韵部头各自带掩码的交叉熵之和（λ 加权）。

        对齐口径：第 i 个位置的隐藏态预测第 i+1 个 token 的属性，标签由调用方
        （train.py）按此口径生成；本函数只做按位置掩码加权的交叉熵，不负责移位。
        逐头口径为 L = -Σ(mask·log p[target])/Σmask（分母为 mask 之和，非位置数），
        mask=0 位置分子分母均不计。回传用的 logits 梯度按
        (probs - one-hot)·mask/Σmask·λ 缓存在 self.tone_dlogits / self.rhyme_dlogits，
        供 backward 在 h 处相加。

        未启用的头直接跳过（贡献 0）；mask 全 0 的头损失为 0、相应头参数梯度全 0 且
        不产生 NaN。启用某头时必须同时提供其 target 与 mask，否则断言失败（防止
        mask 缺省被 np.asarray(None) 静默转成 NaN，导致损失与梯度无声出错）。
        tone_target/tone_mask/rhyme_target/rhyme_pos_mask 可为原生 numpy
        数组，经 np.asarray 搬运到当前计算后端（GPU 下自动拷入显存），无需调用方自行搬运。
        """
        total = 0.0
        if self.use_tone:
            assert tone_target is not None and tone_mask is not None, \
                "启用平仄头时必须提供 tone_target 与 tone_mask"
            L_tone, self.tone_dlogits = self._masked_ce(
                self.tone_logits, tone_target, tone_mask, lam_tone)
            total = total + lam_tone * L_tone
        else:
            self.tone_dlogits = None
        if self.n_rhyme > 0:
            assert rhyme_target is not None and rhyme_pos_mask is not None, \
                "启用韵部头时必须提供 rhyme_target 与 rhyme_pos_mask"
            L_rhyme, self.rhyme_dlogits = self._masked_ce(
                self.rhyme_logits, rhyme_target, rhyme_pos_mask, lam_rhyme)
            total = total + lam_rhyme * L_rhyme
        else:
            self.rhyme_dlogits = None
        return total

    def backward(self):
        """基于最后一次 loss() 回传，计算所有参数的梯度。

        softmax+CE 的加权梯度：d logits = (probs - one-hot)·w / Σw。
        回传链：lm_head -> ln_f -> blocks（逆序）-> 嵌入层。
        嵌入层 x = tok_emb[idx] + pos_emb[offset:offset+T]，其反向需 scatter add：
        同一 token 在多处出现时梯度要累加（np.add.at 自动处理重复索引）。
        """
        # softmax+CE 联合梯度：真实 token 处概率减 1（即 one-hot 差值）
        dflat = self.probs.copy()                  # 副本避免改坏缓存的 probs
        dflat[np.arange(len(self.targets_flat)), self.targets_flat] -= 1
        # 逐位置加权并除以权重之和：全 1 权重时 Σw = N，退化为旧的 /N 口径；
        # 权重 0 的位置乘积恒为 0，梯度贡献为 0（mask 语义）
        dflat *= self.pos_weights_flat[:, None]    # [B*T,V] · [B*T,1] 广播
        dflat /= self.pos_weights_flat.sum()       # 与 loss 的 /Σw 对应
        dy = dflat.reshape(*self.logits.shape)     # [B*T,V] -> [B,T,V]
        # 逐模块逆序回传：每个 backward 返回输入侧梯度，供上一层继续
        dy = self.lm_head.backward(dy)             # 输出投影层 -> h 处梯度
        # 辅助头梯度在 h 处相加：三路（lm_head + 平仄头 + 韵部头）汇合后才进 ln_f，
        # 保证 ln_f.backward 全程只调用一次；禁用时不加任何分支（与旧版逐位一致）
        if self.use_tone:
            dy = dy + self.tone_head.backward(self.tone_dlogits)
        if self.n_rhyme > 0:
            dy = dy + self.rhyme_head.backward(self.rhyme_dlogits)
        dy = self.ln_f.backward(dy)                # 最终 LayerNorm
        for blk in reversed(self.blocks):          # Block 从最后一层逐层往回
            dy = blk.backward(dy)
        # 嵌入层反向：梯度回填到词表与位置矩阵
        idx, T, offset = self.x_cache              # forward 缓存的输入信息
        self.d_tok_emb.fill(0.0)                   # 清零（scatter add 前必须清）
        np.add.at(self.d_tok_emb, idx, dy)         # 词嵌入：按 token 索引累加回填
        self.d_pos_emb[offset:offset + T] += dy.sum(axis=0)  # 位置嵌入：跨 batch 求和（广播加法反向）
        return dy

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 kv_cache=None, logits_processor=None):
        """自回归生成：从 idx [1,T] 续写 max_new_tokens 个 token，返回 [1, T+new]。

        kv_cache 为空则内部初始化（每层一份 [K,V] 空缓存）：prefill 一次把整段
        prompt 的 K/V 填满；之后每步只前向最后 1 个 token（增量解码，offset 自动
        取缓存长度，位置嵌入正确），采样出下一个 token 拼回序列，循环生成。

        logits_processor：可选回调 (logits[V], idx[1,T]) -> logits[V]，采样前对得分做
        一次加工；默认 None 时行为与旧版逐位一致（内核不感知任何外置规则）。
        """
        if kv_cache is None:
            head_dim = self.d_model // self.n_head    # 每头维度（与注意力层一致）
            # 每层缓存 [K, V] 两个空数组，长度 0 起步；MHA 下头数 = n_head
            kv_cache = [[np.zeros((idx.shape[0], self.n_head, 0, head_dim)),
                         np.zeros((idx.shape[0], self.n_head, 0, head_dim))]
                        for _ in range(self.n_layer)]
        self.forward(idx, kv_cache)                   # prefill：整段 prompt 前向，顺带填满缓存
        for _ in range(max_new_tokens):
            logits = self.forward(idx[:, -1:], kv_cache)  # decode：每次只喂最后 1 个 token
            # logits[0, -1, :]：B=1 生成，取末位置得分 -> 一维 [V]
            step_logits = logits[0, -1, :]
            if logits_processor is not None:
                # 通用钩子：采样前对得分做一次加工（句长约束等外置规则），内核不感知规则内容
                step_logits = logits_processor(step_logits, idx)
            next_token = _sample(step_logits, temperature, top_k, top_p)  # 采样 [1,1]
            idx = np.concatenate([idx, next_token], axis=1)   # 新 token 拼回序列
        return idx

    def _named_params(self):
        # 全模型参数按名收集：嵌入 + 每层 Block + 末端 LN/投影；名字全局唯一
        out = [("tok_emb", self.tok_emb), ("pos_emb", self.pos_emb)]
        for i, blk in enumerate(self.blocks):
            out += [(f"blocks.{i}." + a, b) for a, b in blk.get_params()]
        out += [("ln_f." + a, b) for a, b in self.ln_f.get_params()]
        out += [("lm_head." + a, b) for a, b in self.lm_head.get_params()]
        # 辅助头参数按序追加（仅启用时）；禁用时列表与旧版逐项相同
        if self.use_tone:
            out += [("tone_head." + a, b) for a, b in self.tone_head.get_params()]
        if self.n_rhyme > 0:
            out += [("rhyme_head." + a, b) for a, b in self.rhyme_head.get_params()]
        return out

    def _named_grads(self):
        # 与 _named_params 同名字同顺序的梯度表（优化器按名一一对应更新）
        out = [("tok_emb", self.d_tok_emb), ("pos_emb", self.d_pos_emb)]
        for i, blk in enumerate(self.blocks):
            out += [(f"blocks.{i}." + a, blk.get_grad(a)) for a, _ in blk.get_params()]
        out += [("ln_f." + a, self.ln_f.get_grad(a)) for a, _ in self.ln_f.get_params()]
        out += [("lm_head." + a, self.lm_head.get_grad(a)) for a, _ in self.lm_head.get_params()]
        # 辅助头梯度按同样顺序追加（仅启用时）；禁用时列表与旧版逐项相同
        if self.use_tone:
            out += [("tone_head." + a, self.tone_head.get_grad(a))
                    for a, _ in self.tone_head.get_params()]
        if self.n_rhyme > 0:
            out += [("rhyme_head." + a, self.rhyme_head.get_grad(a))
                    for a, _ in self.rhyme_head.get_params()]
        return out

    def dump_params(self):
        # 导出全部参数为 {名字: 副本}（拷贝存档，防止后续训练改动污染已存数据）
        return {name: p.copy() for name, p in self._named_params()}

    def load_params(self, d):
        # 按名写回参数（加载 checkpoint）；p[...]= 原地写入，保持对象引用不变。
        # 写回前经 np.asarray 搬进计算后端：GPU 下 np=cupy，把 numpy 显式拷入显存（隐式整片赋值会报错）
        for name, p in self._named_params():
            p[...] = np.asarray(d[name])

    def save(self, path):
        # 存 npz 文件：一键保存全模型参数（T5 训练产出的 checkpoint）。
        # 落盘永远用原生 numpy（onp），GPU 数组经 as_numpy 显式拷回 CPU 再存，保证文件跨设备可读
        onp.savez(path, **{name: as_numpy(a) for name, a in self.dump_params().items()})

    def load(self, path):
        # 读 npz 文件并写回全部参数（内部复用 load_params 完成设备搬运）
        self.load_params(onp.load(path))

    def zero_grad(self):
        # 清空全模型梯度（backward 用 += 累加，训练每步前必须清零）
        self.d_tok_emb.fill(0.0)
        self.d_pos_emb.fill(0.0)
        for blk in self.blocks:
            blk.zero_grad()
        self.ln_f.zero_grad()
        self.lm_head.zero_grad()
        # 辅助头梯度同样清零（仅启用时；禁用时无此模块）
        if self.use_tone:
            self.tone_head.zero_grad()
        if self.n_rhyme > 0:
            self.rhyme_head.zero_grad()


def _sample(logits, temperature=1.0, top_k=None, top_p=None):
    """按 logits 概率采样一个 token：输入 [V] 得分，返回 [1,1] 的 token 数组。

    temperature：softmax 前把 logits 除以 T。T<1 放大得分差距（分布更尖/更自信），
    T>1 缩小差距（更平/更随机）。top_k：只保留得分前 k 个重归一化后采样，砍长尾；
    top_p：保留累积概率达 p 的最小候选集（核采样，候选数自适应）。三者均可叠加。
    """
    if temperature != 1.0:
        logits = logits / temperature
    m = logits.max()                                  # 减最大值防 exp 溢出
    probs = np.exp(logits - m)
    probs /= probs.sum()                              # softmax -> 概率分布
    if top_k is not None:
        k = min(top_k, len(probs))
        idx = np.argpartition(-probs, k - 1)[:k]      # 得分最高的前 k 个下标
        p2 = probs[idx]
        p2 /= p2.sum()                                # 砍掉长尾后重归一化
        # size=1 显式给大小：cupy 的 choice 不支持不带 size（numpy 可省），统一写法兼容两后端
        t = np.random.choice(idx, size=1, p=p2)[0]
        return np.array([[t]])
    if top_p is not None:
        order = np.argsort(-probs)                    # 概率从高到低排序
        p_sorted = probs[order]
        cum = np.cumsum(p_sorted)                     # 累积概率
        keep = cum - p_sorted <= top_p                # 含自身在内累计达 p 的最小集合
        keep_idx = order[keep]
        p2 = probs[keep_idx]
        p2 /= p2.sum()                                # 重归一化
        t = np.random.choice(keep_idx, size=1, p=p2)[0]
        return np.array([[t]])
    t = np.random.choice(len(probs), size=1, p=probs)[0]   # 原始分布直接采样
    return np.array([[t]])
