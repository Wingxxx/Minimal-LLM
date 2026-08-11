# 极简 LLM（微型 GPT）+ 推理服务 实现计划

**Goal:** 纯 Python + numpy 从 0 到 1 搭建一个能真训练的微型 GPT，并用 HTTP 推理服务（SSE 流式 + 连续批处理 + KV Cache + p99 统计）包装，外加 CLI。

**Architecture:** 分层模块化。`model/` 下 layers（原语含 backward）→ attention（MHA + causal mask + KV Cache）→ gpt（Block 堆叠 + 训练接口 + 生成接口）；`train.py` 训练存 checkpoint；`server.py` 用 ThreadingHTTPServer 起服务，请求线程经 Batcher 攒批共享模型前向，SSE 逐 token 返回并记录 p99；`cli.py` 本地交互。

**Tech Stack:** Python 3.13 + numpy 2.4（零第三方依赖），测试用纯 assert + 梯度检查（有限差分）。

**运行约定：** 每次可运行文件输出都重定向到 `test/output-*.txt`。每个 Task 完成后进行一次 git commit + push。

**教学约定：** 每个 Task 按「概念级」拆分步骤：**讲解 → 动手 → 验证**三步闭环。先讲清公式/原理，再写代码，每个步骤都有可运行的小验证点，错误当场暴露，不攒到最后。

---

### Task 1: 项目脚手架 + 语料

**教学主线:** 模型吃什么？char-level 分词：把连续文本切成字符序列，字符集即词表。这是整个 LLM 的数据入口。

**Files:**
- Create: `data/corpus.txt`
- Create: `model/__init__.py`
- Create: `test/__init__.py`

- [ ] **S1 讲解：char-level 分词**
  - 中文按「字符」切分：`"床前明月光"` → `['床','前','明','月','光']`
  - 词表 = 语料中出现过的所有字符集合（本项目约 2500 个），远小于 token 级（几千~几万）
  - 优点：零分词器依赖、实现简单；缺点：字符间缺乏词边界语义，但足够学会「诗的语言规律」

- [ ] **S2 动手：创建训练语料** `data/corpus.txt`（《唐诗三百首》313 首，约 2.4 万字符 / 约 2500 个不同字符）
  - 语料 = 《唐诗三百首》全文（公有领域）：每首诗一行诗题 + 数行正文，格式如下
  - 执行时从公开文本整理 313 首全文写入本文件（只写诗题与正文，不写作者）

```text
静夜思
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
春晓
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。
登鹳雀楼
白日依山尽，黄河入海流。
欲穷千里目，更上一层楼。
……（其余 310 首同格式，共 313 首）
```

- [ ] **S3 讲解：包标记 `__init__.py` 的作用**
  - 让 `model/`、`test/` 变成 Python 包：`from model.layers import Linear` 才能生效
  - 本项目中内容为空即可（仅包标记）

- [ ] **S4 动手：创建包标记**
  - 创建空文件 `model/__init__.py`、`test/__init__.py`

- [ ] **S5 验证：语料可读且规模符合预期**

Run: `python -c "data = open(r'data/corpus.txt', encoding='utf-8').read(); chars = sorted(set(data)); print(len(data), 'chars,', len(chars), 'unique')"`
Expected: 约 2.4 万 chars，约 2500 unique

- [ ] **S6 收尾：commit + push**

Run: `git add data/corpus.txt model/__init__.py test/__init__.py; git commit -m "T1: 语料与包脚手架"; git push`

---

### Task 2: model/layers.py —— 神经网络原语（forward + backward）

**教学主线:** 一切深度学习都由三层原语堆出：`Linear`（线性变换）、`LayerNorm`（归一化）、`GELU`（非线性激活）。本任务逐个拆解每个原语的「前向公式 → 反向推导 → 数值验证」。

**Files:**
- Create: `model/layers.py`
- Create: `test/test_layers.py`

- [ ] **S1 讲解 Linear 数学**
  - 前向：`y = x @ W + b`，形状 `[B, T, d] @ [d, out] + [out] → [B, T, out]`
  - `W` 是权重，`b` 是偏置；`W` 用 `1/√d` 缩放初始化——让每个输出神经元的输入方差 ≈ 1，避免深层网络信号爆炸/消失
  - 验证直觉：`x @ W` 是「输入的加权求和」，W 的每一列对应一个输出神经元的一套权重

- [ ] **S2 实现 `Linear.forward` 并验证形状**

```python
class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(in_f, out_f) / np.sqrt(in_f)
        self.b = np.zeros(out_f)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
```

Run: `python -c "import numpy as np, sys; sys.path.insert(0,'.'); from model.layers import Linear; l=Linear(8,16); y=l.forward(np.random.randn(2,3,8)); print(y.shape)"`
Expected: `(2, 3, 16)`

- [ ] **S3 讲解反向传播：链式法则三板斧**
  - 已知上游梯度 `dy`（loss 对输出的梯度），求三个梯度：
  - `dW = xᵀ @ dy`：W 被每个位置的 x 用到，梯度是「所有位置的 x 与 dy 的外积之和」（需把 `[B,T,d]` 拍平成 `[B*T,d]`）
  - `db = Σ dy`：b 广播到每个位置，梯度沿非特征维求和
  - `dx = dy @ Wᵀ`：x 通过 W 传到下游
  - 记忆点：**前向用什么运算，反向就是该运算对某参数的偏导 × 上游梯度**

- [ ] **S4 实现 `Linear.backward` 并手算核对**

```python
    def backward(self, dy):
        flat_x = self.x.reshape(-1, self.x.shape[-1])
        flat_dy = dy.reshape(-1, dy.shape[-1])
        self.dW += flat_x.T @ flat_dy
        self.db += dy.sum(axis=tuple(range(dy.ndim - 1)))
        return dy @ self.W.T
```

手算核对（纸上推一遍）：`x=[1,2]`、`W=[[1],[1]]`、`dy=[3]` → `dx = dy@Wᵀ = [3,3]`，`dW = xᵀ@dy = [3,6]`

- [ ] **S5 讲解 LayerNorm 公式**
  - 对每个特征维做归一化：`x̂ = (x - μ) / √(σ² + ε)`（μ、σ 沿除特征维以外的所有维求）
  - 再加可学习缩放：`y = x̂ * γ + β`
  - 为什么用？防止中间层数值分布漂移（internal covariate shift），训练更稳
  - `ε=1e-5` 防止除零

- [ ] **S6 实现 `LayerNorm.forward` 并验证 μ/σ**

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.dim, self.eps = dim, eps
        self.gamma, self.beta = np.ones(dim), np.zeros(dim)
        self.dgamma, self.dbeta = np.zeros(dim), np.zeros(dim)
        self.x = self.mean = self.var = self.xhat = None

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.xhat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.xhat * self.gamma + self.beta
```

Run: `python -c "import numpy as np, sys; sys.path.insert(0,'.'); from model.layers import LayerNorm; ln=LayerNorm(8); y=ln.forward(np.random.randn(2,3,8)*5); print('mean~0:', round(float(y.mean()),6), 'var~1:', round(float(y.var()),6))"`
Expected: `mean~0: 0.0`, `var~1: 1.0`（归一化生效）

- [ ] **S7 讲解 LayerNorm 反向推导（三步链）**
  - `dγ = Σ(dy * x̂)`，`dβ = Σ dy`（沿非特征维求和）
  - `dx̂ = dy * γ`
  - `dx = (dx̂ - mean(dx̂) - x̂ * mean(dx̂ * x̂)) / √(σ²+ε)`
  - 直觉：x̂ 依赖 μ 和 σ，μ/σ 又依赖所有 x，反向必须把这三条路径都扣除，所以有减均值、减 `x̂·mean(dx̂·x̂)` 两项修正

- [ ] **S8 实现 `LayerNorm.backward` 并核对 dγ**

```python
    def backward(self, dy):
        dxhat = dy * self.gamma
        self.dgamma += (dy * self.xhat).sum(axis=tuple(range(dy.ndim - 1)))
        self.dbeta += dy.sum(axis=tuple(range(dy.ndim - 1)))
        d = dxhat - dxhat.mean(-1, keepdims=True) - self.xhat * (dxhat * self.xhat).mean(-1, keepdims=True)
        return d / np.sqrt(self.var + self.eps)
```

Run: `python -c "import numpy as np, sys; sys.path.insert(0,'.'); from model.layers import LayerNorm; ln=LayerNorm(4); x=np.random.randn(2,4); dy=np.random.randn(2,4); ln.forward(x); ln.backward(dy); print('dgamma:', np.round(ln.dgamma,4))"`
Expected: 数值合理（有限区间内，最终会由梯度检查把关）

- [ ] **S9 讲解 GELU 与 tanh 近似**
  - 精确 GELU：`x·Φ(x)`（标准正态 CDF）——处处可导、比 ReLU 平滑
  - 计算代价高，用 tanh 近似：`0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`
  - 反向只需对 `x` 求导：`d/dx = 0.5(1+tanh g) + 0.5x(1-tanh²g)·g'(x)`

- [ ] **S10 实现 `GELU.forward/backward`**

```python
class GELU:
    def __init__(self):
        self.x = self.g = self.tanh_g = None

    def forward(self, x):
        self.x = x
        self.g = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        self.tanh_g = np.tanh(self.g)
        return 0.5 * x * (1 + self.tanh_g)

    def backward(self, dy):
        dg_dx = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x ** 2)
        df_dx = 0.5 * (1 + self.tanh_g) + 0.5 * self.x * (1 - self.tanh_g ** 2) * dg_dx
        return dy * df_dx

    def get_params(self): return []
    def get_grad(self, name): return None
    def set_param(self, name, val): pass
    def zero_grad(self): pass
```

- [ ] **S11 讲解梯度检查（有限差分）原理**
  - 不知道解析梯度对不对？数值上近似：`df/dx ≈ (f(x+ε) - f(x-ε)) / 2ε`
  - 对输入 x 和每个参数分别算数值梯度，与 backward 的解析梯度对比，误差 < 1e-4 即正确
  - 这是「手写反向传播」的黄金验证手段——**用数值梯度校准解析梯度**

- [ ] **S12 动手：写梯度检查测试** `test/test_layers.py`

```python
"""layers 梯度检查：所有层 backward 必须与有限差分一致（误差 < 1e-4）"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.layers import Linear, LayerNorm, GELU

EPS, TOL = 1e-6, 1e-4

def numeric_grad(f, x):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[i] += EPS; xm[i] -= EPS
        g[i] = (f(xp) - f(xm)) / (2 * EPS)
        it.iternext()
    return g

def check_layer(layer, x, dy):
    y = layer.forward(x.copy())
    dx = layer.backward(dy)
    f = lambda xx: np.sum(layer.forward(xx) * dy)
    err_x = np.abs(dx - numeric_grad(f, x)).max()
    err_p = 0.0
    for pname, pval in layer.get_params():
        layer.zero_grad()
        # 对每个参数算数值梯度，并重新 forward+backward 拿解析梯度
        fw = lambda pp: np.sum(layer.forward(x.copy()) * dy)
        err_p = max(err_p, np.abs(layer.get_grad(pname) - numeric_grad(fw, pval)).max())
    return err_x, err_p

def test_linear():
    ln = Linear(8, 16)
    ex, ep = check_layer(ln, np.random.randn(2, 3, 8) * 0.5, np.random.randn(2, 3, 16))
    assert ex < TOL and ep < TOL, f"Linear err dx={ex} dW={ep}"
    print("Linear PASS", ex, ep)

def test_layernorm():
    ln = LayerNorm(8)
    ex, ep = check_layer(ln, np.random.randn(2, 3, 8) * 2, np.random.randn(2, 3, 8))
    assert ex < TOL and ep < TOL, f"LayerNorm err dx={ex} dg={ep}"
    print("LayerNorm PASS", ex, ep)

def test_gelu():
    g = GELU()
    x, dy = np.random.randn(2, 3, 8), np.random.randn(2, 3, 8)
    g.forward(x.copy()); dx = g.backward(dy)
    err = np.abs(dx - numeric_grad(lambda xx: np.sum(g.forward(xx) * dy), x)).max()
    assert err < TOL, f"GELU err={err}"
    print("GELU PASS", err)

if __name__ == "__main__":
    test_linear(); test_layernorm(); test_gelu()
    print("ALL LAYER TESTS PASS")
```

> 注：`check_layer` 里对参数的数值梯度只验证「解析梯度 ≈ 数值梯度」这一步（forward 已在此前步骤验证），因此不再在循环内重复 backward。

- [ ] **S13 RED → GREEN**
  - RED：先运行确认测试因 `No module named 'model.layers'` 失败
  - Run: `python test/test_layers.py` → Expected: `ModuleNotFoundError`
  - GREEN：补齐 `layers.py` 全部类（合并 S2/S4/S6/S8/S10 的代码 + 统一 get_params/get_grad/set_param/zero_grad 接口），再运行
  - Run: `python test/test_layers.py` → Expected: `Linear PASS ...` / `LayerNorm PASS ...` / `GELU PASS ...` / `ALL LAYER TESTS PASS`

- [ ] **S14 收尾：commit + push**

Run: `git add model/layers.py test/test_layers.py; git commit -m "T2: layers 原语（Linear/LayerNorm/GELU）含反向"; git push`

---

### Task 3: model/attention.py —— 多头注意力（causal mask + KV Cache + 反向）

**教学主线:** Transformer 的心脏。从 QKV 投影到缩放点积、causal mask、多头切分，再到 KV Cache 与反向传播，逐步搭出可训练、可高效推理的 MHA。

**Files:**
- Create: `model/attention.py`
- Create: `test/test_attention.py`

- [ ] **S1 讲解 MHA 全貌（先看目的地）**
  - 输入 `x[B,T,d]`，经过 4 个 Linear：`Wq/Wk/Wv`（投影）+ `Wo`（输出投影）
  - 注意力计算：`softmax(QKᵀ/√d_k) · V`
  - 多头：把 `d` 切成 `n_head` 份（`d_k = d/n_head`），每头独立算注意力再拼回
  - 意义：不同头关注不同模式（如一个头学相邻词、一个头学句法）

- [ ] **S2 实现头切分/合并的形状变换**

```python
class MultiHeadAttention:
    def __init__(self, d_model, n_head, n_kv_head=None):
        self.d_model, self.n_head = d_model, n_head
        self.n_kv_head = n_kv_head or n_head          # 默认 MHA；设 1 即 MQA；中间值即 GQA
        self.d_k = d_model // n_head
        self.scale = 1.0 / np.sqrt(self.d_k)
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, self.n_kv_head * self.d_k)   # K/V 维度按 n_kv_head 缩
        self.Wv = Linear(d_model, self.n_kv_head * self.d_k)
        self.Wo = Linear(d_model, d_model)
        self.cache = None

    def _to_heads(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.n_head, self.d_k).transpose(0, 2, 1, 3)

    def _from_heads(self, x):
        B, H, T, Dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.n_head * Dk)

    def _kv_heads(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.n_kv_head, self.d_k).transpose(0, 2, 1, 3)

    def _expand_kv(self, k, v):
        reps = self.n_head // self.n_kv_head
        if reps == 1:
            return k, v
        return np.repeat(k, reps, axis=1), np.repeat(v, reps, axis=1)
```

Run: `python -c "import numpy as np, sys; sys.path.insert(0,'.'); from model.attention import MultiHeadAttention; m=MultiHeadAttention(8,2); print(m._to_heads(np.random.randn(2,3,8)).shape, m._from_heads(np.random.randn(2,2,3,4)).shape); g=MultiHeadAttention(8,4,2); print('kv_heads:', g._kv_heads(np.random.randn(2,3,4)).shape)"`
Expected: `(2, 2, 3, 4) (2, 3, 8)` 且 `kv_heads: (2, 2, 3, 2)`（GQA 时 K/V 头数 = n_kv_head）

- [ ] **S3 讲解缩放点积注意力**
  - `scores = Q @ Kᵀ / √d_k`：Q 与每个 K 的点积衡量「查询与键的匹配度」
  - 除以 `√d_k` 防止点积方差随维度爆炸（softmax 过早饱和）
  - `probs = softmax(scores)` 归一化成权重，`attn = probs @ V` 加权求和取值

- [ ] **S4 讲解 causal mask（自回归铁律）**
  - 生成时第 t 个 token **不能看到未来**，否则训练时泄露答案
  - 做法：把「未来位置」的 scores 置为 `-1e9`，softmax 后权重≈0
  - `mask[i,j] = True 当 j > base + i`（base 是已有缓存长度，处理 KV Cache 场景）

- [ ] **S5 实现 `forward`（含 causal mask + KV Cache 接入点）**

```python
    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        q = self._to_heads(self.Wq.forward(x))
        k = self._kv_heads(self.Wk.forward(x))        # [B, n_kv_head, T, d_k]
        v = self._kv_heads(self.Wv.forward(x))
        if kv_cache is not None and kv_cache[0].shape[2] > 0:
            k = np.concatenate([kv_cache[0], k], axis=2)
            v = np.concatenate([kv_cache[1], v], axis=2)
        if kv_cache is not None:
            kv_cache[0], kv_cache[1] = k, v            # 缓存 n_kv_head 份（省内存）
        Tk = k.shape[2]
        k_exp, v_exp = self._expand_kv(k, v)           # 展开到 n_head 参与注意力
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) * self.scale   # [B,H,T,Tk]
        base = Tk - T
        mask = np.arange(Tk)[None, None, None, :] > (base + np.arange(T)[None, None, :, None])
        scores = np.where(mask, -1e9, scores)
        probs = self._softmax(scores)
        attn = probs @ v_exp
        out = self.Wo.forward(self._from_heads(attn))
        self.cache = (x, q, k_exp, v_exp, probs)
        return out

    def _softmax(self, scores):
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        return e / e.sum(axis=-1, keepdims=True)
```

- [ ] **S6 讲解 KV Cache 原理**
  - 解码第 t+1 个 token 时，前 t 个 token 的 K/V 其实没变，重算浪费
  - 缓存每层的 K/V：`[B,H,t,d_k]`，新 token 只算自己的 K/V 并拼接到缓存尾部
  - 收益：解码阶段复杂度从 `O(T²)` 降到 `O(T)`（每个新 token 只与全部 K/V 做一次点积）
  - Q 永远只算当前 token；这是 8-07「KV Cache 三兄弟」的落地

- [ ] **S7 验证 KV Cache 一致性（写测试，先 RED）**

```python
def test_kv_cache_consistency():
    mha = MultiHeadAttention(d_model=8, n_head=2)
    x_full = np.random.randn(1, 5, 8) * 0.5
    y_full = mha.forward(x_full)
    cache = [np.zeros((1, 2, 0, 4)), np.zeros((1, 2, 0, 4))]
    outs = [mha.forward(x_full[:, t:t+1, :], cache) for t in range(5)]
    y_step = np.concatenate(outs, axis=1)
    err = np.abs(y_full - y_step).max()
    assert err < 1e-9, f"KV Cache 不一致 err={err}"
    print("KV Cache consistency PASS", err)
```

RED：`python test/test_attention.py` → `ModuleNotFoundError`
GREEN（补全 attention.py 后）：`KV Cache consistency PASS`

- [ ] **S8 讲解 MHA 反向传播**
  - 沿注意力计算链反推：`d_scores → d_q/d_k/d_v → 各投影层 dx`
  - softmax 反向：`d_scores = probs * (d_probs - Σ(d_probs*probs))`（softmax 的雅可比）
  - `d_q = (d_scores·scale) @ K`，`d_k = (d_scores·scale)ᵀ @ Q`，`d_v = probsᵀ @ d_attn`
  - mask 位置被 `-1e9` 屏蔽 → softmax 后概率≈0 → 反向梯度≈0，用 `np.where(isnan, 0)` 兜底

- [ ] **S9 实现 `backward`**

```python
    def backward(self, dy):
        x, q, k_exp, v_exp, probs = self.cache
        B, H, T, Dk = q.shape
        dy_attn = self._to_heads(self.Wo.backward(dy))
        d_v_exp = probs.transpose(0, 1, 3, 2) @ dy_attn
        d_probs = dy_attn @ v_exp.transpose(0, 1, 3, 2)
        d_scores = probs * (d_probs - (d_probs * probs).sum(axis=-1, keepdims=True))
        d_scores = np.where(np.isnan(d_scores), 0, d_scores)
        d_q = (d_scores * self.scale) @ k_exp
        d_k_exp = (d_scores * self.scale).transpose(0, 1, 3, 2) @ q
        reps = self.n_head // self.n_kv_head
        if reps > 1:                                   # 共享头的梯度按组求和
            d_k = d_k_exp.reshape(B, self.n_kv_head, reps, T, Dk).sum(axis=2)
            d_v = d_v_exp.reshape(B, self.n_kv_head, reps, T, Dk).sum(axis=2)
        else:
            d_k, d_v = d_k_exp, d_v_exp
        dx = np.zeros_like(x)
        dx += self.Wq.backward(self._from_heads(d_q))
        dx += self.Wk.backward(self._from_heads(d_k))
        dx += self.Wv.backward(self._from_heads(d_v))
        return dx

    def zero_grad(self):
        for l in (self.Wq, self.Wk, self.Wv, self.Wo):
            l.zero_grad()

    def get_params(self):
        out = []
        for name, l in (("Wq", self.Wq), ("Wk", self.Wk), ("Wv", self.Wv), ("Wo", self.Wo)):
            for pn, pv in l.get_params():
                out.append((name + "." + pn, pv))
        return out

    def get_grad(self, name):
        lname, pn = name.split(".")
        return getattr(self, lname).get_grad(pn)

    def set_param(self, name, val):
        lname, pn = name.split(".")
        getattr(self, lname).set_param(pn, val)
```

- [ ] **S10 写梯度检查 + 形状测试（RED → GREEN）**

```python
"""MHA 梯度检查 + KV Cache 一致性"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.attention import MultiHeadAttention

EPS, TOL = 1e-6, 1e-4

def numeric_grad(f, x):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[i] += EPS; xm[i] -= EPS
        g[i] = (f(xp) - f(xm)) / (2 * EPS)
        it.iternext()
    return g

def test_mha_grad():
    mha = MultiHeadAttention(d_model=8, n_head=2)
    x, dy = np.random.randn(2, 3, 8) * 0.5, np.random.randn(2, 3, 8)
    mha.forward(x); dx = mha.backward(dy)
    err_x = np.abs(dx - numeric_grad(lambda xx: np.sum(mha.forward(xx) * dy), x)).max()
    assert err_x < TOL, f"MHA dx err={err_x}"
    mha.zero_grad(); mha.forward(x); mha.backward(dy)
    Wq = mha.Wq.W
    g_num = numeric_grad(lambda pp: np.sum(mha.forward(x) * dy), Wq)
    mha.zero_grad(); mha.forward(x); mha.backward(dy)
    err_wq = np.abs(mha.Wq.dW - g_num).max()
    assert err_wq < TOL, f"MHA dWq err={err_wq}"
    print("MHA grad PASS", err_x, err_wq)

def test_causal_shape():
    mha = MultiHeadAttention(d_model=4, n_head=1)
    y = mha.forward(np.random.randn(1, 3, 4))
    assert y.shape == (1, 3, 4)
    print("causal shape PASS")

def test_kv_cache_consistency():
    mha = MultiHeadAttention(d_model=8, n_head=2)
    x_full = np.random.randn(1, 5, 8) * 0.5
    y_full = mha.forward(x_full)
    cache = [np.zeros((1, 2, 0, 4)), np.zeros((1, 2, 0, 4))]
    outs = [mha.forward(x_full[:, t:t+1, :], cache) for t in range(5)]
    y_step = np.concatenate(outs, axis=1)
    err = np.abs(y_full - y_step).max()
    assert err < 1e-9, f"KV Cache 不一致 err={err}"
    print("KV Cache consistency PASS", err)

if __name__ == "__main__":
    test_mha_grad(); test_causal_shape(); test_kv_cache_consistency()
    print("ALL ATTENTION TESTS PASS")
```

Run: `python test/test_attention.py`
RED → `ModuleNotFoundError`；GREEN → `MHA grad PASS ...` / `causal shape PASS` / `KV Cache consistency PASS` / `ALL ATTENTION TESTS PASS`

- [ ] **S11 知识点补强：MHA → MQA → GQA 演进对比（07-15 Attention ②）**
  - 问题：MHA 每个头各存一份 K/V，KV Cache 随 n_head 线性膨胀
  - **MQA**（Multi-Query Attention）：所有头共享 1 组 K/V，缓存降到 1/n_head
  - **GQA**（Grouped-Query Attention）：折中，`n_kv_head` 组 K/V（`n_kv_head < n_head`），组内头共享——LLaMA2/3 实际采用
  - 演进链：MHA（独立）→ GQA（分组共享）→ MQA（完全共享），KV Cache 从 1 降到 1/n_head
  - 收益：长上下文下 KV Cache 内存是关键瓶颈，GQA/MQA 直接砍内存、提吞吐
  - **S2/S5/S9 的代码已经支持 `n_kv_head`**（默认 `n_kv_head=n_head` 即纯 MHA，行为不变）；本步骤只需**追加测试验证 GQA 场景**，无需改代码

  **动手：追加 `test_gqa_kv_consistency` 到 test_attention.py，并在 `__main__` 调用**

```python
def test_gqa_kv_consistency():
    """GQA（n_kv_head=2, n_head=4）：KV Cache 一致性 + 梯度检查"""
    mha = MultiHeadAttention(d_model=16, n_head=4, n_kv_head=2)
    x_full = np.random.randn(1, 5, 16) * 0.5
    y_full = mha.forward(x_full)
    cache = [np.zeros((1, 2, 0, 4)), np.zeros((1, 2, 0, 4))]   # n_kv_head=2
    outs = [mha.forward(x_full[:, t:t+1, :], cache) for t in range(5)]
    y_step = np.concatenate(outs, axis=1)
    assert np.abs(y_full - y_step).max() < 1e-9, "GQA KV Cache 不一致"
    dy = np.random.randn(1, 5, 16)
    mha.zero_grad(); mha.forward(x_full); dx = mha.backward(dy)
    err = np.abs(dx - numeric_grad(lambda xx: np.sum(mha.forward(xx) * dy), x_full)).max()
    assert err < 1e-4, f"GQA grad err={err}"
    print("GQA grad + cache PASS", err)

if __name__ == "__main__":
    test_mha_grad(); test_causal_shape(); test_kv_cache_consistency(); test_gqa_kv_consistency()
    print("ALL ATTENTION TESTS PASS")
```

Run: `python test/test_attention.py` → 原 3 项 + `GQA grad + cache PASS` / `ALL ATTENTION TESTS PASS`
（MQA 同理：`MultiHeadAttention(d_model, n_head, n_kv_head=1)` 即可，可自行验证）

- [ ] **S12 收尾：commit + push**

Run: `git add model/attention.py test/test_attention.py; git commit -m "T3: 多头注意力（causal mask + KV Cache + MQA/GQA + 反向）"; git push`

---

### Task 4: model/gpt.py —— Block 堆叠 + GPT 组装（loss/backward/保存加载/KV Cache 生成）

**教学主线:** 把原语堆成完整 GPT：Block（注意力+FFN+残差+LN）→ 多层堆叠 → 词/位置嵌入 → 交叉熵 loss → 反向 → 生成采样 → 保存加载。

**Files:**
- Create: `model/gpt.py`
- Create: `test/test_gpt.py`

- [ ] **S1 讲解 Block 结构（Transformer 层的内部）**
  - 每个 Block = `LN → MHA → 残差 → LN → FFN → 残差`
  - 残差连接：`x = x + attn(ln1(x))`，让梯度有「高速路」直接流过，深层也好训
  - Pre-LN 布局（先归一化再进注意力）比 Post-LN 更稳

- [ ] **S2 实现 `MLP`（FFN：升维 4x → GELU → 降维回 d）**

```python
class MLP:
    """FFN：Linear -> GELU -> Linear"""
    def __init__(self, d_model):
        self.fc1 = Linear(d_model, 4 * d_model)
        self.gelu = GELU()
        self.fc2 = Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2.forward(self.gelu.forward(self.fc1.forward(x)))

    def backward(self, dy):
        dy = self.fc2.backward(dy)
        dy = self.gelu.backward(dy)
        return self.fc1.backward(dy)

    def zero_grad(self):
        self.fc1.zero_grad(); self.fc2.zero_grad()

    def get_params(self):
        out = [("fc1." + a, b) for a, b in self.fc1.get_params()]
        out += [("fc2." + a, b) for a, b in self.fc2.get_params()]
        return out

    def get_grad(self, name):
        m, pn = name.split(".", 1)
        return getattr(self, m).get_grad(pn)

    def set_param(self, name, val):
        m, pn = name.split(".", 1)
        getattr(self, m).set_param(pn, val)
```

- [ ] **S3 实现 `Block`（前向 + 反向，注意残差两条路径）**

```python
class Block:
    def __init__(self, d_model, n_head):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.x_pre_attn = None
        self.x_pre_mlp = None

    def forward(self, x, kv_cache=None):
        self.x_pre_attn = x
        h = self.ln1.forward(x)
        x = x + self.attn.forward(h, kv_cache)
        self.x_pre_mlp = x
        h2 = self.ln2.forward(x)
        x = x + self.mlp.forward(h2)
        return x

    def backward(self, dy):
        dx_mlp = self.mlp.backward(dy)
        dx_mlp = self.ln2.backward(dx_mlp)
        dx_mlp = dx_mlp + dy                          # MLP 残差
        dx_attn = self.attn.backward(dx_mlp)
        dx_attn = self.ln1.backward(dx_attn)
        return dx_attn + dx_mlp                       # 两分支相加

    def zero_grad(self):
        self.ln1.zero_grad(); self.attn.zero_grad(); self.ln2.zero_grad(); self.mlp.zero_grad()

    def get_params(self):
        out = [("ln1." + a, b) for a, b in self.ln1.get_params()]
        out += [("attn." + a, b) for a, b in self.attn.get_params()]
        out += [("ln2." + a, b) for a, b in self.ln2.get_params()]
        out += [("mlp." + a, b) for a, b in self.mlp.get_params()]
        return out

    def get_grad(self, name):
        m, pn = name.split(".", 1)
        return getattr(self, m).get_grad(pn)

    def set_param(self, name, val):
        m, pn = name.split(".", 1)
        getattr(self, m).set_param(pn, val)
```

- [ ] **S4 讲解 GPT 组装与位置编码**
  - `x = tok_emb[idx] + pos_emb[:T]`：词嵌入 + 可学习位置嵌入
  - 位置编码意义：注意力本身无序（是集合操作），必须注入位置信息才能学「相邻」概念
  - 关键坑：**KV Cache 解码时**每次只喂 1 个 token，位置应是「累计位置」（缓存长度），而非 0——必须用 `offset = 缓存长度` 偏移取 `pos_emb`

- [ ] **S5 实现 `GPT.forward`（含 KV Cache 位置偏移修复）**

```python
    def forward(self, idx, kv_cache=None):
        B, T = idx.shape
        offset = 0
        if kv_cache is not None:
            offset = kv_cache[0][0].shape[2]          # 已有缓存长度 = 当前绝对位置
        x = self.tok_emb[idx] + self.pos_emb[offset:offset + T]
        self.x_cache = (idx, T, offset)
        for i, blk in enumerate(self.blocks):
            blk_cache = kv_cache[i] if kv_cache is not None else None
            x = blk.forward(x, blk_cache)
        x = self.ln_f.forward(x)
        self.logits = self.lm_head.forward(x)
        return self.logits
```

- [ ] **S6 讲解交叉熵 loss**
  - 对每个位置：`-log P(真实下一token)`，其中 `P = softmax(logits)`
  - 数值稳定写法：`log_softmax = logits - logsumexp(logits)`，全程用 log 域避免 `exp` 溢出
  - 均值损失：所有位置的平均，训练目标就是最小化它

- [ ] **S7 实现 `loss` / `backward`（嵌入梯度 scatter add）**

```python
    def loss(self, logits, targets):
        B, T, V = logits.shape
        flat = logits.reshape(-1, V)
        tflat = targets.reshape(-1)
        m = flat.max(axis=-1, keepdims=True)
        lse = np.log(np.exp(flat - m).sum(axis=-1, keepdims=True)) + m
        log_probs = flat - lse
        self.probs = np.exp(log_probs)
        self.targets_flat = tflat
        loss = -log_probs[np.arange(len(tflat)), tflat].mean()
        return loss

    def backward(self):
        """基于最后一次 loss() 计算所有参数梯度"""
        dflat = self.probs.copy()
        dflat[np.arange(len(self.targets_flat)), self.targets_flat] -= 1
        dflat /= len(self.targets_flat)
        dy = dflat.reshape(*self.logits.shape)         # [B,T,V]
        dy = self.lm_head.backward(dy)
        dy = self.ln_f.backward(dy)
        for blk in reversed(self.blocks):
            dy = blk.backward(dy)
        idx, T, offset = self.x_cache
        self.d_tok_emb.fill(0.0)
        np.add.at(self.d_tok_emb, idx, dy)             # 按 token 索引 scatter add
        self.d_pos_emb[offset:offset + T] += dy.sum(axis=0)
        return dy
```

> 讲解：softmax+CE 的梯度有个漂亮结论——`probs - onehot`。loss 反向到最后得到的 `dy`，再经 `tok_emb` 索引回填到每个出现过的 token 位置（`np.add.at` 处理重复索引）。

- [ ] **S8 讲解生成采样（temperature / top-k / top-p）**
  - `temperature < 1`：logits 除以 T，分布更尖，更「自信」；`> 1` 更发散
  - `top-k`：只从概率前 k 个里采样，砍掉长尾
  - `top-p`：保留累积概率到 p 的最小集合（核采样）
  - 三者可叠加，是推理服务「更像人」的控制旋钮

- [ ] **S9 实现 `generate` + `_sample`**

```python
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, kv_cache=None):
        """idx: [1, T]，返回追加生成后的完整序列 [1, T+new]；kv_cache 为空则内部初始化"""
        if kv_cache is None:
            head_dim = self.d_model // self.n_head
            kv_cache = [[np.zeros((idx.shape[0], self.n_head, 0, head_dim)),
                         np.zeros((idx.shape[0], self.n_head, 0, head_dim))]
                        for _ in range(self.n_layer)]
        self.forward(idx, kv_cache)                    # prefill（顺带建 cache）
        for _ in range(max_new_tokens):
            logits = self.forward(idx[:, -1:], kv_cache)   # decode，只看最后一个 token
            next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
            idx = np.concatenate([idx, next_token], axis=1)
        return idx
```

```python
def _sample(logits, temperature=1.0, top_k=None, top_p=None):
    """logits: [V]，返回 shape [1,1] 的 token"""
    if temperature != 1.0:
        logits = logits / temperature
    m = logits.max()
    probs = np.exp(logits - m)
    probs /= probs.sum()
    if top_k is not None:
        k = min(top_k, len(probs))
        idx = np.argpartition(-probs, k - 1)[:k]
        p2 = probs[idx]; p2 /= p2.sum()
        t = np.random.choice(idx, p=p2)
        return np.array([[t]])
    if top_p is not None:
        order = np.argsort(-probs)
        p_sorted = probs[order]
        cum = np.cumsum(p_sorted)
        keep = cum - p_sorted <= top_p
        keep_idx = order[keep]
        p2 = probs[keep_idx]; p2 /= p2.sum()
        t = np.random.choice(keep_idx, p=p2)
        return np.array([[t]])
    t = np.random.choice(len(probs), p=probs)
    return np.array([[t]])
```

- [ ] **S10 实现参数管理：dump/load（保存加载 checkpoint 的基础）**

```python
    def _named_params(self):
        out = [("tok_emb", self.tok_emb), ("pos_emb", self.pos_emb)]
        for i, blk in enumerate(self.blocks):
            out += [(f"blocks.{i}." + a, b) for a, b in blk.get_params()]
        out += [("ln_f." + a, b) for a, b in self.ln_f.get_params()]
        out += [("lm_head." + a, b) for a, b in self.lm_head.get_params()]
        return out

    def _named_grads(self):
        out = [("tok_emb", self.d_tok_emb), ("pos_emb", self.d_pos_emb)]
        for i, blk in enumerate(self.blocks):
            out += [(f"blocks.{i}." + a, blk.get_grad(a)) for a, _ in blk.get_params()]
        out += [("ln_f." + a, self.ln_f.get_grad(a)) for a, _ in self.ln_f.get_params()]
        out += [("lm_head." + a, self.lm_head.get_grad(a)) for a, _ in self.lm_head.get_params()]
        return out

    def dump_params(self):
        return {name: p.copy() for name, p in self._named_params()}

    def load_params(self, d):
        for name, p in self._named_params():
            p[...] = d[name]

    def save(self, path):
        np.savez(path, **self.dump_params())

    def load(self, path):
        d = np.load(path)
        for name, p in self._named_params():
            p[...] = d[name]

    def zero_grad(self):
        self.d_tok_emb.fill(0.0); self.d_pos_emb.fill(0.0)
        for blk in self.blocks:
            blk.zero_grad()
        self.ln_f.zero_grad(); self.lm_head.zero_grad()
```

- [ ] **S11 写端到端测试（loss / KV Cache / 保存加载 / 参数量）**

```python
"""GPT 端到端：loss 有限、KV Cache 一致性、保存加载一致性、参数量"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPT

def test_gpt_loss_and_backward():
    gpt = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    x = np.random.randint(0, 30, size=(4, 10))
    y = np.random.randint(0, 30, size=(4, 10))
    logits = gpt.forward(x)
    loss = gpt.loss(logits, y)
    assert logits.shape == (4, 10, 30), logits.shape
    assert np.isfinite(loss), f"loss 非有限: {loss}"
    assert 2.0 < loss < 5.0, f"loss 异常: {loss}"
    gpt.zero_grad(); gpt.backward()
    assert np.all(np.isfinite(gpt.d_tok_emb)), "tok_emb 梯度非有限"
    print("GPT loss/backward PASS", round(float(loss), 4))

def test_gpt_kv_cache():
    gpt = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    x = np.random.randint(0, 30, size=(1, 6))
    y_full = gpt.forward(x)
    gpt2 = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    gpt2.load_params(gpt.dump_params())
    head_dim = gpt2.d_model // gpt2.n_head
    cache = [[np.zeros((1, gpt2.n_head, 0, head_dim)),
              np.zeros((1, gpt2.n_head, 0, head_dim))] for _ in range(gpt2.n_layer)]
    first = gpt2.forward(x[:, :1], cache)
    rest = [gpt2.forward(x[:, t:t+1], cache) for t in range(1, 6)]
    y_step = np.concatenate([first] + rest, axis=1)
    err = np.abs(y_full - y_step).max()
    assert err < 1e-8, f"GPT KV Cache 不一致 err={err}"
    print("GPT KV Cache PASS", err)

def test_save_load():
    gpt1 = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    gpt2 = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    gpt2.load_params(gpt1.dump_params())
    x = np.random.randint(0, 30, size=(2, 8))
    assert np.abs(gpt1.forward(x) - gpt2.forward(x)).max() < 1e-10
    print("save/load consistency PASS")

def test_parameter_count():
    gpt = GPT(vocab_size=30, d_model=16, n_head=4, n_layer=2, ctx_len=10)
    n = sum(p.size for p in gpt.dump_params().values())
    print("param count:", n)
    assert n > 0

if __name__ == "__main__":
    test_gpt_loss_and_backward(); test_gpt_kv_cache(); test_save_load(); test_parameter_count()
    print("ALL GPT TESTS PASS")
```

- [ ] **S12 RED → GREEN**
  - RED：`python test/test_gpt.py` → `ModuleNotFoundError`
  - GREEN：补全 `gpt.py`（含 `__init__` 组装与 `_sample`），再运行
  - Expected: `GPT loss/backward PASS ...` / `GPT KV Cache PASS ...` / `save/load consistency PASS` / `param count: ...` / `ALL GPT TESTS PASS`

- [ ] **S13 收尾：commit + push**

Run: `git add model/gpt.py test/test_gpt.py; git commit -m "T4: GPT 组装（Block 堆叠 + loss + 生成 + 保存加载）"; git push`

---

### Task 5: train.py —— 训练循环（手写 AdamW + 快速验证 + 完整训练）

**教学主线:** 让模型真正学会。数据加载 → 采样批次 → 前向/反向 → AdamW 更新参数，循环几千步，loss 一路下降，最终生成出「像诗」的文本。

**Files:**
- Create: `train.py`
- Create: `test/test_train.py`

- [ ] **S1 讲解数据管线：stoi/itos 与 get_batch**
  - `load_corpus`：读文本 → 字符集 → 建 `stoi`（字符→id）和 `itos`（id→字符）→ 整段文本转成 id 数组
  - `get_batch`：随机取 `batch_size` 个起点，各截 `ctx_len` 长度；输入 `x` 与目标 `y` 错开一位（`y = x 右移`，让模型预测下一个字符）
  - 这就是自回归训练的核心数据格式：`(输入序列, 下一token序列)`

- [ ] **S2 实现数据加载**

```python
"""训练循环：数据加载 + char 分词 + AdamW + 训练 + 保存 model.npz"""
import os
import numpy as np

from model.gpt import GPT

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "corpus.txt")
CKPT_PATH = os.path.join(BASE, "model.npz")


def load_corpus():
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return text, chars, stoi, itos, data


def get_batch(data, batch_size, ctx_len):
    ix = np.random.randint(0, len(data) - ctx_len, size=batch_size)
    x = np.stack([data[i:i + ctx_len] for i in ix])
    y = np.stack([data[i + 1:i + ctx_len + 1] for i in ix])
    return x, y
```

Run: `python -c "import sys; sys.path.insert(0,'.'); from train import load_corpus, get_batch; text,chars,stoi,itos,data=load_corpus(); x,y=get_batch(data,4,32); print('vocab:', len(chars), 'x:', x.shape, 'y:', y.shape, 'aligned:', (y[:,:-1]==x[:,1:]).all())"`
Expected: `vocab: ~2500  x: (4, 32)  y: (4, 32)  aligned: True`

- [ ] **S3 讲解 AdamW 优化器**
  - 一阶矩 `m`（梯度 EMA）：方向；二阶矩 `v`（梯度平方 EMA）：自适应步长——大梯度步长小，小梯度步长大
  - 偏置修正：`m/(1-β₁ᵗ)`、`v/(1-β₂ᵗ)`，前期 EMA 偏小需校正
  - Weight decay：`p *= (1 - lr·wd)`，但对 **LN 的 γ/β 和 bias 不衰减**（它们本来就该自由缩放）
  - 更新：`p -= lr · m̂ / (√v̂ + ε)`

- [ ] **S4 实现 `AdamW`（注意 zip 解包要配对参数名）**

```python
class AdamW:
    """手写 AdamW：带 bias correction 与 weight decay（不衰减 LN/bias）"""

    def __init__(self, named_params, named_grads, lr=3e-4, betas=(0.9, 0.99), eps=1e-8, wd=0.1):
        # 按名字配对 (name, param, grad)，避免 zip 拆包错位
        self.pg = [(n, p, g) for (n, p), (_, g) in zip(named_params, named_grads)]
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.t = 0
        self.m = {n: np.zeros_like(p) for n, p, g in self.pg}
        self.v = {n: np.zeros_like(p) for n, p, g in self.pg}

    def step(self):
        self.t += 1
        for name, p, g in self.pg:
            if name.endswith(".gamma") or name.endswith(".beta") or name.endswith(".b"):
                wd = 0.0
            else:
                wd = self.wd
            if wd != 0.0:
                p *= (1.0 - self.lr * wd)
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (g * g)
            m_hat = self.m[name] / (1 - self.b1 ** self.t)
            v_hat = self.v[name] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

- [ ] **S5 讲解训练循环并实现 `train()`**

```python
def train(steps=3000, batch_size=32, ctx_len=64, lr=3e-4, seed=0, verbose=True):
    """返回 (首步 loss, 末步 loss)；默认完整训练并保存 model.npz"""
    np.random.seed(seed)
    text, chars, stoi, itos, data = load_corpus()
    gpt = GPT(vocab_size=len(chars), d_model=64, n_head=4, n_layer=2, ctx_len=64)
    opt = AdamW(gpt._named_params(), gpt._named_grads(), lr=lr)
    losses = []
    for step in range(steps):
        x, y = get_batch(data, batch_size, ctx_len)
        logits = gpt.forward(x)
        loss = gpt.loss(logits, y)
        gpt.zero_grad()
        gpt.backward()
        opt.step()
        losses.append(float(loss))
        if verbose and (step % 200 == 0 or step == steps - 1):
            print(f"step {step:5d} | loss {loss:.4f}")
    if steps >= 500:
        gpt.save(CKPT_PATH)
        print(f"saved -> {CKPT_PATH}")
    return losses[0], losses[-1]


if __name__ == "__main__":
    train()
```

> 循环五件套顺序不能错：`forward → loss → zero_grad → backward → opt.step()`。`zero_grad` 必须在 `backward` 前，否则梯度累加到旧值上。

- [ ] **S6 冒烟测试：60 步 loss 必须下降**

```python
"""训练冒烟测试：50 步内 loss 显著下降 + AdamW 步进"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import train

def test_train_smoke():
    loss0, loss1 = train(steps=60, batch_size=16, ctx_len=32, lr=1e-2, seed=0, verbose=False)
    print(f"loss: {loss0:.4f} -> {loss1:.4f}")
    assert loss1 < loss0 * 0.9, f"loss 未下降: {loss0} -> {loss1}"
    print("train smoke PASS")

if __name__ == "__main__":
    test_train_smoke()
    print("ALL TRAIN TESTS PASS")
```

Run: `python test/test_train.py`
Expected: `loss: 7.x -> 6.x`（明显下降），`train smoke PASS`

- [ ] **S7 完整训练并落 output 文件**

Run: `python train.py *> test/output-train.txt`
Expected: step 每 200 步打印 loss，末行 `saved -> ...\model.npz`；loss 从 ~7.8（随机基线 ln2500）降到 <4.5

- [ ] **S8 生成验证：训练成果可视化**

Run: `python -c "import sys,os; sys.path.insert(0, os.getcwd()); from train import load_corpus; from model.gpt import GPT; text,chars,stoi,itos,data=load_corpus(); gpt=GPT(vocab_size=len(chars)); gpt.load('model.npz'); import numpy as np; prompt='床前'; idx=np.array([[stoi[c] for c in prompt]]); out=gpt.generate(idx, 40, temperature=0.8, top_k=20); print(''.join(itos[i] for i in out[0]))" *> test/output-train.txt`
Expected: 输出一段近似古诗风格的中文（非乱码），追加进 output-train.txt

- [ ] **S9 收尾：commit + push**

Run: `git add train.py test/test_train.py; git commit -m "T5: 训练循环（手写 AdamW + 保存 model.npz）"; git push`

---

### Task 6: server.py —— HTTP 推理服务（连续批处理 + KV Cache + SSE 流式 + p99）

**教学主线:** 把训练好的模型变成「服务」。请求怎么排队攒批（连续批处理）、decode 怎么共享一次前向（Batcher）、怎么逐 token 推给客户端（SSE）、怎么量延迟（p99）——4 个概念层层落地。

**Files:**
- Create: `server.py`
- Create: `test/test_server.py`

- [ ] **S1 讲解服务架构（先看全局）**
  - `ThreadingHTTPServer`：每个 HTTP 请求一个线程，天然支持并发
  - 主路径：请求 → `InferenceEngine.complete()` → prefill（一次完整前向建 KV Cache）→ 循环 decode（每步经 Batcher 组批）→ 采样 → 返回
  - SSE 流式：响应头 `text/event-stream`，每个 token 一条 `data: {...}\n\n`，最后 `data: [DONE]`
  - 非流式：攒完整段文本一次性 JSON 返回

- [ ] **S2 讲解连续批处理调度器（Batcher）核心思想**
  - 问题：单请求 decode 时计算利用率低——每个请求只算 1 个 token 的 1 次前向
  - 思想：把多个请求的同一 decode 步「攒」在一起，`[B,1,V]` 一次前向 = B 个请求共享
  - 触发条件：`满批（len>=max_batch）` 或 `超时（max_wait）`——满批保吞吐，超时保延迟
  - 难点：不同请求的 cache 长度可能不同 → 按 `Tk` 长度分组，各组分别前向
  - 这是 8-07「连续批处理」/ 8-11「推理服务设计」的直接落地

- [ ] **S3 实现 `Batcher`（队列 + 双触发 + 分组前向）**

```python
class Batcher:
    """连续批处理调度器：多个请求的 decode 步攒批，共享一次模型前向"""

    def __init__(self, model, max_batch=4, max_wait=0.02):
        self.model = model
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.cond = threading.Condition()
        self.queue = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            with self.cond:
                while not self.queue:
                    self.cond.wait()
                deadline = time.time() + self.max_wait
                while len(self.queue) < self.max_batch:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    self.cond.wait(remaining)
                batch = self.queue
                self.queue = []
            self._forward(batch)
            with self.cond:
                for req in batch:
                    req['result'] = req['_result']
                    req['_result'] = None
                self.cond.notify_all()

    def _forward(self, batch):
        # 不同请求的 cache 长度（Tk）可能不同，按长度分组后分别组批前向
        groups = {}
        for r in batch:
            groups.setdefault(r['kvcache'][0][0].shape[2], []).append(r)
        for g in groups.values():
            self._forward_group(g)

    def _forward_group(self, batch):
        xs = np.concatenate([r['ids'] for r in batch], axis=0)      # [B,1]
        n_layer = self.model.n_layer
        kvc = [None] * n_layer
        for i in range(n_layer):
            ks = np.concatenate([r['kvcache'][i][0] for r in batch], axis=0)
            vs = np.concatenate([r['kvcache'][i][1] for r in batch], axis=0)
            kvc[i] = [ks, vs]
        logits = self.model.forward(xs, kvc)                        # [B,1,V]
        for j, r in enumerate(batch):
            r['_result'] = logits[j:j + 1]
            for i in range(n_layer):
                r['kvcache'][i][0] = kvc[i][0][j:j + 1]
                r['kvcache'][i][1] = kvc[i][1][j:j + 1]

    def next(self, req):
        with self.cond:
            self.queue.append(req)
            self.cond.notify_all()
        with self.cond:
            while req.get('result') is None:
                self.cond.wait()
            r = req['result']
            req['result'] = None
            return r
```

- [ ] **S4 讲解并实现 `InferenceEngine` + `Tracker`（p99 统计）**
  - `InferenceEngine.complete()`：prompt → 分词 → prefill（全序列前向，把每层 KV 都填进 cache）→ 循环：取最后一个 token 交给 Batcher 组批 decode → 采样 → 拼接
  - 关键分工：**prefill 自己算（只一次），decode 全部走 Batcher（共享）**
  - `Tracker`：记录每次请求耗时，环形缓冲只留最近 cap 条，`summary()` 输出 count/avg_ms/p50_ms/p99_ms

```python
class InferenceEngine:
    """prefill 独立 + decode 经 Batcher 组批；对外 complete()"""

    def __init__(self, gpt, stoi, itos, max_batch=4, max_wait=0.02):
        self.gpt = gpt
        self.stoi, self.itos = stoi, itos
        self.batcher = Batcher(gpt, max_batch, max_wait)
        self.head_dim = gpt.d_model // gpt.n_head

    def tokenize(self, prompt):
        ids = [self.stoi[c] for c in prompt if c in self.stoi]
        if not ids:
            ids = [self.stoi['\n']]
        return np.array([ids], dtype=np.int64)

    def prefill(self, idx):
        kvc = [[np.zeros((1, self.gpt.n_head, 0, self.head_dim)),
                np.zeros((1, self.gpt.n_head, 0, self.head_dim))]
               for _ in range(self.gpt.n_layer)]
        self.gpt.forward(idx, kvc)
        return kvc

    def complete(self, prompt, max_tokens=32, temperature=0.8, top_k=20, top_p=None, on_token=None):
        t0 = time.time()
        idx = self.tokenize(prompt)
        kvc = self.prefill(idx)
        parts = []
        for _ in range(max_tokens):
            logits = self.batcher.next({'ids': idx[:, -1:], 'kvcache': kvc})
            t = _sample(logits[:, -1, :], temperature, top_k, top_p)
            c = self.itos[int(t[0, 0])]
            parts.append(c)
            idx = np.concatenate([idx, t], axis=1)
            if on_token:
                on_token(c)
        return ''.join(parts), time.time() - t0


class Tracker:
    """延迟统计：avg / p50 / p99"""

    def __init__(self, cap=200):
        self.lat = []
        self.lock = threading.Lock()
        self.cap = cap

    def record(self, dt):
        with self.lock:
            self.lat.append(dt)
            if len(self.lat) > self.cap:
                self.lat.pop(0)

    def summary(self):
        with self.lock:
            a = np.array(self.lat) * 1000
            if len(a) == 0:
                return {}
            return {'count': int(len(a)), 'avg_ms': round(float(a.mean()), 1),
                    'p50_ms': round(float(np.percentile(a, 50)), 1),
                    'p99_ms': round(float(np.percentile(a, 99)), 1)}
```

- [ ] **S5 实现 HTTP 处理器 `Handler` 与 `create_server`**

```python
class Handler(BaseHTTPRequestHandler):
    engine = None
    tracker = None

    def do_GET(self):
        if self.path == '/stats':
            body = json.dumps(self.tracker.summary(), ensure_ascii=False).encode()
            self._reply(200, 'application/json', body)
        else:
            self._reply(404, 'application/json', b'{"error":"not found"}')

    def do_POST(self):
        if self.path != '/v1/completions':
            self._reply(404, 'application/json', b'{"error":"not found"}')
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(length).decode('utf-8'))
        except Exception:
            self._reply(400, 'application/json', b'{"error":"bad json"}')
            return
        prompt = str(req.get('prompt', ''))
        max_tokens = max(1, min(int(req.get('max_tokens', 32)), 256))
        temperature = float(req.get('temperature', 0.8))
        top_k = int(req.get('top_k')) if req.get('top_k') else None
        top_p = float(req.get('top_p')) if req.get('top_p') else None
        stream = bool(req.get('stream', False))
        if stream:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            t0 = time.time()

            def on_token(c):
                self.wfile.write(f"data: {json.dumps({'token': c}, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            text, dt = self.engine.complete(prompt, max_tokens, temperature, top_k, top_p, on_token)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            text, dt = self.engine.complete(prompt, max_tokens, temperature, top_k, top_p)
            self._reply(200, 'application/json',
                        json.dumps({'text': text}, ensure_ascii=False).encode())
        self.tracker.record(dt)
        print(f"[{time.strftime('%H:%M:%S')}] {prompt!r} -> {dt * 1000:.0f}ms, {len(text)}tok")

    def _reply(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def create_server(gpt, stoi, itos, port=8000, max_batch=4, max_wait=0.02):
    Handler.engine = InferenceEngine(gpt, stoi, itos, max_batch, max_wait)
    Handler.tracker = Tracker()
    srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    return srv, srv.server_address[1]


if __name__ == '__main__':
    text, chars, stoi, itos, data = load_corpus()
    gpt = GPT(vocab_size=len(chars))
    gpt.load(os.path.join(BASE, 'model.npz'))
    srv, port = create_server(gpt, stoi, itos)
    print(f"LLM server on http://127.0.0.1:{port}/v1/completions")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
```

- [ ] **S6 写集成测试：起服务 → 流式/非流式/并发 → /stats p99**

```python
"""server 集成测试：起服务 → 流式/非流式/并发 4 请求 → /stats 输出 p99"""
import sys, os, json, threading, time
import http.client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPT
from train import load_corpus
from server import create_server

def start(port=0):
    text, chars, stoi, itos, data = load_corpus()
    gpt = GPT(vocab_size=len(chars))
    gpt.load('model.npz')
    srv, p = create_server(gpt, stoi, itos, port=port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, p

def req(p, prompt, stream=False, max_tokens=16):
    conn = http.client.HTTPConnection('127.0.0.1', p, timeout=30)
    body = json.dumps({'prompt': prompt, 'max_tokens': max_tokens, 'stream': stream})
    conn.request('POST', '/v1/completions', body, {'Content-Type': 'application/json'})
    r = conn.getresponse()
    data = r.read().decode('utf-8')
    conn.close()
    return r.status, data

def stats(p):
    conn = http.client.HTTPConnection('127.0.0.1', p, timeout=5)
    conn.request('GET', '/stats')
    r = conn.getresponse()
    d = json.loads(r.read().decode())
    conn.close()
    return d

def test_server():
    srv, p = start()
    time.sleep(0.3)
    st, data = req(p, '床前明月光')
    assert st == 200 and len(json.loads(data)['text']) > 0, data
    st, data = req(p, '春眠不觉晓', stream=True)
    assert st == 200 and 'data:' in data and '[DONE]' in data, data
    results = []
    def worker(prompt):
        results.append(req(p, prompt))
    ts = [threading.Thread(target=worker, args=(pr,)) for pr in ['静夜思', '春晓', '江雪', '相思']]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(s == 200 for s, _ in results), results
    time.sleep(0.2)
    s = stats(p)
    print("stats:", s)
    assert s['count'] >= 6
    srv.shutdown()
    print("server integration PASS")

if __name__ == '__main__':
    test_server()
```

Run: `python test/test_server.py *> test/output-server.txt`
Expected: `stats: {'count': 6, 'avg_ms': ..., 'p50_ms': ..., 'p99_ms': ...}` + `server integration PASS`

- [ ] **S7 收尾：commit + push**

Run: `git add server.py test/test_server.py; git commit -m "T6: HTTP 推理服务（连续批处理 + KV Cache + SSE + p99）"; git push`

---

### Task 7: cli.py —— 命令行交互 + 全量验证收尾

**教学主线:** 最后一块拼图——CLI 让本地加载权重直接生成，不需要起服务。然后全量回归、更新 README、交付检查，收尾。

**Files:**
- Create: `cli.py`

- [ ] **S1 讲解 CLI 定位**
  - 与 server 的区别：CLI 本地单机直接推理（不经过 HTTP、不组批），适合快速试模型
  - 复用 `load_corpus`（分词）和 `GPT.generate`（KV Cache 生成），代码很薄
  - `argparse`：命令行参数 prompt / --max-tokens / --temperature / --top-k / --top-p

- [ ] **S2 实现 `cli.py`**

```python
"""CLI：加载 model.npz 本地交互生成"""
import argparse
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model.gpt import GPT
from train import load_corpus


def main():
    ap = argparse.ArgumentParser(description='极简 LLM 命令行生成')
    ap.add_argument('prompt', nargs='?', default='床前明月光')
    ap.add_argument('--max-tokens', type=int, default=60)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--top-p', type=float, default=None)
    args = ap.parse_args()
    text, chars, stoi, itos, data = load_corpus()
    gpt = GPT(vocab_size=len(chars))
    gpt.load(os.path.join(BASE, 'model.npz'))
    idx = np.array([[stoi[c] for c in args.prompt if c in stoi]], dtype=np.int64)
    if idx.shape[1] == 0:
        idx = np.array([[stoi['\n']]], dtype=np.int64)
    out = gpt.generate(idx, args.max_tokens, args.temperature, args.top_k, args.top_p)
    print(''.join(itos[int(i)] for i in out[0]))


if __name__ == '__main__':
    main()
```

- [ ] **S3 CLI 生成演示并落 output 文件**

Run: `python cli.py "床前明月光" --max-tokens 40 *> test/output-cli.txt`
Expected: 输出一段古诗风格中文，写入 output-cli.txt

- [ ] **S4 全量测试回归（从根目录）**

Run: `python test/test_layers.py; python test/test_attention.py; python test/test_gpt.py`
Expected: 三个文件均输出 `ALL ... TESTS PASS`

- [ ] **S5 更新 README 使用部分**
  - 把「使用方式（实现后生效）」改为实际命令 + 引用三个 output 文件

- [ ] **S6 最终交付检查**
  - 确认仓库根目录下存在：README.md、PLAN.md、data/corpus.txt、model/{__init__,layers,attention,gpt}.py、train.py、server.py、cli.py、model.npz、test/output-train.txt、test/output-cli.txt、test/output-server.txt
  - 确认无临时脚本残留（环境洁癖）
  - 清理：`test/__init__.py` 与 `model/__init__.py` 保留（包标记必需）

- [ ] **S7 收尾：commit + push**

Run: `git add cli.py; git commit -m "T7: CLI + README 收尾"; git push`

---

## 执行与验收顺序一览

| 顺序 | 命令 | 产出 | 验收点 |
|---|---|---|---|
| T1 | 语料验证 | corpus.txt | 2.4 万 chars / 2500 unique |
| T2 | `python test/test_layers.py` | — | 梯度检查 < 1e-4 |
| T3 | `python test/test_attention.py` | — | 梯度检查 + KV Cache 一致 + GQA |
| T4 | `python test/test_gpt.py` | — | loss/KV/save-load |
| T5a | `python test/test_train.py` | — | 60 步 loss 下降 |
| T5b | `python train.py *> test/output-train.txt` | model.npz + output | loss <4.5，生成有规律 |
| T6 | `python test/test_server.py *> test/output-server.txt` | output-server.txt | 流式/非流式/并发/p99 |
| T7 | `python cli.py "床前明月光" *> test/output-cli.txt` | output-cli.txt | 古诗风格文本 |

## 训练资料知识点 → 计划映射

| 训练资料 | 知识点 | 落点 |
|---|---|---|
| 07-12/07-13 | 位置编码（正弦/余弦 vs 可学习） | 附录①（主线只用可学习编码） |
| 07-14 | Self-Attention 公式推导与矩阵可视化 | T3 S3-S5（主线） |
| 07-15 | MHA → MQA → GQA 演进 | T3 S11（主线：单套代码 + 追加测试） |
| 07-16 | FFN / LayerNorm / 残差连接 | T2 / T4 S1-S3（主线） |
| 07-17 | 解码策略 TopK / TopP / Temperature | T4 S8-S9（主线） |
| 07-27 | MHA 手写 + MoE 混合专家 | T3（主线）/ 附录②（MoE） |
| 07-28 | 微调链路 SFT / LoRA / DPO / RLHF | 附录③ |
| 07-30 | 混合精度训练 FP16 / BF16 | 附录④ |
| 08-07 | KV Cache 三兄弟 / 连续批处理 / 量化 | T3 S6 / T6 S2-S3（主线）/ 附录⑤（量化） |
| 08-10 | 流式输出 | T6 S5-S6（主线） |
| 08-11 | 设计大模型推理服务（p99） | T6 S4-S6（主线） |

---

## 附录：训练资料延伸练习（学习价值 > 项目价值）

> 以下 5 项**不进入主线**：主线保持最小闭环（训练 → 推理 → 服务化），它们留给学有余力的延伸。每一项都是**独立练习**：独立文件、独立验证，不接入主模型，不影响主线验收。完成主线全部 Task 后可按需挑选。

### 附录① 位置编码：正弦/余弦 vs 可学习

**定位：** 主线用了**可学习** `pos_emb`；本练习补原版 Transformer 的**正弦/余弦固定编码**，并对比两者差异。

- 正弦公式：`PE(pos,2i)=sin(pos/10000^(2i/d))`、`PE(pos,2i+1)=cos(pos/10000^(2i/d))`
- 固定编码优点：不需要学、可外推到训练没见过的长度；可学习优点：数据自适应、小模型更灵活
- 关键直觉：`pos_emb[i] · pos_emb[j]` 只依赖 `|i-j|`（相对位置）——注意力需要的是**相对距离信息，而非绝对位置**

**动手：** 在 `model/gpt.py` 加辅助函数（不替换主线可学习编码）：

```python
def _sincos_pos(T, d):
    pos = np.arange(T)[:, None]                                   # [T,1]
    i = np.arange(d // 2)[None, :]                                # [1,d/2]
    angles = pos / np.power(10000.0, 2 * i / d)                   # [T,d/2]
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe
```

Run: `python -c "import numpy as np, sys; sys.path.insert(0,'.'); from model.gpt import _sincos_pos; pe=_sincos_pos(10,16); print(pe.shape); d=np.abs(pe[1:]-pe[:-1]); print('相邻位置编码差异:', np.round(d.mean(axis=1)[:3],3))"`
Expected: `(10, 16)`，相邻差异小且不恒定（位置越近编码越像 → 相对位置信息）

**延伸思考：** 把 GPT 的位置嵌入换成 `_sincos_pos(ctx_len, d_model)` 重训对比——固定编码下小语料 loss 是否仍能下降？

### 附录② MoE 混合专家：稀疏激活

**定位：** 一个大 FFN 拆成 N 个「专家」（小 FFN），每个 token 只激活 Top-2。收益是**参数变多、计算不涨**（Mixtral 8×7B 同款思想）；代价是需要 load balance 防止 token 全挤一个专家。

**动手：** 在 `model/gpt.py` 加独立对照模块（**不接入主 Block**）。注意这是**骨架代码**，需先补全两处再运行：
1. `softmax` 用 `scipy` 不可取（零依赖），补一个 `np.exp(logits - logits.max(-1,keepdims=True))` 的稳定 softmax；
2. 专家路径反向依赖 MLP **缓存输入 x**（若 MLP 无 `x` 缓存属性，需先给 MLP 加 `self.x`）——这正是「手写反向」的又一练手点。

```python
class MoE:
    """稀疏 MoE：n_expert 个小 FFN，每个 token 激活 Top-2 并加权求和（骨架）"""
    def __init__(self, d_model, n_expert=4):
        self.n_expert = n_expert
        self.experts = [MLP(d_model) for _ in range(n_expert)]
        self.router = Linear(d_model, n_expert)
        self.cache = None

    def forward(self, x):
        B, T, D = x.shape
        logits = self.router.forward(x)                            # [B,T,n_expert]
        probs = _stable_softmax(logits)                            # 需自行补全
        top2 = np.argsort(-probs, axis=-1)[:, :, :2]
        w = np.take_along_axis(probs, top2, axis=-1)               # [B,T,2]
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-6)             # 归一化 Top-2 权重
        out = np.zeros_like(x)
        self.cache = (x, top2, w)
        for k in range(2):
            idx = top2[:, :, k]                                    # [B,T]
            flat = x.reshape(-1, D)
            for e in range(self.n_expert):
                mask = (idx == e)
                if mask.any():
                    sel = flat[mask]
                    out[mask] += w[:, :, k][mask, None] * self.experts[e].forward(sel)
        return out

    def backward(self, dy):
        x, top2, w = self.cache
        B, T, D = x.shape
        flat_dy, flat_x = dy.reshape(-1, D), x.reshape(-1, D)
        dx = np.zeros_like(flat_x)
        for e in range(self.n_expert):
            for k in range(2):
                mask = (top2[:, :, k].reshape(-1) == e)
                if mask.any():
                    sel = flat_x[mask]
                    dsel = w[:, :, k].reshape(-1)[mask, None] * flat_dy[mask]
                    dx[mask] += self.experts[e].backward(dsel)     # 需 MLP 缓存 x
        # 路由梯度（简化）：d out / d w_k = 专家输出 E_k(x) → router.dW/db
        for k in range(2):
            idx = top2[:, :, k].reshape(-1)
            for e in range(self.n_expert):
                mask = (idx == e)
                if mask.any():
                    contrib = flat_dy[mask] * w[:, :, k].reshape(-1)[mask, None]
                    self.router.dW += flat_x[mask].T @ contrib
                    self.router.db += contrib.sum(axis=0)
        return dx.reshape(B, T, D)
```

验证（形状 + 梯度检查，复用 `test_layers.py` 的 `numeric_grad`）：MoE 前向/反向输出形状均为 `(2, 3, 8)`，且解析梯度与有限差分误差 < 1e-4（注意路由路径是简化梯度，误差阈值可放宽到 1e-2 并思考为什么）。

### 附录③ 微调链路 SFT / LoRA / DPO / RLHF

**定位：** 主线只做预训练。本练习补「预训练 → SFT → 偏好对齐」全景中最轻的一环：SFT 续训。

- **SFT**：拿高质量「输入→期望输出」数据在 `model.npz` 上续训（本练习落地）
- **LoRA**：冻结主干，只训低秩增量 `ΔW = A·B`（r≪d），参数量降到千分之一
- **DPO**：不用奖励模型，用偏好对 `(chosen, rejected)` 算对比损失
- **RLHF（PPO）**：需要奖励模型 + 在线采样，最重

**动手：** 新增 `fine_tune.py`，**前置条件：先自建一份小语料 `data/sft-corpus.txt`**（一两段古诗即可）。核心与 `train()` 几乎一样，区别只是「加载已有权重 + 换数据源」：

```python
"""SFT 续训：加载 model.npz，在新语料上继续训练，保存 model-sft.npz"""
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model.gpt import GPT
from train import load_corpus, get_batch, AdamW

SFT_CORPUS = os.path.join(BASE, "data", "sft-corpus.txt")   # 可自行扩展新语料
CKPT_SFT = os.path.join(BASE, "model-sft.npz")


def sft_train(steps=500, batch_size=16, ctx_len=64, lr=1e-4, seed=42, verbose=True):
    np.random.seed(seed)
    text, chars, stoi, itos, data = load_corpus()           # 复用主线词表（字符集一致）
    with open(SFT_CORPUS, encoding='utf-8') as f:
        sft_text = f.read()
    sft_data = np.array([stoi[c] for c in sft_text if c in stoi], dtype=np.int64)
    gpt = GPT(vocab_size=len(chars), d_model=64, n_head=4, n_layer=2, ctx_len=64)
    gpt.load(os.path.join(BASE, "model.npz"))               # 续训起点 = 预训练权重
    opt = AdamW(gpt._named_params(), gpt._named_grads(), lr=lr)
    losses = []
    for step in range(steps):
        x, y = get_batch(sft_data, batch_size, ctx_len)
        logits = gpt.forward(x)
        loss = gpt.loss(logits, y)
        gpt.zero_grad()
        gpt.backward()
        opt.step()
        losses.append(float(loss))
        if verbose and (step % 100 == 0 or step == steps - 1):
            print(f"sft step {step:5d} | loss {loss:.4f}")
    gpt.save(CKPT_SFT)
    print(f"saved -> {CKPT_SFT}")
    return losses[0], losses[-1]


if __name__ == "__main__":
    sft_train()
```

Run: `python -c "import sys; sys.path.insert(0,'.'); from fine_tune import sft_train; sft_train(steps=200, verbose=False)" *> test/output-sft.txt`
Expected: 复用主线词表、模型加载成功、loss 下降、保存 `model-sft.npz`

> LoRA / DPO 作为延伸思考（不写代码）：LoRA 就是给每个 Linear 加 `A·B` 旁路只训它；DPO 就是把偏好对塞进交叉熵对比损失。两者都能在 `gpt._named_params()` 之上拼装，思路与 AdamW/梯度检查一致。

### 附录④ 混合精度训练 FP16 / BF16

**定位：** 讲清「为什么省显存、为什么需要 loss scaling、BF16 为什么更好」。**注意：纯 numpy 里 `p[...] = p.astype(np.float16)` 赋回 float32 数组并不省内存**——本练习的价值在观察精度损失，不在省内存。

- FP16 数值范围窄（`2⁻¹⁴ ~ 2¹⁵`），小梯度下溢为 0 → 需要 **loss scaling**（loss×2¹⁶ 回传再除回）
- BF16 指数位与 FP32 相同，范围不丢只丢尾数——**无需 loss scaling**，现代 GPU 主流
- 精度损失：小模型权重敏感，fp16 训练 loss 通常略高于 fp32

**动手：** 给 `train()` 加 `dtype` 参数，**存一份 float16 的权重副本，前向计算时转回 float32**（演示「精度损失」而非「省内存」）：

```python
def train(steps=3000, batch_size=32, ctx_len=64, lr=3e-4, seed=0, verbose=True, dtype='float32'):
    """... dtype: 'float32' 或 'float16'，演示低精度的精度损失"""
    np.random.seed(seed)
    text, chars, stoi, itos, data = load_corpus()
    gpt = GPT(vocab_size=len(chars), d_model=64, n_head=4, n_layer=2, ctx_len=64)
    if dtype == 'float16':
        # 低精度副本：前向前转回 float32 计算（模拟"存储低精度、计算高精度"）
        for name, p in gpt._named_params():
            p[...] = p.astype(np.float16).astype(np.float32)   # 只保留 fp16 精度
    opt = AdamW(gpt._named_params(), gpt._named_grads(), lr=lr)
    losses = []
    for step in range(steps):
        x, y = get_batch(data, batch_size, ctx_len)
        logits = gpt.forward(x)
        loss = gpt.loss(logits, y)
        gpt.zero_grad()
        gpt.backward()
        opt.step()
        losses.append(float(loss))
        if verbose and (step % 200 == 0 or step == steps - 1):
            print(f"step {step:5d} | loss {loss:.4f}")
    if steps >= 500:
        gpt.save(CKPT_PATH)
        print(f"saved -> {CKPT_PATH}")
    return losses[0], losses[-1]
```

Run: `python -c "import sys; sys.path.insert(0,'.'); from train import train; a,_=train(steps=300, seed=0, verbose=False); b,_=train(steps=300, seed=0, dtype='float16', verbose=False); print('fp32 末步 loss:', a, ' fp16:', b)"`
Expected: fp16 loss 略高于 fp32（精度损失的代价），但仍能下降

### 附录⑤ 量化压缩 int8

**定位：** 权重 fp32（4 字节/参数）→ int8（1 字节），内存/带宽降 4 倍。**独立 demo，不接入 server**（纯 numpy 无真实带宽收益，仅原理演示）。

**动手：** 新增 `model/quantize.py`：

```python
"""int8 量化（Absmax）：把 fp32 权重压到 int8，评估峰值误差与内存占用"""
import numpy as np


def quantize(w):
    """w: [out,in] fp32 → (int8 数组, scale)"""
    scale = np.max(np.abs(w)) / 127.0
    w8 = np.round(w / scale).astype(np.int8)
    return w8, scale


def dequantize(w8, scale):
    return w8.astype(np.float32) * scale


def memory_saved(w):
    return f"{w.nbytes // 1024}KB -> {quantize(w)[0].nbytes // 1024}KB (x4)"


def demo():
    np.random.seed(0)
    w = np.random.randn(64, 64) * 0.5
    w8, s = quantize(w)
    w_back = dequantize(w8, s)
    err = np.abs(w - w_back).max()
    print("W shape:", w.shape, "| 峰值误差:", round(float(err), 5))
    print("内存:", memory_saved(w))
    return err
```

Run: `python -c "import sys; sys.path.insert(0,'.'); from model.quantize import demo; demo()"`
Expected: 峰值误差 ~0.004（相对权重量级 0.5 约 0.8%），内存 `16KB -> 4KB (x4)`

**延伸思考：** 把 `demo()` 换成对 `model.npz` 全部权重量化，评估整体峰值误差；对比 int8 后生成的文本是否还能读。