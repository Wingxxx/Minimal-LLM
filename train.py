"""训练循环：数据加载 + char 分词 + AdamW + 真续训 checkpoint + val loss

GPU 加速：设环境变量 MINIMAL_GPU=1（且已安装 cupy）即全链路跑 GPU，
否则默认纯 numpy/CPU。计算后端统一由 model.backend 提供。

S2 管线升级要点：
  - 模型尺寸参数化：train() 可传 d_model/n_head/n_layer（默认=旧值 d64/h4/2 层，CPU 回归无损）
  - anneal_total 与段长解耦：学习率按「全局步数 + anneal_total」余弦回放，
    step >= anneal_total 后冻结在峰值 10%；分段续训时各段曲线自然衔接
  - 真续训 checkpoint：模型参数 + 优化器 m/v/t + 词表快照 chars + 调度签名
    {anneal_total, warmup_steps, peak_lr} + 格式版本；载入时词表/签名不一致即拒载
  - 原子存档：临时文件 + os.replace；训练循环 try/finally，中断也先落盘再退出
  - val loss：独立验证集（data/val.txt）按固定随机窗口求平均，分辨"学规律 vs 背语料"
"""
import os

# 计算后端：默认 numpy(CPU)；MINIMAL_GPU=1 时 np=cupy(GPU)。
# onp 恒为原生 numpy：语料处理留在 CPU（数据量小、切片快），进模型前再搬设备
from model.backend import np, onp, as_numpy
from model.gpt import GPT
from meter import make_meter_processor

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "corpus.txt")   # 训练语料（全唐诗简体净流）
VAL_PATH = os.path.join(BASE, "data", "val.txt")       # 独立验证集（400 首，与训练同构）
META_PATH = os.path.join(BASE, "data", "meta.npz")     # 逐位置元信息（权重/平仄/韵部/区标）
L_BASE_PATH = os.path.join(BASE, "model-ext.npz")      # 遗忘基线 L_base 的锚模型（扩词表迁移档）
CKPT_PATH = os.path.join(BASE, "model.npz")            # 训练/续训存档位
CKPT_VERSION = 2                                       # checkpoint 格式版本：2 = 真续训
# demo 默认 prompts：程序化校验 ⊆ 词表，不全即跳过（新语料词表大，防 KeyError）
DEMO_PROMPTS = ["床前明月光", "白日依山尽", "春眠不觉晓", "海上生明月", "大漠孤烟直"]
# 诗体特殊 token：顺序须与 tools/build_corpus.POEM_TAGS 逐项相同（防两处定义漂移），
# 词表构造时追加在纯文本字符之后（id 8196..8200）。跨模块一致性由 test/test_tokenizer.py 锁定。
SPECIAL_TOKENS = ("<五绝>", "<七绝>", "<五律>", "<七律>", "<杂言>")


def _strip_special_tokens(text):
    """剥去文本中全部诗体特殊 token，返回纯文本（用于构造不含 < > 的字符词表）。"""
    for tok in SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text


_DEFAULT_STOI = None    # encode 省略 stoi 参数时的默认词表（首次使用时构建并缓存）


def _default_stoi():
    """返回默认语料的完整词表（含特殊 token），首次调用时构建并缓存。"""
    global _DEFAULT_STOI
    if _DEFAULT_STOI is None:
        _DEFAULT_STOI = load_corpus()[2]
    return _DEFAULT_STOI


def encode(text, stoi=None):
    """把文本编码为 token id 序列：先匹配特殊 token，未匹配处逐字符查 stoi。

    特殊 token（如 <五绝>）在文本中占 4 字符、在 token 流中计 1 个 id；
    `<` 不出现在纯文本字符集中，故按 SPECIAL_TOKENS 逐个前缀匹配无歧义。
    词表外字符跳过（OOV skip，与验证集编码口径一致）；stoi 省略时使用默认语料词表。
    返回与现有 data 同 dtype 的 onp.int64 数组。
    """
    if stoi is None:
        stoi = _default_stoi()
    specials = [(tok, stoi[tok]) for tok in SPECIAL_TOKENS if tok in stoi]
    ids = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "<":                 # 特殊 token 均以 < 起头，先做廉价前缀判定
            for tok, tid in specials:
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    break
            else:
                i += 1                     # 非特殊 token 的 < → 按词表外字符跳过
            continue
        c = text[i]
        if c in stoi:
            ids.append(stoi[c])
        i += 1
    return onp.array(ids, dtype=onp.int64)


def load_corpus(data_path=None):
    """读语料文本，建「纯文本字符 + 诗体特殊 token」词表，整段编码成 id 数组。

    chars 由「剥去特殊 token 后的纯文本」构造：语料每首诗首行前插有诗体标签
    （如 <五绝>），若直接对原文取字符集会把 < > 收进词表，故先剥离标签再取字符集。
    stoi 先放纯文本字符（id 0..len(chars)-1），再把 SPECIAL_TOKENS 追加在末尾。

    返回 (text, chars, stoi, itos, data, special_ids)：
      text        —— 语料原文（含诗体标签）。
      chars       —— 纯文本字符表（剥去标签后的字符集，按编码排序）。
      stoi/itos   —— 含特殊 token 的完整映射（长度 len(chars) + 5）。
      data        —— 全文 token 序列（每个特殊 token 计 1 个 id，训练切片用）。
      special_ids —— 5 个特殊 token 的 id 列表（顺序与 SPECIAL_TOKENS 一致）。
    """
    with open(data_path or DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(_strip_special_tokens(text)))     # 纯文本字符（不含标签字符 < >）
    stoi = {c: i for i, c in enumerate(chars)}           # 字符 -> id（0..len(chars)-1）
    base = len(chars)
    special_ids = []
    for k, tok in enumerate(SPECIAL_TOKENS):             # 特殊 token 追加在末尾
        stoi[tok] = base + k
        special_ids.append(base + k)
    itos = {i: c for c, i in stoi.items()}               # id -> 字符（stoi 的逆映射）
    data = encode(text, stoi)                            # 含特殊 token id 的完整序列
    return text, chars, stoi, itos, data, special_ids


def load_val(val_path, stoi):
    """读独立验证集并按训练词表编码；词表外字符（OOV）跳过——val 只评估模型见过的字。

    编码口径与 encode 一致：先匹配诗体特殊 token，未匹配处逐字符查 stoi，
    无法匹配的字符跳过。文件缺失时返回 None（关闭 val 监控）。
    """
    if not os.path.exists(val_path):
        return None
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    return encode(text, stoi)


def get_batch(data, batch_size, ctx_len):
    """随机取 batch_size 个起点，各截 ctx_len 长；y 是 x 右移一位（预测下一个字）。

    自回归数据格式：位置 t 的输入是 data[i+t]，目标是 data[i+t+1]，
    故 y[:, :-1] == x[:, 1:] 恒成立。返回形状均为 (batch_size, ctx_len)。
    切片在 CPU（onp）做，返回前搬到计算设备（np.asarray：CPU 零拷贝，GPU 拷入显存）。
    """
    ix = onp.random.randint(0, len(data) - ctx_len, size=batch_size)
    x = onp.stack([data[i:i + ctx_len] for i in ix])
    y = onp.stack([data[i + 1:i + ctx_len + 1] for i in ix])
    return np.asarray(x), np.asarray(y)


def load_meta(path=None):
    """载入逐位置元信息 data/meta.npz，返回原生 numpy 数组字典。

    四数组长度均等于语料 token 数 L，下标为 token 在 data 中的绝对位置：
      weights      float32 —— 该位置 token 作为 target 时的主损失权重；
      tone_labels  int8    —— 平仄标签（0=平 1=仄 2=未覆盖/非汉字）；
      rhyme_labels int16   —— 韵部 id（0=unknown，1..106=韵部）；
      zone_tags    int8    —— 位置属性（0=常态 1=韵脚 2=半句末标点 3=特殊 token 4=对仗区）。
    返回原生 numpy（onp）数组，训练期由 get_batch_ext 在 CPU 切片后再搬设备。
    """
    d = dict(onp.load(path or META_PATH))
    return {"weights": d["weights"], "tone_labels": d["tone_labels"],
            "rhyme_labels": d["rhyme_labels"], "zone_tags": d["zone_tags"]}


def derive_masks(tone_labels, rhyme_labels, weights):
    """派生辅助头逐位置掩码，返回 (tone_mask, rhyme_mask) 两个 bool 数组。

    tone_mask  = (tone_labels != 2)             覆盖「非汉字」与「韵书未见字」
    rhyme_mask = (weights == 2.0) & (rhyme_labels != 0)

    韵脚掩码仅以「离线原始 weights == 2.0」为判据，**禁止**由 zone_tags == 1 派生：
    zone_tags 优先级为「特殊 token(3) > 对仗区(4) > 韵脚(1) > 半句末标点(2) > 常态(0)」，
    颔联/颈联内的韵脚会被标为 4 而丢失监督。weights 取值仅 {1.0, 1.5, 2.0}，2.0 与韵脚
    严格一一对应。传入的 weights 必须是**未融合 duizhang_weight 的离线原始权重**；
    融合后的 w 在 duizhang_weight == 2.0 时会引入对仗区假韵脚，故取批时须先派生掩码、
    再做融合。后半项兑现「韵脚字未被韵书覆盖（rhyme_labels == 0）则跳过」。
    """
    tone_mask = tone_labels != 2
    rhyme_mask = (weights == 2.0) & (rhyme_labels != 0)
    return tone_mask, rhyme_mask


def get_batch_ext(data, meta, batch_size, ctx_len, duizhang_weight=1.0):
    """取加权训练批次 + 辅助头标签与掩码，返回 7 元组。

    返回 (x, y, w, tone_t, tone_m, rhyme_t, rhyme_m)，形状均 (batch_size, ctx_len)：
      x       = data[s : s+T]              输入；
      y       = data[s+1 : s+1+T]          目标（右移一位）；
      w       = maximum(weights[s+1:s+1+T], (zone_tags[s+1:s+1+T] == 4) * duizhang_weight)
                逐位置主损失权重：先取离线权重，再按「取最大」（非叠乘）融合对仗区权重；
      tone_t  = tone_labels[s+1 : s+1+T]   平仄标签（与 y 同起点同长度）；
      tone_m  = tone_t != 2                平仄头掩码；
      rhyme_t = rhyme_labels[s+1 : s+1+T] 韵部标签（与 y 同起点同长度）；
      rhyme_m = (weights[s+1:s+1+T] == 2.0) & (rhyme_t != 0)  韵部头掩码。

    起点 s 满足 s + 1 + T <= len(data)（断言保护）；掩码一律基于**未融合**的原始
    weights 派生后再做融合。**纯切片、零解析**：除布尔比较与 maximum 外不做任何
    Python 逐字符处理。切片在 CPU（onp）做，返回前 np.asarray 搬到计算设备
    （与 get_batch 同风格：CPU 零拷贝，GPU 拷入显存）。
    """
    T = ctx_len
    weights = meta["weights"]
    tone_labels = meta["tone_labels"]
    rhyme_labels = meta["rhyme_labels"]
    zone_tags = meta["zone_tags"]
    n = len(data)
    assert n == len(weights) == len(tone_labels) == len(rhyme_labels) == len(zone_tags), (
        f"data 与 meta 数组长度不一致：data={n} weights={len(weights)} "
        f"tone={len(tone_labels)} rhyme={len(rhyme_labels)} zone={len(zone_tags)}")
    s = onp.random.randint(0, n - T, size=batch_size)
    assert int(s.max()) + 1 + T <= n, (
        f"起点越界：max(s)+1+T={int(s.max()) + 1 + T} > len(data)={n}")
    x = onp.stack([data[i:i + T] for i in s])
    y = onp.stack([data[i + 1:i + 1 + T] for i in s])
    w_raw = onp.stack([weights[i + 1:i + 1 + T] for i in s])       # 未融合的原始权重
    zone = onp.stack([zone_tags[i + 1:i + 1 + T] for i in s])
    tone_t = onp.stack([tone_labels[i + 1:i + 1 + T] for i in s])
    rhyme_t = onp.stack([rhyme_labels[i + 1:i + 1 + T] for i in s])
    # 先派生掩码（基于未融合 w_raw），再做对仗区融合——顺序不可颠倒
    tone_m, rhyme_m = derive_masks(tone_t, rhyme_t, w_raw)
    w = onp.maximum(w_raw, (zone == 4) * duizhang_weight)
    assert x.shape == y.shape == (batch_size, T), f"x/y 形状异常：{x.shape}/{y.shape}"
    assert w.shape == tone_t.shape == tone_m.shape == (batch_size, T), \
        f"w/tone_t/tone_m 形状异常：{w.shape}/{tone_t.shape}/{tone_m.shape}"
    assert rhyme_t.shape == rhyme_m.shape == (batch_size, T), \
        f"rhyme_t/rhyme_m 形状异常：{rhyme_t.shape}/{rhyme_m.shape}"
    assert onp.array_equal(x[:, 1:], y[:, :-1]), "x/y 自回归对齐错误：x[:,1:] 应等于 y[:,:-1]"
    return (np.asarray(x), np.asarray(y), np.asarray(w), np.asarray(tone_t),
            np.asarray(tone_m), np.asarray(rhyme_t), np.asarray(rhyme_m))


class AdamW:
    """AdamW 优化器：m 稳方向、v 定步长、权重衰减与 Adam 解耦（防过拟合）。

    每个参数 p 独立维护两份状态，更新规则（t 为已走步数）：
      m = b1*m + (1-b1)*g                一阶矩：梯度的滑动平均（惯性，平滑抖动）
      v = b2*v + (1-b2)*g^2              二阶矩：梯度平方的滑动平均（该参数梯度量级）
      m_hat = m/(1-b1^t), v_hat = v/(1-b2^t)   偏差校正：头几步 m/v 从 0 起步偏小，放大校准
      p -= lr * m_hat / (sqrt(v_hat)+eps)      自适应步长：梯度大 => v 大 => 步子小
      p -= lr * wd * p                           解耦权重衰减：把参数整体往 0 拉一点
    """

    def __init__(self, model, lr=3e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1):
        self.model = model
        self.lr = lr
        self.b1, self.b2 = betas          # 一/二阶矩的滑动平均系数（记忆衰减速度）
        self.eps = eps                    # 防除零小常数（分母 sqrt(v)+eps）
        self.wd = weight_decay            # 权重衰减强度（与 lr 相乘后使用）
        self.t = 0                        # 已更新步数（偏差校正用）
        # 每个参数配一份 m/v 状态，形状与参数一致，零起步
        self.m = {name: np.zeros_like(p) for name, p in model._named_params()}
        self.v = {name: np.zeros_like(p) for name, p in model._named_params()}

    def step(self):
        """用当前梯度更新全部参数（调用前须先 backward 且梯度有效）。"""
        self.t += 1
        for (name, p), (_, g) in zip(self.model._named_params(), self.model._named_grads()):
            m = self.m[name]
            v = self.v[name]
            m *= self.b1
            m += (1 - self.b1) * g        # 更新一阶矩：保留旧方向 + 吸收新梯度
            v *= self.b2
            v += (1 - self.b2) * g * g    # 更新二阶矩：保留旧量级 + 吸收新梯度平方
            m_hat = m / (1 - self.b1 ** self.t)   # 偏差校正（指数衰减使初始偏小放大）
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # 自适应步长更新
            p -= self.lr * self.wd * p    # 解耦权重衰减（单独一步，独立于梯度）


def get_lr(step, anneal_total, warmup_steps, peak_lr):
    """学习率调度：前 warmup_steps 线性热身升到峰值，之后余弦退火降到峰值 10%。

    热身期：lr = 峰值 * (step+1) / warmup_steps（0 -> 峰值，防开头冲飞）
    退火期：lr = 峰值 * 0.5 * (1+cos(pi*进度))，进度 0->1（峰值 -> 峰值*0.1）
    anneal_total 是全局退火终点（与单段长度解耦）：分段续训时各段都按
    「全局步数 step + anneal_total」回放同一余弦，曲线天然连续；step >=
    anneal_total 后进度封顶 1，lr 恒为峰值 10%（小步精修，不再下降）。
    """
    anneal_total = max(anneal_total, warmup_steps + 1)   # 防退火区间为负/零（除零）
    if step < warmup_steps:
        # 热身段：从 0 线性爬到峰值（step 从 0 数起，故 +1 使第 0 步非零）
        return peak_lr * (step + 1) / warmup_steps
    # 退火段：进度 p 从 0（热身结束）线性走向 1（退火终点）
    p = (step - warmup_steps) / (anneal_total - warmup_steps)
    p = min(p, 1.0)                                # 防越界：p 封顶 1（退火后冻结）
    # 余弦退火：cos 从 1 到 -1，lr 从峰值平滑降到峰值*0.1（保留小步精修能力）
    return peak_lr * 0.5 * (1 + np.cos(np.pi * p)) * 0.9 + peak_lr * 0.1


def _demo_text(model, stoi, itos, prompts=None, new_tokens=16, enforce_meter=False):
    """对多条 prompt 续写并拼成一行；enforce_meter=True 时叠加句长/标点约束处理器。

    默认 False：展示模型**未经约束**的原生输出，用于目检格律是否已内化；
    需对照「约束关」效果时显式传 True。
    """
    if prompts is None:
        prompts = DEMO_PROMPTS
    punct_ids = [stoi[c] for c in ("，", "。") if c in stoi]
    # 非汉字 token（换行、其他标点等）：半句内一律禁绝，且不计入句长额度
    non_hanzi_ids = [i for c, i in stoi.items() if not ("\u4e00" <= c <= "\u9fff")]
    newline_id = stoi.get("\n")
    outs = []
    for prompt in prompts:
        if not all(c in stoi for c in prompt):
            continue
        processor = None
        if enforce_meter and len(punct_ids) == 2:   # 每条独立处理器
            processor = make_meter_processor(punct_ids, stoi["，"], stoi["。"],
                                             non_hanzi_ids=non_hanzi_ids,
                                             newline_id=newline_id)
        pid = np.array([[stoi[c] for c in prompt]], dtype=np.int64)
        out = model.generate(pid, new_tokens, temperature=1.0, top_k=5,
                             logits_processor=processor)
        outs.append(prompt + "".join(itos[int(i)] for i in out[0][len(prompt):]))
    return " | ".join(outs) if outs else "(全部 prompt 含词表外字符，已跳过)"


def estimate_val_loss(model, val_data, ctx_len, n_batch=4, seed=1234):
    """val loss：在独立验证集上固定抽 n_batch 个随机窗口求平均交叉熵。

    评估用独立 RandomState（默认 seed 固定）→ 不推进训练抽题用的全局 RNG，
    同 seed 训练可复现性不被破坏，且历次 val loss 可比。语料 2.5 万字符以上，
    4 个 batch×ctx_len 窗口的均值抖动已足够分辨趋势。
    """
    rng = onp.random.RandomState(seed)              # 独立随机源（评估专用）
    losses = []
    for _ in range(n_batch):
        i = rng.randint(0, len(val_data) - ctx_len)
        x = val_data[i:i + ctx_len][None, :]        # 补 batch 维 -> (1, ctx_len)
        y = val_data[i + 1:i + ctx_len + 1][None, :]
        logits = model.forward(np.asarray(x))
        losses.append(model.loss(logits, np.asarray(y)).item())
    return float(onp.mean(losses))


def _aux_accuracy(model, tone_t, tone_m, rhyme_t, rhyme_m):
    """按掩码统计辅助头 argmax 准确率，返回 (tone_acc, rhyme_acc)，各项可为 None。

    对启用且当前批次含有效位（mask 求和 > 0）的头，取 argmax(预测) 与 target 逐位比对、
    仅在 mask 为真处求均值；未启用的头（对应 logits 为 None）或无有效位的头返回 None。
    两头皆为 None 时整体返回 None（供打印端输出「aux -」）。

    设备搬运一律经 as_numpy 显式拷回 CPU 后再做 numpy 统计：logits 与标签/掩码在
    GPU(cupy) 后端下为设备数组，禁止用 onp.asarray 隐式转换（cupy 会直接抛
    TypeError）；CPU 下 as_numpy 为零拷贝透传，数值结果不变。
    """
    tone_acc = None
    if getattr(model, "tone_logits", None) is not None:
        pred = as_numpy(model.tone_logits).argmax(axis=-1)
        m = as_numpy(tone_m).astype(bool)
        if m.any():
            tone_acc = float((pred[m] == as_numpy(tone_t)[m]).mean())
    rhyme_acc = None
    if getattr(model, "rhyme_logits", None) is not None:
        pred = as_numpy(model.rhyme_logits).argmax(axis=-1)
        m = as_numpy(rhyme_m).astype(bool)
        if m.any():
            rhyme_acc = float((pred[m] == as_numpy(rhyme_t)[m]).mean())
    if tone_acc is None and rhyme_acc is None:
        return None
    return tone_acc, rhyme_acc


# ───────────────────────── checkpoint（真续训格式）─────────────────────────

def _savez_atomic(path, payload):
    """原子落盘：先写同目录临时文件，成功后 os.replace 原子替换。

    中断只残留 .tmp 且会被清理，已存在的旧档永不被半截文件破坏。
    """
    tmp = f"{path}.tmp{os.getpid()}.npz"   # .npz 结尾：np.savez 不再自动补后缀
    try:
        onp.savez(tmp, **payload)
        os.replace(tmp, path)                       # 同盘原子改名（写盘完成才替换）
    finally:
        if os.path.exists(tmp):                     # 失败路径清理临时文件
            os.remove(tmp)


def save_checkpoint(path, model, opt, chars, anneal_total, warmup_steps, peak_lr):
    """存真续训 checkpoint：模型参数 + 优化器 m/v/t + 词表快照 + 调度签名 + 版本。

    落盘一律用原生 numpy（GPU 数组经 as_numpy 显式拷回 CPU），保证文件跨设备可读；
    m/v 按「_opt.m.{参数名}」平铺命名；chars 存字符串数组作词表快照。
    """
    payload = {"_ckpt_version": onp.asarray(CKPT_VERSION),          # 格式版本
               "_chars": onp.asarray(list(chars))}                  # 词表快照
    payload.update({name: as_numpy(a) for name, a in model.dump_params().items()})
    payload["_opt.t"] = onp.asarray(opt.t)                          # 已训步数（续训起点）
    payload.update({f"_opt.m.{name}": as_numpy(a) for name, a in opt.m.items()})
    payload.update({f"_opt.v.{name}": as_numpy(a) for name, a in opt.v.items()})
    payload["_sched.anneal_total"] = onp.asarray(anneal_total)      # 调度签名
    payload["_sched.warmup_steps"] = onp.asarray(warmup_steps)
    payload["_sched.peak_lr"] = onp.asarray(peak_lr)
    _savez_atomic(path, payload)


def load_checkpoint(path, model, opt=None, expect_chars=None, expect_sched=None):
    """载入真续训 checkpoint，返回已训步数 t（= 存档 opt.t，续训的权威起点）。

    拒载校验（防静默错位）：
      - 旧格式（三百首时代，仅模型参数无元数据）→ ValueError，提示从头训
      - 版本号 != CKPT_VERSION → ValueError
      - expect_chars 传入且与档内词表不一致（换了语料）→ ValueError
      - expect_sched 传入：warmup_steps/peak_lr 必须相等；anneal_total 若给定
        且小于档内值（退火终点缩短）→ ValueError（只允许延长）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    d = dict(onp.load(path))
    if "_ckpt_version" not in d:
        raise ValueError(f"{path} 为旧格式（仅模型参数，无 m/v/词表/调度元数据），"
                         "已拒绝续训；如需继续请从头训")
    if int(d["_ckpt_version"]) != CKPT_VERSION:
        raise ValueError(f"{path} 的 checkpoint 版本 {int(d['_ckpt_version'])} "
                         f"!= 当前 {CKPT_VERSION}，拒绝续训")
    if expect_chars is not None:
        if list(d["_chars"]) != list(expect_chars):
            raise ValueError("checkpoint 词表与当前语料不一致（换语料了？），拒绝续训")
    if expect_sched is not None:
        saved = dict(anneal_total=int(d["_sched.anneal_total"]),
                     warmup_steps=int(d["_sched.warmup_steps"]),
                     peak_lr=float(d["_sched.peak_lr"]))
        for key in ("warmup_steps", "peak_lr"):
            if key in expect_sched and abs(saved[key] - expect_sched[key]) > 1e-12:
                raise ValueError(f"调度签名 {key} 不一致：档内 {saved[key]} "
                                 f"vs 传入 {expect_sched[key]}，拒绝续训")
        if ("anneal_total" in expect_sched
                and expect_sched["anneal_total"] < saved["anneal_total"]):
            raise ValueError(f"anneal_total {expect_sched['anneal_total']} < 档内 "
                             f"{saved['anneal_total']}（退火终点只可延长不可缩短），拒绝续训")
    # 参数写回（非 "_" 前缀键即模型参数）；m/v/t 恢复
    params = {k: v for k, v in d.items() if not k.startswith("_")}
    model.load_params(params)
    t = int(d["_opt.t"])
    if opt is not None:
        for name, _ in model._named_params():
            opt.m[name][...] = np.asarray(d[f"_opt.m.{name}"])   # np.asarray：CPU 零拷贝 / GPU 显式拷入
            opt.v[name][...] = np.asarray(d[f"_opt.v.{name}"])
        opt.t = t
    return t


def load_params_partial(path, model):
    """按名载入 checkpoint 中**模型已有**的参数，返回缺失（新增）参数个数。

    跨里程碑续训专用（M1 -> M2 增挂辅助头等）：仅写入双方共有的参数名，模型多出的
    新参数保留自身初始化值。**严禁**用 load_checkpoint 承担此职责——它逐名读取
    `_opt.m.{name}`，M1 档缺 `_opt.m.tone_head.*` 键会直接 KeyError（且调度签名与
    词表快照也不匹配）。本函数只读非 "_" 前缀键（即模型参数），天然跳过
    `_opt.*` / `_sched.*` / `_chars` / `_ckpt_version` 等元数据键。
    形状不一致时断言失败；载入后打印缺失参数个数。
    """
    d = dict(onp.load(path))
    missing = 0
    for name, p in model._named_params():
        if name in d:
            arr = onp.asarray(d[name])
            assert tuple(arr.shape) == tuple(p.shape), (
                f"参数 {name} 形状不一致：档内 {tuple(arr.shape)} vs 模型 {tuple(p.shape)}")
            p[...] = np.asarray(arr)            # np.asarray：CPU 零拷贝 / GPU 显式拷入
        else:
            missing += 1
    print(f"按名载参 {path}：缺失 {missing} 个新增参数，按初始化保留")
    return missing


# ─────────────────────────────── 主训练循环 ────────────────────────────────

def train(max_steps=3000, batch_size=8, ctx_len=64,
          warmup_steps=200, peak_lr=3e-4, print_every=200,
          demo_every=500, seed=0, resume_path=None, start_step=0,
          d_model=64, n_head=4, n_layer=2,
          anneal_total=None, save_every=None, demo_prompts=None,
          data_path=None, val_path=None, ckpt_path=None, val_n_batch=16,
          base_path=None, n_rhyme=0, use_tone=False, lam_tone=0.3, lam_rhyme=0.5,
          rhyme_classes=107, meta_path=None, duizhang_weight=1.0, lr_scale=1.0,
          l_base_path=None):
    """主训练循环：抽题（含权重/辅助标签） -> 前向 -> 加权损失+辅助损失 -> 反向 -> AdamW。

    每 print_every 步为监控节拍：算 val loss 与 val_ratio（= val_loss / L_base）并打印
    step/lr/loss/val/val_ratio/aux_acc；每 demo_every 步对多条 prompt 续写目检生成进化；
    正常结束存终档 ckpt_path；save_every 间隔额外存中间档；KeyboardInterrupt/异常时
    finally 先落盘断点再退出（中断兜底）。

    分段/里程碑接线：
      - base_path 非 None：按名载入共有参数（load_params_partial），模型新增参数
        保留初始化；**跨里程碑严禁走 load_checkpoint**（M1 档缺 _opt.m.tone_head.* 键）。
      - n_rhyme>0 / use_tone：挂载韵部头/平仄头；n_rhyme>0 时须等于 rhyme_classes。
      - meta_path：逐位置元信息（默认 data/meta.npz），供加权主损失与辅助标签切片；
        必须与语料同长，长度不符直接抛 AssertionError（不退化）；自定义语料须显式传入
        与之匹配的 meta_path，否则须重建 meta（tools/build_corpus.py）。
      - duizhang_weight：对仗区权重（取最大融合，M3 起设 1.5）。
      - lr_scale：运行期学习率标量，乘在 get_lr 之后（遗忘红线的折半开关）。
      - l_base_path：遗忘基线 L_base 的锚模型（默认 model-ext.npz），仅 val 可用时计算。

    续训：resume_path 指向真格式 checkpoint——载入参数/m/v 与词表、调度签名校验，
    已训步数取档内 opt.t（自动衔接）。anneal_total 为全局退火终点（默认 = 本段起点 +
    max_steps）。save_every 须显式等于 print_every（保证每个监控点都有可回滚中间档）。
    """
    if n_rhyme > 0:
        assert n_rhyme == rhyme_classes, (
            f"n_rhyme（{n_rhyme}）必须与 rhyme_classes（{rhyme_classes}）相等："
            "二者同源，防韵部类别数漂移")
    if save_every is not None and save_every != print_every:
        raise ValueError(f"save_every（{save_every}）必须等于 print_every（{print_every}），"
                         "以保证每个监控点都有中间档可回滚")
    onp.random.seed(seed)   # 抽题 RNG 恒在 CPU（onp）：与计算后端无关，保证同 seed 可复现
    np.random.seed(seed)    # 参数初始化 RNG：CPU 下 np==onp（同上），GPU 下为 cupy 随机源
    text, chars, stoi, itos, data, special_ids = load_corpus(data_path)   # 训练语料
    val_data = load_val(val_path or VAL_PATH, stoi)                # 验证集（无则 None）
    # 逐位置元信息（默认 data/meta.npz）：必须与语料同源同长，长度不符直接断言失败，不退化
    meta = load_meta(meta_path)
    assert len(data) == len(meta["weights"]) == len(meta["tone_labels"]) == \
        len(meta["rhyme_labels"]) == len(meta["zone_tags"]), (
            f"语料与 meta 长度不一致：语料={len(data)} weights={len(meta['weights'])} "
            f"tone={len(meta['tone_labels'])} rhyme={len(meta['rhyme_labels'])} "
            f"zone={len(meta['zone_tags'])}。语料与 meta 必须同源；自定义语料请显式传入"
            "与之匹配的 meta_path，否则请重建 meta（tools/build_corpus.py）")
    model = GPT(vocab_size=len(stoi), d_model=d_model, n_head=n_head,
                n_layer=n_layer, ctx_len=ctx_len, n_rhyme=n_rhyme, use_tone=use_tone)
    if base_path is not None:                      # 按名载参（跨里程碑续训，严禁 load_checkpoint）
        load_params_partial(base_path, model)
    opt = AdamW(model, lr=peak_lr)                 # fresh AdamW：不继承任何优化器状态
    # ── 续训：真格式校验 + 读档；起点取档内 opt.t（权威）──
    start = start_step
    if resume_path is not None:
        expect_sched = dict(warmup_steps=warmup_steps, peak_lr=peak_lr)
        if anneal_total is not None:
            expect_sched["anneal_total"] = anneal_total
        start = load_checkpoint(resume_path, model, opt,
                                expect_chars=chars, expect_sched=expect_sched)
        print(f"续训：已从 {resume_path} 载入（checkpoint 已训 {start} 步）")
    anneal_total = anneal_total if anneal_total is not None else start + max_steps
    n_param = sum(p.size for _, p in model._named_params())
    print(f"训练 {max_steps} 步（起点 {start}，退火终点 {anneal_total}）| 词表 {len(stoi)} "
          f"| d{d_model}/h{n_head}/L{n_layer} | 参数 {n_param:,}"
          + (" | 启用 val loss" if val_data is not None else "")
          + (f" | 韵部头 {n_rhyme}" if n_rhyme > 0 else "")
          + (" | 平仄头" if use_tone else ""))
    # 遗忘基线 L_base：在扩词表迁移档 model-ext.npz 上、同 ctx/val 口径算一次 val loss，
    # 使各里程碑的 val_ratio 可比（不可用旧 model.npz 数值充数：词表已 8196 → 8201）
    l_base = None
    if val_data is not None:
        l_base_path = l_base_path or L_BASE_PATH
        if os.path.exists(l_base_path):
            anchor = GPT(vocab_size=len(stoi), d_model=d_model, n_head=n_head,
                         n_layer=n_layer, ctx_len=ctx_len)          # 同架构临时锚模型
            load_params_partial(l_base_path, anchor)
            l_base = estimate_val_loss(anchor, val_data, ctx_len, val_n_batch)
            del anchor
            print(f"L_base（{l_base_path}）= {l_base:.4f}")
        else:
            print(f"L_base：未找到锚模型 {l_base_path}，本次不输出 val_ratio")
    done = False                                 # 正常完成标记：finally 据此判断是否兜底存档
    over_line = 0                                # val_ratio 连续越线计数（遗忘红线预案 1）
    try:
        for i in range(max_steps):
            step = start + i                      # 全局步数：续训从档内步数接着数
            # ① 抽题：主损失权重 + 平仄/韵部标签与掩码（纯切片，零解析）
            x, y, w, tone_t, tone_m, rhyme_t, rhyme_m = get_batch_ext(
                data, meta, batch_size, ctx_len, duizhang_weight)
            opt.lr = get_lr(step, anneal_total, warmup_steps, peak_lr) * lr_scale  # ② lr（含预案）
            logits = model.forward(x)                                 # ③ 前向
            loss = model.loss(logits, y, w)                           # ④ 加权主损失
            if n_rhyme > 0 or use_tone:                               # ⑤ 辅助损失（仅启用时）
                model.aux_loss(tone_t, tone_m, rhyme_t, rhyme_m, lam_tone, lam_rhyme)
            model.zero_grad()                                         # ⑥ 清账
            model.backward()                                          # ⑦ 反向（含辅助头回传）
            opt.step()                                                # ⑧ AdamW 改参数
            monitor = (step % print_every == 0) or (i == max_steps - 1)
            v = None
            val_ratio = None
            acc = None
            if monitor:
                # 先取辅助头准确率：estimate_val_loss 会前向验证窗口并覆写 tone_logits/
                # rhyme_logits 缓存，故须在算 val 之前基于本训练批次的前向缓存取值
                acc = _aux_accuracy(model, tone_t, tone_m, rhyme_t, rhyme_m)
            if monitor and val_data is not None:                      # 先算 val_ratio（存档标记需要）
                v = estimate_val_loss(model, val_data, ctx_len, val_n_batch)
                if l_base is not None:
                    val_ratio = v / l_base
            if val_ratio is not None:                                 # 预案 1：连续 2 次越线 → lr 折半
                over_line = over_line + 1 if val_ratio > 1.05 else 0
                if over_line >= 2:
                    lr_scale *= 0.5
                    print(f"  [预案] val_ratio 连续 2 次 > 1.05，lr_scale 折半为 {lr_scale:g}")
                    over_line = 0
            if save_every and step and step % save_every == 0:        # 中间存档（带合格回滚点标记）
                mid = _mid_ckpt_path(ckpt_path or CKPT_PATH, step)
                save_checkpoint(mid, model, opt, chars, anneal_total, warmup_steps, peak_lr)
                if val_ratio is not None and val_ratio <= 1.05:
                    print(f"  [存档] {mid}（合格回滚点 val_ratio={val_ratio:.4f}）")
                else:
                    shown = "无" if val_ratio is None else f"{val_ratio:.4f}"
                    print(f"  [存档] {mid}（不合格 val_ratio={shown}）")
            if monitor:
                line = f"step {step:5d}  lr {opt.lr:.2e}  loss {loss.item():.4f}"
                if v is not None:                                     # 每监控节拍报 val loss
                    line += f"  val {v:.4f}"
                if val_ratio is not None:
                    line += f"  val_ratio {val_ratio:.4f}"
                if acc is None:
                    line += "  aux -"
                else:
                    ta, ra = acc
                    parts = ([f"t{ta:.3f}"] if ta is not None else []) \
                        + ([f"r{ra:.3f}"] if ra is not None else [])
                    line += "  aux " + "/".join(parts)
                print(line)
            if demo_every and step and step % demo_every == 0:
                print(f"  生成: {_demo_text(model, stoi, itos, demo_prompts)}")
        done = True
        save_checkpoint(ckpt_path or CKPT_PATH, model, opt, chars,
                        anneal_total, warmup_steps, peak_lr)
        print(f"完成，已保存 {ckpt_path or CKPT_PATH}")
    finally:
        # 中断兜底：没正常跑完（Ctrl-C/异常）也先落盘，保证下次能续训
        if not done:
            save_checkpoint(ckpt_path or CKPT_PATH, model, opt, chars,
                            anneal_total, warmup_steps, peak_lr)
            print(f"中断兜底：已保存断点 {ckpt_path or CKPT_PATH}"
                  "（续训请 resume_path 指向该文件）")


def _mid_ckpt_path(ckpt_path, step):
    """中间档路径：在文件名主体插入步数，如 model.npz -> model-2500.npz。"""
    root, ext = os.path.splitext(ckpt_path)
    return f"{root}-{step}{ext}"


if __name__ == "__main__":
    train()
