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

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "corpus.txt")   # 训练语料（全唐诗简体净流）
VAL_PATH = os.path.join(BASE, "data", "val.txt")       # 独立验证集（400 首，与训练同构）
CKPT_PATH = os.path.join(BASE, "model.npz")            # 训练/续训存档位
CKPT_VERSION = 2                                       # checkpoint 格式版本：2 = 真续训
# demo 默认 prompts：程序化校验 ⊆ 词表，不全即跳过（新语料词表大，防 KeyError）
DEMO_PROMPTS = ["床前明月光", "白日依山尽", "春眠不觉晓", "海上生明月", "大漠孤烟直"]


def load_corpus(data_path=None):
    """读语料文本，建字符级分词：stoi（字符→id）与 itos（id→字符），整段转 id 数组。

    返回 (text, chars, stoi, itos, data)：data 是全文的 token id 序列，
    训练时按随机窗口切片（见 get_batch），生成时用 itos 把 id 还原成汉字。
    """
    with open(data_path or DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))              # 全部唯一字符（按编码排序保证确定性）
    stoi = {c: i for i, c in enumerate(chars)}   # 字符 -> id
    itos = {i: c for c, i in stoi.items()}       # id -> 字符（stoi 的逆映射）
    data = onp.array([stoi[c] for c in text], dtype=onp.int64)   # 语料留 CPU（onp）
    return text, chars, stoi, itos, data


def load_val(val_path, stoi):
    """读独立验证集并按训练词表编码；词表外字符（OOV）跳过——val 只评估模型见过的字。

    验证集同构于训练（一联一行），但不参与训练；文件缺失时返回 None（关闭 val 监控）。
    """
    if not os.path.exists(val_path):
        return None
    with open(val_path, encoding="utf-8") as f:
        text = f.read()
    ids = [stoi[c] for c in text if c in stoi]
    return onp.array(ids, dtype=onp.int64)


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


def _demo_text(model, stoi, itos, prompts=None, new_tokens=16):
    """对多条 prompt 各续写一段，拼接返回；prompt 有字符不在词表即跳过该条（防 KeyError）。

    prompts 为 None 时用模块级 DEMO_PROMPTS；词表变化（换语料）后旧 prompt 可能
    含生僻新字，程序化校验保证了永不因缺字崩溃。
    """
    if prompts is None:
        prompts = DEMO_PROMPTS
    outs = []
    for prompt in prompts:
        if not all(c in stoi for c in prompt):
            continue                                # 词表外字符 → 本条跳过
        pid = np.array([[stoi[c] for c in prompt]], dtype=np.int64)   # prompt 编码成 id
        out = model.generate(pid, new_tokens, temperature=1.0, top_k=5)  # KV Cache 增量生成
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


# ─────────────────────────────── 主训练循环 ────────────────────────────────

def train(max_steps=3000, batch_size=8, ctx_len=64,
          warmup_steps=200, peak_lr=3e-4, print_every=200,
          demo_every=500, seed=0, resume_path=None, start_step=0,
          d_model=64, n_head=4, n_layer=2,
          anneal_total=None, save_every=None, demo_prompts=None,
          data_path=None, val_path=None, ckpt_path=None, val_n_batch=4):
    """主训练循环：抽题 -> 前向 -> loss -> 反向 -> AdamW 更新（学习率走山坡）。

    每 print_every 步打印 loss（+ val loss）；每 demo_every 步对多条 prompt 续写
    目检生成进化；正常结束存终档 ckpt_path；save_every 间隔额外存 model-{step}.npz
    中间档；KeyboardInterrupt/异常时 finally 先落盘断点再退出（中断兜底）。

    续训：resume_path 指向真格式 checkpoint——载入参数/m/v 与词表、调度签名校验，
    已训步数取档内 opt.t（自动衔接，不依赖调用方数步）。anneal_total 为全局退火
    终点（默认 = 本段起点 + max_steps，兼容旧单段语义；S4/S5 分段续训请显式传
    全局终点，如 30000，使各段 lr 曲线连续）。
    """
    onp.random.seed(seed)   # 抽题 RNG 恒在 CPU（onp）：与计算后端无关，保证同 seed 可复现
    np.random.seed(seed)    # 参数初始化 RNG：CPU 下 np==onp（同上），GPU 下为 cupy 随机源
    text, chars, stoi, itos, data = load_corpus(data_path)         # 训练语料
    val_data = load_val(val_path or VAL_PATH, stoi)                # 验证集（无则 None）
    model = GPT(vocab_size=len(chars), d_model=d_model, n_head=n_head,
                n_layer=n_layer, ctx_len=ctx_len)
    opt = AdamW(model, lr=peak_lr)                 # AdamW 的记忆按 lr 峰值初始化
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
    print(f"训练 {max_steps} 步（起点 {start}，退火终点 {anneal_total}）| 词表 {len(chars)} "
          f"| d{d_model}/h{n_head}/L{n_layer} | 参数 {n_param:,}"
          + (" | 启用 val loss" if val_data is not None else ""))
    done = False                                 # 正常完成标记：finally 据此判断是否兜底存档
    try:
        for i in range(max_steps):
            step = start + i                      # 全局步数：续训从档内步数接着数
            opt.lr = get_lr(step, anneal_total, warmup_steps, peak_lr)  # 全局调度回放
            x, y = get_batch(data, batch_size, ctx_len)               # ① 抽题
            logits = model.forward(x)                                 # ② 做题（前向）
            loss = model.loss(logits, y)                              # ③ 判卷
            model.zero_grad()                                         # ④ 清账
            model.backward()                                          # ⑤ 追责到每层
            opt.step()                                                # ⑥ AdamW 改参数
            if save_every and step and step % save_every == 0:        # 中间存档（可断点续）
                mid = _mid_ckpt_path(ckpt_path or CKPT_PATH, step)
                save_checkpoint(mid, model, opt, chars, anneal_total, warmup_steps, peak_lr)
                print(f"  [存档] 已保存中间档 {mid}")
            if step % print_every == 0 or i == max_steps - 1:
                line = f"step {step:5d}  lr {opt.lr:.2e}  loss {loss.item():.4f}"
                if val_data is not None:                              # 每打印间隔报 val loss
                    v = estimate_val_loss(model, val_data, ctx_len, val_n_batch)
                    line += f"  val {v:.4f}"
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
