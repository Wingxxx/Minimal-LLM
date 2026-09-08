"""训练循环：数据加载 + char 分词 + AdamW + 训练 + 保存 model.npz

GPU 加速：设环境变量 MINIMAL_GPU=1（且已安装 cupy）即全链路跑 GPU，
否则默认纯 numpy/CPU。计算后端统一由 model.backend 提供。
"""
import os

# 计算后端：默认 numpy(CPU)；MINIMAL_GPU=1 时 np=cupy(GPU)。
# onp 恒为原生 numpy：语料处理留在 CPU（数据量小、切片快），进模型前再搬设备
from model.backend import np, onp
from model.gpt import GPT

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "corpus.txt")
CKPT_PATH = os.path.join(BASE, "model.npz")


def load_corpus():
    """读语料文本，建字符级分词：stoi（字符→id）与 itos（id→字符），整段转 id 数组。

    返回 (text, chars, stoi, itos, data)：data 是全文的 token id 序列，
    训练时按随机窗口切片（见 get_batch），生成时用 itos 把 id 还原成汉字。
    """
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))              # 全部唯一字符（按编码排序保证确定性）
    stoi = {c: i for i, c in enumerate(chars)}   # 字符 -> id
    itos = {i: c for c, i in stoi.items()}       # id -> 字符（stoi 的逆映射）
    data = onp.array([stoi[c] for c in text], dtype=onp.int64)   # 语料留 CPU（onp）
    return text, chars, stoi, itos, data


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


def get_lr(step, max_steps, warmup_steps, peak_lr):
    """学习率调度：前 warmup_steps 线性热身升到峰值，之后余弦退火降到峰值 10%。

    热身期：lr = 峰值 * (step+1) / warmup_steps（0 -> 峰值，防开头冲飞）
    退火期：lr = 峰值 * 0.5 * (1+cos(pi*进度))，进度 0->1（峰值 -> 峰值*0.1）
    曲线呈"山坡"：先缓加速、中段高峰、后期小步精修。
    """
    if step < warmup_steps:
        # 热身段：从 0 线性爬到峰值（step 从 0 数起，故 +1 使第 0 步非零）
        return peak_lr * (step + 1) / warmup_steps
    # 退火段：进度 p 从 0（热身结束）线性走向 1（训练结束）
    p = (step - warmup_steps) / (max_steps - warmup_steps)
    p = min(p, 1.0)                                # 防越界：p 封顶 1
    # 余弦退火：cos 从 1 到 -1，lr 从峰值平滑降到峰值*0.1（保留小步精修能力）
    return peak_lr * 0.5 * (1 + np.cos(np.pi * p)) * 0.9 + peak_lr * 0.1


def _demo_text(model, stoi, itos, prompt="床前明月光", new_tokens=16):
    """用当前模型对固定 prompt 续写一段，看训练过程中生成质量逐步提升。"""
    pid = np.array([[stoi[c] for c in prompt]], dtype=np.int64)   # prompt 编码成 id
    out = model.generate(pid, new_tokens, temperature=1.0, top_k=5)  # KV Cache 增量生成
    return prompt + "".join(itos[int(i)] for i in out[0][len(prompt):])


def train(max_steps=3000, batch_size=8, ctx_len=64,
          warmup_steps=200, peak_lr=3e-4, print_every=200,
          demo_every=500, seed=0, resume_path=None, start_step=0):
    """主训练循环：抽题 -> 前向 -> loss -> 反向 -> AdamW 更新（学习率走山坡）。

    每 print_every 步打印一次 loss（下降即训练有效）；每 demo_every 步用当前
    模型续写一句 demo（肉眼见证从乱码到诗句的进化）；训练结束存盘 model.npz。

    续训：resume_path 传 checkpoint 路径即从旧权重接着训，start_step 是已训步数。
    续训期学习率冻结在退火终点值（峰值 10%），不再重新规划余弦曲线——否则曲线
    重映射会让 lr 跳升、破坏已学好的权重。注意：checkpoint 只存模型参数，m/v 从零重建。
    """
    onp.random.seed(seed)   # 抽题 RNG 恒在 CPU（onp）：与计算后端无关，保证同 seed 可复现
    np.random.seed(seed)    # 参数初始化 RNG：CPU 下 np==onp（同上），GPU 下为 cupy 随机源
    text, chars, stoi, itos, data = load_corpus()
    model = GPT(vocab_size=len(chars), d_model=64, n_head=4, n_layer=2, ctx_len=ctx_len)
    if resume_path is not None:                    # 续训：载入旧权重，不重新初始化
        model.load(resume_path)
        print(f"续训：已从 {resume_path} 载入（已训 {start_step} 步）")
    total = start_step + max_steps                 # 累计总步数（仅展示用；续训不再依它规划 lr）
    opt = AdamW(model, lr=peak_lr)                 # AdamW 的记忆按 lr 峰值初始化
    print(f"训练 {max_steps} 步（累计 {total}） | 词表 {len(chars)} | 参数 {sum(p.size for _, p in model._named_params()):,}")

    for i in range(max_steps):
        step = start_step + i                      # 全局步数：续训从上次终点接着数
        if resume_path is not None:
            opt.lr = peak_lr * 0.1                 # 续训：冻结在退火终点（峰值 10%），小步精修不跳升
        else:
            opt.lr = get_lr(step, total, warmup_steps, peak_lr)   # 从头训：每步取当日学习率
        x, y = get_batch(data, batch_size, ctx_len)               # ① 抽题
        logits = model.forward(x)                                 # ② 做题（前向）
        loss = model.loss(logits, y)                              # ③ 判卷
        model.zero_grad()                                         # ④ 清账
        model.backward()                                          # ⑤ 追责到每层
        opt.step()                                                # ⑥ AdamW 改参数
        if step % print_every == 0 or i == max_steps - 1:
            print(f"step {step:5d}  lr {opt.lr:.2e}  loss {loss.item():.4f}")
        if step % demo_every == 0:
            print(f"  生成: {_demo_text(model, stoi, itos)}")

    model.save(CKPT_PATH)                                         # ⑦ 存档
    print(f"完成，已保存 {CKPT_PATH}")


if __name__ == "__main__":
    train()
